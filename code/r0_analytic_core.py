# -*- coding: utf-8 -*-
"""R0 · 解析外推核对照（构想 F 最小验证，第三阶段开题实验）。

判据预注册（跑前写死）：
  三臂（最小集 h192，5 seeds，同口径 train/eval）：
    analytic  解析外推核：ẑ = 2z_t − z_{t−1} + s·tanh(r(zc))，外推算子**零可学参数**，残差 s=0.5
    koop_s05  Q4.0 GO 臂（学习线性核 + 同结构残差，a=+0.239, m=+0.077 归档 5seed）
    mlp       纯学习基线（a=+0.6669, m=+0.0920 归档 5seed）
  R0 PASS ⇔ analytic：a_mean ≤ 0.239（不劣于 koop GO 点）AND margin_mean ≥ 0.8×0.077=0.0616。
  解析可设计性对拍：外推算子 E=2I−S 特征方程 ρ²−2ρ+1=0 ⇒ 重根 ρ=1 ⇒ **公式预测 a=0**（临界）；
    实测 analytic 臂的 a 与 0 的偏差 = 残差贡献 + 有限样本误差，即为「公式 vs 实测」的第一次对拍。
  参数账本：analytic 可学参数 = 残差 MLP；koop = K(4096) + 残差 MLP。
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from q12_min_train import (  # noqa: E402
    LATENT, CTX, HIDDEN, EPOCHS, BATCH, load_latents, make_pairs, to_norm,
    margin_and_cos, twin_a, TRAIN_LAT, VAL_LAT,
)
from t2b_route import DenseDynamics  # noqa: E402
from q40_koopman import KoopResidual  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "r0_analytic_core.json")
S_RES = 0.5
SEEDS = (0, 1, 2, 3, 4)
REF_KOOP = dict(a=0.2392, margin=0.0770)
REF_MLP = dict(a=0.6669, margin=0.0920)
MARGIN_FLOOR = 0.8 * REF_KOOP["margin"]

t0 = time.time()


class AnalyticExtrap(nn.Module):
    """ẑ = 2z_t − z_{t−1} + s·tanh(r(zc))。外推算子固定零参数；接口与 KoopResidual 对齐。

    解析性质：扰动传播 (δ_{t+1}, δ_t) ← E(δ_t, δ_{t−1})，E=2I−S 特征方程 ρ²−2ρ+1=0，
    重根 ρ=1 ⇒ λ=0（临界，Jordan 二阶线性增长）。**a 的公式预测 = 0**。
    """

    def __init__(self, hidden=HIDDEN, s=S_RES):
        super().__init__()
        self.r = nn.Sequential(nn.Linear(CTX * LATENT, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, LATENT))
        self.s = float(s)

    def forward(self, zc, acts=None, feats=None):
        extrap = 2 * zc[:, -1] - zc[:, -2]
        res = self.s * torch.tanh(self.r(zc.flatten(1)))
        return extrap + res, None

    def spec_pen(self, iters=5, rho0=None):
        z = torch.zeros(1, device=next(self.parameters()).device)
        return torch.relu(z + 0.0) ** 2, torch.tensor(1.0)   # 谱固定：pen=0，σ=1（临界）


def train(model, ztr, seed, epochs=EPOCHS, spec=False):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A, B, Y = make_pairs(ztr, 1)
    A = to_norm(A.reshape(-1, LATENT), *model_stats(ztr))
    Bn = to_norm(B.reshape(-1, LATENT), *model_stats(ztr))
    Yn = to_norm(Y.reshape(-1, 1, LATENT), *model_stats(ztr))
    N = A.shape[0]
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    for ep in range(epochs):
        idx = torch.randperm(N)[:BATCH]
        zc = torch.stack([A[idx].to(DEV), Bn[idx].to(DEV)], dim=1)
        p, _ = model(zc, None, None)
        loss = F.mse_loss(p, Yn[idx].to(DEV)[:, 0])
        if spec:
            sp, _ = model.spec_pen()
            loss = loss + 1.0 * sp
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return model


_stats_cache = {}


def model_stats(ztr):
    key = id(ztr)
    if key not in _stats_cache:
        flat = ztr.reshape(-1, LATENT)
        _stats_cache[key] = (flat.mean(0), flat.std(0))
    return _stats_cache[key]


def main():
    ztr = load_latents(TRAIN_LAT)
    zva = load_latents(VAL_LAT)
    mean, std = ztr.reshape(-1, LATENT).mean(0), ztr.reshape(-1, LATENT).std(0)

    results = {}
    plans = {
        "analytic": (lambda: AnalyticExtrap(hidden=HIDDEN, s=S_RES), False),
        "koop_s05": (lambda: KoopResidual(hidden=HIDDEN, s=0.5), True),
        "mlp": (lambda: DenseDynamics(hidden=HIDDEN, ctx=CTX, act_dim=0), False),
    }
    for nm, (cls_f, spec) in plans.items():
        as_, ms_, n_par = [], [], None
        for sd in SEEDS:
            m = cls_f().to(DEV)
            if n_par is None:
                n_par = sum(p.numel() for p in m.parameters() if p.requires_grad)
            m = train(m, ztr, sd, spec=spec)
            mc = margin_and_cos(m, zva, mean, std)
            ta = twin_a(m, zva, mean, std)
            as_.append(ta["a"]); ms_.append(mc["margin"])
            del m
            torch.cuda.empty_cache()
        results[nm] = dict(a_mean=float(np.mean(as_)), a_sd=float(np.std(as_, ddof=1)),
                           margin_mean=float(np.mean(ms_)), margin_sd=float(np.std(ms_, ddof=1)),
                           per_seed_a=[round(x, 4) for x in as_], n_params=int(n_par))
        print(f"  {nm:<10} a={results[nm]['a_mean']:+.4f}±{results[nm]['a_sd']:.4f} "
              f"margin={results[nm]['margin_mean']:+.4f}±{results[nm]['margin_sd']:.4f} 参数 {n_par}")

    an = results["analytic"]
    r0_pass = bool(an["a_mean"] <= REF_KOOP["a"] and an["margin_mean"] >= MARGIN_FLOOR)
    print(f"\nR0 判定：analytic a={an['a_mean']:+.4f}（门限 ≤{REF_KOOP['a']}）｜"
          f"margin={an['margin_mean']:+.4f}（门限 ≥{MARGIN_FLOOR:.4f}）→ "
          f"{'PASS：解析先验有增益' if r0_pass else 'FAIL：解析先验无增益（如实记录）'}")
    print(f"解析可设计性对拍：公式预测 a=0（外推算子重根 ρ=1）｜实测 a={an['a_mean']:+.4f} "
          f"⇒ 残差+有限样本贡献 {an['a_mean']:+.4f}")

    out = dict(prereg=dict(rule="a≤0.239 AND margin≥0.8×0.077", s_res=S_RES, seeds=list(SEEDS),
                           analytic_prediction_a=0.0),
               results=results, ref=dict(koop=REF_KOOP, mlp=REF_MLP),
               r0_pass=r0_pass, elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → {OUT_JSON}（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
