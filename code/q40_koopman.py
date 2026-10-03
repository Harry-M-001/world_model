# -*- coding: utf-8 -*-
"""Q4.0 · T6 重启：有界残差 Koopman + 有限时间口径监督（构想一）。

机制假设（纲领/T3 链路）：谱约束结构代价仅 0.012，代价全在「线性」上 ⇒
    p = K·b + s·tanh(r(zc))
    K 受谱范数惩罚（ρ0=1.0，幂迭代 5 步可微），残差输出有界 s·tanh。
若 a-sup（μ=1，与 Q1.1 同口径）压 a 时残差不再跑飞（输出有界）⇒ margin 可能不再崩。

判据预注册（跑前写死；对照 = 归档 single 5seed (a=0.6669, margin=+0.0920) 与
μ=1 MLP 5seed (a=0.3185, margin=−0.0072)，Q1.4 前沿内部区间 (0.32, 0.66)×margin>0 为空）：
  主判据（go/no-go，用户 2026-09-24 预注册）：∃ config 使
      a_mean ≤ 0.50  AND  margin_mean ≥ +0.03
  成立 ⇒ 「设计故事线」打开，补到 5 seed 复核；
  不成立 ⇒ 关闭设计故事线，按「度量论文」定位收口（项目依然成立，不算失败）。
  3 seed ⇒ 迹象级；正式判定需 ≥5 seed（与本项目纪律一致）。
口径：评测函数（margin_and_cos / twin_a / 常量）全部从 q12_min_train 导入，零漂移。
"""
import argparse
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

from q12_min_train import (  # noqa: E402  常量与评测口径从定义处导入
    LATENT, CTX, HIDDEN, EPOCHS, BATCH, EPS_A,
    load_latents, make_pairs, to_norm, margin_and_cos, twin_a,
    TRAIN_LAT, VAL_LAT, OUTDIR, ROOT as _ROOT,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LAMBDA_SPEC = 1.0        # 谱范数惩罚权重（预注册）
RHO0 = 1.0               # K 的谱范数目标上界（预注册）
OUT_JSON = os.path.join(_ROOT, "logs", "q40_koopman.json")


class KoopResidual(nn.Module):
    """p = K·b + s·tanh(r(zc))；zc=(B,ctx,L)，b=最后一帧。接口与 DenseDynamics 对齐。"""

    def __init__(self, hidden=HIDDEN, s=0.25):
        super().__init__()
        self.K = nn.Parameter(torch.eye(LATENT) * 0.9 + 0.01 * torch.randn(LATENT, LATENT))
        self.r = nn.Sequential(nn.Linear(CTX * LATENT, hidden), nn.SiLU(),
                               nn.Linear(hidden, hidden), nn.SiLU(),
                               nn.Linear(hidden, LATENT))
        self.s = float(s)

    def forward(self, zc, acts=None, feats=None):
        b = zc[:, -1]                       # (B, L)
        lin = b @ self.K.T
        res = self.s * torch.tanh(self.r(zc.flatten(1)))
        return lin + res, None

    def spec_pen(self, iters=5, rho0=None):
        """σ_max(K) 的幂迭代估计（固定初始向量，可微）；penalty = relu(σ−ρ0)²。"""
        if rho0 is None:
            rho0 = RHO0
        g = torch.Generator(device="cpu").manual_seed(123)
        v = torch.randn(LATENT, generator=g).to(self.K.device)
        v = v / v.norm()
        u = v
        sigma = torch.tensor(0.0, device=self.K.device)
        for _ in range(iters):
            u = self.K @ v
            u = u / u.norm().clamp_min(1e-9)
            v = self.K.T @ u
            v = v / v.norm().clamp_min(1e-9)
            sigma = (u @ self.K @ v).abs()
        return torch.relu(sigma - rho0) ** 2, sigma.detach()


def train_koop(ztr, mean, std, s, mu_a, seed, epochs=EPOCHS):
    torch.manual_seed(seed)
    np.random.seed(seed)
    A, B, Y = make_pairs(ztr, 1)
    A = to_norm(A.reshape(-1, LATENT), mean, std)
    B = to_norm(B.reshape(-1, LATENT), mean, std)
    Yn = to_norm(Y.reshape(-1, 1, LATENT), mean, std)
    N = A.shape[0]
    m = KoopResidual(hidden=HIDDEN, s=s).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    t0 = time.time()
    sigma_last = 0.0
    for ep in range(epochs):
        idx = torch.randperm(N)[:BATCH]
        a, b, y = A[idx].to(DEV), B[idx].to(DEV), Yn[idx].to(DEV)
        zc = torch.stack([a, b], dim=1)
        p, _ = m(zc, None, None)
        loss = F.mse_loss(p, y[:, 0])
        if mu_a and mu_a > 0:
            g = torch.Generator(device="cpu").manual_seed(1000 + ep)
            d = torch.randn(zc.shape, generator=g).to(DEV)
            dn = d.flatten(1).norm(dim=1)
            d = d / dn.view(-1, 1, 1) * EPS_A
            p2, _ = m(zc + d, None, None)
            amp = (p2 - p).norm(dim=1) / dn
            loss = loss + mu_a * torch.log(amp.clamp_min(1e-6)).mean()
        sp, sigma_last = m.spec_pen()
        loss = loss + LAMBDA_SPEC * sp
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
    return m, dict(s=s, mu_a=mu_a, seed=seed, epochs=epochs,
                   sigma_K=float(sigma_last), train_s=round(time.time() - t0, 1))


def eval_model(m, zva, mean, std, tag):
    mc = margin_and_cos(m, zva, mean, std)
    ta = twin_a(m, zva, mean, std)
    r = dict(tag=tag, **mc, **{k: v for k, v in ta.items() if k in ("a", "a_rel", "n_drop_fix", "n_pairs")})
    print(f"  [{tag}] a={r['a']:+.4f}  margin={r['margin']:+.4f}  val_cos={r['val_cos']:.4f}")
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    a = ap.parse_args()
    seeds = [int(x) for x in a.seeds.split(",")]
    t0 = time.time()

    ztr = load_latents(TRAIN_LAT)
    zva = load_latents(VAL_LAT)
    flat = ztr.reshape(-1, LATENT)
    mean, std = flat.mean(0), flat.std(0)
    print(f"Q4.0 有界残差 Koopman｜seeds={seeds} epochs={a.epochs}｜"
          f"对照归档 single(a=0.6669,m=+0.0920) μ1-MLP(a=0.3185,m=−0.0072)")

    results = []

    # ---- 阶段1：结构扫描（无 a-sup，s 网格）----
    for s in (0.1, 0.25, 0.5):
        for sd in seeds:
            m, cfg = train_koop(ztr, mean, std, s=s, mu_a=0.0, seed=sd, epochs=a.epochs)
            r = eval_model(m, zva, mean, std, tag=f"koop_s{s}_plain")
            results.append(dict(config=cfg, **r))
            del m

    # ---- 阶段2：s* = margin 最高且 a≤0.6 的 s；a-sup μ∈{1,3} ----
    cands = [r for r in results if r["a"] <= 0.6]
    pool = cands if cands else results
    s_star = max(pool, key=lambda r: r["margin"])["config"]["s"]
    print(f"\n  s* = {s_star}（阶段1 margin 最高且 a≤0.6）")
    for mu in (1.0, 3.0):
        for sd in seeds:
            m, cfg = train_koop(ztr, mean, std, s=s_star, mu_a=mu, seed=sd, epochs=a.epochs)
            r = eval_model(m, zva, mean, std, tag=f"koop_s{s_star}_mu{mu:g}")
            results.append(dict(config=cfg, **r))
            del m

    # ---- 聚合 + go/no-go 判定 ----
    import collections
    groups = collections.defaultdict(list)
    for r in results:
        groups[r["tag"]].append(r)
    agg = {}
    for tag, rs in groups.items():
        as_ = [x["a"] for x in rs]
        ms = [x["margin"] for x in rs]
        agg[tag] = dict(a_mean=float(np.mean(as_)), a_std=float(np.std(as_, ddof=1)) if len(as_) > 1 else None,
                        margin_mean=float(np.mean(ms)),
                        margin_std=float(np.std(ms, ddof=1)) if len(ms) > 1 else None,
                        val_cos=float(np.mean([x["val_cos"] for x in rs])), n_seeds=len(rs))
        print(f"  {tag:<22} a={agg[tag]['a_mean']:+.4f}  margin={agg[tag]['margin_mean']:+.4f}  "
              f"val_cos={agg[tag]['val_cos']:.4f}  (n={agg[tag]['n_seeds']})")

    hit = [(t, v) for t, v in agg.items()
           if v["a_mean"] <= 0.50 and v["margin_mean"] >= 0.03]
    if hit:
        best = max(hit, key=lambda kv: kv[1]["margin_mean"])
        verdict = (f"GO：{best[0]} a={best[1]['a_mean']:+.4f} margin={best[1]['margin_mean']:+.4f} "
                   f"→ 设计故事线打开（迹象级，补 5 seed 复核）")
    else:
        verdict = "NO-GO：预注册区间内无 (a, margin) 同时改善的点 → 关闭设计故事线（按度量论文定位收口）"
    print(f"\n  判定：{verdict}")

    out = dict(preregistered=dict(success="a_mean<=0.50 AND margin_mean>=+0.03",
                                  ref_single=dict(a=0.6669, margin=0.0920),
                                  ref_mu1_mlp=dict(a=0.3185, margin=-0.0072),
                                  lambda_spec=LAMBDA_SPEC, rho0=RHO0,
                                  note="3 seed=迹象级；正式判定需≥5 seed"),
               groups=agg, per_run=results, verdict=verdict,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  产物 → {OUT_JSON}")


if __name__ == "__main__":
    main()
