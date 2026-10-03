# -*- coding: utf-8 -*-
"""X5 · a–有效步数解耦实验（X3 遗留格）。

预注册（跑前写死）：
  假说（T_crash 公式的极限）：T_crash = ln(1+δ*(e^a−1)/η)/a；a→0 极限 = δ*/η
    ⇒ **极低 a 时首崩由单步固有误差 η 主导，a 的收益只在 η 小时可见**。
  实验：single（a=0.66）/ koop（a=0.24）两模型 × η_inj 注入扫描
    {0, 0.01, 0.03, 0.1, 0.3, 1.0}（标准化空间每步输出加高斯噪声）× 128 序列 × 18 步。
  E1（η 主导检验）：同一 η_inj 下两模型有效步数曲线差 < 0.3 步 ⇒ 有效步数由 η 主导、
    与模型身份无关（X3 K1 不成立的机制解释）；
  E2（a 独立贡献检验）：η_inj=0（无注入）时 koop 开环有效步数 ≥ single + 1 步 ⇒ a 有独立贡献。
  两判据可同时成立（η 主导主区 + a 在低 η 区显效）⇒ 解耦的完整刻画。
  附加：两模型的 η₀（无注入单步 MSE，标准化空间）直接测量 —— η 差异是首崩同刻的候选解释。
"""
import copy
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from q12_min_train import (  # noqa: E402
    load_latents, TRAIN_LAT, VAL_LAT, LATENT, CTX, EPOCHS, BATCH, twin_a,
)
from t2b_route import DenseDynamics  # noqa: E402
from q40_koopman import KoopResidual  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_ROLL, N_SEQ, COS_THR = 18, 128, 0.9
ETA_GRID = [0.0, 0.01, 0.03, 0.1, 0.3, 1.0]
OUT_JSON = os.path.join(ROOT, "logs", "x5_decouple.json")
t0 = time.time()


def train_single(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = DenseDynamics(hidden=192, ctx=CTX, act_dim=0).to(DEV)
    return m


def train_koop(seed=0):
    torch.manual_seed(seed); np.random.seed(seed)
    m = KoopResidual(hidden=192, s=0.5).to(DEV)
    return m


def train_model(m, ztr, mean_t, std_t):
    T = ztr.shape[1]
    rng = np.random.default_rng(1)
    n_pairs = 24000
    si = rng.integers(0, ztr.shape[0], n_pairs)
    ti = rng.integers(0, T - 2, n_pairs)
    Az = torch.tensor(ztr[si, ti]).float()
    Bz = torch.tensor(ztr[si, ti + 1]).float()
    Yz = torch.tensor(ztr[si, ti + 2]).float()
    An, Bn, Yn = (Az - mean_t) / std_t, (Bz - mean_t) / std_t, (Yz - mean_t) / std_t
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    N = len(An)
    spec = isinstance(m, KoopResidual)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(N)[:BATCH]
        zc = torch.stack([An[idx].to(DEV), Bn[idx].to(DEV)], dim=1)
        p, _ = m(zc, None, None)
        loss = F.mse_loss(p, Yn[idx].to(DEV))
        if spec:
            sp_pen, _ = m.spec_pen()
            loss = loss + sp_pen
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan {type(m).__name__} ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()


@torch.no_grad()
def eff_curve(m, zva, eta_inj, seed=99):
    """η 注入扫描：每步输出加 η·N(0,1)（标准化空间）。返回有效步数均值。"""
    g = torch.Generator(device="cpu").manual_seed(seed)
    tot = 0
    with torch.no_grad():
        for k in range(min(N_SEQ, zva.shape[0])):
            ep = zva[k]
            zc = torch.tensor(ep[0:2]).float().unsqueeze(0).to(DEV)
            for t in range(2, 2 + H_ROLL):
                p, _ = m(zc, None, None)
                p = p[0]
                if eta_inj > 0:
                    p = p + eta_inj * torch.randn(p.shape, generator=g).float().to(DEV)
                z_true = torch.tensor(ep[t]).float().to(DEV)
                c = F.cosine_similarity(p[None].float(), z_true[None].float()).item()
                if c >= COS_THR:
                    tot += 1
                zc = torch.stack([zc[:, -1], p.unsqueeze(0)], dim=1)
    return tot / min(N_SEQ, zva.shape[0])


@torch.no_grad()
def eta0_of(m, zva):
    """无注入单步 MSE（标准化空间）+ 首崩实测。"""
    mse, first_min = [], []
    with torch.no_grad():
        for k in range(min(N_SEQ, zva.shape[0])):
            ep = zva[k]
            zc = torch.tensor(ep[0:2]).float().unsqueeze(0).to(DEV)
            for t in range(2, 2 + H_ROLL):
                p, _ = m(zc, None, None)
                p = p[0]
                z_true = torch.tensor(ep[t]).float().to(DEV)
                mse.append(float(((p - z_true) ** 2).mean()))
                zc = torch.stack([zc[:, -1], p.unsqueeze(0)], dim=1)
    return float(np.mean(mse))


def main():
    ztr = load_latents(TRAIN_LAT)
    zva = load_latents(VAL_LAT)
    mean_t, std_t = ztr.reshape(-1, LATENT).mean(0), ztr.reshape(-1, LATENT).std(0)
    mean, std = mean_t, std_t

    models = {}
    for nm, mk in (("single", train_single), ("koop", train_koop)):
        m = mk().to(DEV)
        m = train_model(m, ztr, mean_t, std_t)
        ta = twin_a(m, zva, mean, std)
        models[nm] = dict(m=m, a=ta["a"])
        print(f"{nm}: a=+{ta['a']:.4f}", flush=True)

    eta0 = {nm: eta0_of(d["m"], zva) for nm, d in models.items()}
    print(f"η₀（无注入单步 MSE）: {eta0}")

    curves = {}
    for nm, d in models.items():
        curves[nm] = {}
        for eta in ETA_GRID:
            curves[nm][eta] = eff_curve(d["m"], zva, eta)
        print(f"{nm}: " + "  ".join(f"η={e}:{curves[nm][e]:.3f}" for e in ETA_GRID), flush=True)

    # E1: 同 η 下两模型差
    diffs = {e: abs(curves["single"][e] - curves["koop"][e]) for e in ETA_GRID}
    e1 = bool(max(diffs.values()) < 0.3)
    # E2: η=0 时 koop − single
    e2 = bool(curves["koop"][0.0] >= curves["single"][0.0] + 1.0)
    print(f"\nE1 η 主导（同 η 差 <0.3 步）: max 差 {max(diffs.values()):.3f} → "
          f"{'成立' if e1 else '不成立（a 有独立贡献）'}")
    print(f"E2 a 独立贡献（η=0 时 koop ≥ single+1 步）: "
          f"koop {curves['koop'][0.0]:.3f} vs single {curves['single'][0.0]:.3f} → "
          f"{'成立' if e2 else '不成立'}")

    out = dict(prereg=dict(e1="同η差<0.3步", e2="η=0 koop≥single+1步", eta_grid=ETA_GRID),
               models_a={nm: d["a"] for nm, d in models.items()},
               eta0=eta0, curves=curves, diffs=diffs,
               e1_eta_dominant=bool(e1), e2_a_independent=bool(e2),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "x5_decouple.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/x5_decouple.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
