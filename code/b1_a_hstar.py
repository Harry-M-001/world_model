# -*- coding: utf-8 -*-
"""B1 · a–h* 定量关系：η 扫描的 (η, h*) 曲线 vs T_crash 公式理论曲线的拟合检验。

预注册（跑前写死）：
  两模型（single a=0.66 / koop a=0.26，MM 域）× η 网格 10 点（log 均匀 0.005-1.0）×
  128 序列 × 18 步。h* 定义（margin 口径，X5 教训）：逐序列 margin(cos pred−cos copy)
  过零视界的中位数。
  理论：T_crash(η) = ln(1+δ*(e^a−1)/η)/a，δ* 为唯一拟合参数（各模型独立拟合，log 尺度最小二乘）。
  判定：R² ≥ 0.8（两模型）⇒ a 预测 h* 的定量关系成立；R² < 0.5 ⇒ 无关系（负结果）；
  中间 ⇒ 弱关系（如实记录）。附加：两模型的 δ* 一致性（同一域应共享崩坏阈值）。
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
H_ROLL, N_SEQ = 18, 128
ETA_GRID = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
OUT_JSON = os.path.join(ROOT, "logs", "b1_a_hstar.json")
t0 = time.time()


def train_single():
    torch.manual_seed(0); np.random.seed(0)
    return DenseDynamics(hidden=192, ctx=CTX, act_dim=0).to(DEV)


def train_koop():
    torch.manual_seed(0); np.random.seed(0)
    return KoopResidual(hidden=192, s=0.5).to(DEV)


def train_model(m, ztr, mean_t, std_t):
    T = ztr.shape[1]
    rng = np.random.default_rng(1)
    si = rng.integers(0, ztr.shape[0], 24000); ti = rng.integers(0, T - 2, 24000)
    An = (torch.tensor(ztr[si, ti]).float() - mean_t) / std_t
    Bn = (torch.tensor(ztr[si, ti + 1]).float() - mean_t) / std_t
    Yn = (torch.tensor(ztr[si, ti + 2]).float() - mean_t) / std_t
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    spec = isinstance(m, KoopResidual)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(len(An))[:BATCH]
        zc = torch.stack([An[idx].to(DEV), Bn[idx].to(DEV)], dim=1)
        p, _ = m(zc, None, None)
        loss = F.mse_loss(p, Yn[idx].to(DEV))
        if spec:
            sp_pen, _ = m.spec_pen(); loss = loss + sp_pen
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    return m.eval()


def hstar_curve(m, zva, eta_inj, seed=99):
    """η 注入 rollout：逐序列 margin 过零视界。返回 (h*_中位, h*_均值, 有效 n)。"""
    rng = np.random.default_rng(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    firsts = []
    with torch.no_grad():
        for k in range(N_SEQ):
            ep = zva[k]
            zc = torch.tensor(ep[0:2]).float().unsqueeze(0).to(DEV)
            first = None
            for t in range(2, 2 + H_ROLL):
                p, _ = m(zc, None, None)
                p = p[0]
                if eta_inj > 0:
                    nz = torch.randn(p.shape, generator=g).float().to(DEV)
                    p = p + eta_inj * nz / nz.norm() * math.sqrt(len(p))
                z_true = torch.tensor(ep[t]).float().to(DEV)
                zh = zc[:, -1][0]
                cp = F.cosine_similarity(p[None].float(), z_true[None].float()).item()
                cc = F.cosine_similarity(zh[None].float(), z_true[None].float()).item()
                if first is None and (cp - cc) < 0:
                    first = t - 1
                zc = torch.stack([zc[:, -1], p.unsqueeze(0)], dim=1)
            firsts.append(first if first is not None else H_ROLL)
    return float(np.median(firsts)), float(np.mean(firsts)), len(firsts)


def fit_tcrash(etas, hstars, a):
    """log 尺度最小二乘拟合 δ*。返回 δ*, R²。"""
    best = None
    for log_d in np.linspace(-4, 2, 200):
        d_star = math.exp(log_d)
        pred = []
        for eta in etas:
            t = math.log(1 + d_star * (math.exp(a) - 1) / eta) / a if a > 1e-6 else float("inf")
            pred.append(min(t, H_ROLL))
        pred = np.array(pred)
        resid = np.array(hstars) - pred
        ss_res = float((resid ** 2).sum())
        if best is None or ss_res < best[2]:
            best = (d_star, pred, ss_res)
    d_star, pred, ss_res = best
    ss_tot = float(((np.array(hstars) - np.mean(hstars)) ** 2).sum())
    r2 = 1 - ss_res / max(ss_tot, 1e-9)
    return d_star, r2, pred


def main():
    ztr = load_latents(TRAIN_LAT)
    zva = load_latents(VAL_LAT)
    mean_t, std_t = ztr.reshape(-1, LATENT).mean(0), ztr.reshape(-1, LATENT).std(0)
    mean, std = mean_t, std_t

    models = {}
    for nm, mk in (("single", train_single), ("koop", train_koop)):
        m = train_model(mk(), ztr, mean_t, std_t)
        ta = twin_a(m, zva, mean, std)
        models[nm] = dict(m=m, a=ta["a"])
        print(f"{nm}: a=+{ta['a']:.4f}", flush=True)

    curves = {}
    for nm, d in models.items():
        curves[nm] = {}
        for eta in ETA_GRID:
            med, mean_v, n = hstar_curve(d["m"], zva, eta, seed=int(100 + eta * 1000))
            curves[nm][eta] = dict(median=med, mean=mean_v)
        print(f"{nm} h*(η): " + " ".join(f"{e}:{curves[nm][e]['median']:.0f}" for e in ETA_GRID), flush=True)

    # 拟合
    fits = {}
    for nm, d in models.items():
        a = d["a"]
        hstars = [curves[nm][e]["median"] for e in ETA_GRID]
        d_star, r2, pred = fit_tcrash(ETA_GRID, hstars, a)
        fits[nm] = dict(a=a, delta_star=d_star, r2=r2,
                        pred=[round(float(v), 2) for v in pred])
        print(f"[{nm}] δ*={d_star:.4f}｜R²={r2:.3f}｜理论: " + " ".join(f"{v:.1f}" for v in pred))
    r2_min = min(f["r2"] for f in fits.values())
    if r2_min >= 0.8:
        verdict = "a 预测 h* 的定量关系成立（R²≥0.8）"
    elif r2_min < 0.5:
        verdict = "定量关系不成立（负结果）"
    else:
        verdict = "弱关系（如实记录）"
    delta_consistent = abs(fits["single"]["delta_star"] - fits["koop"]["delta_star"]) / max(
        fits["single"]["delta_star"], fits["koop"]["delta_star"], 1e-9) < 0.5
    print(f"\n判定：{verdict}")
    print(f"δ* 一致性（同域共享崩坏阈值）: {delta_consistent}")

    out = dict(prereg=dict(r2_pass=0.8, r2_fail=0.5, hstar_def="margin 过零视界中位"),
               eta_grid=ETA_GRID, curves=curves, fits=fits,
               delta_consistent=bool(delta_consistent), verdict=verdict,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "b1_a_hstar.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)

    # 图
    import matplotlib
    matplotlib.use("Agg")
    sys.path.insert(0, HERE)
    from figstyle import setup_matplotlib, dark_axes, legend  # noqa: E402
    setup_matplotlib()
    import matplotlib.pyplot as plt  # noqa: E402
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    colors = dict(single="#5b8cff", koop="#00c2a8")
    for nm in ("single", "koop"):
        f = fits[nm]
        etas_f = np.linspace(ETA_GRID[0], ETA_GRID[-1], 60)
        pred_f = [min(math.log(1 + f["delta_star"] * (math.exp(f["a"]) - 1) / e) / f["a"], H_ROLL)
                  for e in etas_f]
        ax.plot(etas_f, pred_f, "--", color=colors[nm], lw=1.2, alpha=0.6)
        ax.plot(ETA_GRID, [curves[nm][e]["median"] for e in ETA_GRID], "-o", color=colors[nm],
                ms=5, lw=1.8, label=f"{nm} (a=+{f['a']:.2f}, R²={f['r2']:.2f})")
    dark_axes(ax)
    ax.set_xscale("log")
    ax.set_xlabel("注入噪声 η")
    ax.set_ylabel("h*（margin 过零视界，步）")
    ax.set_title("B1 · h*(η) 实测 vs T_crash 理论（虚线）——a 预测 h* 的定量检验")
    legend(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ROOT, "figs", "b1_a_hstar.png"), dpi=140)
    print(f"图 → figs/b1_a_hstar.png")


if __name__ == "__main__":
    main()
