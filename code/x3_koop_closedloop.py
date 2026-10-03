# -*- coding: utf-8 -*-
"""X3 · koop 架构进闭环复测（Q3.3 遗留登记 → 第三阶段滚动项）。

预注册（跑前写死）：
  口径与 q33_closed_loop.py 逐字同源（MM 域、128 val 序列、18 步、cos≥0.9 有效步数、
  CollapseWarner 3σ），唯一差异 = 动力学模型换成 q40 KoopResidual(hidden=192, s=0.5, 无 a-sup)。
  先 twin_a 确认 koop 的 a ≈ 0.24 量级（Q4.0 归档）。
  K1：koop 开环有效步数 > single 开环有效步数（归档 0.16）⇒ 架构层直接延长闭环可用时长；
  K2：koop 下 warner_reset−open_loop 增量 ≥ single 下增量（+0.10）⇒ 重置收益放大验证。
  渲染臂跳过（Q3.3 J3 已证渲染=输出端，闭环成功率无关）。
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
    load_latents, TRAIN_LAT, VAL_LAT, LATENT, CTX, twin_a,
)
from q40_koopman import KoopResidual  # noqa: E402
from q12_min_train import EPOCHS, BATCH  # noqa: E402
from warn_component import CollapseWarner  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
H_ROLL, N_SEQ, COS_THR = 18, 128, 0.9
OUT_JSON = os.path.join(ROOT, "logs", "x3_koop_closedloop.json")
SINGLE_ARCHIVE = dict(open_eff=0.164, warner_eff=0.266, delta=0.102)  # q33 归档
t0 = time.time()


def cos_step(z_hat, z_true):
    return float(F.cosine_similarity(z_hat[None].float(), z_true[None].float()).item())


def main():
    ztr = load_latents(TRAIN_LAT)
    zva = load_latents(VAL_LAT)
    mean = ztr.reshape(-1, LATENT).mean(0)
    std = ztr.reshape(-1, LATENT).std(0)
    T = ztr.shape[1]

    # ---- 训练 koop（无 a-sup，s=0.5，seed 0，与 q12/q33 同 EPOCHS）----
    torch.manual_seed(0); np.random.seed(0)
    m = KoopResidual(hidden=192, s=0.5).to(DEV)
    A, B, Y = None, None, None
    flat = ztr.reshape(-1, LATENT)
    mean_t, std_t = flat.mean(0), flat.std(0)
    n_pairs = 24000
    idx_rng = np.random.default_rng(1)
    si = idx_rng.integers(0, ztr.shape[0], n_pairs)
    ti = idx_rng.integers(0, T - 2, n_pairs)
    Az = torch.tensor(ztr[si, ti]).float()
    Bz = torch.tensor(ztr[si, ti + 1]).float()
    Yz = torch.tensor(ztr[si, ti + 2]).float()
    An = (Az - mean_t) / std_t
    Bn = (Bz - mean_t) / std_t
    Yn = (Yz - mean_t) / std_t
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    N = len(An)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(N)[:BATCH]
        zc = torch.stack([An[idx].to(DEV), Bn[idx].to(DEV)], dim=1)
        p, _ = m(zc, None, None)
        loss = F.mse_loss(p, Yn[idx].to(DEV))
        sp_pen, _ = m.spec_pen()
        loss = loss + 1.0 * sp_pen
        if not torch.isfinite(loss):
            raise RuntimeError(f"koop train nan ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    ta = twin_a(m, zva, mean, std)
    print(f"koop 训练完成：a=+{ta['a']:.4f}（Q4.0 归档 ≈0.24 量级）")

    # ---- warner（训练范数分布）----
    norms = flat.norm(dim=1)
    warner = CollapseWarner(float(norms.mean()), float(norms.std()), n_sigma=3.0)

    # ---- 三臂复测（口径 = q33）----
    def run_arm(mode):
        res = []
        with torch.no_grad():
            for k in range(min(N_SEQ, zva.shape[0])):
                ep = zva[k]
                zc = torch.tensor(ep[0:2]).float().unsqueeze(0).to(DEV)
                eff, first, resets, coss = 0, None, 0, []
                for t in range(2, 2 + H_ROLL):
                    p, _ = m(zc, None, None)
                    p = p[0]
                    z_true = torch.tensor(ep[t]).float().to(DEV)
                    c = cos_step(p, z_true)
                    coss.append(c)
                    if c >= COS_THR:
                        eff += 1
                    if first is None and c < COS_THR:
                        first = t
                    if mode == "warner" and warner.online_check(float(p.norm())):
                        p = z_true
                        resets += 1
                    elif mode == "periodic" and (t % 2) == 0:
                        p = z_true
                        resets += 1
                    zc = torch.stack([zc[:, -1], p.unsqueeze(0)], dim=1)
                res.append(dict(eff=eff, first=first if first is not None else H_ROLL,
                                resets=resets))
        return res

    arms = dict(open_loop=run_arm("none"), warner_reset=run_arm("warner"),
                periodic=run_arm("periodic", ) if True else None)
    # periodic K：与 q33 同法——按 koop warner 臂重置频率取 K
    r2 = float(np.mean([r["resets"] for r in arms["warner_reset"]]))
    K = max(1, round(H_ROLL / max(r2, 1e-9)))
    arms["periodic"] = run_arm("periodic", ) if False else None
    # 重新跑 periodic（用定好的 K）
    def run_periodic(K_):
        res = []
        with torch.no_grad():
            for k in range(min(N_SEQ, zva.shape[0])):
                ep = zva[k]
                zc = torch.tensor(ep[0:2]).float().unsqueeze(0).to(DEV)
                eff, first, resets = 0, None, 0
                for t in range(2, 2 + H_ROLL):
                    p, _ = m(zc, None, None)
                    p = p[0]
                    z_true = torch.tensor(ep[t]).float().to(DEV)
                    c = cos_step(p, z_true)
                    if c >= COS_THR:
                        eff += 1
                    if first is None and c < COS_THR:
                        first = t
                    if (t % K_) == 0:
                        p = z_true; resets += 1
                    zc = torch.stack([zc[:, -1], p.unsqueeze(0)], dim=1)
                res.append(dict(eff=eff, first=first if first is not None else H_ROLL,
                                resets=resets))
        return res
    arms["periodic"] = run_periodic(K)

    for nm, rs in arms.items():
        print(f"  {nm:<13} 有效步数 {np.mean([r['eff'] for r in rs]):.3f}｜"
              f"首崩中位 {np.median([r['first'] for r in rs]):.0f}｜"
              f"重置 {np.mean([r['resets'] for r in rs]):.2f}")

    eff_open = float(np.mean([r["eff"] for r in arms["open_loop"]]))
    eff_warner = float(np.mean([r["eff"] for r in arms["warner_reset"]]))
    eff_per = float(np.mean([r["eff"] for r in arms["periodic"]]))
    d_koop = eff_warner - eff_open
    d_single = SINGLE_ARCHIVE["warner_eff"] - SINGLE_ARCHIVE["open_eff"]
    k1 = bool(eff_open > SINGLE_ARCHIVE["open_eff"])
    k2 = bool(d_koop >= d_single)
    print(f"\nK1 koop 开环有效步数 {eff_open:.3f} vs single {SINGLE_ARCHIVE['open_eff']:.3f} "
          f"→ {'架构层延长 ✓' if k1 else '未延长'}")
    print(f"K2 重置增量：koop Δ=+{d_koop:.3f} vs single Δ=+{d_single:.3f} → "
          f"{'收益放大 ✓' if k2 else '未放大'}")
    print(f"   periodic 对照（K={K}）：{eff_per:.3f}（{'预警仍有增量' if eff_warner > eff_per else '与预警相当'}）")

    out = dict(prereg=dict(K1="koop open_eff > single 0.164",
                           K2="koop delta ≥ single delta 0.102"),
               koop_a=ta["a"], K=K,
               arms={k2_: [dict(eff=r["eff"], first=r["first"], resets=r["resets"])
                           for r in v] for k2_, v in arms.items()},
               k1=bool(k1), k2=bool(k2),
               delta_koop=d_koop, delta_single=d_single,
               single_archive=SINGLE_ARCHIVE,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → {OUT_JSON}（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
