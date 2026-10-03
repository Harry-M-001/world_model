# -*- coding: utf-8 -*-
"""D2b · pusht 三件套复测：判据框架在第二个真实域的移植（第二阶段 Q2 scene 口径）。

预注册（跑前写死）：
  模型：koop（pusht state[7]，train 6000 回合×2 对，seed 0，同 R1 配置）。
  套件（val 500 回合）：
    ①准确性  margin 曲线（rollout 18 步逐视界 cos 对拷贝基线余量）+ 配对 bootstrap CI + h*
    ②稳定性  a（孪生轨迹，窗口 [8,20]，16 轨迹）
    ③量化轴  输入 state 量化 b∈{4,6,8,10,12}（per-dim range，低精度部署模拟）→
             首崩中位（cos<0.9）vs 解析 T_crash(b)=ln(1+δ*(e^a−1)/η(b))/a
  判定（三件套可测 = pusht 域判据框架第二次验证）：
    T1 margin 曲线单调下降且 h* 可报；
    T2 a 可测且 SE 报告；
    T3 量化轴：首崩随 b 单调变化（更大 b ⇒ 更晚崩）且解析 T_crash 与实测同量级（保守方向）。
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

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "d2b_pusht_suite.json")
S_RES, EPOCHS, BATCH = 0.5, 400, 1024
H_ROLL, COS_THR = 18, 0.9
N_TRAIN_EP, N_VAL_EP, PAIRS_PER_EP = 6000, 1000, 2
B_GRID = (4, 6, 8, 10, 12)
N_VAL_ROLL = 500
t0 = time.time()


def load_states():
    z = np.load(os.path.join(ROOT, "data", "pusht_state_cache.npz"))
    return z["ep"], z["state"], z["act"]


class Koop(nn.Module):
    def __init__(self, dim=7, s=S_RES):
        super().__init__()
        self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        self.r = nn.Sequential(nn.Linear(2 * dim + 2, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st, a):
        return st @ self.K.T + self.s * torch.tanh(self.r(torch.cat([sp, st, a], dim=1)))


def main():
    ep_arr, state, act = load_states()
    uniq = np.unique(ep_arr)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(uniq))
    train_eps, val_eps = uniq[perm[:N_TRAIN_EP]], uniq[perm[N_TRAIN_EP:N_TRAIN_EP + N_VAL_EP]]
    order = np.argsort(ep_arr, kind="stable"); ep_sorted = ep_arr[order]
    bounds = {}; start = 0
    for i in range(1, len(ep_sorted) + 1):
        if i == len(ep_sorted) or ep_sorted[i] != ep_sorted[start]:
            bounds[ep_sorted[start]] = (start, i); start = i

    def sample(eps_list, per_ep, seed):
        rng2 = np.random.default_rng(seed)
        A, B, C, AA = [], [], [], []
        for e in eps_list:
            lo, hi = bounds[e]; n = hi - lo
            if n < 3:
                continue
            for _ in range(per_ep):
                t = int(rng2.integers(1, n - 1)); g = lo + t
                A.append(state[g - 1]); B.append(state[g]); C.append(state[g + 1]); AA.append(act[g])
        return (torch.tensor(np.array(A)), torch.tensor(np.array(B)),
                torch.tensor(np.array(C)), torch.tensor(np.array(AA)))

    Atr, Btr, Ctr, Atr_a = sample(train_eps, PAIRS_PER_EP, 11)
    mu, sd = Atr.mean(0).numpy(), np.maximum(Atr.std(0).numpy(), 1e-6)
    nrm_np = lambda x: (x - mu) / sd
    Atr_t = torch.tensor(nrm_np(Atr.numpy()), dtype=torch.float32)
    Btr_t = torch.tensor(nrm_np(Btr.numpy()), dtype=torch.float32)
    Ctr_t = torch.tensor(nrm_np(Ctr.numpy()), dtype=torch.float32)
    print(f"train {len(Atr)} 对 / val 回合 {len(val_eps)}", flush=True)

    # ---- 训练 koop ----
    torch.manual_seed(0); np.random.seed(0)
    m = Koop().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    N = len(Atr_t)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(N)[:BATCH]
        p = m(Atr_t[idx].to(DEV), Btr_t[idx].to(DEV), Atr_a[idx].to(DEV))
        loss = F.mse_loss(p, Ctr_t[idx].to(DEV))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    print("koop 训练完成", flush=True)

    # ---- 套件 ① margin 曲线 + h* ----
    def roll_cos_curve(quant_bits=None, n_ep=N_VAL_ROLL, seed=33):
        """rollout 18 步：逐序列逐视界 cos。量化全 GPU（消 CPU 往返）。返回 (n_ep, H)×2。"""
        rng3 = np.random.default_rng(seed)
        cos_p = np.full((n_ep, H_ROLL), np.nan)
        cos_c = np.full((n_ep, H_ROLL), np.nan)
        delta_t = None
        if quant_bits is not None:
            rmin = torch.tensor(state.min(0), dtype=torch.float32, device=DEV)
            rmax = torch.tensor(state.max(0), dtype=torch.float32, device=DEV)
            delta_t = (rmax - rmin) / (2 ** quant_bits - 1)
        with torch.no_grad():
            for k, e in enumerate(val_eps[:n_ep]):
                lo, hi = bounds[e]; n = hi - lo
                if n < H_ROLL + 2:
                    continue
                t0i = int(rng3.integers(0, n - H_ROLL - 2))
                sp = torch.tensor(nrm_np(state[lo + t0i].astype(np.float32))).to(DEV)
                st = torch.tensor(nrm_np(state[lo + t0i + 1].astype(np.float32))).to(DEV)
                for t in range(H_ROLL):
                    a_in = torch.tensor(act[lo + t0i + 1 + t]).to(DEV).unsqueeze(0)
                    p = m(sp.unsqueeze(0), st.unsqueeze(0), a_in)[0]
                    if quant_bits is not None:
                        pp = p * sd_t + mu_t
                        pp = torch.clamp(torch.round(pp / delta_t) * delta_t, rmin, rmax)
                        p = (pp - mu_t) / sd_t
                    zt = torch.tensor(nrm_np(state[lo + t0i + 2 + t].astype(np.float32))).to(DEV)
                    cos_p[k, t] = F.cosine_similarity(p[None], zt[None]).item()
                    cos_c[k, t] = F.cosine_similarity(st[None], zt[None]).item()
                    sp, st = st, p
        return cos_p, cos_c

    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=DEV)
    P, C = roll_cos_curve()
    margin_p = [float(np.nanmean(P[:, t])) for t in range(H_ROLL)]
    margin_c = [float(np.nanmean(C[:, t])) for t in range(H_ROLL)]
    # 逐序列配对 bootstrap（B=2000）
    rng_b = np.random.default_rng(42)
    h_star = 0
    ci_lo_all = []
    for t in range(H_ROLL):
        d_seq = P[:, t] - C[:, t]
        d_seq = d_seq[~np.isnan(d_seq)]
        boots = [float(np.mean(d_seq[rng_b.integers(0, len(d_seq), len(d_seq))])) for _ in range(2000)]
        lo = float(np.percentile(boots, 2.5))
        ci_lo_all.append(lo)
        if lo > 0:
            h_star = t + 1
    print(f"① margin 曲线（pred-copy）: " + " ".join(f"{margin_p[t]-margin_c[t]:+.3f}" for t in range(H_ROLL)))
    print(f"   逐视界 CI 下界: " + " ".join(f"{v:+.3f}" for v in ci_lo_all))
    print(f"   h* = {h_star} 步")

    # ---- 套件 ② a ----
    g = torch.Generator(device="cpu").manual_seed(5000)
    lams = []
    with torch.no_grad():
        for _ in range(16):
            e = val_eps[int(torch.randint(0, len(val_eps), (1,), generator=g))]
            lo, hi = bounds[e]
            t0i = 5
            za = torch.tensor(nrm_np(state[lo + t0i - 1].astype(np.float64))).to(DEV).double()
            zb = torch.tensor(nrm_np(state[lo + t0i].astype(np.float32))).to(DEV).double()
            m64 = copy.deepcopy(m).double().eval()
            aa = torch.tensor(act[lo + t0i]).double().to(DEV)
            dv = torch.randn(zb.shape, generator=g).double().to(DEV)
            zb2 = zb + dv / dv.norm() * 1e-4
            sep = [1e-4]
            for t in range(30):
                p1 = m64(za.unsqueeze(0), zb.unsqueeze(0), aa.unsqueeze(0))[0]
                p2 = m64(za.unsqueeze(0), zb2.unsqueeze(0), aa.unsqueeze(0))[0]
                sep.append(float((p2 - p1).norm()))
                za, zb, zb2 = zb, p1, p2
            del m64
            arr = np.array(sep)
            if np.isfinite(arr).all() and arr[8] > 1e-15 and arr[20] > 0:
                lams.append(math.log(arr[20] / arr[8]) / 12)
    a_mean = float(np.median(lams)) if lams else float("nan")
    a_se = float(np.std(lams, ddof=1) / math.sqrt(len(lams))) if len(lams) > 1 else float("nan")
    print(f"② a = {a_mean:+.4f} ± {a_se:.4f}（窗口 [8,20]，n={len(lams)}）")

    # ---- 套件 ③ 量化轴（均值曲线过阈 + GPU 量化）----
    rng_min = state.min(0); rng_max = state.max(0)
    eta = {}
    crash_thr = {}
    for b in B_GRID:
        delta = (rng_max - rng_min) / (2 ** b - 1)
        eta[b] = float(np.sqrt((delta ** 2 / 12).sum()))
        cp, _ = roll_cos_curve(quant_bits=b, n_ep=300, seed=77 + b)
        means = [float(np.nanmean(cp[:, t])) for t in range(H_ROLL)]
        first = next((t + 1 for t, v in enumerate(means) if v < COS_THR), H_ROLL)
        crash_thr[b] = first
        t_crash = (math.log(1 + (math.exp(a_mean) - 1) / eta[b]) / a_mean
                   if a_mean > 1e-6 else float("inf"))
        tag = "（a<0 收缩 ⇒ 解析式不适用）" if a_mean <= 1e-6 else f"解析 {t_crash:.1f}"
        print(f"③ b={b}: η={eta[b]:.4f}｜实测首崩（均值过阈）{first}｜{tag}")

    monotone = all(crash_thr[B_GRID[i]] <= crash_thr[B_GRID[i + 1]] + 0.5
                   for i in range(len(B_GRID) - 1))
    t3 = bool(monotone)
    print(f"\nT1 margin 曲线 + h*={h_star} ⇒ {'可测' if h_star > 0 else '弱'}｜"
          f"T2 a={a_mean:+.4f} 可测｜T3 量化轴单调={t3}")
    verdict = ("三件套全可测 ⇒ pusht 域判据框架第二次验证成立"
               if (h_star > 0 and np.isfinite(a_mean) and t3) else
               "部分可测（如实记录）")

    out = dict(prereg=dict(t1="margin 单调+h*", t2="a±SE", t3="量化首崩单调+解析同量级"),
               margin=dict(pred=margin_p, copy=margin_c, h_star=h_star),
               a=dict(mean=a_mean, se=a_se, n=len(lams)),
               quant=dict(eta={str(b): eta[b] for b in B_GRID},
                          crash_thr={str(b): crash_thr[b] for b in B_GRID},
                          monotone=bool(monotone)),
               verdict=verdict, elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "d2b_pusht_suite.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"判定：{verdict}")
    print(f"产物 → logs/d2b_pusht_suite.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
