# -*- coding: utf-8 -*-
"""R1b · pusht 混合臂扩展：action 驱动解析 + 有界残差（对照 R1 外推形式）。

预注册（跑前写死）：
  数据：LeWM pusht state[7]（同 R1 管线：train 6000 回合×2 / val 1000×5，seeds 0-4）。
  四臂（全部带 s=0.5 有界残差，可训练）：
    extrap_res   2s_t−s_{t−1} + res（=R1 analytic 的可训练同款）
    action_res   agent 位置 x,y += K·a（**K 最小二乘解析估计、固定不进梯度**），其余维保持 + res
                 ——「公式形式已知、增益由数据估计」的诚实设定
    koop / mlp   同 R1 对照
  判据：
    J-A：action_res a_mean < extrap_res a_mean AND margin10 ≥ 0.8×koop margin10
         ⇒ 解析形式质量决定混合臂成败（R1 外推 vs R1b action 驱动的正反对照）；
    J-B：action_res 的解析覆盖度消融：残差置 0（纯解析）单步 cos 显著 > 0（公式真的在干活）。
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
OUT_JSON = os.path.join(ROOT, "logs", "r1b_hybrid_action.json")
S_RES, EPOCHS, BATCH = 0.5, 400, 1024
SEEDS = (0, 1, 2, 3, 4)
H_ROLL, COS_THR = 10, 0.9
N_TRAIN_EP, N_VAL_EP, PAIRS_PER_EP, VAL_PER_EP = 6000, 1000, 2, 5
t0 = time.time()


def load_states():
    z = np.load(os.path.join(ROOT, "data", "pusht_state_cache.npz"))
    return z["ep"], z["state"], z["act"]


class Arm(nn.Module):
    """kind: extrap_res / action_res / koop / mlp。残差输入 [s_prev, s_t, a]（16 维）。"""

    def __init__(self, kind, k=None, dim=7, s=S_RES):
        super().__init__()
        self.kind = kind
        self.s = float(s)
        self.r = nn.Sequential(nn.Linear(2 * dim + 2, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        if kind == "koop":
            self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        if kind == "action_res":
            assert k is not None
            self.register_buffer("k", torch.tensor(k, dtype=torch.float32))

    def forward(self, sp, st, a):
        r_in = torch.cat([sp, st, a], dim=1)
        res = self.s * torch.tanh(self.r(r_in))
        if self.kind == "extrap_res":
            return 2 * st - sp + res
        if self.kind == "action_res":
            # 物理单位公式：agent 位置 += K·a；回标准化
            st_p = st * self.sd + self.mu
            sp_p = sp * self.sd + self.mu
            a_p = a * self.act_sd
            base = st_p.clone()
            base[:, 0] = base[:, 0] + self.k[0] * a_p[:, 0]
            base[:, 1] = base[:, 1] + self.k[1] * a_p[:, 1]
            base_n = (base - self.mu) / self.sd
            return base_n + res
        if self.kind == "koop":
            return st @ self.K.T + res
        return self.r(r_in)

    def set_stats(self, mu, sd, act_sd):
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.register_buffer("act_sd", torch.tensor(act_sd, dtype=torch.float32))


def main():
    ep, state, act = load_states()
    uniq = np.unique(ep)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(uniq))
    train_eps, val_eps = uniq[perm[:N_TRAIN_EP]], uniq[perm[N_TRAIN_EP:N_TRAIN_EP + N_VAL_EP]]
    order = np.argsort(ep, kind="stable"); ep_sorted = ep[order]
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
    Ava, Bva, Cva, Ava_a = sample(val_eps, VAL_PER_EP, 22)
    mu, sd = Atr.mean(0).numpy(), Atr.std(0).clamp_min(1e-6).numpy()
    act_sd = max(float(Atr_a.std()), 1e-6)
    nrm = lambda t: (t - mu) / sd
    Atr_t = torch.tensor(nrm(Atr.numpy()), dtype=torch.float32)
    Btr_t = torch.tensor(nrm(Btr.numpy()), dtype=torch.float32)
    Ctr_t = torch.tensor(nrm(Ctr.numpy()), dtype=torch.float32)
    Ava_t = torch.tensor(nrm(Ava.numpy()), dtype=torch.float32)
    Bva_t = torch.tensor(nrm(Bva.numpy()), dtype=torch.float32)
    Cva_t = torch.tensor(nrm(Cva.numpy()), dtype=torch.float32)
    print(f"训练对 {len(Atr)}｜val 对 {len(Ava)}", flush=True)

    # K 最小二乘解析估计（agent 位置维 0/1：Δpos 对 action 回归，过原点）
    dpos = (Ctr[:, :2] - Btr[:, :2]).numpy()
    kk = []
    for d in range(2):
        aa = Atr_a[:, d].numpy()
        k_est = float((aa * dpos[:, d]).sum() / max((aa * aa).sum(), 1e-9))
        kk.append(k_est)
    print(f"K 最小二乘估计：k_x={kk[0]:.2f} k_y={kk[1]:.2f}（像素/单位 action）")

    def make(kind):
        if kind == "action_res":
            m = Arm(kind, k=kk); m.set_stats(mu, sd, np.array([act_sd, act_sd])); return m
        if kind == "koop":
            return Arm(kind)
        return Arm(kind)

    def train_one(kind, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = make(kind).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        N = len(Atr_t)
        for ep_ in range(EPOCHS):
            idx = torch.randperm(N)[:BATCH]
            p = m(Atr_t[idx].to(DEV), Btr_t[idx].to(DEV), Atr_a[idx].to(DEV))
            loss = F.mse_loss(p, Ctr_t[idx].to(DEV))
            if not torch.isfinite(loss):
                raise RuntimeError(f"train nan: {kind} seed{seed} ep{ep_}")
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return m

    def eval_arm(m, kind):
        cs = []
        with torch.no_grad():
            for i in range(0, len(Ava_t), 2048):
                p = m(Ava_t[i:i+2048].to(DEV), Bva_t[i:i+2048].to(DEV), Ava_a[i:i+2048].to(DEV))
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"eval nan 单步 {kind}")
                cs.append(F.cosine_similarity(p, Cva_t[i:i+2048].to(DEV), dim=1).cpu())
        step1 = float(torch.cat(cs).mean())
        roll = []
        for e in val_eps[:500]:
            lo, hi = bounds[e]
            if hi - lo < H_ROLL + 2:
                continue
            sp = torch.tensor(nrm(state[lo].astype(np.float32))).to(DEV)
            st = torch.tensor(nrm(state[lo + 1].astype(np.float32))).to(DEV)
            for t in range(H_ROLL):
                a_in = torch.tensor(act[lo + 1 + t]).to(DEV).unsqueeze(0)
                p = m(sp.unsqueeze(0), st.unsqueeze(0), a_in)[0]
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"eval nan rollout {kind} t={t}")
                zt = torch.tensor(nrm(state[lo + 2 + t].astype(np.float32))).to(DEV)
                roll.append(float(F.cosine_similarity(p, zt, dim=0)))
                sp, st = st, p
        copy_v = []
        for e in val_eps[:500]:
            lo, hi = bounds[e]
            if hi - lo < H_ROLL + 2:
                continue
            for t in range(H_ROLL):
                zt = torch.tensor(nrm(state[lo + 2 + t].astype(np.float32))).to(DEV)
                zh = torch.tensor(nrm(state[lo + 1].astype(np.float32))).to(DEV)
                copy_v.append(float(F.cosine_similarity(zh, zt, dim=0)))
        return step1, float(np.mean(roll)), float(np.mean(copy_v))

    # 残差置 0 消融（action_res 纯解析覆盖度）
    @torch.no_grad()
    def pure_analytic_cos(m):
        cs = []
        for i in range(0, len(Ava_t), 2048):
            st_n = Bva_t[i:i+2048].to(DEV)
            a_in = Ava_a[i:i+2048].to(DEV)
            st_p = st_n * m.sd + m.mu
            base = st_p.clone()
            base[:, 0] += m.k[0] * a_in[:, 0] * m.act_sd[0]
            base[:, 1] += m.k[1] * a_in[:, 1] * m.act_sd[1]
            base_n = (base - m.mu) / m.sd
            cs.append(F.cosine_similarity(base_n, Cva_t[i:i+2048].to(DEV), dim=1).cpu())
        return float(torch.cat(cs).mean())

    arms = {}
    for kind in ("extrap_res", "action_res", "koop", "mlp"):
        a_list, m_list, s1_list = [], [], []
        for sd_ in SEEDS:
            m = train_one(kind, sd_)
            ta = None
            if kind != "mlp":
                m64 = copy.deepcopy(m).double().eval()
                g = torch.Generator(device="cpu").manual_seed(5000)
                lams = []
                with torch.no_grad():
                    for _ in range(16):
                        e = val_eps[int(torch.randint(0, len(val_eps), (1,), generator=g))]
                        lo, hi = bounds[e]
                        t0i = 5
                        za = torch.tensor(nrm(state[lo + t0i - 1].astype(np.float32))).double().to(DEV)
                        zb = torch.tensor(nrm(state[lo + t0i].astype(np.float32))).double().to(DEV)
                        aa_ = torch.tensor(act[lo + t0i]).double().to(DEV)
                        dv = torch.randn(zb.shape, generator=g).double().to(DEV)
                        zb2 = zb + dv / dv.norm() * 1e-4
                        sep = [1e-4]
                        for t in range(40):
                            p1 = m64(za.unsqueeze(0), zb.unsqueeze(0), aa_.unsqueeze(0))[0]
                            p2 = m64(za.unsqueeze(0), zb2.unsqueeze(0), aa_.unsqueeze(0))[0]
                            sep.append(float((p2 - p1).norm()))
                            za, zb, zb2 = zb, p1, p2
                        arr = np.array(sep)
                        if np.isfinite(arr).all() and arr[8] > 1e-15 and arr[20] > 0:
                            lams.append(math.log(arr[20] / arr[8]) / 12)
                a_list.append(float(np.median(lams)) if lams else float("nan"))
                del m64
            s1, m10, _ = eval_arm(m, kind)
            m_list.append(m10); s1_list.append(s1)
            if kind != "mlp" and (math.isnan(a_list[-1])):
                a_list[-1] = float("nan")
            del m
            torch.cuda.empty_cache()
        arms[kind] = dict(a_mean=float(np.nanmean(a_list)) if kind != "mlp" else float("nan"),
                          margin10=float(np.mean(m_list)), step1_cos=float(np.mean(s1_list)),
                          per_seed_a=[round(x, 4) for x in a_list])
        print(f"  {kind:<11} a={arms[kind]['a_mean'] if kind != 'mlp' else float('nan'):+.4f} "
              f"margin10={arms[kind]['margin10']:+.4f} step1cos={arms[kind]['step1_cos']:.4f}")

    # 纯解析覆盖度（action_res 残差置 0）
    m0 = make("action_res").to(DEV)
    # 用 seed0 训练后的残差网络置零需要重训——改为训练后评估：载入同 seed 模型再置零
    torch.manual_seed(0); np.random.seed(0)
    m0 = train_one("action_res", 0)
    with torch.no_grad():
        m0.s = 0.0   # 残差置 0（纯解析）
    pure_cos = pure_analytic_cos(m0)
    print(f"action_res 纯解析（残差置 0）单步 cos = {pure_cos:.4f}（解析覆盖度）")

    ja = bool(arms["action_res"]["a_mean"] < arms["extrap_res"]["a_mean"]
              and arms["action_res"]["margin10"] >= 0.8 * arms["koop"]["margin10"])
    print(f"\nJ-A 判定：action_res a={arms['action_res']['a_mean']:+.4f} vs extrap "
          f"{arms['extrap_res']['a_mean']:+.4f}｜margin10 {arms['action_res']['margin10']:+.4f} vs "
          f"koop {arms['koop']['margin10']:+.4f} → {'PASS：解析形式质量决定混合臂成败' if ja else 'FAIL'}")

    out = dict(prereg=dict(ja="action a<extrap a AND margin≥0.8×koop"),
               k_ls=kk, pure_analytic_cos=pure_cos, arms=arms, ja=bool(ja),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "r1b_hybrid_action.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/r1b_hybrid_action.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
