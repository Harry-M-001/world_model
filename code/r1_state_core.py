# -*- coding: utf-8 -*-
"""R1 · 显式状态空间解析核（R0 负结果后的修正路线）。

预注册（跑前写死）：
  数据：LeWM pusht state[7]（显式物理坐标：位置/角度/目标），18685 回合。
    train 6000 回合（每回合随机 2 个三元组）／val 1000 回合（每回合 5 个），按回合划分。
  三臂（全部输入 [s_{t-1}, s_t, a_t]，s=0.5 残差，5 seeds）：
    analytic   ẑ = 2s_t − s_{t-1} + s·tanh(r(·))   外推核零参数（公式 a 预测 = 0，临界）
    koop       p = K·s_t + s·tanh(r(·))             学习线性核
    mlp        纯 MLP
  指标：单步 cos（标准化空间）；10 步 rollout cos 均值（margin 口径：对拷贝基线的余量）；
    a（孪生口径，动作固定重复）。
  R1 PASS ⇔ analytic：a_mean ≤ koop a_mean AND margin10 ≥ 0.8×koop margin10。
    PASS ⇒ 显式坐标上解析先验有效（与 R0 潜空间负结果构成坐标错位假说的正反对照）；
    FAIL ⇒ 坐标对但解析形式错（匀速外推不适合 PushT 推板动力学），如实记录并转向
    「解析部分=可学习线性化的约束版」或承认解析核仅适用于保守系统。
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

import lance  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
LANCE = "E:/ai-self_data/lewm_pusht.lance/data/lewm_pusht.lance"
OUT_JSON = os.path.join(ROOT, "logs", "r1_state_core.json")
S_RES, EPOCHS, BATCH = 0.5, 400, 1024
SEEDS = (0, 1, 2, 3, 4)
H_ROLL = 10
COS_THR = 0.9
N_TRAIN_EP, N_VAL_EP, PAIRS_PER_EP, VAL_PER_EP = 6000, 1000, 2, 5
t0 = time.time()


def load_states():
    cache = os.path.join(ROOT, "data", "pusht_state_cache.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["ep"], z["state"], z["act"]
    ds = lance.dataset(LANCE)
    t = ds.to_table(columns=["episode_idx", "state", "action"])
    ep = np.array(t["episode_idx"])
    state = np.array(t["state"].combine_chunks().values).reshape(-1, 7).astype(np.float32)
    act = np.array(t["action"].combine_chunks().values).reshape(-1, 2).astype(np.float32)
    np.savez(cache, ep=ep, state=state, act=act)
    return ep, state, act


class ResArms(nn.Module):
    def __init__(self, kind, dim=7):
        super().__init__()
        self.kind = kind
        inp = 2 * dim + 2  # [s_{t-1}(7), s_t(7), a(2)] = 16
        self.r = nn.Sequential(nn.Linear(inp, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        if kind == "koop":
            self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))

    def forward(self, s_prev, s_t, a):
        r_in = torch.cat([s_prev, s_t, a], dim=1)
        res = S_RES * torch.tanh(self.r(r_in))
        if self.kind == "analytic":
            return 2 * s_t - s_prev + res
        if self.kind == "koop":
            return s_t @ self.K.T + res
        return self.mlp(r_in)


class MLPArm(nn.Module):
    def __init__(self, dim=7):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim + 2, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, dim))

    def forward(self, s_prev, s_t, a):
        return self.mlp(torch.cat([s_prev, s_t, a], dim=1))


def main():
    ep, state, act = load_states()
    uniq = np.unique(ep)
    rng = np.random.default_rng(7)
    perm = rng.permutation(len(uniq))
    train_eps, val_eps = uniq[perm[:N_TRAIN_EP]], uniq[perm[N_TRAIN_EP:N_TRAIN_EP + N_VAL_EP]]
    ep_index = {e: i for i, e in enumerate(uniq)}
    # 按回合切片
    order = np.argsort(ep, kind="stable")
    ep_sorted = ep[order]
    bounds = {}
    start = 0
    for i in range(1, len(ep_sorted) + 1):
        if i == len(ep_sorted) or ep_sorted[i] != ep_sorted[start]:
            bounds[ep_sorted[start]] = (start, i)
            start = i
    print(f"回合 {len(uniq)}｜train {len(train_eps)} / val {len(val_eps)}", flush=True)

    def sample(eps_list, per_ep, seed):
        rng2 = np.random.default_rng(seed)
        A, B, C, AA = [], [], [], []
        for e in eps_list:
            lo, hi = bounds[e]
            n = hi - lo
            if n < 3:
                continue
            for _ in range(per_ep):
                t = int(rng2.integers(1, n - 1))
                g = lo + t
                A.append(state[g - 1]); B.append(state[g]); C.append(state[g + 1])
                AA.append(act[g])
        return (torch.tensor(np.array(A)), torch.tensor(np.array(B)),
                torch.tensor(np.array(C)), torch.tensor(np.array(AA)))

    Atr, Btr, Ctr, Atr_a = sample(train_eps, PAIRS_PER_EP, 11)
    Ava, Bva, Cva, Ava_a = sample(val_eps, VAL_PER_EP, 22)
    mu, sd = Atr.mean(0), Atr.std(0).clamp_min(1e-6)
    print(f"Atr finite={bool(torch.isfinite(Atr).all())} mu finite={bool(torch.isfinite(mu).all())} "
          f"sd min={float(sd.min()):.4f} sd={np.round(sd.numpy(), 2)}", flush=True)
    nrm = lambda t: ((t - mu) / sd)
    print(f"训练对 {len(Atr)}｜val 对 {len(Ava)}", flush=True)

    class KoopWrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.arm = ResArms("koop")

        def forward(self, sp, st, a):
            return self.arm(sp, st, a)

        def spec_pen(self):
            g = torch.Generator(device="cpu").manual_seed(123)
            v = torch.randn(7, generator=g).to(DEV); v = v / v.norm()
            u = v
            for _ in range(5):
                u = self.arm.K @ v; u = u / u.norm().clamp_min(1e-9)
                v = self.arm.K.T @ u; v = v / v.norm().clamp_min(1e-9)
                sigma = (u @ self.arm.K @ v).abs()
            return torch.relu(sigma - 1.0) ** 2, sigma.detach()

    def make(kind):
        if kind == "mlp":
            return MLPArm()
        if kind == "koop":
            return KoopWrap()
        return ResArms(kind)

    def train_one(kind, seed):
        torch.manual_seed(seed); np.random.seed(seed)
        m = make(kind).to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        N = len(Atr)
        for ep_ in range(EPOCHS):
            idx = torch.randperm(N)[:BATCH]
            sp = nrm(Atr[idx]).to(DEV); st = nrm(Btr[idx]).to(DEV)
            aa = Atr_a[idx].to(DEV); y = nrm(Ctr[idx]).to(DEV)
            p = m(sp, st, aa)
            loss = F.mse_loss(p, y)
            if not torch.isfinite(loss):
                diff = Atr[idx] - mu
                bad_diff = int((~torch.isfinite(diff)).sum())
                bad_in = int((~torch.isfinite(torch.cat([sp, st, aa], 1))).sum())
                bad_y = int((~torch.isfinite(y)).sum())
                bmax = float(sp.abs().max()), float(st.abs().max()), float(aa.abs().max()), float(y.abs().max())
                raise RuntimeError(f"train nan: kind={kind} epoch={ep_} "
                                   f"bad_diff={bad_diff} bad_in={bad_in} bad_y={bad_y} absmax={bmax} "
                                   f"mu={mu.tolist()} sd={sd.tolist()}")
            if kind == "koop":
                sp_pen, _ = m.spec_pen()
                loss = loss + sp_pen
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        return m

    @torch.no_grad()
    def eval_arm(m):
        # 单步 cos
        cs = []
        for i in range(0, len(Ava), 2048):
            p = m(nrm(Ava[i:i+2048]).to(DEV), nrm(Bva[i:i+2048]).to(DEV), Ava_a[i:i+2048].to(DEV))
            if not torch.isfinite(p).all():
                raise RuntimeError(f"eval nan: 单步输出 kind 见调用栈")
            y = nrm(Cva[i:i+2048]).to(DEV)
            cs.append(F.cosine_similarity(p, y, dim=1).cpu())
        step1 = float(torch.cat(cs).mean())
        # 10 步 rollout（每序列前 10 步真动作链）
        roll_cos = []
        n_roll = 0
        for e in val_eps[:500]:
            lo, hi = bounds[e]
            if hi - lo < H_ROLL + 2:
                continue
            s_prev = nrm(torch.tensor(state[lo])).to(DEV)
            s_t = nrm(torch.tensor(state[lo + 1])).to(DEV)
            for t in range(H_ROLL):
                a_in = torch.tensor(act[lo + 1 + t]).to(DEV).unsqueeze(0)
                p = m(s_prev.unsqueeze(0), s_t.unsqueeze(0), a_in)[0]
                if not torch.isfinite(p).all():
                    raise RuntimeError(f"eval nan: rollout step={t}")
                z_true = nrm(torch.tensor(state[lo + 2 + t])).to(DEV)
                roll_cos.append(float(F.cosine_similarity(p, z_true, dim=0)))
                s_prev, s_t = s_t, p
                n_roll += 1
        margin10 = float(np.mean(roll_cos)) - float(np.mean([psnr_copy for psnr_copy in copy_vals]))
        return step1, float(np.mean(roll_cos)), margin10

    # 拷贝基线（真值 s_t 保持）
    copy_vals = []
    for e in val_eps[:500]:
        lo, hi = bounds[e]
        if hi - lo < H_ROLL + 2:
            continue
        for t in range(H_ROLL):
            z_true = nrm(torch.tensor(state[lo + 2 + t])).to(DEV)
            z_hold = nrm(torch.tensor(state[lo + 1])).to(DEV)
            copy_vals.append(float(F.cosine_similarity(z_hold, z_true, dim=0)))
    copy10 = float(np.mean(copy_vals))
    print(f"拷贝基线 10 步 cos 均值: {copy10:.4f}")

    arms = {}
    for kind in ("analytic", "koop", "mlp"):
        a_list, m_list, s1_list = [], [], []
        for sd_i in SEEDS:
            m = train_one(kind, sd_i)
            s1, m10, _ = eval_arm(m)
            # a：孪生口径（动作固定重复）
            m64 = copy.deepcopy(m).double().eval()
            g = torch.Generator(device="cpu").manual_seed(5000)
            lam_all = []
            with torch.no_grad():
                for _ in range(16):
                    e = val_eps[int(torch.randint(0, len(val_eps), (1,), generator=g))]
                    lo, hi = bounds[e]
                    t0i = 5
                    za = nrm(torch.tensor(state[lo + t0i - 1])).double().to(DEV)
                    zb = nrm(torch.tensor(state[lo + t0i])).double().to(DEV)
                    dv = torch.randn(zb.shape, generator=g).double().to(DEV)
                    zb2 = zb + dv / dv.norm() * 1e-4
                    a_rep = torch.tensor(act[lo + t0i]).double().to(DEV)
                    sep = [1e-12]
                    for t in range(40):
                        p1 = m64(za.unsqueeze(0), zb.unsqueeze(0), a_rep.unsqueeze(0))[0]
                        p2 = m64(za.unsqueeze(0), zb2.unsqueeze(0), a_rep.unsqueeze(0))[0]
                        sep.append(float((p2 - p1).norm()))
                        za, zb, zb2 = zb, p1, p2
                    arr = np.array(sep)
                    if arr[8] > 1e-15 and arr[20] > 0:
                        lam_all.append(math.log(arr[20] / arr[8]) / 12)
            a_list.append(float(np.median(lam_all)))
            m_list.append(m10); s1_list.append(s1)
            del m, m64
            torch.cuda.empty_cache()
        arms[kind] = dict(a_mean=float(np.mean(a_list)), a_sd=float(np.std(a_list, ddof=1)),
                          margin10=float(np.mean(m_list)), step1_cos=float(np.mean(s1_list)),
                          per_seed_a=[round(x, 4) for x in a_list])
        print(f"  {kind:<9} a={arms[kind]['a_mean']:+.4f}±{arms[kind]['a_sd']:.4f} "
              f"margin10={arms[kind]['margin10']:+.4f} step1cos={arms[kind]['step1_cos']:.4f}")

    an, kp = arms["analytic"], arms["koop"]
    r1_pass = bool(an["a_mean"] <= kp["a_mean"] and an["margin10"] >= 0.8 * kp["margin10"])
    verdict = ("PASS：显式坐标上解析先验有效（与 R0 潜空间负结果构成坐标错位假说的正反对照）"
               if r1_pass else
               "FAIL：坐标对但解析形式不适合 PushT 动力学（外推核≠正确解析形式），如实记录")
    print(f"\nR1 判定：{'PASS' if r1_pass else 'FAIL'}——{verdict}")

    out = dict(prereg=dict(rule="analytic a≤koop a AND margin10≥0.8×koop", seeds=list(SEEDS),
                           analytic_prediction_a=0.0),
               data=dict(episodes_total=len(uniq), train_ep=len(train_eps), val_ep=len(val_eps),
                          train_pairs=len(Atr), copy10=copy10),
               arms=arms, r1_pass=r1_pass, verdict=verdict,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "r1_state_core.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/r1_state_core.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
