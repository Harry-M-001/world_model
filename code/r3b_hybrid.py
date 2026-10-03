# -*- coding: utf-8 -*-
"""R3b · 混合臂实验：analytic 公式打底 + 有界残差（R 线最后一块——工程可用形态）。

预注册（跑前写死）：
  四臂 × 三档（uniform/gravity/scatter，数据管线与 R3 逐字同源）：
    analytic  纯公式零参数（R3 归档：uniform 0 / gravity 6.14e-2 / scatter 8.27e-1）
    hybrid    档位公式打底 + s·tanh(r(zc)) 有界残差（可训练，残差只学公式没覆盖的部分）
    koop      学习线性核 + 残差
    mlp       纯学习
  判据：
    H1 scatter（核心）：hybrid 单步 MSE < analytic 纯公式（8.27e-1）且 < koop ⇒
       「公式打底+残差补差」在部分已知域有效（R 线工程形态成立）；
    H2 uniform（不伤害对照）：hybrid ≈ analytic（残差应学到 ≈0）⇒ 公式完美时残差不伤害；
    H3 gravity：hybrid < analytic 自估（6.14e-2）。
"""
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

from multiphys import LIM, V_MAX, G_CHOICES, N_DIGIT, rand_state, step_state  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "r3b_hybrid.json")
S_RES = 0.5
EPOCHS, BATCH = 400, 1024
SEEDS = (0, 1, 2, 3, 4)
N_TRAIN, N_VAL = 6000, 1000
DIM = 8
t0 = time.time()


def gen_pairs(proc, n_ep, seed):
    rng = np.random.default_rng(seed)
    A, B, C = [], [], []
    for _ in range(n_ep):
        st = rand_state(rng, N_DIGIT).reshape(-1).astype(np.float64)
        g = float(rng.choice(G_CHOICES)) if proc == "gravity" else 0.0
        traj = [st]
        for t in range(12):
            st, _ = step_state(st.reshape(N_DIGIT, 4), proc, g=g)
            traj.append(st.reshape(-1))
        traj = np.array(traj)
        k = int(rng.integers(1, 11))
        A.append(traj[k - 1]); B.append(traj[k]); C.append(traj[k + 1])
    return np.array(A), np.array(B), np.array(C)


def analytic_step(zc, proc):
    """档位公式（物理单位）。zc (B,2,8)。gravity 的 g 从 vy 差分中位数自估。"""
    z_t, z_prev = zc[:, -1], zc[:, -2]
    out = z_t.clone()
    for i in range(N_DIGIT):
        x, y = z_t[:, i * 4], z_t[:, i * 4 + 1]
        vx, vy = z_t[:, i * 4 + 2], z_t[:, i * 4 + 3]
        if proc == "gravity":
            g_est = torch.median(torch.stack([vy - z_prev[:, i * 4 + 3],
                                              z_t[:, i * 4 + 3] - z_prev[:, i * 4 + 3]]))
            vy = torch.clamp(vy + g_est, -V_MAX, V_MAX)
        nx, ny = x + vx, y + vy
        ref_x = (nx < 0) | (nx > LIM)
        ref_y = (ny < 0) | (ny > LIM)
        vx = torch.where(ref_x, -vx, vx)
        vy = torch.where(ref_y, -vy, vy)
        out[:, i * 4] = nx.clamp(0, LIM)
        out[:, i * 4 + 1] = ny.clamp(0, LIM)
        out[:, i * 4 + 2] = vx
        out[:, i * 4 + 3] = vy
    return out


class Analytic(nn.Module):
    def __init__(self, proc):
        super().__init__()
        self.proc = proc

    def forward(self, sp, st):
        return analytic_step(torch.stack([sp, st], dim=1), self.proc)


class Hybrid(nn.Module):
    def __init__(self, proc, mu, sd, dim=DIM, s=S_RES):
        super().__init__()
        self.proc = proc
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))
        self.r = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st):
        # 公式必须作用在物理坐标（R3 教训：标准化空间跑公式=形式错配复刻）
        sp_p = sp * self.sd + self.mu
        st_p = st * self.sd + self.mu
        base = analytic_step(torch.stack([sp_p, st_p], dim=1), self.proc)
        base_n = (base - self.mu) / self.sd
        return base_n + self.s * torch.tanh(self.r(torch.cat([sp, st], dim=1)))


class Koop(nn.Module):
    def __init__(self, dim=DIM, s=S_RES):
        super().__init__()
        self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        self.r = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st):
        return st @ self.K.T + self.s * torch.tanh(self.r(torch.cat([sp, st], dim=1)))

    def spec_pen(self, rho0=1.0):
        g = torch.Generator(device="cpu").manual_seed(123)
        v = torch.randn(DIM, generator=g).to(DEV); v = v / v.norm()
        for _ in range(5):
            u = self.K @ v; u = u / u.norm().clamp_min(1e-9)
            v = self.K.T @ u; v = v / v.norm().clamp_min(1e-9)
            sigma = (u @ self.K @ v).abs()
        return torch.relu(sigma - rho0) ** 2, sigma.detach()


class MLP(nn.Module):
    def __init__(self, dim=DIM):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(),
                                 nn.Linear(128, dim))

    def forward(self, sp, st):
        return self.mlp(torch.cat([sp, st], dim=1))


def main():
    out = dict(prereg=dict(h1="scatter hybrid<analytic 且 <koops", h2="uniform hybrid≈analytic",
                           h3="gravity hybrid<analytic"), arms={}, )
    table = {}
    for proc in ("uniform", "gravity", "scatter"):
        Atr, Btr, Ctr = gen_pairs(proc, N_TRAIN, seed=hash(proc) % 1000)
        Ava, Bva, Cva = gen_pairs(proc, N_VAL, seed=777)
        mu, sd = Atr.mean(0), np.maximum(Atr.std(0), 1e-6)
        nrm = lambda t: (t - mu) / sd
        Atr_t = torch.tensor(nrm(Atr), dtype=torch.float32)
        Btr_t = torch.tensor(nrm(Btr), dtype=torch.float32)
        Ctr_t = torch.tensor(nrm(Ctr), dtype=torch.float32)
        Ava_t = torch.tensor(nrm(Ava), dtype=torch.float32)
        Bva_t = torch.tensor(nrm(Bva), dtype=torch.float32)

        def phys_mse(pred_n):
            return float(((pred_n * sd + mu - Cva) ** 2).mean())

        arms = {}
        # analytic（零训练）
        ana = Analytic(proc).to(DEV)
        with torch.no_grad():
            for i in range(0, len(Ava_t), 2048):
                p = ana(Ava_t[i:i+2048].to(DEV), Bva_t[i:i+2048].to(DEV)).cpu().numpy()
                if i == 0:
                    parts = [p]
                else:
                    parts.append(p)
        arms["analytic"] = dict(mse=phys_mse(np.concatenate(parts)), trained=False)

        for nm, cls, spec in (("hybrid", Hybrid, False), ("koop", Koop, True), ("mlp", MLP, False)):
            mse_list = []
            for sd_ in SEEDS:
                torch.manual_seed(sd_); np.random.seed(sd_)
                m = (cls(proc, mu, sd) if nm == "hybrid" else cls()).to(DEV)
                if nm == "hybrid":
                    m = m.to(DEV)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
                N = len(Atr_t)
                for ep_ in range(EPOCHS):
                    idx = torch.randperm(N)[:BATCH]
                    p = m(Atr_t[idx].to(DEV), Btr_t[idx].to(DEV))
                    loss = F.mse_loss(p, Ctr_t[idx].to(DEV))
                    if spec:
                        sp_pen, _ = m.spec_pen()
                        loss = loss + sp_pen
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nan: {proc}/{nm} ep{ep_}")
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                with torch.no_grad():
                    for i in range(0, len(Ava_t), 2048):
                        pn = m(Ava_t[i:i+2048].to(DEV), Bva_t[i:i+2048].to(DEV)).cpu().numpy()
                        mse_list.append(((pn * sd + mu - Cva[i:i+2048]) ** 2).mean())
                del m
                torch.cuda.empty_cache()
            arms[nm] = dict(mse=float(np.mean(mse_list)), sd=float(np.std(mse_list)), trained=True)
            print(f"  [{proc}] {nm:<8} MSE={arms[nm]['mse']:.4e}", flush=True)
        print(f"  [{proc}] analytic {arms['analytic']['mse']:.4e}（零训练）")
        table[proc] = arms

    # 判定
    h1 = bool(table["scatter"]["hybrid"]["mse"] < table["scatter"]["analytic"]["mse"]
              and table["scatter"]["hybrid"]["mse"] < table["scatter"]["koop"]["mse"])
    rel_h2 = table["uniform"]["hybrid"]["mse"] / max(table["uniform"]["analytic"]["mse"], 1e-12)
    h2 = bool(table["uniform"]["hybrid"]["mse"] < 1e-9)
    h3 = bool(table["gravity"]["hybrid"]["mse"] < table["gravity"]["analytic"]["mse"])
    print(f"\nH1 scatter：hybrid {table['scatter']['hybrid']['mse']:.3e} vs analytic "
          f"{table['scatter']['analytic']['mse']:.3e} vs koop {table['scatter']['koop']['mse']:.3e} → "
          f"{'PASS：公式打底+残差补差成立' if h1 else 'FAIL'}")
    print(f"H2 uniform 不伤害：hybrid {table['uniform']['hybrid']['mse']:.3e}（analytic "
          f"{table['uniform']['analytic']['mse']:.3e}）→ {'PASS' if h2 else '部分'}")
    print(f"H3 gravity：hybrid {table['gravity']['hybrid']['mse']:.3e} vs analytic "
          f"{table['gravity']['analytic']['mse']:.3e} → {'PASS' if h3 else 'FAIL'}")

    out = dict(table={k: {kk: vv for kk, vv in v.items()} for k, v in table.items()},
               h1=bool(h1), h2=bool(h2), h3=bool(h3),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "r3b_hybrid.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/r3b_hybrid.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
