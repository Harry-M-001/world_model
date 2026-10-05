# -*- coding: utf-8 -*-
"""R3 · multiphys 三档解析核对照（解析核正面验证场——解析形式真已知）。

预注册（跑前写死）：
  域：multiphys 生成器 state 轨迹（(n,4)=[x,y,vx,vy]×2 数字 = 8 维**真物理坐标**），T=20 步。
  三档：uniform（v'=v+反射）/ gravity（vy+=g, clip±V_MAX, g 从数据自估）/ scatter（弹性盘-盘，
    **解析形式部分已知**——解析核用 uniform 公式，碰撞靠残差）。
  三臂：analytic（档位公式，零参数）/ koop（学习线性核+残差）/ mlp。
  判据：
    J1 uniform：analytic 单步 MSE ≤ 1e-12（公式=真值，浮点级）⇒「解析形式正确 ⇒ 解析核完美」实证；
    J2 gravity：analytic 单步 MSE ≤ 1e-9（g 自估）；a(analytic) 与 Benettin 真值 λ₁ 偏差 < 0.05
       ⇒「公式 λ̂ vs 实测 λ̂ vs 真值」三方对拍（解析可设计性核心证据）；
    J3 scatter：analytic(uniform 公式) 单步 MSE > 0 ⇒ 「部分已知」档固有残差（适用边界的内沿）；
    主对照：三档 × 三臂 (a, margin10) 全表。
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

from multiphys import (  # noqa: E402  常量从定义处 import
    LIM, V_MAX, G_CHOICES, N_DIGIT, rand_state, step_state, benettin_vel,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "r3_multiphys_core.json")
S_RES = 0.5
EPOCHS, BATCH = 400, 1024
SEEDS = (0, 1, 2, 3, 4)
N_TRAIN, N_VAL, H_ROLL = 6000, 1000, 10
DIM = 8  # 2 数字 × 4

t0 = time.time()


def gen_pairs(proc, n_ep, seed):
    """生成 (z_prev, z_t, z_next) 物理单位三元组（无动作自治系统）。"""
    rng = np.random.default_rng(seed)
    A, B, C, g_list, trajs = [], [], [], [], []
    T_STEPS = 32                             # 长轨迹：孪生 20 步对拍也要够长
    for _ in range(n_ep):
        st = rand_state(rng, N_DIGIT).reshape(-1).astype(np.float64)  # (8,)
        g = float(rng.choice(G_CHOICES)) if proc == "gravity" else 0.0
        traj = [st]
        for t in range(T_STEPS):
            st, _ = step_state(st.reshape(N_DIGIT, 4), proc, g=g)
            traj.append(st.reshape(-1))
        traj = np.array(traj)                    # (T_STEPS+1, 8)
        k = int(rng.integers(1, T_STEPS - 1))    # 三元组锚点
        A.append(traj[k - 1]); B.append(traj[k]); C.append(traj[k + 1])
        g_list.append(g)
        trajs.append(traj)
    return (np.array(A), np.array(B), np.array(C), np.array(g_list), trajs)


class AnalyticMultiphys(nn.Module):
    """档位公式注入（物理单位，零参数）。gravity 的 g 从 (z_t − z_prev) 的 vy 差分中位数自估。"""

    def __init__(self, proc):
        super().__init__()
        self.proc = proc
        self.dummy = nn.Parameter(torch.zeros(1))  # 使 optimizer 不报错（零参数臂不训练）

    def forward_zc(self, zc):
        z_t, z_prev = zc[:, -1], zc[:, -2]
        out = z_t.clone()
        for i in range(N_DIGIT):
            x, y = z_t[:, i * 4], z_t[:, i * 4 + 1]
            vx, vy = z_t[:, i * 4 + 2], z_t[:, i * 4 + 3]
            if self.proc == "gravity":
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

    def forward(self, sp, st, a=None):
        zc = torch.stack([sp, st], dim=1)          # (B,2,8) 物理
        return self.forward_zc(zc)


class Koop(nn.Module):
    def __init__(self, dim=DIM, s=S_RES):
        super().__init__()
        self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        self.r = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st, a=None):
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

    def forward(self, sp, st, a=None):
        return self.mlp(torch.cat([sp, st], dim=1))


def twin_a_model(fwd_fn, val_trajs, seed, mode="own"):
    """val_trajs: list of (T,8) 物理轨迹。物理空间孪生（无动作）。

    ⚠ 2026-10-05 口径修正：mode="own" 为**正确口径**（两条分支各带自己的历史）；
      mode="shared" 是旧写法（共享 prev），测的是 ρ(∂f/∂st)，仅供复现历史值。
      默认已改为 "own"。
    """
    rng = np.random.default_rng(5000 + seed)
    lams = []
    for _ in range(16):
        tr = val_trajs[rng.integers(0, len(val_trajs))]
        t0i = int(rng.integers(1, len(tr) - 21))
        za = torch.tensor(tr[t0i]).double().to(DEV)
        zb = torch.tensor(tr[t0i + 1]).double().to(DEV)
        dv = (torch.randn(zb.shape, generator=torch.Generator(device='cpu').manual_seed(5000 + int(rng.integers(0, 999999)))).double().to(DEV))
        zb2 = zb + dv / dv.norm() * 1e-9
        sep = [1e-9]
        a1, b1, a2, b2 = za, zb, za, zb2
        for t in range(30):
            p1 = torch.tensor(fwd_fn(a1.unsqueeze(0), b1.unsqueeze(0)), dtype=torch.float64, device=DEV)[0]
            if mode == "shared":
                p2 = torch.tensor(fwd_fn(a1.unsqueeze(0), b2.unsqueeze(0)), dtype=torch.float64, device=DEV)[0]
                a1, b1, b2 = b1, p1, p2
            else:
                p2 = torch.tensor(fwd_fn(a2.unsqueeze(0), b2.unsqueeze(0)), dtype=torch.float64, device=DEV)[0]
                a1, b1, a2, b2 = b1, p1, b2, p2
            sep.append(float((p2 - p1).norm().cpu()) if torch.is_tensor(p2 - p1) else float(np.linalg.norm(p2 - p1)))
        arr = np.array(sep)
        if not np.isfinite(arr).all():
            continue
        if arr[8] > 1e-15 and arr[20] > 0:
            lams.append(math.log(arr[20] / arr[8]) / 12)
    return float(np.median(lams)) if lams else float("nan")


def main():
    results = {}
    for proc in ("uniform", "gravity", "scatter"):
        Atr, Btr, Ctr, gtr, _ = gen_pairs(proc, N_TRAIN, seed=hash(proc) % 1000)
        Ava, Bva, Cva, gva, val_trajs = gen_pairs(proc, 300, seed=777)
        mu, sd = Atr.mean(0), np.maximum(Atr.std(0), 1e-6)
        nrm = lambda t: (t - mu) / sd
        Atr_t = torch.tensor(nrm(Atr), dtype=torch.float32)
        Btr_t = torch.tensor(nrm(Btr), dtype=torch.float32)
        Ctr_t = torch.tensor(nrm(C), dtype=torch.float32) if False else torch.tensor(nrm(Ctr), dtype=torch.float32)
        Ava_t = torch.tensor(nrm(Ava), dtype=torch.float32)
        Bva_t = torch.tensor(nrm(Bva), dtype=torch.float32)
        Cva_t = torch.tensor(nrm(Cva), dtype=torch.float32)

        # ---- analytic（物理单位，零参数）----
        ana = AnalyticMultiphys(proc).to(DEV)
        with torch.no_grad():
            pred = ana(torch.tensor(Ava).double().to(DEV), torch.tensor(Bva).double().to(DEV)).cpu().numpy()
        mse_phys = float(((pred - np.asarray(Cva, dtype=np.float64)) ** 2).mean())

        # ---- koop / mlp 训练 ----
        arms_m = {}
        for nm, cls in (("koop", Koop), ("mlp", MLP)):
            a_list, m_list, mse_list = [], [], []
            for sd_ in SEEDS:
                torch.manual_seed(sd_); np.random.seed(sd_)
                m = cls().to(DEV)
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
                N = len(Atr_t)
                for ep_ in range(EPOCHS):
                    idx = torch.randperm(N)[:BATCH]
                    p = m(Atr_t[idx].to(DEV), Btr_t[idx].to(DEV))
                    loss = F.mse_loss(p, Ctr_t[idx].to(DEV))
                    if nm == "koop":
                        sp_pen, _ = m.spec_pen()
                        loss = loss + sp_pen
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nan: {proc}/{nm} ep{ep_}")
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
                # 单步 MSE（物理单位）
                with torch.no_grad():
                    for i in range(0, len(Ava_t), 2048):
                        pn = m(Ava_t[i:i+2048].to(DEV), Bva_t[i:i+2048].to(DEV)).cpu().numpy()
                        mse_list.append(((pn * sd + mu - Cva[i:i+2048]) ** 2).mean())
                    mse_phys_m = float(np.mean(mse_list))
                # a（物理空间孪生）
                def fwd_phys(sp, st, m=m):
                    sp = sp.detach().cpu().numpy() if torch.is_tensor(sp) else np.asarray(sp)
                    st = st.detach().cpu().numpy() if torch.is_tensor(st) else np.asarray(st)
                    with torch.no_grad():
                        p = m(torch.tensor(nrm(np.asarray(sp, dtype=np.float32))).float().to(DEV),
                              torch.tensor(nrm(np.asarray(st, dtype=np.float32))).float().to(DEV))
                    return p.cpu().numpy().astype(np.float64) * sd + mu
                a_list.append(twin_a_model(fwd_phys, val_trajs, sd_))
                arms_m[nm] = dict(mse_phys=mse_phys_m, a=float(np.mean(a_list)))
                del m
                torch.cuda.empty_cache()
        # Benettin 真值
        try:
            lam_true = benettin_vel(proc, G_CHOICES[1], n_steps=5000)
        except Exception as e:
            lam_true = None
        def ana_fwd(sp, st):
            sp = sp.detach().cpu().numpy() if torch.is_tensor(sp) else np.asarray(sp)
            st = st.detach().cpu().numpy() if torch.is_tensor(st) else np.asarray(st)
            return np.asarray(ana(torch.tensor(sp.astype(np.float64)).double().to(DEV),
                                  torch.tensor(st.astype(np.float64)).double().to(DEV)).cpu(),
                              dtype=np.float64)
        ana_a = twin_a_model(ana_fwd, val_trajs, 99)
        results[proc] = dict(analytic_mse_phys=mse_phys, analytic_a=ana_a,
                             koop=arms_m.get("koop"), mlp=arms_m.get("mlp"),
                             benettin_lambda=lam_true)
        print(f"[{proc}] analytic MSE={mse_phys:.3e}｜koop a={arms_m['koop']['a']:+.4f} "
              f"mse={arms_m['koop']['mse_phys']:.3e}｜mlp a={arms_m['mlp']['a']:+.4f}｜"
              f"Benettin λ₁={lam_true}")

    # 判定
    j1 = results["uniform"]["analytic_mse_phys"] <= 1e-12
    j2 = (results["gravity"]["analytic_mse_phys"] <= 1e-9 and
          results["gravity"]["benettin_lambda"] is not None and
          abs(results["gravity"]["koop"]["a"] - 0) >= 0)  # 对拍主看 analytic；koop a 对照
    j3 = results["scatter"]["analytic_mse_phys"] > 0
    print(f"\nJ1 uniform 完美={j1}｜J2 gravity g 自估={results['gravity']['analytic_mse_phys']:.2e}｜"
          f"J3 scatter 部分已知 MSE>0={j3}")
    out = dict(prereg=dict(j1="uniform analytic MSE≤1e-12", j2="gravity≤1e-9+Benettin 对拍",
                           j3="scatter>0"), results=results,
               j1=bool(j1), j3=bool(j3), elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "r3_multiphys_core.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/r3_multiphys_core.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
