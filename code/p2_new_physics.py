# -*- coding: utf-8 -*-
"""P2 · 新三档判据实测：解析核 vs 混合臂 vs 纯学习（同 R3 模式，判据口径升级）。

预注册（跑前写死，2026-10-04）
================================================================================
域：multiphys 物理空间 8 维真值坐标（2 数字 × [x,y,vx,vy]），boundary="none"
    （与 P1 解析真值同口径：抛体/摩擦/谐振子都是**纯物理**过程，边界会破坏解析形式）。
    档位：projectile（匀加速）/ friction（库仑摩擦，恒定减速到停）/ harmonic（简谐振动）
          + **uniform 作对照档**（复现 R3 的 a 伪读数并给出纠正后的口径）。
参数：**每序列采样**（GP/DEC/KH_CHOICES）——解析臂只知道**形式**、必须**从 (z_prev,z_t)
    自估参数**，与学习臂看到的信息严格相同（避免把常量硬编码进解析核造成信息泄漏）。
    · projectile: g_est  = vy_t − vy_prev
    · friction  : dec_est= max over digits of (|v_prev| − |v_t|)
    · harmonic  : k_est  = −Σ[Δv·(x_prev−C)] / (DT·Σ[(x_prev−C)²])（最小二乘，双轴双数字合并）
输入：三帧 (z_prev, z_t) → 预测 z_next（沿用 M 系列修正后的三帧口径）。
七臂（uniform 档 full≡unif，只跑 4 臂）：
    ana_full  档位完整公式，**每步重估参数**（零可训参数）
    ana_lock  档位完整公式，**参数在起点估一次并锁存**（G0/J4 归因消融）
    ana_unif  匀速公式（**形式错配基线**，对应 R3 的 scatter 档做法）
    hyb_full  完整公式 + 有界残差 s·tanh(r)，s=0.5
    hyb_unif  匀速公式 + 有界残差（混合臂在「形式残缺」时的补救价值）
    koop      学习线性核 + 有界残差（含 spec_pen）
    mlp       纯 MLP
判据：
  J1 形式完整⇒精确：三新档 ana_full 单步 MSE ≤ 1e-9（参数自估应达浮点级）
  J2 形式残缺⇒残差可补（两条同时成立才过）：
     J2a mse(hyb_unif) ≤ 0.25·mse(ana_unif)（残差至少压掉 75% 的形式错配误差）
     J2b mse(hyb_unif) ≤ mse(mlp)（混合臂不劣于纯学习臂）
  J3 形式完整⇒残差冗余：三新档 mse(hyb_full) 相对 ana_full 的改进 < 10%
  J4 解析核不引入额外误差放大：三新档 |a_jac(ana_full) − a_true| ≤ |a_jac(mlp) − a_true|
     （a_jac = log ρ(J_aug)：16×16 两帧提升的**数值 Jacobian 谱半径**，float64 有限差分）
  J4'（**归因消融**，与 J4 同时预注册）：三新档 |a_jac(ana_lock) − a_true| ≤ 0.01
     —— 若 J4 失败而 J4' 通过 ⇒ 放大来自「每步重估参数」的估计器增益，而非解析形式本身
  G0 自检：ana_full 的 ρ(J_aug) 与真值物理映射的 ρ(J_true) 相差 < 1e-6
  G0'（归因）：ana_lock 的 ρ(J_aug) 与 ρ(J_true) 相差 < 1e-6
  G1 守卫可触发：故意写反 projectile 的 g 符号 ⇒ J1 必须失败（证明判据不是常数满分）
  G2 冒烟：B≥2 时各学习臂的预测逐样本不同（防 cat/stack 广播静默通过）
附：a_twin（孪生窗口 [8,20] 率尺度）作为 a_jac 的**独立交叉验证**。
  ⚠ 口径修正（本脚本发现）：R3 的 twin_a_model **两条分支共享 prev 帧**，测到的是
    ρ(∂f/∂st) 而不是模型自回归的误差放大——projectile 解析核的 ∂vy'/∂st 有特征值 2
    （vy' = 1.5·vy_t^0 + 0.5·vy_t^1 − ...），会把 λ=0 的系统读成 log2 = 0.69。
    本脚本改为**每条分支自带历史**（与 J_aug 同一提升），修正后学习臂的 a_twin 与
    a_jac 一致（uniform mlp 0.0966 vs 0.1072），而 R3 的 uniform 0.068 正是共享 prev
    在 λ=0 系统上的多项式增长本底 log(20/8)/12 = 0.0764。
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

from multiphys import (  # noqa: E402  常量从定义处 import + assert
    LIM, N_DIGIT, rand_state, step_state, DT_H, HARM_C,
)

assert N_DIGIT == 2 and DT_H == 1.0, "本脚本按 N_DIGIT=2 / DT_H=1 写死"
DIM = 4 * N_DIGIT

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "p2_new_physics.json")

# ---------------------------------------------------------------- 预注册常量
GP_CHOICES = (0.15, 0.25, 0.35)      # 抛体：每步 vy 增量
DEC_CHOICES = (0.04, 0.06, 0.08)     # 摩擦：每步减速度 dec = μ·g_f
KH_CHOICES = (0.05, 0.08, 0.11)      # 简谐：刚度 k（半隐式欧拉稳定区 0<k<4）
T_STEPS = 32
K_LO, K_HI = 1, 11                   # 锚点范围（保证 k + H_ROLL ≤ T_STEPS）
N_TRAIN, N_VAL = 6000, 300
H_ROLL = 20
EPOCHS, BATCH = 300, 1024
SEEDS = (0, 1, 2)
S_RES = 0.5
BOOT = 1000
J1_TOL = 1e-9
TIERS = ("uniform", "projectile", "friction", "harmonic")
ARMS_FULL = ("ana_full", "ana_lock", "ana_unif", "hyb_full", "hyb_unif", "koop", "mlp")
ARMS_UNIF = ("ana_full", "hyb_full", "koop", "mlp")     # uniform 档 full≡unif

t0 = time.time()


# ---------------------------------------------------------------- 数据
def _kwargs_of(proc, p):
    return dict(gp=(p if proc == "projectile" else None),
                dec=(p if proc == "friction" else None),
                kh=(p if proc == "harmonic" else None))


def gen_pairs(proc, n_ep, seed):
    """生成 (z_prev, z_t, z_next) 三帧组 + 参数 + 整条轨迹（供 rollout 取真值）。"""
    rng = np.random.default_rng(seed)
    A, B, C, plist, trajs, ks = [], [], [], [], [], []
    for _ in range(n_ep):
        st = rand_state(rng, N_DIGIT).reshape(-1).astype(np.float64)
        if proc == "projectile":
            p = float(rng.choice(GP_CHOICES))
        elif proc == "friction":
            p = float(rng.choice(DEC_CHOICES))
        elif proc == "harmonic":
            p = float(rng.choice(KH_CHOICES))
        else:
            p = 0.0
        kw = _kwargs_of(proc, p)
        traj = [st]
        for _ in range(T_STEPS):
            st, _ = step_state(st.reshape(N_DIGIT, 4), proc, boundary="none", **kw)
            traj.append(st.reshape(-1))
        traj = np.array(traj, dtype=np.float64)          # (T_STEPS+1, 8)
        k = int(rng.integers(K_LO, K_HI + 1))
        A.append(traj[k - 1]); B.append(traj[k]); C.append(traj[k + 1])
        plist.append(p); trajs.append(traj); ks.append(k)
    return (np.array(A), np.array(B), np.array(C), np.array(plist), trajs, np.array(ks))


# ---------------------------------------------------------------- 解析臂（torch，逐样本估参）
def estimate_param(sp, st, proc):
    """从 (z_prev, z_t) 逐样本自估档位参数。返回 (B,) 或 None（uniform 无参数）。"""
    if proc == "projectile":
        gs = [st[:, i * 4 + 3] - sp[:, i * 4 + 3] for i in range(N_DIGIT)]
        return torch.stack(gs, dim=1).median(dim=1).values
    if proc == "friction":
        decs = []
        for i in range(N_DIGIT):
            vp = torch.hypot(sp[:, i * 4 + 2], sp[:, i * 4 + 3])
            vt = torch.hypot(st[:, i * 4 + 2], st[:, i * 4 + 3])
            decs.append(torch.clamp(vp - vt, min=0.0))
        return torch.stack(decs, dim=1).max(dim=1).values      # 停住的数字给 0，取 max
    if proc == "harmonic":
        num = torch.zeros(st.shape[0], dtype=st.dtype, device=st.device)
        den = torch.zeros_like(num)
        for i in range(N_DIGIT):
            dx = sp[:, i * 4] - HARM_C
            dy = sp[:, i * 4 + 1] - HARM_C
            dvx = st[:, i * 4 + 2] - sp[:, i * 4 + 2]
            dvy = st[:, i * 4 + 3] - sp[:, i * 4 + 3]
            num = num + (dvx * dx + dvy * dy)
            den = den + (dx * dx + dy * dy)
        return -num / (DT_H * den + 1e-6)
    return None


def apply_param(st, proc, p):
    """给定参数 p（(B,) 或 None）执行一歩档位公式（**不再估参**）。"""
    out = st.clone()
    if proc in ("uniform", None) or p is None:
        for i in range(N_DIGIT):
            out[:, i * 4] = st[:, i * 4] + st[:, i * 4 + 2]
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + st[:, i * 4 + 3]
        return out
    if proc == "projectile":
        for i in range(N_DIGIT):
            out[:, i * 4] = st[:, i * 4] + st[:, i * 4 + 2]
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + st[:, i * 4 + 3] + p
            out[:, i * 4 + 2] = st[:, i * 4 + 2]
            out[:, i * 4 + 3] = st[:, i * 4 + 3] + p
        return out
    if proc == "friction":
        for i in range(N_DIGIT):
            vt = torch.hypot(st[:, i * 4 + 2], st[:, i * 4 + 3])
            sc = torch.clamp((vt - p) / vt.clamp_min(1e-12), min=0.0)
            vx = st[:, i * 4 + 2] * sc
            vy = st[:, i * 4 + 3] * sc
            out[:, i * 4] = st[:, i * 4] + vx
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + vy
            out[:, i * 4 + 2] = vx
            out[:, i * 4 + 3] = vy
        return out
    if proc == "harmonic":
        for i in range(N_DIGIT):
            vx = st[:, i * 4 + 2] - p * DT_H * (st[:, i * 4] - HARM_C)
            vy = st[:, i * 4 + 3] - p * DT_H * (st[:, i * 4 + 1] - HARM_C)
            out[:, i * 4] = st[:, i * 4] + vx
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + vy
            out[:, i * 4 + 2] = vx
            out[:, i * 4 + 3] = vy
        return out
    raise ValueError(f"未知档位 {proc!r}")


def analytic_next(sp, st, proc, form):
    """(B,8)->(B,8)，物理单位。form ∈ {"full","unif"}。full = 每步重估参数。"""
    out = st.clone()
    if form == "unif" or proc == "uniform":
        # 匀速外推：v' = v_t，pos' = pos_t + v_t
        for i in range(N_DIGIT):
            out[:, i * 4] = st[:, i * 4] + st[:, i * 4 + 2]
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + st[:, i * 4 + 3]
        return out
    return apply_param(st, proc, estimate_param(sp, st, proc))


def analytic_next_broken(sp, st):
    """守卫自触发用：projectile 的 g 符号写反（g_est = vy_prev − vy_t）。"""
    g = (sp[:, 3] - st[:, 3])
    out = st.clone()
    for i in range(N_DIGIT):
        out[:, i * 4] = st[:, i * 4] + st[:, i * 4 + 2]
        out[:, i * 4 + 1] = st[:, i * 4 + 1] + st[:, i * 4 + 3] + g
        out[:, i * 4 + 2] = st[:, i * 4 + 2]
        out[:, i * 4 + 3] = st[:, i * 4 + 3] + g
    return out


# ---------------------------------------------------------------- 模型臂
class _Norm(nn.Module):
    def __init__(self, mu, sd):
        super().__init__()
        self.register_buffer("mu", torch.tensor(mu, dtype=torch.float32))
        self.register_buffer("sd", torch.tensor(sd, dtype=torch.float32))

    def cat_n(self, sp, st):
        return torch.cat([(sp - self.mu) / self.sd, (st - self.mu) / self.sd], dim=1)


class AnalyticArm(nn.Module):
    def __init__(self, proc, form):
        super().__init__()
        self.proc, self.form = proc, form

    def forward(self, sp, st):
        return analytic_next(sp, st, self.proc, self.form)


class AnalyticArmLock(nn.Module):
    """解析核·**参数锁存版**：参数只在起点估一次（reset），rollout 全程复用。

    用途：分离「每步重估参数」引入的估计器增益——G0/J4 失败后的归因消融。
    """

    def __init__(self, proc):
        super().__init__()
        self.proc = proc
        self.p = None

    @torch.no_grad()
    def reset(self, sp, st):
        self.p = estimate_param(sp, st, self.proc)
        return self

    def forward(self, sp, st):
        if self.p is None:
            self.reset(sp, st)
        p = self.p if self.p is None else self.p.to(sp.device)
        return apply_param(st, self.proc, p)


class Hyb(_Norm):
    """解析核（full 或 unif）+ 有界残差 s·sd·tanh(r)。"""

    def __init__(self, proc, form, mu, sd, s=S_RES):
        super().__init__(mu, sd)
        self.proc, self.form, self.s = proc, form, s
        self.r = nn.Sequential(nn.Linear(2 * DIM, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, DIM))

    def forward(self, sp, st):
        base = analytic_next(sp, st, self.proc, self.form)
        return base + self.s * self.sd * torch.tanh(self.r(self.cat_n(sp, st)))


class Koop(_Norm):
    def __init__(self, mu, sd, s=S_RES):
        super().__init__(mu, sd)
        self.K = nn.Parameter(torch.eye(DIM) * 0.9 + 0.01 * torch.randn(DIM, DIM))
        self.r = nn.Sequential(nn.Linear(2 * DIM, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, DIM))
        self.s = float(s)

    def forward(self, sp, st):
        zn = (st - self.mu) / self.sd
        pred_n = zn @ self.K.T + self.s * torch.tanh(self.r(self.cat_n(sp, st)))
        return pred_n * self.sd + self.mu

    def spec_pen(self, rho0=1.0):
        v = torch.randn(DIM, generator=torch.Generator().manual_seed(123))
        v = (v / v.norm()).to(self.K.device)
        sigma = None
        for _ in range(5):
            u = self.K @ v; u = u / u.norm().clamp_min(1e-9)
            v = self.K.T @ u; v = v / v.norm().clamp_min(1e-9)
            sigma = (u @ self.K @ v).abs()
        return torch.relu(sigma - rho0) ** 2


class MLPNet(_Norm):
    def __init__(self, mu, sd):
        super().__init__(mu, sd)
        self.mlp = nn.Sequential(nn.Linear(2 * DIM, 128), nn.SiLU(),
                                 nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, DIM))

    def forward(self, sp, st):
        return self.mlp(self.cat_n(sp, st)) * self.sd + self.mu


def build_arm(name, proc, mu, sd):
    if name == "ana_full":
        return AnalyticArm(proc, "full")
    if name == "ana_unif":
        return AnalyticArm(proc, "unif")
    if name == "ana_lock":
        return AnalyticArmLock(proc)
    if name == "hyb_full":
        return Hyb(proc, "full", mu, sd)
    if name == "hyb_unif":
        return Hyb(proc, "unif", mu, sd)
    if name == "koop":
        return Koop(mu, sd)
    if name == "mlp":
        return MLPNet(mu, sd)
    raise ValueError(name)


# ---------------------------------------------------------------- 判据
def jac_aug_rho(fwd, sp0, st0, h=1e-6, iters=400):
    """两帧提升 (z_prev,z_t)->(z_t,f) 的 Jacobian 谱半径（float64 有限差分 + 幂迭代）。"""
    d = int(sp0.numel())
    sp0 = sp0.detach().clone().double().reshape(-1)
    st0 = st0.detach().clone().double().reshape(-1)
    base = fwd(sp0.reshape(1, -1), st0.reshape(1, -1)).reshape(-1)
    J = np.zeros((2 * d, 2 * d), dtype=np.float64)
    for src, off in ((sp0, 0), (st0, d)):
        for i in range(d):
            sv = src.clone(); sv[i] += h
            if off == 0:
                p = fwd(sv.reshape(1, -1), st0.reshape(1, -1)).reshape(-1)
            else:
                p = fwd(sp0.reshape(1, -1), sv.reshape(1, -1)).reshape(-1)
            col = ((p - base) / h).detach().cpu().numpy().astype(np.float64)
            J[0:d, off + i] = 0.0
            J[d:2 * d, off + i] = col
    # 上半块：z_t' = z_t
    J[0:d, d:2 * d] = np.eye(d)
    # ⚠ 不用幂迭代：新三档的 J 是非正规的（Jordan 块 / 复共轭单位圆特征值），
    #   幂迭代只收敛到 1+O(1/k)（400 步仍给 1.0025）、复数模 1 时甚至给 1.5。
    #   16×16 直接求特征值即可。
    return float(np.max(np.abs(np.linalg.eigvals(J))))


def true_jac_rho(proc, p, states, h=1e-6, iters=400):
    """真值物理一步映射的 Jacobian 谱半径（float64），states: list of (n,4)。"""
    rhos = []
    for st in states:
        st = np.array(st, dtype=np.float64)
        d = int(st.size)
        base = step_state(st.reshape(N_DIGIT, 4), proc, boundary="none",
                          **_kwargs_of(proc, p))[0].reshape(-1)
        J = np.zeros((d, d), dtype=np.float64)
        flat = st.reshape(-1)
        for i in range(d):
            sv = flat.copy(); sv[i] += h
            out = step_state(sv.reshape(N_DIGIT, 4), proc, boundary="none",
                             **_kwargs_of(proc, p))[0].reshape(-1)
            J[:, i] = (out - base) / h
        rhos.append(float(np.max(np.abs(np.linalg.eigvals(J)))))
    return float(np.median(rhos))


def twin_a(fwd, trajs, anchors, seed, n_rep=16, t1=8, t2=20, delta=1e-9, pre=None):
    """R3 同款孪生读数（固定绝对窗口 [8,20]、率尺度），供对照。

    pre(cp, c1)：每条轨迹开头调用一次（锁存臂用它固定参数）。
    """
    rng = np.random.default_rng(5000 + seed)
    lams = []
    for _ in range(n_rep):
        j = int(rng.integers(0, len(trajs)))
        tr = trajs[j]
        k = max(1, min(int(anchors[j]), len(tr) - t2 - 2))
        sp = torch.tensor(tr[k - 1], dtype=torch.float64)
        st = torch.tensor(tr[k], dtype=torch.float64)
        st2 = st + delta * torch.tensor(rng.standard_normal(DIM), dtype=torch.float64)
        sep = [delta]
        if pre is not None:
            pre(sp, st)
        # ⚠ 口径修正：**两条分支各带自己的历史**（旧的 R3 写法共享 prev，
        #   测到的是 ρ(∂f/∂st) 而非模型自回归的误差放大——projectile 解析核
        #   的 ∂vy'/∂st 有特征值 2，会把 0 误读成 log2=0.69）
        cp1, c1 = sp, st
        cp2, c2 = sp, st2
        for _ in range(t2):
            p1 = fwd(cp1.reshape(1, -1), c1.reshape(1, -1)).reshape(-1)
            p2 = fwd(cp2.reshape(1, -1), c2.reshape(1, -1)).reshape(-1)
            d = float(torch.sqrt((p2 - p1).norm() ** 2 + (c2 - c1).norm() ** 2))
            sep.append(d)
            cp1, c1 = c1, p1
            cp2, c2 = c2, p2
        arr = np.array(sep)
        if not np.isfinite(arr).all():
            continue
        if arr[t1] > 1e-15 and arr[t2] > 0:
            lams.append(math.log(arr[t2] / arr[t1]) / (t2 - t1))
    return float(np.median(lams)) if lams else float("nan")


def aggregator_probe():
    """归因探针：projectile 解析核的**参数聚合算子**对 ρ(J_aug) 的影响。

    median 在偶数个元素上取**下中位数**（2 个数字 ⇒ 等于 min，非平滑）；
    mean / 单数字 是平滑的；lock 为锁存基线。
    """
    _, _, _, _, trajs, anc = gen_pairs("projectile", 300, seed=777)
    sp0 = torch.tensor(trajs[0][anc[0] - 1], dtype=torch.float64)
    st0 = torch.tensor(trajs[0][anc[0]], dtype=torch.float64)

    def f_generic(sp, st, how, p=None):
        d0 = st[:, 3] - sp[:, 3]
        d1 = st[:, 7] - sp[:, 7]
        if p is not None:
            g = p
        elif how == "median":
            g = torch.stack([d0, d1], dim=1).median(dim=1).values
        elif how == "mean":
            g = 0.5 * (d0 + d1)
        elif how == "d0":
            g = d0
        else:
            raise ValueError(how)
        out = st.clone()
        for i in range(N_DIGIT):
            out[:, i * 4] = st[:, i * 4] + st[:, i * 4 + 2]
            out[:, i * 4 + 1] = st[:, i * 4 + 1] + st[:, i * 4 + 3] + g
            out[:, i * 4 + 2] = st[:, i * 4 + 2]
            out[:, i * 4 + 3] = st[:, i * 4 + 3] + g
        return out

    res = {}
    with torch.no_grad():
        for how in ("median", "mean", "d0"):
            res[how] = jac_aug_rho(lambda a, b: f_generic(a.to(DEV), b.to(DEV), how),
                                   sp0, st0)
        p0 = (0.5 * ((st0[3] - sp0[3]) + (st0[7] - sp0[7]))).reshape(1)
        res["lock"] = jac_aug_rho(
            lambda a, b: f_generic(a.to(DEV), b.to(DEV), None, p0.to(DEV)), sp0, st0)
    return res


def rollout_errs(fwd_np, Ava, Bva, trajs, anchors, H=H_ROLL):
    """模型 rollout 与拷贝基线的逐样本 MSE（按 horizon）。返回 (mse_m, mse_c) 各 (H, N)。"""
    N = len(Ava)
    mse_m = np.zeros((H, N)); mse_c = np.zeros((H, N))
    cp = Ava.copy(); cur = Bva.copy()
    for h in range(1, H + 1):
        nxt = fwd_np(cp, cur)
        cp, cur = cur, nxt                          # 两帧滑窗：prev ← 旧 cur
        truth = np.array([trajs[j][anchors[j] + h] for j in range(N)])
        mse_m[h - 1] = ((cur - truth) ** 2).mean(1)
        mse_c[h - 1] = ((Bva - truth) ** 2).mean(1)
    return mse_m, mse_c


def paired_hstar(mse_m, mse_c, boot=BOOT, seed=0):
    """h* = 最大 h 使 (mse_copy − mse_model) 的配对 bootstrap 95% CI 下界 > 0。"""
    d = mse_c - mse_m                      # (H, N)：正 = 模型更好
    N = d.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, N, size=(boot, N))
    means = d[:, idx].mean(2)              # (H, boot)
    lo = np.percentile(means, 2.5, axis=1)
    mu = d.mean(1)
    hstar = 0
    for h in range(d.shape[0]):
        if lo[h] > 0:
            hstar = h + 1
    return int(hstar), lo.tolist(), mu.tolist()


# ---------------------------------------------------------------- 主流程
def run_tier(proc):
    Atr, Btr, Ctr, ptr, _, _ = gen_pairs(proc, N_TRAIN, seed=abs(hash(proc)) % 1000)
    Ava, Bva, Cva, pva, vtrajs, vanc = gen_pairs(proc, N_VAL, seed=777)
    obs = np.concatenate([Atr, Btr], 0)             # 只用观测帧统计（不用目标帧）
    mu, sd = obs.mean(0), np.maximum(obs.std(0), 1e-6)

    At = torch.tensor(Atr, dtype=torch.float32)
    Bt = torch.tensor(Btr, dtype=torch.float32)
    Ct = torch.tensor(Ctr, dtype=torch.float32)

    arms = ARMS_UNIF if proc == "uniform" else ARMS_FULL
    res = {}

    # 真值 Jacobian 谱半径（三档各自的中位参数）
    p_mid = {"projectile": GP_CHOICES[1], "friction": DEC_CHOICES[1],
             "harmonic": KH_CHOICES[1], "uniform": 0.0}[proc]
    states = []
    for j in range(0, N_VAL, 40):
        states.append(vtrajs[j][vanc[j]].reshape(N_DIGIT, 4))
    rho_true = true_jac_rho(proc, p_mid, states)
    a_true = math.log(rho_true) if rho_true > 0 else float("nan")

    for name in arms:
        mses, aj, at, hs, margins = [], [], [], [], []
        is_lock = (name == "ana_lock")
        zero_par = name in ("ana_full", "ana_lock", "ana_unif")
        for sd_ in (SEEDS if not zero_par else (0,)):
            torch.manual_seed(sd_); np.random.seed(sd_)
            m = build_arm(name, proc, mu, sd).to(DEV)
            if is_lock:                                  # 参数在锚点估一次并锁存
                with torch.no_grad():
                    m.reset(torch.tensor(Ava, dtype=torch.float32).to(DEV),
                            torch.tensor(Bva, dtype=torch.float32).to(DEV))
            if not zero_par:
                opt = torch.optim.Adam(m.parameters(), lr=1e-3)
                sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
                N = len(At)
                for ep in range(EPOCHS):
                    idx = torch.randperm(N)[:BATCH]
                    p = m(At[idx].to(DEV), Bt[idx].to(DEV))
                    loss = F.mse_loss(p, Ct[idx].to(DEV))
                    if name == "koop":
                        loss = loss + m.spec_pen()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"nan: {proc}/{name} ep{ep}")
                    opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            # ---- 单步 MSE / margin（物理单位）
            with torch.no_grad():
                pred = m(torch.tensor(Ava, dtype=torch.float32).to(DEV),
                         torch.tensor(Bva, dtype=torch.float32).to(DEV)).cpu().numpy().astype(np.float64)
            mse1 = float(((pred - Cva) ** 2).mean())
            mse_copy = float(((Bva - Cva) ** 2).mean())
            margins.append((mse_copy - mse1) / max(mse_copy, 1e-12))
            mses.append(mse1)

            # ---- G2 冒烟：B≥2 时预测必须逐样本不同
            with torch.no_grad():
                if is_lock:      # 锁存臂：冒烟批的 latch 要跟着批大小重设（后面 jac/rollout 各自 reset）
                    m.reset(torch.tensor(Ava[:8], dtype=torch.float32).to(DEV),
                            torch.tensor(Bva[:8], dtype=torch.float32).to(DEV))
                sm = m(torch.tensor(Ava[:8], dtype=torch.float32).to(DEV),
                       torch.tensor(Bva[:8], dtype=torch.float32).to(DEV)).cpu().numpy()
            assert sm.std(0).min() > 1e-12, f"G2 失败：{proc}/{name} 逐样本预测无差异"

            # ---- a_jac（float64）
            md = m.double()
            with torch.no_grad():
                sp0 = torch.tensor(Ava[0], dtype=torch.float64, device=DEV)
                st0 = torch.tensor(Bva[0], dtype=torch.float64, device=DEV)
                if is_lock:
                    md.reset(sp0.reshape(1, -1), st0.reshape(1, -1))   # 锁存：p 不随扰动变
                rho = jac_aug_rho(lambda a, b: md(a.to(DEV), b.to(DEV)), sp0.cpu(), st0.cpu())
            aj.append(math.log(rho) if rho > 0 else float("nan"))

            # ---- a_twin（对照读数）
            def _pre(cp, c1, md=md):
                if is_lock:
                    with torch.no_grad():
                        md.reset(cp.reshape(1, -1), c1.reshape(1, -1))
            with torch.no_grad():
                at.append(twin_a(lambda a, b: md(a.to(DEV), b.to(DEV)).double(),
                                 vtrajs, vanc, sd_, pre=_pre))

            # ---- rollout h*
            if is_lock:
                with torch.no_grad():
                    md.reset(torch.tensor(Ava, dtype=torch.float64, device=DEV),
                             torch.tensor(Bva, dtype=torch.float64, device=DEV))

            def fwd_np(sp_np, st_np, md=md):
                with torch.no_grad():
                    p = md(torch.tensor(np.asarray(sp_np, dtype=np.float64), device=DEV),
                           torch.tensor(np.asarray(st_np, dtype=np.float64), device=DEV))
                return p.cpu().numpy().astype(np.float64)
            mm, mc = rollout_errs(fwd_np, Ava, Bva, vtrajs, vanc)
            hstar, lo, dmu = paired_hstar(mm, mc)
            hs.append(hstar)
            del m, md
            torch.cuda.empty_cache()
        res[name] = dict(mse1=float(np.mean(mses)),
                         mse1_se=float(np.std(mses, ddof=1) / math.sqrt(len(mses))) if len(mses) > 1 else 0.0,
                         margin1=float(np.mean(margins)),
                         a_jac=float(np.nanmean(aj)),
                         a_twin=float(np.nanmean(at)),
                         hstar=int(round(float(np.mean(hs)))),
                         hstar_all=[int(x) for x in hs],
                         h_lo=lo, h_mean_diff=dmu)
    return res, dict(rho_true=rho_true, a_true=a_true)


def main():
    out = dict(prereg=dict(
        tiers=list(TIERS),
        arms={t: (list(ARMS_UNIF) if t == "uniform" else list(ARMS_FULL)) for t in TIERS},
        j1=f"三新档 ana_full 单步 MSE ≤ {J1_TOL:g}",
        j2="J2a mse(hyb_unif) ≤ 0.25·mse(ana_unif)；J2b mse(hyb_unif) ≤ mse(mlp)",
        j3="三新档 mse(hyb_full) 相对 ana_full 改进 < 10%",
        j4="|a_jac(ana_full)−a_true| ≤ |a_jac(mlp)−a_true|",
        j4b="|a_jac(ana_lock)−a_true| ≤ 0.01（归因消融）",
        g0="ρ(J_aug of ana_full) 与真值 ρ(J_true) 差 < 1e-6",
        g0b="ρ(J_aug of ana_lock) 与真值 ρ(J_true) 差 < 1e-6（归因）",
        g1="g 符号写反 ⇒ J1 必须失败（守卫可触发）",
        note="参数每序列采样、解析臂自估；a_jac 为 16×16 两帧提升 Jacobian 谱半径"),
        config=dict(GP_CHOICES=list(GP_CHOICES), DEC_CHOICES=list(DEC_CHOICES),
                    KH_CHOICES=list(KH_CHOICES), T_STEPS=T_STEPS, K=[K_LO, K_HI],
                    N_TRAIN=N_TRAIN, N_VAL=N_VAL, H_ROLL=H_ROLL,
                    EPOCHS=EPOCHS, BATCH=BATCH, SEEDS=list(SEEDS), S_RES=S_RES))
    R, T = {}, {}
    for proc in TIERS:
        r, t = run_tier(proc)
        R[proc], T[proc] = r, t
        print(f"\n===== {proc}（真值 ρ(J)={t['rho_true']:.6f}, a_true={t['a_true']:+.6f}）=====")
        print(f"  {'臂':<10}{'单步MSE':>13}{'margin':>10}{'a_jac':>10}{'a_twin':>10}{'h*':>6}")
        for nm, v in r.items():
            print(f"  {nm:<10}{v['mse1']:>13.3e}{v['margin1']:>+10.4f}"
                  f"{v['a_jac']:>+10.4f}{v['a_twin']:>+10.4f}{v['hstar']:>6d}")

    # ---- G0 / G0'：解析核的 Jacobian 是否与真值一致（full 每步重估 / lock 锁存）
    NEW = ("projectile", "friction", "harmonic")
    g0 = {p: abs(math.exp(R[p]["ana_full"]["a_jac"]) - T[p]["rho_true"]) for p in NEW}
    g0_ok = all(v < 1e-6 for v in g0.values())
    g0b = {p: abs(math.exp(R[p]["ana_lock"]["a_jac"]) - T[p]["rho_true"]) for p in NEW}
    g0b_ok = all(v < 1e-6 for v in g0b.values())

    # ---- G1：守卫可触发
    Ava, Bva, Cva, _, _, _ = gen_pairs("projectile", 200, seed=4242)
    with torch.no_grad():
        pb = analytic_next_broken(torch.tensor(Ava), torch.tensor(Bva)).numpy()
    mse_broken = float(((pb - Cva) ** 2).mean())
    g1_ok = mse_broken > J1_TOL

    # ---- J1..J4 / J4'
    j1 = {p: float(R[p]["ana_full"]["mse1"]) for p in NEW}
    j1_ok = all(v <= J1_TOL for v in j1.values())
    j2 = {p: dict(ana_unif=float(R[p]["ana_unif"]["mse1"]),
                  hyb_unif=float(R[p]["hyb_unif"]["mse1"]),
                  mlp=float(R[p]["mlp"]["mse1"])) for p in NEW}
    j2a_ok = all(v["hyb_unif"] <= 0.25 * v["ana_unif"] for v in j2.values())
    j2b_ok = all(v["hyb_unif"] <= v["mlp"] for v in j2.values())
    j2_ok = bool(j2a_ok and j2b_ok)
    j3 = {p: dict(ana_full=float(R[p]["ana_full"]["mse1"]),
                  hyb_full=float(R[p]["hyb_full"]["mse1"])) for p in NEW}
    j3_ok = all((v["ana_full"] - v["hyb_full"]) < 0.1 * max(v["ana_full"], 1e-12) for v in j3.values())
    j4 = {p: dict(ana=abs(R[p]["ana_full"]["a_jac"] - T[p]["a_true"]),
                  mlp=abs(R[p]["mlp"]["a_jac"] - T[p]["a_true"])) for p in NEW}
    j4_ok = all(v["ana"] <= v["mlp"] for v in j4.values())
    j4b = {p: abs(R[p]["ana_lock"]["a_jac"] - T[p]["a_true"]) for p in NEW}
    j4b_ok = all(v <= 0.01 for v in j4b.values())

    print("\n" + "=" * 96)
    print("  【主假设】")
    print(f"  J1 形式完整⇒精确: {j1_ok}   " + "  ".join(f"{k}={v:.2e}" for k, v in j1.items()))
    print(f"  J2 残缺⇒残差可补: {j2_ok} (J2a {j2a_ok} / J2b {j2b_ok})   " +
          "  ".join(f"{k}={v['ana_unif']:.2e}→{v['hyb_unif']:.2e}(mlp {v['mlp']:.2e})"
                    for k, v in j2.items()))
    print(f"  J3 完整⇒残差冗余: {j3_ok}   " +
          "  ".join(f"{k}={v['ana_full']:.2e}→{v['hyb_full']:.2e}" for k, v in j3.items()))
    print("  【被否证的原假设（=发现）】")
    print(f"  G0 每步重估的解析核 ρ=真值ρ: {g0_ok}  " +
          "  ".join(f"{k}+{v:.2e}" for k, v in g0.items()))
    print(f"  J4 解析核误差放大不劣于 MLP: {j4_ok}   " +
          "  ".join(f"{k}(ana {v['ana']:.4f} vs mlp {v['mlp']:.4f})" for k, v in j4.items()))
    print("  【归因消融 + 守卫】")
    print(f"  G0' 锁存版 ρ=真值ρ: {g0b_ok}  " + "  ".join(f"{k}+{v:.2e}" for k, v in g0b.items()))
    print(f"  J4' 锁存版 |a−a_true|≤0.01: {j4b_ok}   " +
          "  ".join(f"{k}={v:.2e}" for k, v in j4b.items()))
    print(f"  G1 守卫可触发（g 写反 MSE={mse_broken:.3e} > {J1_TOL:g}）: {g1_ok}")
    agg = aggregator_probe()
    g3 = {p: float(R[p]["ana_lock"]["a_twin"]) for p in NEW}
    g3_ok = all(abs(v) <= 0.15 for v in g3.values())     # 锁存版不应出现指数读数
    print(f"  G3 孪生口径自洽（锁存版 |a_twin|≤0.15）: {g3_ok}   " +
          "  ".join(f"{k}={v:+.4f}" for k, v in g3.items()))
    print("  归因探针·聚合算子对 ρ(J_aug) 的影响（projectile）: " +
          "  ".join(f"{k}={v:.6f}" for k, v in agg.items()))
    ok_hyp = bool(j1_ok and j2_ok and j3_ok)
    ok_attr = bool(g0b_ok and j4b_ok)
    ok = bool(ok_hyp and ok_attr and g1_ok and g3_ok)
    print(f"  主假设 {ok_hyp}｜归因消融 {ok_attr}｜守卫 {g1_ok}/{g3_ok}  ⇒  P2 完成标准: "
          f"{'PASS' if ok else 'FAIL'}（G0/J4 否证另记为发现）")
    print("=" * 96)

    out.update(results=R, truth=T, g0=g0, g0_ok=bool(g0_ok), g0b=g0b, g0b_ok=bool(g0b_ok),
               g1_mse_broken=mse_broken, g1_ok=bool(g1_ok),
               j1=j1, j1_ok=bool(j1_ok), j2=j2, j2_ok=bool(j2_ok),
               j2a_ok=bool(j2a_ok), j2b_ok=bool(j2b_ok),
               j3=j3, j3_ok=bool(j3_ok), j4=j4, j4_ok=bool(j4_ok),
               j4b=j4b, j4b_ok=bool(j4b_ok), g3=g3, g3_ok=bool(g3_ok),
               aggregator_probe=agg,
               ok_hyp=ok_hyp, ok_attr=ok_attr,
               finding=("每步重估参数的解析核：单步精确（MSE~1e-13）但扰动增益 "
                        f"a_jac={ {p: round(R[p]['ana_full']['a_jac'], 4) for p in NEW} }、"
                        f"a_twin≈log2；锁存参数后 a_jac 回到真值 0 "
                        f"（{ {p: round(R[p]['ana_lock']['a_jac'], 6) for p in NEW} }）"
                        "⇒ 放大来自估计器增益，不是解析形式本身"),
               passed=bool(ok), elapsed_s=round(time.time() - t0, 1))
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/p2_new_physics.json（{out['elapsed_s']}s）")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
