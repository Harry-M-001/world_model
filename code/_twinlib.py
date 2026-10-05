# -*- coding: utf-8 -*-
"""误差放大率的**唯一正确口径**库（2026-10-05 建立，替代全库散落的 twin_a 写法）。

历史教训（P0）——旧写法 `za, zb, zb2 = zb, p1, p2` 让两条孪生分支**共享 prev 帧**，
测到的是 ρ(∂f/∂st)，而不是模型自回归的误差放大：
  · 正确递推：ε_{t+1} = A·ε_{t−1} + B·ε_t   （A=∂f/∂sp，B=∂f/∂st）
  · 共享 prev：ε_{t+1} = B·ε_t              （A 项被强制置零）
后果：λ=0 的系统被读成 +0.66（q12 最小集 0.6669、T14 0.5323 均属此类伪读数）；
      抛体解析核的 ∂vy'/∂st 有特征值 2，λ=0 被读成 log2=0.69。

本库提供三个口径，**必须至少用两个互相印证**：
  L1 jac_rho(fwd, sp, st)          —— 两帧提升 Jacobian 的谱半径（float64 + eigvals）
  L2 twin_lam(fwd, ...)            —— 孪生，两条分支各带自己的历史
  L4 twin_lam_shared(fwd, ...)     —— 【废弃】旧口径，仅供对照复现，不得用于新结论

⚠ 两个铁律：
  ① 谱半径**不能用幂迭代**：Jordan 块只收敛到 1+1/k（400 步仍 1.0025）、
     复共轭单位圆特征值给 1.5 ⇒ 一律 np.linalg.eigvals 取 max|λ|。
  ② 误差增长是**三阶段**的（早期瞬态 ≈0.11 / 饱和段 ≈0.005，差 17~24×），
     单标量 λ 不可概括 ⇒ 报 λ 必须同时报窗口。
"""
import math

import numpy as np
import torch


# ---------------------------------------------------------------- L1 Jacobian
def jac_aug(fwd, sp0, st0, h=1e-6):
    """两帧提升 (z_prev,z_t)->(z_t,f) 的 Jacobian（float64 有限差分）。

    fwd(sp, st) -> next，sp/st 为 (1, d)。返回 (2d, 2d) numpy float64。
    """
    sp0 = sp0.detach().clone().double().reshape(-1)
    st0 = st0.detach().clone().double().reshape(-1)
    d = int(sp0.numel())
    base = fwd(sp0.reshape(1, -1), st0.reshape(1, -1)).reshape(-1)
    J = np.zeros((2 * d, 2 * d), dtype=np.float64)
    for src, off in ((sp0, 0), (st0, d)):
        for i in range(d):
            sv = src.clone(); sv[i] += h
            p = fwd(sv.reshape(1, -1), st0.reshape(1, -1)) if off == 0 \
                else fwd(sp0.reshape(1, -1), sv.reshape(1, -1))
            J[d:2 * d, off + i] = ((p.reshape(-1) - base) / h).detach().cpu().numpy()
    J[0:d, d:2 * d] = np.eye(d)
    return J


def jac_rho(fwd, sp0, st0, h=1e-6):
    """ρ(J_aug)：直接求特征值（不用幂迭代）。"""
    return float(np.max(np.abs(np.linalg.eigvals(jac_aug(fwd, sp0, st0, h)))))


def lam_jac(fwd, sp0, st0, h=1e-6):
    rho = jac_rho(fwd, sp0, st0, h)
    return math.log(rho) if rho > 0 else float("nan")


# ---------------------------------------------------------------- L2 孪生（正确口径）
def twin_lam(fwd, sp, st, t1=8, t2=20, delta=1e-9, window=None):
    """两条分支**各带自己的历史**。fwd(sp, st) -> next，sp/st 为 (1, d) float64。

    返回窗口 [t1, t2] 的率尺度 λ̂ = ln(sep_t2 / sep_t1) / (t2 − t1)。
    window=(a,b) 可覆盖默认窗口（报 λ 必须同时报窗口！）。
    """
    if window is not None:
        t1, t2 = window
    st2 = st + delta * torch.randn(st.shape, dtype=st.dtype, device=st.device)
    a1, b1 = sp, st          # 分支1 (prev, cur)
    a2, b2 = sp, st2         # 分支2 (prev, cur)
    sep = [delta]
    for _ in range(t2):
        p1 = fwd(a1.reshape(1, -1), b1.reshape(1, -1)).reshape(-1)
        p2 = fwd(a2.reshape(1, -1), b2.reshape(1, -1)).reshape(-1)
        sep.append(float(torch.sqrt((p2 - p1).norm() ** 2 + (b2 - b1).norm() ** 2)))
        a1, b1 = b1, p1
        a2, b2 = b2, p2
    if len(sep) <= t2 or sep[t1] <= 0 or sep[t2] <= 0:
        return float("nan")
    return math.log(sep[t2] / sep[t1]) / (t2 - t1)


def twin_lam_shared(fwd, sp, st, t1=8, t2=20, delta=1e-9):
    """【废弃·仅供复现旧值】两条分支共享 prev 帧。新结论一律不得使用。"""
    st2 = st + delta * torch.randn(st.shape, dtype=st.dtype, device=st.device)
    ca, c1, c2 = sp, st, st2
    sep = [delta]
    for _ in range(t2):
        p1 = fwd(ca.reshape(1, -1), c1.reshape(1, -1)).reshape(-1)
        p2 = fwd(ca.reshape(1, -1), c2.reshape(1, -1)).reshape(-1)
        sep.append(float((p2 - p1).norm()))
        ca, c1, c2 = c1, p1, p2
    if len(sep) <= t2 or sep[t1] <= 0 or sep[t2] <= 0:
        return float("nan")
    return math.log(sep[t2] / sep[t1]) / (t2 - t1)


# ---------------------------------------------------------------- 自检
def self_check():
    """守卫：恒定映射（λ 必须为 0）与纯放大映射（λ 必须为 ln k）上验证三个口径。"""
    d = 4
    sp = torch.zeros(d, dtype=torch.float64)
    st = torch.ones(d, dtype=torch.float64)

    def f_identity(a, b):
        return b                                  # 恒等：ρ=1 → λ=0
    def f_amp(a, b, k=2.0):
        return k * b                              # 纯放大：ρ=k → λ=ln k

    torch.manual_seed(0)
    l1 = lam_jac(f_identity, sp, st)
    l2 = twin_lam(f_identity, sp, st)
    l4 = twin_lam_shared(f_identity, sp, st)
    a1 = lam_jac(f_amp, sp, st)
    a2 = twin_lam(f_amp, sp, st)
    ok = (abs(l1) < 1e-6 and abs(l2) < 0.1 and abs(a1 - math.log(2)) < 1e-6
          and abs(a2 - math.log(2)) < 0.05)
    print(f"  恒等映射：L1={l1:+.6f}  L2={l2:+.4f}  旧口径L4={l4:+.4f}")
    print(f"  2× 放大：  L1={a1:+.6f}（应 ln2={math.log(2):.6f}）  L2={a2:+.4f}")
    print(f"  自检：{'PASS' if ok else 'FAIL'}")
    return bool(ok)


if __name__ == "__main__":
    self_check()
