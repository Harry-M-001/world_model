"""
T2b 前置 · 多物理过程生成器（四档：匀速 / 重力抛射 / 盘-盘弹性碰撞 / 阻尼）

规格：
  ① 四档覆盖 λ₁ 的行为谱：**匀速 λ₁=0 / 重力 λ₁=0（扰动至多线性增长）/ 碰撞 λ₁>0**
     / **阻尼 λ_v=log(1−γ)<0（速度子空间收缩；状态最大指数仍为 0，见 benettin pert 口径）**
     —— 阻尼档是唯一 λ₁<0 的机制（速度子空间收缩），与谱约束实验对应
  ② **碰撞档的接触几何必须是曲面**：用盘-盘弹性接触（`collide.resolve_pair`，已被 T18
     的两轮诊断打磨过），**不是平面等距反射** —— 后者与 Moving MNIST 现有机制等价
     （真值 λ₁=0），不构成新档。
  ③ **标签只用于事后评测**：`labels` 不进模型输入、不进损失。
  ④ **每条序列内只切换一次**（`t_switch`）—— 序列内固定存在退化解：路由只需在
     序列开头判断一次并锁存，判据仍会通过，测到的是序列级分类而非逐帧物理感知。
     切换一次后判据可升级为「路由是否在已知切换点换专家」（有 ground truth，不泄漏标签）。
  ⑤ **复用现有 VAE**：三档都不引入新视觉元素 —— 画布 / 灰度 / 数字来源 / 背景 /
     是否重叠 / 取整方式**全部与 gray_s3 一致**（见下方 CONFIG，已显式写死）。
  ⑥ 三档共用**同一个竞技场**（反射边界，与 gray_s3 一致）：边界是共享元素，
     因此档间的唯一差别就是**速度演化规则**（单变量）。

口径说明：`labels[t]` 标注的是「**从帧 t 出发的那次转移所用的过程**」。
   因此切换点在 t = t_switch 处表现为 label 从 proc_a 跳到 proc_b，
   而**状态是连续的**（只有规则变，位置/速度不回滚）。
   模型看不到标签，只能从若干帧的运动证据里推断 ⇒ 判据必须带窗口（见 t2b 评测）。

用法：
  python multiphys.py --self-check          # 生成器自检 + 三档可区分性统计
  python multiphys.py --truth               # 四档真值 λ₁（Benettin，同 t18_truth 口径）
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from collide import resolve_pair, rand_state, RADIUS, LIM, DIGIT   # noqa: E402

# ---------------------------------------------------------------- 渲染/物理约定（写死）
H = W = 64
V_MAX = 3
T_FRAMES = 20
N_DIGIT = 2                      # 与 gray_s3 / T18 一致（VAE 的训练分布）
SPEEDS = (1, 2, 3)
T_SWITCH = 10                    # 序列内唯一一次切换（帧索引）
G_CHOICES = (0.4, 0.5, 0.6)      # 重力场强度（每序列一个，均匀场 → 作用于所有数字）
E_REST = 1.0                     # 恢复系数（弹性）
PROCESSES = ("uniform", "gravity", "scatter", "damped")
GAMMA_D = 0.05                  # 阻尼档：每步速度 ×(1−γ)；λ_v = log(1−γ) ≈ −0.0513/步
ACT_DV = 0.6                    # 动作脉冲幅值（相对 SPEEDS=1..3 保守）；ux/uy ∈ {−1,0,1} 且非零
                                # 动作只对单个数字施加（共享动作叠加 clip 会破坏伽利略不变性，使碰撞消失）
                                # damped 必须排在末尾（索引 3）：标签与缓存依赖前三个索引不变

CONFIG = dict(
    canvas=[H, W], channels="grayscale", levels=255, background=0,
    digit_source="MNIST train split（train 划分）/ test split（val 划分）",
    digit_size=DIGIT, n_digit=N_DIGIT, combine="pixel-wise max",
    offset="integer pixel（round 后 clamp 到 [0, LIM]）",
    overlap="允许（max 合成）", boundary="reflect（与 gray_s3 逐字一致）",
    v_max=V_MAX, speeds=list(SPEEDS), n_frames=T_FRAMES, t_switch=T_SWITCH,
    g_choices=list(G_CHOICES), e_rest=E_REST,
    arena="反射边界（三档共享 → 档间唯一差别是速度演化规则）",
    processes={
        "uniform": "速度恒定 v'=v（与 gray_s3 同规则）",
        "gravity": "重力场 v_y' = clip(v_y + g, ±V_MAX)，g 为每序列采样的均匀场强度",
        "scatter": "速度恒定 + 盘-盘弹性接触（曲面接触，非平面反射）",
    },
)


# ---------------------------------------------------------------- 单步推进
def step_state(st, proc, g=0.0, e=E_REST, v_max=V_MAX, boundary="reflect"):
    """推进一帧。st: (n,4) float64 [x,y,vx,vy]（左上角坐标）。返回 (新状态, 碰撞次数)。

    与 t15/t18 保持同一顺序：**先改速度（重力）→ 再移动 → 边界处理 → 盘-盘接触**。
    boundary="reflect" 时与 collide.step_state 完全一致（仅动作项被换成过程规则）。
    """
    st = np.array(st, dtype=np.float64, copy=True)
    if proc == "gravity":
        st[:, 3] = np.clip(st[:, 3] + g, -v_max, v_max)
    elif proc == "damped":
        st[:, 2] *= (1.0 - GAMMA_D)            # 各向同性线性阻力（速度只衰减不反向）
        st[:, 3] *= (1.0 - GAMMA_D)
    elif proc in ("uniform", "scatter"):
        pass                                   # 速度不变
    else:
        raise ValueError(f"未知过程 {proc!r}")
    st[:, 0] += st[:, 2]
    st[:, 1] += st[:, 3]
    if boundary == "periodic":
        st[:, 0] = np.mod(st[:, 0], float(LIM))
        st[:, 1] = np.mod(st[:, 1], float(LIM))
    else:
        for k in range(st.shape[0]):
            if st[k, 0] < 0.0 or st[k, 0] > LIM:
                st[k, 2] = -st[k, 2]
                st[k, 0] = min(max(st[k, 0], 0.0), float(LIM))
            if st[k, 1] < 0.0 or st[k, 1] > LIM:
                st[k, 3] = -st[k, 3]
                st[k, 1] = min(max(st[k, 1], 0.0), float(LIM))
    ncol = 0
    if proc == "scatter":
        for i in range(st.shape[0]):
            for j in range(i + 1, st.shape[0]):
                ncol += int(resolve_pair(st, i, j, e))
    return st, ncol


# ---------------------------------------------------------------- 生成
def make_seq(digits, rng, proc_a, proc_b, t_switch=T_SWITCH, n_frames=T_FRAMES,
             n_digit=N_DIGIT, boundary="reflect", with_actions=False):
    """生成一条含**一次过程切换**的序列。

    返回 (frames (T,64,64) uint8, labels (T,) int8, meta dict)
      labels[t] = 从帧 t 出发的那次转移所用的过程编号（见模块 docstring）
    """
    assert proc_a != proc_b, "切换两侧必须不同过程"
    assert 0 < t_switch < n_frames, f"切换点越界 {t_switch}"
    ia, ib = PROCESSES.index(proc_a), PROCESSES.index(proc_b)
    frames = np.zeros((n_frames, H, W), dtype=np.uint8)
    labels = np.zeros(n_frames, dtype=np.int8)
    st = rand_state(rng, n_digit)               # 互不重叠的初值（拒绝采样）
    g = float(rng.choice(G_CHOICES))
    cols = [0, 0]
    states = np.zeros((n_frames, n_digit, 4), dtype=np.float64)   # 真值状态（供线性读出校验）
    acts = np.zeros((n_frames, 3), dtype=np.float64)              # [digit, ux, uy]（转移 t 的动作）
    for t in range(n_frames):
        for i in range(n_digit):                # 先渲染当前帧（与 t15/t18 同序）
            xi = int(min(max(round(st[i, 0]), 0), LIM))
            yi = int(min(max(round(st[i, 1]), 0), LIM))
            np.maximum(frames[t, yi:yi + DIGIT, xi:xi + DIGIT], digits[i],
                       out=frames[t, yi:yi + DIGIT, xi:xi + DIGIT])
        if t < t_switch:
            proc, labels[t] = proc_a, ia
            ph = 0
        else:
            proc, labels[t] = proc_b, ib
            ph = 1
        states[t] = st
        if with_actions:
            di = int(rng.integers(0, n_digit))
            ux = uy = 0
            while ux == 0 and uy == 0:
                ux = int(rng.integers(-1, 2))
                uy = int(rng.integers(-1, 2))
            st[di, 2] += ACT_DV * ux                  # 施力在物理规则演化之前（转移 t 的一部分）
            st[di, 3] += ACT_DV * uy
            acts[t] = (di, ux, uy)
        st, nc = step_state(st, proc, g=g, boundary=boundary)
        cols[ph] += nc
    meta = dict(proc_a=proc_a, proc_b=proc_b, t_switch=int(t_switch),
                g=g, cols_a=int(cols[0]), cols_b=int(cols[1]),
                n_digit=int(n_digit), boundary=boundary,
                states=states,          # (T, n_digit, 4) = [x, y, vx, vy]
                actions=acts if with_actions else None)

    return frames, labels, meta


def build_dataset(n, seed, train, pairs=("uniform", "gravity"),
                  t_switch=T_SWITCH, n_frames=T_FRAMES, n_digit=N_DIGIT,
                  mix=None, verbose=True, with_actions=False):
    """生成 n 条序列。

    pairs: 允许的**无序**过程对列表（如 [("uniform","gravity")]）；每条序列随机取一对并随机
           决定先后（A→B 或 B→A）—— 保证「切换方向」不被固定，否则路由可以用「时间位置」作弊。
           —— 保证「切换方向」不被固定，否则路由可以用「时间位置」作弊。
    mix  : 可选 dict，指定每个无序对的采样权重（默认等权）
    返回 (frames (n,T,64,64) uint8, labels (n,T) int8, metas list[dict])
    """
    from variant2 import GRAY
    digits_all = GRAY[train]
    F_, L_, M_ = [], [], []
    t0 = time.time()
    for i in range(n):
        rng = np.random.default_rng((seed * 1_000_003 + i) % (2 ** 32))
        if mix:
            keys = list(mix)
            w = np.array([mix[k] for k in keys], dtype=np.float64)
            pa, pb = keys[int(rng.choice(len(keys), p=w / w.sum()))]
        else:
            pa, pb = pairs[int(rng.integers(0, len(pairs)))]
        if bool(rng.integers(0, 2)):
            pa, pb = pb, pa
        order = rng.permutation(len(digits_all))[:n_digit]
        digs = [digits_all[k] for k in order]
        f, lab, meta = make_seq(digs, rng, pa, pb, t_switch=t_switch,
                                n_frames=n_frames, n_digit=n_digit,
                                with_actions=with_actions)
        F_.append(f)
        L_.append(lab)
        M_.append(meta)
    if verbose:
        counts = {}
        for m in M_:
            counts[(m["proc_a"], m["proc_b"])] = counts.get((m["proc_a"], m["proc_b"]), 0) + 1
        print(f"    生成 {n} 条：{time.time()-t0:.0f}s   "
              f"过程对分布 {counts}   "
              f"平均碰撞（a 相 {np.mean([m['cols_a'] for m in M_]):.2f} / "
              f"b 相 {np.mean([m['cols_b'] for m in M_]):.2f}）")
    return np.stack(F_), np.stack(L_), M_


# ---------------------------------------------------------------- 真值 λ₁（Benettin）
def benettin_proc(proc, g, n_digit=N_DIGIT, n_steps=20000, delta0=1e-9, d_max=0.5,
                  warm=2000, seed=0, e=E_REST, boundary="reflect", pert="pos",
                  d_min=None):
    """单过程、单条轨迹的 Benettin 估计（与 t18_truth.benettin 同口径）。

    使用**反射边界**（与生成数据、与 gray_s3 一致）；已知反射边界本身
      贡献约 +0.0013 的假扩张（无碰撞系统上实测），读数时按此量级理解。

    收缩方向（damped 的速度子空间，λ_v<0）：d 只缩不涨，永不触发 d_max 重归一化，
      单侧归一化会使 acc 恒 0、误报 λ=0。
      传 d_min（如 delta0/2）启用**双侧重归一化**：d < d_min 时拉回 delta0 并累计
      log(d/delta0)（负值）。默认 None = 旧单侧行为，既有测量逐位不变。
    """
    rng = np.random.default_rng(seed)
    st = rand_state(rng, n_digit)
    for _ in range(warm):
        st, _ = step_state(st, proc, g=g, e=e, boundary=boundary)
    st2 = st.copy()
    if pert == "pos":
        st2[0, 0] += delta0
    elif pert == "vel":
        st2[0, 2] += delta0                    # 只扰速度 → 量速度子空间的收缩/扩张
    else:
        raise ValueError(f"未知 pert {pert!r}")
    acc, nren, ncol = 0.0, 0, 0
    for _ in range(n_steps):
        st, c1 = step_state(st, proc, g=g, e=e, boundary=boundary)
        st2, c2 = step_state(st2, proc, g=g, e=e, boundary=boundary)
        ncol += c1
        d = float(np.linalg.norm((st2 - st).ravel()))
        if d > d_max:
            acc += math.log(d / delta0)
            nren += 1
            st2 = st + (st2 - st) * (delta0 / d)
        elif d_min is not None and d < d_min:
            acc += math.log(d / delta0)        # 收缩：log 值为负
            nren += 1
            st2 = st + (st2 - st) * (delta0 / d)
    return acc / n_steps, nren, ncol / n_steps


def benettin_vel(proc, g, n_steps=20000, delta0=1e-9, warm=2000, seed=0,
                 d_lo=0.5, d_hi=2.0):
    """速度子空间的 Benettin：距离只取 δv 分量，双侧重归一化。

    用途：damped 的 λ_v（速度图是线性乘法 ⇒ 预期精确 = log(1−γ)）。
    全状态距离在此失效：δv 衰减时位置偏移积分饱和于 δv/γ ⇒ 范数不归零、
    永不过阈值，单侧归一化会误报为 0。
    """
    rng = np.random.default_rng(seed)
    st = rand_state(rng, N_DIGIT)
    for _ in range(warm):
        st, _ = step_state(st, proc, g=g)
    st2 = st.copy()
    st2[0, 2] += delta0
    acc, nren = 0.0, 0
    for _ in range(n_steps):
        st, _ = step_state(st, proc, g=g)
        st2, _ = step_state(st2, proc, g=g)
        d = float(np.linalg.norm((st2[:, 2:4] - st[:, 2:4]).ravel()))
        if d > d_hi * delta0 or d < d_lo * delta0:
            acc += math.log(d / delta0)
            nren += 1
            st2[:, 2:4] = st[:, 2:4] + (st2[:, 2:4] - st[:, 2:4]) * (delta0 / d)
    return acc / n_steps, nren


# ---------------------------------------------------------------- 自检
def self_check(n=64, seed=99):
    """① 视觉分布与 gray_s3 一致（墨迹占比）；② 三档的运动统计可区分。"""
    from variant2 import GRAY, make_seq_gray
    print("=" * 100)
    print("  多物理生成器 · 自检")
    print("=" * 100)

    # ① 墨迹占比：与 gray_s3 同分布说明没有引入新视觉元素
    rng = np.random.default_rng(0)
    ink_ref = float(np.mean([(make_seq_gray(GRAY[True], rng, SPEEDS, 0.0) > 0).mean()
                             for _ in range(16)]))
    print(f"\n  ① 视觉约定对照（同一 n_digit={N_DIGIT}、同一速度集合）")
    print(f"     gray_s3（参照）墨迹占比 {ink_ref*100:.3f}%")
    stats = {}
    for pa, pb in (("uniform", "gravity"), ("uniform", "scatter"),
                   ("gravity", "scatter"), ("uniform", "damped"),
                   ("scatter", "damped")):
        F, L, M = build_dataset(n, seed, True, pairs=[(pa, pb)], verbose=False)
        ink = float((F > 0).mean())
        stats[f"{pa}->{pb}"] = dict(ink=ink, cols_a=float(np.mean([m["cols_a"] for m in M])),
                                    cols_b=float(np.mean([m["cols_b"] for m in M])))
        print(f"     {pa:>7}->{pb:<8} 墨迹占比 {ink*100:.3f}%   "
              f"（差 {(ink-ink_ref)*100:+.3f} 个百分点）")

    # ② 各档独立的运动统计（直接从真值状态算，不经过渲染）
    print(f"\n  ② {len(PROCESSES)} 档的运动统计（每条 200 步、64 条，直接读真值状态）")
    print(f"     {'过程':<9}{'Δvy 均值':>11}{'Δvy 标准差':>12}{'平均速度':>10}"
          f"{'|Δv|>0 的比例':>14}{'碰撞/步':>9}")
    # ③ T18 护栏：scatter+动作 的碰撞率不得塌缩（共享动作+clip 曾致碰撞消失）
    print(f"\n  ③ 动作条件化护栏（每步对单个数字施加 ±{ACT_DV} 脉冲）")
    g_cols = float(np.mean([np.mean([mm["cols_a"] + mm["cols_b"] for mm in M])
                            for M in [build_dataset(64, 5, True, pairs=[("uniform", "scatter")],
                                                    verbose=False, with_actions=True)[2]]]))
    g_cols0 = float(np.mean([mm["cols_a"] + mm["cols_b"]
                             for mm in build_dataset(64, 5, True, pairs=[("uniform", "scatter")],
                                                     verbose=False)[2]]))
    print(f"     scatter 碰撞/步：无动作 {g_cols0/200:.4f} vs 有动作 {g_cols/200:.4f}"
          f"（比值 {g_cols/max(g_cols0,1e-9):.2f}，须 > 0.5）")
    sep = {}
    for proc in PROCESSES:
        dvs, sps, nz, cols = [], [], [], []
        for k in range(64):
            rng = np.random.default_rng(500 + k)
            st = rand_state(rng, N_DIGIT)
            g = float(rng.choice(G_CHOICES))
            for _ in range(200):
                prev_v = st[:, 2:4].copy()          # (n,2) = [vx, vy]
                st, nc = step_state(st, proc, g=g)
                dvs.append(float((st[:, 3] - prev_v[:, 1]).mean()))     # Δvy
                sps.append(float(np.linalg.norm(st[:, 2:4], axis=1).mean()))
                nz.append(float((np.abs(st[:, 2:4] - prev_v).max(axis=1) > 1e-9).mean()))
                cols.append(nc)
        sep[proc] = dict(dvy_mean=float(np.mean(dvs)), dvy_std=float(np.std(dvs)),
                         speed=float(np.mean(sps)), nonzero=float(np.mean(nz)),
                         cols=float(np.mean(cols)))
        print(f"     {proc:<9}{np.mean(dvs):>+11.4f}{np.std(dvs):>12.4f}"
              f"{np.mean(sps):>10.3f}{np.mean(nz)*100:>13.1f}%{np.mean(cols):>9.4f}")
    print("\n  说明（⚠ 这一行原写作「Δvy 均值把重力分开」—— **被我自己的数据否证**："
          "速度饱和 + 边界反弹后 Δvy 的长期均值≈0）。真正可分的三个统计是：")
    print(f"    · **|Δv|>0 的比例**：uniform {sep['uniform']['nonzero']*100:.1f}%（只有撞墙） vs "
          f"gravity {sep['gravity']['nonzero']*100:.1f}%（几乎每步都在变） vs "
          f"scatter {sep['scatter']['nonzero']*100:.1f}%（撞墙 + 接触）")
    print(f"    · **碰撞/步**：scatter {sep['scatter']['cols']:.4f} vs 另两档精确 0")
    print(f"    · **平均速度**：uniform {sep['uniform']['speed']:.3f} / "
          f"gravity {sep['gravity']['speed']:.3f} / scatter {sep['scatter']['speed']:.3f}"
          f"（gravity 偏低：位置夹回在墙面吃掉能量）")
    return dict(ink_ref=ink_ref, pairs=stats, sep=sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-check", action="store_true")
    ap.add_argument("--truth", action="store_true")
    ap.add_argument("--ic", type=int, default=12)
    ap.add_argument("--steps", type=int, default=20000)
    a = ap.parse_args()
    res = {}
    if a.self_check:
        res["self_check"] = self_check()
    if a.truth:
        print("\n" + "=" * 100)
        print("  四档真值 λ₁（Benettin，反射边界、与生成数据同一竞技场）")
        print("=" * 100)
        print(f"  {a.ic} 初值 × {a.steps} 步；g 取集合中位 {sorted(G_CHOICES)[len(G_CHOICES)//2]}")
        print(f"  ⚠ 反射边界的本底假扩张 ≈ +0.0013（T18 实测，n=1 无碰撞系统）\n")
        print(f"  {'过程':<9}{'λ₁ 均值±SE':>18}{'中位':>10}{'范围':>22}{'碰撞/步':>9}")
        rows = []
        g_mid = sorted(G_CHOICES)[len(G_CHOICES) // 2]
        for proc in PROCESSES:
            lams, cols = [], []
            for i in range(a.ic):
                lam, nr, nc = benettin_proc(proc, g_mid, n_steps=a.steps, seed=i)
                lams.append(lam)
                cols.append(nc)
            L = np.array(lams)
            se = float(L.std(ddof=1) / math.sqrt(len(L)))
            rng_s = "[{:+.4f},{:+.4f}]".format(L.min(), L.max())
            extra = {}
            if proc == "damped":
                Lv = []
                for i in range(a.ic):
                    lam, _ = benettin_vel(proc, g_mid, n_steps=a.steps, seed=i)
                    Lv.append(lam)
                Lv = np.array(Lv)
                extra = dict(lam_vel_mean=float(Lv.mean()),
                             lam_vel_se=float(Lv.std(ddof=1) / math.sqrt(len(Lv))),
                             lam_vel_theory=math.log(1.0 - GAMMA_D), gamma=GAMMA_D)
                print(f"      ↳ damped 速度扰动口径 λ_v = {Lv.mean():+.5f}±{Lv.std(ddof=1)/math.sqrt(len(Lv)):.5f}"
                      f"（理论 log(1−γ) = {math.log(1.0-GAMMA_D):+.5f}）；"
                      f"上表 λ_max 是位置扰动口径（物理上应为 ≈0，位置方向边际）")
            rows.append(dict(proc=proc, lam_mean=float(L.mean()), lam_se=se,
                             lam_med=float(np.median(L)), lam_min=float(L.min()),
                             lam_max=float(L.max()), lams=[float(x) for x in L],
                             collisions_per_step=float(np.mean(cols)), **extra))
            print(f"  {proc:<9}{L.mean():>+11.5f}±{se:<6.5f}{np.median(L):>+10.5f}"
                  f"{rng_s:>22}{np.mean(cols):>9.4f}")
        res["truth"] = dict(config=dict(n_ic=a.ic, n_steps=a.steps, g=g_mid), rows=rows)

    if res:
        os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
        out = os.path.join(ROOT, "logs", "multiphys_check.json")
        prev = json.load(open(out, encoding="utf-8")) if os.path.exists(out) else {}
        prev.update(res)
        prev["config_snapshot"] = CONFIG
        with open(out, "w", encoding="utf-8") as f:
            json.dump(prev, f, ensure_ascii=False, indent=2)
        print(f"\n  → {out}")


if __name__ == "__main__":
    main()
