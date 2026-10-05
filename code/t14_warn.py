"""
T14 · 崩坏预警对照（判据已预注册）

任务：量化 rollout 的「何时该停」判定。真值崩溃时刻只用于**评测**（环路外）。
基础模型：动作条件化模型（moe_lb0.01_act，满足世界模型第一条判据）。
量化：潜空间逐激活伪量化（t4b 口径，lo/hi 取训练潜分布 0.1%/99.9% 分位），
      与零参数公式的 η(b) = (hi−lo)/(2·(2^b−1)) 严格对应。

三臂（全部在动力学环路之外——只读、不写）：
  A1 解析判据：T_crash 零参数公式 T(b) = (1/λ)·log(1 + δ*(e^λ−1)/η(b))，
     λ 用孪生轨迹法在**同一动作序列**上实测（模型自身轨迹，无需参照）。
  A2a 手写规则·范数 OOD：‖z_t‖ 离开训练分布 μ ± 3σ 即停。
  A2b 手写规则·二阶差分：‖z_t − 2z_{t−1} + z_{t−2}‖ 超过校准分位即停。
  A3 独立判断器：Laya（开放权重 421M）零样本读序列化指标，逐帧输出「继续/停」。

预注册预期：A2 手写规则很可能已经足够，这本身是有效的负面结果；
   没有 A1/A2 两臂对照，只能证明「A3 能预警」，不能证明「A3 值得用」。

真值崩溃：δ_t = ‖z^q_t − z^fp32_t‖（训练统计归一化空间）首次 > δ* = 1.0（T18 口径）。
产物 logs/t14_warn.json
"""
import argparse
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

from t16_plan import quantize_model          # noqa: E402  （权重口径备用，主口径为激活量化）
from vae import LATENT, CKPT                 # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX = 2
T_FRAMES = 20
H_ROLL = 40                                  # rollout 步数（超出数据窗口，纯模型推演）
DELTA_STAR = 1.0                             # T18 口径
BS = (4, 5, 6, 8, 10, 12, 16)
N_ROLL = 64
LEAD = 10                                    # 预警提前窗：T_stop ∈ [T_true−LEAD, T_true]
HIT_W = 5                                    # |T_stop − T_true| ≤ HIT_W 记「命中」
OUT = None                                  # 运行时按 tag 决定（防同名覆盖）


def mtime(p):
    import datetime
    return datetime.datetime.fromtimestamp(os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")


# ---------------------------------------------------------------- 模型
def load_act_model(tag, B=8):
    """载入 T2b 第三轮 act 模型（结构与 t2b_route 一致，避免循环导入就地重建）。"""
    import torch.nn as nn
    from t2b_route import MoEDynamics, DenseDynamics, ACT_DIM, N_PROC

    d = torch.load(os.path.join(CKPT, f"t2b_route_{tag}_B{B}_ep400.pt"),
                   map_location=DEV, weights_only=False)
    if d["arm"] == "dense":
        m = DenseDynamics(hidden=d["hidden"], act_dim=ACT_DIM)
    else:
        m = MoEDynamics(n_expert=d["B"], hidden=d["hidden"], topk=d["topk"],
                        act_dim=ACT_DIM)
    m.load_state_dict(d["state"])
    m.eval()
    return m.to(DEV), d["mean"].to(DEV), d["std"].to(DEV)


def act_feats(act_t):
    """act (…,3)[digit,ux,uy] → (…,4) 特征。"""
    d1h = F.one_hot(act_t[..., 0].long(), 2).float()
    return torch.cat([d1h, act_t[..., 1:3]], dim=-1)


def q_lat(z, bits, lo, hi):
    """逐激活伪量化（t4b 口径）。z: (…, L)。"""
    if bits >= 32:
        return z
    step = (hi - lo) / (2 ** bits - 1)
    return lo + torch.clamp(torch.round((z - lo) / step), 0, 2 ** bits - 1) * step


def rollout(m, z0, z1, acts_f, bits, lo, hi, mean, std):
    """闭环 rollout H_ROLL 步。z0,z1: (1,L) 原始潜；acts_f: (H,4)。
    返回归一化轨迹 (H,L)。量化施加在**每步输入潜**上（t4b 激活口径）。"""
    zs = []
    za = (z0.to(DEV) - mean) / std
    zb = (z1.to(DEV) - mean) / std
    for t in range(len(acts_f)):
        a = acts_f[t:t + 1].to(DEV)
        zc = torch.stack([za, zb], dim=1)   # (1,2,L)：forward 里 zt=zc[:,-1] 期望 ctx 这一维
        if bits < 32:
            # 上下文形状为 (1, ctx, L)：per-dim lo/hi 铺两份后再补 batch 维，
            # 与 zc 同形状 (1, 2, L) 才能广播
            lo2 = torch.stack([lo, lo], dim=0).unsqueeze(0)
            hi2 = torch.stack([hi, hi], dim=0).unsqueeze(0)
            zc = q_lat(zc, bits, (lo2 - mean) / std, (hi2 - mean) / std)
        p, _ = m(zc, None, a)
        zs.append(p)
        za, zb = zb, p
    return torch.cat(zs, dim=0)            # (H, L) 归一化空间


def twin_lambda(m, acts_f, zva, mean, std, eps=1e-4, n_pairs=8, H=None, mode="own"):
    """模型自身轨迹的有效扩张率（**两帧提升口径**，与 code/_twinlib.py 一致）。

    ⚠ 2026-10-05 口径修正（两处，缺一不可）：
      ① **各带历史**：分支2 的 prev 是它自己的前一帧。旧写法两条分支共用 za（prev），
         测的是 ρ(∂f/∂st)，会把 λ≈0 的模型读成 +0.53。
      ② **用 augmented 范数、不再每步重归一化**：原实现对**单帧**映射做 Benettin
         （每步把分离度重置到 eps），用在两帧模型上语义混乱——next 帧分离度
         = A·e_prev + B·e_cur，两帧都带残差时会读出虚高的值（实测 3.199）。
         两帧提升的正确做法是分离度取 sqrt(‖e_prev‖²+‖e_cur‖²)，在固定窗口
         [T1, T2] 上取率尺度；float64 下 1e-4 起步不会下溢，无需重归一化。
      mode="shared" 保留旧口径，仅供复现历史值。
    """
    H = H or H_ROLL
    T1, T2 = 8, min(20, H - 1)
    lams, n_drop = [], 0
    import copy
    m64 = copy.deepcopy(m).double()
    mean64, std64 = mean.double(), std.double()
    with torch.no_grad():
        for k in range(n_pairs):
            s = int(k) % zva.shape[0]
            t0 = CTX + 3
            pa1 = ((zva[s, t0 - 1].to(DEV).double() - mean64) / std64).unsqueeze(0)
            cb1 = ((zva[s, t0].to(DEV).double() - mean64) / std64).unsqueeze(0)
            g = torch.Generator(device="cpu").manual_seed(2000 + k)
            dirv = torch.randn(pa1.shape, generator=g).to(DEV).double()
            dirv = dirv / dirv.norm() * eps
            pa2, cb2 = pa1, cb1 + dirv           # 扰动只加在 cur 上
            sep = [float(torch.sqrt(((cb2 - cb1).norm() ** 2)
                                    + ((pa2 - pa1).norm() ** 2)))]
            for t in range(T2 + 1):
                a = acts_f[min(t, len(acts_f) - 1):min(t, len(acts_f) - 1) + 1].to(DEV).double()
                p, _ = m64(torch.stack([pa1, cb1], dim=1), None, a)
                p2, _ = m64(torch.stack([pa2, cb2], dim=1), None, a)
                if mode == "shared":       # 旧口径：分支2 的 prev 也用分支1 的 cur
                    pa1, cb1 = cb1, p
                    pa2, cb2 = pa1, p2
                else:                      # 正确口径：各带自己的历史
                    pa1, cb1 = cb1, p
                    pa2, cb2 = cb2, p2
                sep.append(float(torch.sqrt(((cb2 - cb1).norm() ** 2)
                                            + ((pa2 - pa1).norm() ** 2))))
            arr = np.array(sep)
            if not np.isfinite(arr).all() or arr[T1] <= 0 or arr[T2] <= 0:
                n_drop += 1
                continue
            lams.append(math.log(arr[T2] / arr[T1]) / (T2 - T1))
    if not lams:
        return float("nan"), float("nan"), 0
    return float(np.mean(lams)), float(np.std(lams) / math.sqrt(len(lams))), n_drop


# ---------------------------------------------------------------- 主流程
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="dense_h480_act")
    ap.add_argument("--n-roll", type=int, default=N_ROLL)
    ap.add_argument("--skip-laya", action="store_true")

    ap.add_argument("--act", action="store_true",
                    help="底座为 act 模型（t2b_route 的模块级 ACT 由此开启）")
    a = ap.parse_args()

    t0 = time.time()
    print("=" * 100)
    print(f"  T14 · 崩坏预警对照（底座 {a.tag}，H={H_ROLL}，δ*={DELTA_STAR}，b={BS}）")
    print("=" * 100)

    m, mean, std = load_act_model(a.tag)
    D = torch.load(os.path.join(CKPT, "t2b_route_data_act.pt"),
                   map_location="cpu", weights_only=False)
    zva, tsw = D["z_va"], D["tsw_va"]
    print(f"    数据缓存 mtime {mtime(os.path.join(CKPT, 't2b_route_data_act.pt'))}"
          f"  val {tuple(zva.shape)}")

    # 训练分布潜范数（A2a 的 μ/σ）与潜分位（η(b) 的 lo/hi）
    with torch.no_grad():
        norms = zva.norm(dim=-1).flatten().numpy()
        mu_n, sd_n = float(norms.mean()), float(norms.std())
        flat = zva.reshape(-1, LATENT)
        lo_p, hi_p = np.percentile(flat.numpy(), [0.1, 99.9], axis=0)
        lo_t = torch.tensor(lo_p, dtype=torch.float32, device=DEV)
        hi_t = torch.tensor(hi_p, dtype=torch.float32, device=DEV)
        eta = {b: float((hi_p - lo_p).mean() / (2 ** b - 1) / 2) for b in BS}
    print(f"    潜范数 μ={mu_n:.3f} σ={sd_n:.3f}  η(b)={ {k: round(v, 5) for k, v in eta.items()} }")

    # 动作序列（每条 rollout 一份，三臂共享；分布与训练一致）
    rng = np.random.default_rng(7)
    H = H_ROLL
    acts = np.zeros((a.n_roll, H, 3), dtype=np.float64)
    for i in range(a.n_roll):
        for t in range(H):
            di = int(rng.integers(0, 2))
            ux = uy = 0
            while ux == 0 and uy == 0:
                ux, uy = int(rng.integers(-1, 2)), int(rng.integers(-1, 2))
            acts[i, t] = (di, ux, uy)
    acts_f_all = torch.cat([F.one_hot(torch.tensor(acts[:, :, 0]).long(), 2).float(),
                            torch.tensor(acts[:, :, 1:3]).float()], dim=-1)  # (n,H,4)

    # λ（孪生，模型自身轨迹、共享动作）——仅对连续映射（dense）有效
    is_dense = a.tag.startswith("dense")
    lam, lam_se, nren = None, None, 0
    jump_probe = None
    if is_dense:
        lam, lam_se, nren = twin_lambda(m, acts_f_all[0], zva, mean, std)
        print(f"    孪生 λ_model = {lam:+.5f}±{lam_se:.5f}（{nren} 次重归一化）")
    else:
        # MoE top-1 硬路由使映射不连续，λ₁ 无良好定义。不连续探针：
        #    同一状态、扰动 δ ∈ {1e-2, 1e-3, 1e-4}（扰末帧），测输出放大率
        print("    ⚠ MoE top-1 硬路由：映射不连续，λ₁(自身轨迹) 无良好定义 → A1 不适用")
        jump_probe = {}
        with torch.no_grad():
            for delta in (1e-2, 1e-3, 1e-4):
                amps = []
                for s in range(12):
                    t0 = CTX + 3
                    za = ((zva[s, t0 - 1].to(DEV) - mean) / std).unsqueeze(0)
                    zb = ((zva[s, t0].to(DEV) - mean) / std).unsqueeze(0)
                    aa = act_feats(torch.tensor([acts[s, t0]], dtype=torch.float32)).to(DEV)
                    p, _ = m(torch.stack([za, zb], 1), None, aa)
                    pb, _ = m(torch.stack([za, zb + delta / torch.sqrt(zb.norm())], 1),
                              None, aa)
                    amps.append(float((pb - p).norm() / delta))
                jump_probe[str(delta)] = float(np.mean(amps))
        print(f"    不连续探针（扰末帧放大率）: {jump_probe}")

    # A1：解析判据的预测崩溃步（每 b 一个常数）
    def t_crash_pred(b):
        """零参数公式；λ<0 时 δ_∞ = η/(1−e^λ)，η ≤ δ*(1−e^λ) 则永不崩溃（∞ → None）。"""
        if abs(lam) < 1e-6:
            return None
        e = math.exp(lam)
        val = 1.0 + DELTA_STAR * (e - 1) / max(eta[b], 1e-12)
        if val <= 0:
            return None
        return max(1, round(math.log(val) / lam))
    A1 = {b: t_crash_pred(b) for b in BS} if is_dense else {b: None for b in BS}
    if is_dense:
        print(f"    A1 解析判据 T̂_crash(b) = {A1}")

    # rollouts
    rows = []
    zva_d = zva.to(DEV)
    with torch.no_grad():
        for b in BS:
            for i in range(a.n_roll):
                s0 = int(rng.integers(0, zva.shape[0] - 1))
                t_start = int(max(tsw[s0], CTX))          # 从切换后的段起滚（过程已切换）
                z_start = zva_d[s0, t_start - 1].unsqueeze(0)
                z1 = zva_d[s0, t_start].unsqueeze(0)
                af = acts_f_all[i]
                zq = rollout(m, z_start, z1, af, b, lo_t, hi_t, mean, std)
                zf = rollout(m, z_start, z1, af, 32, lo_t, hi_t, mean, std)
                delta = (zq - zf).norm(dim=-1).cpu().numpy()      # (H,)
                nz = zq.norm(dim=-1).cpu().numpy()
                d2 = (zq[2:] - 2 * zq[1:-1] + zq[:-2]).norm(dim=-1).cpu().numpy()
                crash = np.argmax(delta > DELTA_STAR) + 1 if (delta > DELTA_STAR).any() else None
                rows.append(dict(b=b, i=i, delta=delta.tolist(), norm=nz.tolist(),
                                 d2=d2.tolist(), crash=int(crash) if crash else None,
                                 acts=acts[i].tolist()))
    n_crash = sum(1 for r in rows if r["crash"])
    print(f"    rollouts {len(rows)}  真实崩溃 {n_crash}（未崩 {len(rows)-n_crash}）")

    # ---------------------------------------------------------------- 三臂判定
    def arm_a1(r):
        return A1.get(r["b"])

    def arm_a2a(r):
        for t, v in enumerate(r["norm"]):
            if abs(v - mu_n) > 3 * sd_n:
                return t + 1
        return None

    # A2b 阈值：校准分位（用 b=16 全部 d2 的 99 分位 —— 校准不碰真值标签）
    cal = np.concatenate([np.array(r["d2"]) for r in rows if r["b"] == 16])
    thr_d2 = float(np.percentile(cal, 99))

    def arm_a2b(r):
        for t, v in enumerate(r["d2"]):
            if v > thr_d2:
                return t + 3            # 二阶差分在 t 检测到，外推 3 步内失效
        return None

    laya_arm = None
    if not a.skip_laya:
        try:
            from laya import Router
            rtr = Router()
            qs = {"verdict": {"type": "choice",
                              "instructions":
                                  "You watch a latent world-model rollout online. "
                                  "Each line is one step: step index, latent norm, "
                                  "norm deviation from training mean (in sigmas), "
                                  "second-difference magnitude, and the applied action. "
                                  "Decide whether to CONTINUE trusting the rollout or "
                                  "STOP because it is becoming unreliable.",
                              "criteria": {
                                  "continue": "The rollout still looks self-consistent; "
                                               "norm and second difference are within their "
                                               "usual range.",
                                  "stop": "The rollout is drifting: norm deviates from "
                                          "the training distribution or the second "
                                          "difference is growing abnormally."}}}

            def arm_laya(r):
                lines = []
                mu, sd = mu_n, sd_n
                for t in range(len(r["norm"])):
                    lines.append(f"step {t}: norm {r['norm'][t]:.2f} "
                                 f"dev {(r['norm'][t]-mu)/sd:+.2f} sigma "
                                 f"d2 {r['d2'][t]:.3f} "
                                 f"action digit={int(r['acts'][t][0])} "
                                 f"dv=({int(r['acts'][t][1])},{int(r['acts'][t][2])})")
                    state = chr(10).join(lines[-8:])
                    out = rtr.predict(state, qs, model="english")
                    if out["answers"]["verdict"]["choice"] == "stop":
                        return t
                return None
            laya_arm = arm_laya
            print("    Laya Router 就绪")
        except Exception as e:
            print(f"    ⚠ Laya 不可用（{type(e).__name__}: {str(e)[:120]}）→ 只跑 A1/A2 两臂")

    def evaluate(stop_fn, name):
        hits, fa, late, used = 0, 0, 0, 0
        errs = []
        for r in rows:
            if r["crash"] is None:
                continue
            used += 1
            ts = stop_fn(r)
            if ts is None:
                continue
            errs.append(ts - r["crash"])
            if abs(ts - r["crash"]) <= HIT_W:
                hits += 1
            elif ts < r["crash"] - LEAD:
                fa += 1
            elif ts > r["crash"]:
                late += 1
        res = dict(arm=name, n=used, hit=hits, false_alarm=fa, late=late,
                   hit_rate=hits / max(used, 1),
                   mean_err=float(np.mean(errs)) if errs else None)
        print(f"    {name:8s} 命中 {hits}/{used}（{res['hit_rate']*100:.1f}%）"
              f"  误报（提前>10步）{fa}  迟到 {late}  平均误差 "
              f"{res['mean_err'] if res['mean_err'] is None else round(res['mean_err'],2)}")
        return res

    res = []
    if is_dense:
        res.append(evaluate(arm_a1, "A1解析"))
    res.append(evaluate(arm_a2a, "A2a范数"))
    res.append(evaluate(arm_a2b, "A2b二阶差分"))
    if laya_arm is not None:
        t_l = time.time()
        try:
            res.append(evaluate(laya_arm, "A3Laya"))
        except Exception as e:      # ★ Laya 臂依赖未存档的逐行输出；缺失时降级而非中断
            print(f"    ⚠ A3Laya 跳过（{type(e).__name__}: {str(e)[:100]}）"
                  f"——A3 逐行输出未存档，需单独重跑才能拆位宽")
        print(f"    （A3 耗时 {time.time()-t_l:.0f}s）")

    Q = dict(config=dict(tag=a.tag, H=H_ROLL, delta_star=DELTA_STAR, bs=list(BS),
                         n_roll=a.n_roll, lead=LEAD, hit_w=HIT_W,
                         lam=lam, lam_se=lam_se, eta=eta, is_dense=is_dense,
                         jump_probe=jump_probe,
                         a1_pred={str(k): v for k, v in A1.items()},
                         thr_d2=thr_d2, mu_n=mu_n, sd_n=sd_n),
             rows=[{k: v for k, v in r.items() if k != "acts"} | {"acts": r["acts"]}
                   for r in rows],
             arms=res,
             laya_available=laya_arm is not None,
             elapsed_s=round(time.time() - t0, 1))
    out = os.path.join(ROOT, "logs", f"t14_warn_{a.tag}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(Q, f, ensure_ascii=False)
    print(f"\n  → {out}")


if __name__ == "__main__":
    main()
