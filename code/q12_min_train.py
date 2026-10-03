# -*- coding: utf-8 -*-
"""下一批 · 最小集：3 个训练目标 × 1 seed（只判「迹象」，不出结论）

规格（下一批工作_定稿规格.html）：
  · 目标集合：single（单步 MSE，基线）／multistep K=5／a-sup（把每步放大率写进损失，正对照）
  · 统一 400 epochs、不早停；同数据、同架构、同 seed 处理 ⇒ **只换训练目标**
  · 每臂必记 (a, margin)：a = 孪生纯动力学口径（相对窗口）；margin = val_cos − 拷贝上一帧基线
  · 最小集判据（只决定投不投）：存在任一臂 |Δlog a| ≥ **0.5 × Δ_thr** ⇒ 继续补 seed
      Δ_thr = 2 × σ_used；1 seed 时 σ_used 取 T12 先验上界 **0.094**（加固①：不得用本批样本估）
  · ⚠ 本阶段**不许出现「a 动了」这类结论句**

自包含：基线臂就在本批里（同数据同架构同 epochs），所以不依赖外部 a=1.139 才能比较。

产物：logs/q12_min/<arm>_s0.json + logs/q12_min_summary.json
用法：python q12_min_train.py            # 跑三个目标
      python q12_min_train.py --arm single
"""
import argparse
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

from t2b_route import DenseDynamics  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_LAT = os.path.join(ROOT, "ckpt", "latents_gray_s3__n_seq12000_seed0.pt")
VAL_LAT = os.path.join(ROOT, "ckpt", "latents_gray_s3__n_seq128_seed999.pt")
OUTDIR = os.path.join(ROOT, "logs", "q12_min")
SUMMARY = os.path.join(ROOT, "logs", "q12_min_summary.json")
SUMMARY_TAG = None   # main 里按 --tag 覆盖

CTX, LATENT, K_MS = 2, 64, 5
HIDDEN, EPOCHS, BATCH = 192, 400, 256
SEED = 0
EPS_TWIN = 1e-4
N_PAIRS = 16
REL_ORDERS = 100.0        # 相对窗口起点：分离越过初始值 2 个数量级（**诊断用**）
WIN = 12                  # 窗口长度（写死，不许事后挑）
FIX_T0 = 8                # ★ 主口径窗口起点：固定绝对窗口 [8, 20]
#   为什么是 8：Q1.1 探针（code/q11_probe.py）里"持续段"就是把窗口网格定成
#   1→3 / 3→8 / **8→20** / 20→40 —— 8 是**预先定义**的持续段起点，不是看着
#   最小集结果挑的。相对窗口在各臂漂移（8 / 9 / 26 步）⇒ 跨臂不可比，故换固定窗。
SIGMA_PRIOR = 0.094       # 加固①：n=5 时的先验上界（本批只有 1 seed，不得用样本估）
MU_A = 1.0                # a-sup 的惩罚系数
EPS_A = 1e-3              # a-sup 里有限差分的扰动量（归一化空间）


def load_latents(p):
    d = torch.load(p, map_location="cpu", weights_only=False)
    return d["z"].float()


def make_pairs(z, K):
    """返回 (ctx_a, ctx_b, targets(K 步真值))；全程在**原始**空间，训练时再归一化。"""
    T = z.shape[1]
    a, b, y = [], [], []
    for t in range(1, T - K):
        a.append(z[:, t - 1])
        b.append(z[:, t])
        y.append(z[:, t + 1:t + 1 + K])
    return torch.stack(a, 1), torch.stack(b, 1), torch.stack(y, 1)   # (N,T',L)


def to_norm(x, mean, std):
    return (x - mean) / std


def train_arm(arm, ztr, zva, mean, std, epochs=EPOCHS, seed=SEED, mu_a=None, hidden=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    K = K_MS if arm == "multistep" else 1
    mu_a = MU_A if mu_a is None else float(mu_a)
    hid = hidden if hidden is not None else HIDDEN
    A, B, Y = make_pairs(ztr, K)
    # ⚠ 样本是 (序列 × 时刻) 两个维度的乘积，不是序列数 —— 必须合并 dim0/dim1
    A = A.reshape(-1, LATENT)
    B = B.reshape(-1, LATENT)
    Y = Y.reshape(-1, K, LATENT)
    N = A.shape[0]
    A = to_norm(A, mean, std)
    B = to_norm(B, mean, std)
    Yn = to_norm(Y, mean, std)                 # (N, T', K, L)
    m = DenseDynamics(hidden=hid, ctx=CTX, act_dim=0).to(DEV)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        idx = torch.randperm(N)[:BATCH]
        a, b, y = A[idx].to(DEV), B[idx].to(DEV), Yn[idx].to(DEV)
        zc = torch.stack([a, b], dim=1)
        if arm == "multistep":
            loss = 0.0
            ca, cb = a, b
            for k in range(K):
                p, _ = m(torch.stack([ca, cb], dim=1), None, None)
                loss = loss + F.mse_loss(p, y[:, k])
                ca, cb = cb, p
            loss = loss / K
        else:
            p, _ = m(zc, None, None)
            mse_part = F.mse_loss(p, y[:, 0])
            loss = mse_part
            if arm == "a_sup":
                g = torch.Generator(device="cpu").manual_seed(1000 + ep)
                d = torch.randn(zc.shape, generator=g).to(DEV)      # (B, ctx, L)
                dn = d.flatten(1).norm(dim=1)                       # (B,)
                d = d / dn.view(-1, 1, 1) * EPS_A                    # 按样本展平范数归一（三维广播）
                p2, _ = m(zc + d, None, None)
                amp = (p2 - p).norm(dim=1) / dn
                loss = loss + mu_a * torch.log(amp.clamp_min(1e-6)).mean()
                mse_part_last = float(mse_part.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
        sch.step()
        hist.append(float(loss.detach()))
        if ep % 100 == 0 or ep == epochs - 1:
            print(f"    [{arm}] ep {ep:>3}  loss {hist[-1]:.5f}")
    return m, dict(arm=arm, K=K, hidden=hid, epochs=epochs, seed=seed, mu_a=mu_a,
                   n_params=n_par, loss_last=hist[-1], loss_head=hist[0],
                   train_s=round(time.time() - t0, 1))


# ------------------------------------------------------------------ 评测
def margin_and_cos(m, zva, mean, std):
    """val 单步余弦 + 拷贝上一帧基线 + 余量（全部在原始空间算）。"""
    z = zva
    a = z[:, :-2].reshape(-1, LATENT)
    b = z[:, 1:-1].reshape(-1, LATENT)
    y = z[:, 2:].reshape(-1, LATENT)
    with torch.no_grad():
        zc = torch.stack([to_norm(a, mean, std), to_norm(b, mean, std)], dim=1).to(DEV)
        p, _ = m(zc, None, None)
        p = p * std.to(DEV) + mean.to(DEV)
        cos = float(F.cosine_similarity(p, y.to(DEV), dim=1).mean())
    cos_deg = float(F.cosine_similarity(b, y, dim=1).mean())      # 拷贝上一帧
    return dict(val_cos=cos, degenerate_prev=cos_deg, margin=cos - cos_deg)


def twin_a(m, zva, mean, std, eps=EPS_TWIN, n_pairs=N_PAIRS):
    """孪生纯动力学。λ̂ 主口径 = **固定绝对窗口 [FIX_T0, FIX_T0+WIN]**（跨臂可比）；
    另报相对窗口值（"越过初始值 2 个数量级"）作诊断 —— 它会随臂漂移。"""
    import copy as _c
    m64 = _c.deepcopy(m).double()
    mean64, std64 = mean.to(DEV).double(), std.to(DEV).double()
    H = 60
    sl = []
    with torch.no_grad():
        for k in range(n_pairs):
            s = k % zva.shape[0]
            t0 = 5
            za = ((zva[s, t0 - 1].to(DEV).double() - mean64) / std64).unsqueeze(0)
            zb = ((zva[s, t0].to(DEV).double() - mean64) / std64).unsqueeze(0)
            g = torch.Generator(device="cpu").manual_seed(5000 + k)
            dv = torch.randn(zb.shape, generator=g).to(DEV).double()
            zb2 = zb + dv / dv.norm() * eps
            d = [eps]
            for t in range(H):
                p, _ = m64(torch.stack([za, zb], dim=1), None, None)
                p2, _ = m64(torch.stack([za, zb2], dim=1), None, None)
                d.append(float((p2 - p).norm()))
                za, zb, zb2 = zb, p, p2
            sl.append(d)
    d = np.array(sl)                       # (n_pairs, H+1)

    def _lam(t0_, t1_):
        """给定窗口的 λ̂ 中位；同时报被丢弃的样本数（扰动下溢/非正）。"""
        vals, drop = [], 0
        for row in d:
            if t1_ < len(row) and row[t0_] > 0 and row[t1_] > 0:
                vals.append(math.log(row[t1_] / row[t0_]) / WIN)
            else:
                drop += 1
        return vals, drop

    # ---- 主口径：固定绝对窗口 [FIX_T0, FIX_T0+WIN] ----
    v_fix, drop_fix = _lam(FIX_T0, FIX_T0 + WIN)
    # ---- 诊断：相对窗口（起点 = 中位轨迹越过 2 个数量级的步）----
    med = np.median(d, axis=0)
    start = max(int(np.argmax(med / med[0] >= REL_ORDERS)), 1)
    v_rel, drop_rel = _lam(start, start + WIN)
    return dict(a=float(np.median(v_fix)),                 # ★ 主口径（跨臂只用它）
                a_p10=float(np.percentile(v_fix, 10)),
                a_p90=float(np.percentile(v_fix, 90)),
                a_rel=float(np.median(v_rel)) if v_rel else None,
                window=[FIX_T0, FIX_T0 + WIN], win_rel=[start, start + WIN],
                start_step=start, n_pairs=len(v_fix),
                n_drop_fix=drop_fix, n_drop_rel=drop_rel)


def _summary_path(tag=""):
    """带 tag 时落到 q12_min_summary_<tag>.json（μ 扫描各档分开留档）。"""
    if not tag:
        return SUMMARY
    return os.path.join(ROOT, "logs", f"q12_min_summary_{tag.strip('_')}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="all", choices=["all", "single", "multistep", "a_sup"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--mu_a", type=float, default=None, help="a-监督的惩罚系数（默认 MU_A）")
    ap.add_argument("--tag", default="", help="产物文件名后缀，如 mu2")
    ap.add_argument("--hidden", type=int, default=None,
                    help="模型隐层宽度（默认 HIDDEN=192）")
    ap.add_argument("--seeds", default="0",
                    help="逗号分隔，如 0,1,2,3,4。跨臂比较用各 seed 的 log a 均值；"
                         "≥5 个 seed 才允许出正式判定")
    a = ap.parse_args()
    ep_n = a.epochs
    seeds = [int(x) for x in str(a.seeds).split(",") if x.strip() != ""]
    hidden = a.hidden if a.hidden is not None else HIDDEN
    mu_a = a.mu_a
    tag = ("_" + a.tag) if a.tag else ""
    os.makedirs(OUTDIR, exist_ok=True)

    ztr, zva = load_latents(TRAIN_LAT), load_latents(VAL_LAT)
    flat = ztr.reshape(-1, LATENT)
    mean, std = flat.mean(0), flat.std(0)
    print(f"数据：train {tuple(ztr.shape)} / val {tuple(zva.shape)}")
    print(f"配置：hidden={HIDDEN} epochs={ep_n} batch={BATCH} ctx={CTX} seeds={seeds}")
    print(f"主口径窗口：固定绝对窗口 [{FIX_T0}, {FIX_T0 + WIN}]"
          f"（相对窗口仅作诊断）\n")

    arms = ["single", "multistep", "a_sup"] if a.arm == "all" else [a.arm]
    res = {}                                    # res[arm][seed] = r
    for arm in arms:
        for sd in seeds:
            print(f"=== {arm} / seed {sd} ===")
            m, cfg = train_arm(arm, ztr, zva, mean, std, epochs=ep_n, seed=sd, mu_a=mu_a, hidden=hidden)
            mc = margin_and_cos(m, zva, mean, std)
            ta = twin_a(m, zva, mean, std)
            r = dict(config=cfg, **mc, **ta)
            res.setdefault(arm, {})[sd] = r
            json.dump(r, open(os.path.join(OUTDIR, f"{arm}_s{sd}{tag}.json"), "w",
                              encoding="utf-8"), ensure_ascii=False, indent=1)
            print(f"  → a={r['a']:.4f}（固定窗）｜a_rel={r['a_rel']:.4f}（起点第 {r['start_step']} 步）"
                  f"｜val_cos={r['val_cos']:.4f}｜margin={r['margin']:+.4f}"
                  f"｜丢弃 {r['n_drop_fix']}/{r['n_pairs'] + r['n_drop_fix']}\n")

    # ---------------- 聚合 ----------------
    def agg(arm):
        rs = [res[arm][sd] for sd in seeds]
        la = [r["a"] for r in rs]   # 率尺度：a 本身即每步对数放大率，不可再取 log
        mg = [r["margin"] for r in rs]
        return dict(a=float(np.mean([r["a"] for r in rs])),
                    a_mean=float(np.mean(la)),
                    a_std=float(np.std(la, ddof=1)) if len(la) > 1 else None,
                    margin=float(np.mean(mg)),
                    margin_std=float(np.std(mg, ddof=1)) if len(mg) > 1 else None,
                    a_rel=float(np.mean([r["a_rel"] for r in rs if r["a_rel"] is not None])),
                    val_cos=float(np.mean([r["val_cos"] for r in rs])),
                    degenerate_prev=rs[0]["degenerate_prev"],
                    n_seeds=len(seeds), train_s=sum(r["config"]["train_s"] for r in rs))

    A = {arm: agg(arm) for arm in arms}

    print("=" * 104)
    print("  聚合（跨臂只用 (a, margin)：a_sup 的 loss 含 log(amp) 项，不可跨臂比）")
    print("=" * 104)
    print(f"  {'臂':<12}{'a(固定窗)':>11}{'a_rel':>9}{'margin':>10}{'val_cos':>9}{'seed 数':>8}")
    for arm in arms:
        t = A[arm]
        print(f"  {arm:<12}{t['a']:>11.4f}{t['a_rel']:>9.4f}{t['margin']:>+10.4f}"
              f"{t['val_cos']:>9.4f}{t['n_seeds']:>8}")

    base = A.get("single")
    if base is None:
        print(f"\n  ⚠ 没有 single 基线臂 ⇒ 跳过臂间比较")
        json.dump(dict(config=dict(hidden=HIDDEN, epochs=ep_n, batch=BATCH, seeds=seeds,
                                   fix_window=[FIX_T0, FIX_T0 + WIN]),
                       arms=A, per_seed=res, verdict="未判定：缺 single 基线臂"),
                  open(_summary_path(tag), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        print(f"  → {_summary_path(tag)}")
        return res

    # ---------------- 噪声底（加固①）----------------
    if base["a_std"] is not None:
        s_batch = base["a_std"]
        sigma_used = max(SIGMA_PRIOR, 2.372 * s_batch)      # n=5 时 σ 的 95% 上界
        src = f"max(先验 {SIGMA_PRIOR}, 2.372×s_batch {s_batch:.4f})"
    else:
        sigma_used = SIGMA_PRIOR
        src = f"先验 {SIGMA_PRIOR}（1 seed，不得用样本估）"
    delta_thr = 2 * sigma_used
    formal = len(seeds) >= 5

    print(f"\n  噪声底：σ_used = {sigma_used:.4f} = {src}")
    print(f"  Δ_thr = 2σ = {delta_thr:.4f}；正式判定需 ≥5 seed（当前 {len(seeds)}）")
    print(f"\n  {'臂':<12}{'Δlog a':>10}{'|Δ|/Δ_thr':>11}{'Δmargin':>11}  判读")
    verdicts = {}
    for arm in arms:
        t = A[arm]
        d = t["a_mean"] - base["a_mean"]
        dm = t["margin"] - base["margin"]
        moved = abs(d) >= delta_thr
        kept = t["margin"] > 0            # 不低于"拷贝上一帧"基线
        if arm == "single":
            verdicts[arm] = "基线臂"
        elif not formal:
            verdicts[arm] = "迹象级（seed 不足，不得作结论）" if abs(d) >= 0.5 * delta_thr else "无迹象"
        elif moved and kept:
            verdicts[arm] = "★ 真旋钮（a 降且 margin 保持）"
        elif moved and not kept:
            verdicts[arm] = "a 降但 margin 崩 ⇒ 不是旋钮，是更差的模型"
        else:
            verdicts[arm] = "a 未被该目标推动"
        print(f"  {arm:<12}{d:>+10.4f}{abs(d) / delta_thr:>11.2f}{dm:>+11.4f}  {verdicts[arm]}")

    if not formal:
        keep = any(verdicts[arm].startswith("迹象级") for arm in arms if arm != "single")
        verdict = ("存在迹象 ⇒ 继续补 seed（≥5）再做正式判定" if keep
                   else "无迹象 ⇒ 按规格停手：记「a 在该目标集合下无迹象」")
    else:
        verdict = "正式判定：" + "；".join(f"{k}={v}" for k, v in verdicts.items() if k != "single")
    print(f"\n  → {verdict}")
    if not formal:
        print("\n  ⚠ 纪律：本阶段任何「a 动了 / a 没动」的结论都不成立（seed 不足）。")

    json.dump(dict(config=dict(hidden=HIDDEN, epochs=ep_n, batch=BATCH, seeds=seeds,
                               ctx=CTX, K_ms=K_MS, mu_a=MU_A if mu_a is None else mu_a, eps_a=EPS_A,
                               sigma_prior=SIGMA_PRIOR, sigma_used=sigma_used,
                               delta_thr=delta_thr, fix_window=[FIX_T0, FIX_T0 + WIN],
                               win_len=WIN, train_latents=os.path.basename(TRAIN_LAT),
                               val_latents=os.path.basename(VAL_LAT)),
                   arms=A, per_seed=res, verdicts=verdicts,
                   formal_verdict=formal, verdict=verdict),
              open(_summary_path(tag), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\n  产物 → {_summary_path(tag)}")
    return res


if __name__ == "__main__":
    main()
