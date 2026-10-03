# -*- coding: utf-8 -*-
"""M2 · 音频三判据全套实测（第四阶段主结果）。

在 M1i 的 koop（音频 PCA16 潜码）上跑完整判据套件：
  ①a_alt（Jacobian 谱口径，M1i 已证可用）+ 孪生对照（记录口径分裂）；
  ②margin 曲线 + h*（配对 bootstrap）；
  ③η(b) 量化轴五档（b∈{4,6,8,10,12}，音频潜码 per-dim range 标定）→
    实测首崩（均值曲线过阈）vs 解析 T_crash(b)=ln(1+δ*(e^a−1)/η(b))/a；
  ④平凡基线：拷贝（上一帧，已内建于 margin）+ 静音（零向量）；
  ⑤模态对照表：音频（本次）vs 视觉 MM latent（X5/B1）vs 视觉状态 pusht（D2b）——
    同一套判据流程在三模态-域组合跑通 = 模态无关性的操作验证。
预注册：T3 量化首崩随 b 单调（更大 b ⇒ 更晚崩）；音频量级预期不跨模态（独立验收）。
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

from vae import MovingMNIST  # noqa: E402
from scipy import ndimage  # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
SR, FPS = 24000, 12.5
SAMPLES_PER_FRAME = int(SR / FPS)
DIM_SRC = 64
H_ROLL, COS_THR = 18, 0.9
EPOCHS, BATCH = 400, 256
N_SEQ = 64
B_GRID = (4, 6, 8, 10, 12)
OUT_JSON = os.path.join(ROOT, "logs", "m2_audio_suite.json")
t0 = time.time()


def detect(f):
    lab, n = ndimage.label((f > 0).astype(np.uint8))
    dets = []
    for oi in range(1, n + 1):
        ys, xs = np.where(lab == oi)
        if len(xs) < 8:
            continue
        dets.append(dict(cx=float(xs.mean()), cy=float(ys.mean())))
    return sorted(dets, key=lambda d: d["cx"])


def synth_audio(raw, bounce_events=None):
    """M1g 能量调制：频率固定（440/988Hz），音量随 x 大范围变化。撞墙冲击叠加。"""
    T = raw.shape[0]
    cents = [detect(raw[t]) for t in range(T)]
    audio = np.zeros((T, SAMPLES_PER_FRAME), dtype=np.float64)
    tt = np.arange(SAMPLES_PER_FRAME) / SR
    bounces = []
    if bounce_events is not None:
        bounces = list(bounce_events)
    else:
        for di in range(2):
            xs = [c[di]["cx"] for c in cents if len(c) > di]
            if len(xs) < 3:
                continue
            for t in range(1, len(xs) - 1):
                if (xs[t] - xs[t - 1]) * (xs[t + 1] - xs[t]) < 0 and abs(xs[t]) > 8:
                    bounces.append(dict(digit=di, frame=t))
                    break
    for t in range(T):
        c = cents[t]
        seg = np.zeros(SAMPLES_PER_FRAME)
        if len(c) >= 1:
            amp1 = 0.05 + 0.95 * (c[0]["cx"] / 64.0)
            seg += 0.4 * amp1 * np.sin(2 * np.pi * 440 * tt)
        if len(c) >= 2:
            amp2 = 0.05 + 0.95 * (c[1]["cx"] / 64.0)
            seg += 0.4 * amp2 * np.sin(2 * np.pi * 988 * tt)
        for b in bounces:
            if b["frame"] == t:
                n_imp = int(SR * 0.06)
                ti = np.arange(n_imp) / SR
                rng = np.random.default_rng(1000 + b["frame"])
                imp = rng.standard_normal(n_imp) * np.exp(-ti * 60) * 0.9
                seg[:n_imp] += imp / max(np.abs(imp).max(), 1e-9)
        audio[t] = seg
    return audio, bounces


def build_filterbank_full():
    fb = np.zeros((N_MEL := 64, SAMPLES_PER_FRAME // 2 + 1), dtype=np.float32)
    freqs = np.fft.rfftfreq(SAMPLES_PER_FRAME, 1 / SR)
    fmax = freqs[-1]
    centers = np.linspace(0, fmax, N_MEL + 2)
    for m in range(1, N_MEL + 1):
        lo, ce, hi = centers[m - 1], centers[m], centers[m + 1]
        for j, f in enumerate(freqs):
            if lo < f <= ce:
                fb[m - 1, j] = (f - lo) / (ce - lo)
            elif ce < f < hi:
                fb[m - 1, j] = (hi - f) / (hi - ce)
    return fb


FB_FULL = build_filterbank_full()


def frame_to_latent(audio_seq):
    """(T, SPF) → (T, 64)：全窗 FFT → 64 三角带能量 → log1p（零训练固定编码器）。"""
    T = audio_seq.shape[0]
    out = np.zeros((T, N_MEL), dtype=np.float32)
    for t in range(T):
        spec = np.abs(np.fft.rfft(audio_seq[t])) ** 2
        out[t] = np.log1p(FB_FULL @ spec)
    return out


N_MEL = 64


class Koop(nn.Module):
    def __init__(self, dim, s=0.5):
        super().__init__()
        self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        self.r = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st):
        return st @ self.K.T + self.s * torch.tanh(self.r(torch.cat([sp, st], dim=1)))


def jacobian_spectral_a(m, points, dim, eps=1e-4):
    rhos = []
    with torch.no_grad():
        for z0 in points:
            z = torch.tensor(z0, dtype=torch.float32, device=DEV).unsqueeze(0)
            J = np.zeros((dim, dim))
            for j in range(dim):
                dz = torch.zeros_like(z); dz[0, j] = eps
                p_plus = m(z, z + dz)[0]
                p_minus = m(z, z - dz)[0]
                J[:, j] = ((p_plus - p_minus) / (2 * eps)).cpu().numpy()
            w = np.linalg.eigvals(J)
            rhos.append(float(np.max(np.abs(w))))
    rho_med = float(np.median(rhos))
    return rho_med, math.log(max(rho_med, 1e-9))


def main():
    ds = MovingMNIST(n=N_SEQ, seed=999, train=False)
    lat_seqs, bounce_info = [], []
    for i in range(N_SEQ):
        raw = ds[i].numpy()
        audio, bounces = synth_audio(raw)
        lat_seqs.append(frame_to_latent(audio))
        bounce_info.append(dict(seq=i, bounces=bounces))
    lat_seqs = np.array(lat_seqs)
    N = len(lat_seqs)
    n_train = int(N * 0.75)

    # PCA16
    flat_all = lat_seqs[:n_train].reshape(-1, N_MEL).astype(np.float64)
    mu_p = flat_all.mean(0)
    C = np.cov((flat_all - mu_p).T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1][:16]
    evr = w[order] / w.sum()
    P = V[:, order]
    lat_seqs = ((lat_seqs.reshape(-1, N_MEL) - mu_p) @ P).reshape(len(lat_seqs), -1, 16)
    DIM = 16
    print(f"[PCA] 前 16 主成分解释方差比: {evr.sum():.3f}｜潜码 {DIM} 维", flush=True)
    mu = lat_seqs[:n_train].reshape(-1, DIM).mean(0)
    sd = np.maximum(lat_seqs[:n_train].reshape(-1, DIM).std(0), 1e-6)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=DEV)
    sd_t = torch.tensor(sd, dtype=torch.float32, device=DEV)
    nrm = lambda x: (x - mu) / sd

    Atr, Btr = [], []
    for k in range(n_train):
        tr = lat_seqs[k]
        for t in range(1, tr.shape[0]):
            Atr.append(tr[t - 1]); Btr.append(tr[t])
    Atr = torch.tensor(nrm(np.array(Atr)), dtype=torch.float32)
    Btr = torch.tensor(nrm(np.array(Btr)), dtype=torch.float32)

    torch.manual_seed(0); np.random.seed(0)
    m = Koop(dim=DIM).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(len(Atr))[:BATCH]
        p = m(Atr[idx].to(DEV), Btr[idx].to(DEV))
        loss = F.mse_loss(p, Btr[idx].to(DEV))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    print("koop 训练完成", flush=True)

    # ---- 红线：瞬态保留 ----
    ratios = []
    for bi, binfo in enumerate(bounce_info):
        if not binfo["bounces"]:
            continue
        tr = nrm(lat_seqs[bi])
        for b in binfo["bounces"]:
            if b["frame"] < 2 or b["frame"] >= tr.shape[0] - 1:
                continue
            d_e = float(np.linalg.norm(tr[b["frame"]] - tr[b["frame"] - 1]))
            d_c = float(np.linalg.norm(tr[b["frame"] - 1] - tr[b["frame"] - 2]))
            if d_c > 1e-9:
                ratios.append(d_e / max(d_c, 1e-9))
    ratio = float(np.median(ratios)) if ratios else float("nan")
    redline = bool(ratios and ratio > 1.5)
    print(f"红线：{ratio:.2f} → {'PASS' if redline else 'FAIL'}", flush=True)

    # ---- ① a_alt（Jacobian 谱）----
    val_lat = [nrm(lat_seqs[k]) for k in range(n_train, N)]
    eval_pts = np.array([tr[t] for tr in val_lat for t in range(2, tr.shape[0] - 1)][:64])
    rho_med, a_alt = jacobian_spectral_a(m, eval_pts, DIM)
    print(f"① a_alt = log(ρ(J)) = {a_alt:+.4f}（ρ={rho_med:.4f}）", flush=True)

    # ---- ② margin 曲线 + h* ----
    cos_p = np.full((len(val_lat), H_ROLL), np.nan)
    cos_c = np.full((len(val_lat), H_ROLL), np.nan)
    with torch.no_grad():
        for k, tr in enumerate(val_lat):
            if tr.shape[0] < H_ROLL + 2:
                continue
            sp = torch.tensor(tr[0]).float().to(DEV)
            st = torch.tensor(tr[1]).float().to(DEV)
            for t in range(H_ROLL):
                p = m(sp.unsqueeze(0), st.unsqueeze(0))[0]
                zt = torch.tensor(tr[t + 2]).float().to(DEV)
                cos_p[k, t] = F.cosine_similarity(p[None].float(), zt[None].float()).item()
                cos_c[k, t] = F.cosine_similarity(st[None].float(), zt[None].float()).item()
                sp, st = st, p
    margin = [float(np.nanmean(cos_p[:, t] - cos_c[:, t])) for t in range(H_ROLL)]
    rng_b = np.random.default_rng(42)
    h_star = 0
    for t in range(H_ROLL):
        d_seq = cos_p[:, t] - cos_c[:, t]
        d_seq = d_seq[~np.isnan(d_seq)]
        if len(d_seq) < 5:
            continue
        boots = [float(np.mean(d_seq[rng_b.integers(0, len(d_seq), len(d_seq))])) for _ in range(2000)]
        if float(np.percentile(boots, 2.5)) > 0:
            h_star = t + 1
    print(f"② margin: " + " ".join(f"{v:+.3f}" for v in margin) + f"｜h* = {h_star}", flush=True)

    # ---- ③ η(b) 量化轴五档 ----
    # robust range：1-99 分位（min/max 被极端值主导会让量化直接摧毁信号）
    rmin = np.percentile(lat_seqs.reshape(-1, DIM), 1, axis=0)
    rmax = np.percentile(lat_seqs.reshape(-1, DIM), 99, axis=0)
    rmin_t = torch.tensor(rmin, dtype=torch.float32, device=DEV)
    rmax_t = torch.tensor(rmax, dtype=torch.float32, device=DEV)
    eta = {}
    crash_thr = {}
    for b in B_GRID:
        delta = (rmax - rmin) / (2 ** b - 1)
        delta_t = torch.clamp((rmax_t - rmin_t) / (2 ** b - 1), min=1e-6)  # 低方差维防除零
        eta[b] = float(np.sqrt((delta ** 2 / 12).sum()))
        # 量化 rollout（全 GPU）：均值曲线过阈
        means = []
        with torch.no_grad():
            for k, tr in enumerate(val_lat):
                if tr.shape[0] < H_ROLL + 2:
                    continue
                sp = torch.tensor(tr[0]).float().to(DEV)
                st = torch.tensor(tr[1]).float().to(DEV)
                for t in range(H_ROLL):
                    p = m(sp.unsqueeze(0), st.unsqueeze(0))[0]
                    if True:  # 量化段恒量化
                        pp = p * sd_t + mu_t
                        pp = torch.round(pp / delta_t) * delta_t
                        p = (pp - mu_t) / sd_t
                    zt = torch.tensor(tr[t + 2]).float().to(DEV)
                    means.append(float(F.cosine_similarity(p[None].float(), zt[None].float()).item()))
                    sp, st = st, p
        # 按序列 reshape 求均值曲线 → 首崩
        arr = np.array(means).reshape(-1, H_ROLL)
        curve = [float(np.nanmean(arr[:, t])) for t in range(H_ROLL)]
        first = next((t + 1 for t, v in enumerate(curve) if v < COS_THR), H_ROLL)
        crash_thr[b] = first
        t_crash = (math.log(1 + 0.1 * (math.exp(a_alt) - 1) / eta[b]) / a_alt
                   if a_alt > 1e-6 else float("inf"))
        print(f"③ b={b}: η={eta[b]:.4f}｜实测首崩 {first}｜解析 T_crash(δ*=0.1) {t_crash:.1f}", flush=True)
    monotone = all(crash_thr[B_GRID[i]] <= crash_thr[B_GRID[i + 1]] + 0.5
                   for i in range(len(B_GRID) - 1))
    t3 = bool(monotone)
    print(f"   量化轴单调 = {monotone}")

    # ---- ④ 静音基线 ----
    silent_cos = []
    with torch.no_grad():
        for k, tr in enumerate(val_lat):
            z0 = np.zeros(DIM, dtype=np.float32)
            zt = tr[2]
            silent_cos.append(float(F.cosine_similarity(
                torch.tensor(z0).to(DEV)[None].float(), torch.tensor(zt).to(DEV)[None].float()).item()))
    silent_cos = float(np.mean(silent_cos))
    print(f"④ 静音基线 cos = {silent_cos:.4f}")

    # ---- ⑤ 模态对照表（读归档）----
    modal = {}
    try:
        x5 = json.load(open(os.path.join(ROOT, "logs", "x5_decouple.json"), encoding="utf-8"))
        modal["visual_mm"] = dict(a=0.66, eta0=x5["eta0"]["single"], h_star=float(
            np.median([np.nan])))
        modal["visual_mm"].pop("h_star")
    except Exception:
        pass
    try:
        d2b = json.load(open(os.path.join(ROOT, "logs", "d2b_pusht_suite.json"), encoding="utf-8"))
        modal["state_pusht"] = dict(a=d2b["a"]["mean"], h_star=d2b["margin"]["h_star"])
    except Exception:
        pass
    modal["audio_mm"] = dict(a_alt=a_alt, h_star=h_star, redline=ratio)

    g1 = bool(np.isfinite(a_alt))
    gate = bool(redline and g1 and (h_star > 0) and t3)
    verdict = ("音频三判据全套可测（a_alt/margin/量化轴）——模态无关性操作验证成立" if gate else
               "部分可测（如实记录）")
    print(f"\n判定：{verdict}")

    out = dict(a_alt=a_alt, rho=rho_med, margin=margin, h_star=h_star,
               eta={str(b): eta[b] for b in B_GRID},
               crash_thr={str(b): crash_thr[b] for b in B_GRID},
               monotone=bool(monotone), silent_cos=silent_cos,
               modal=modal, gate=bool(gate), verdict=verdict,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/m2_audio_suite.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
