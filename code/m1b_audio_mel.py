# -*- coding: utf-8 -*-
"""M1.1 · 宽带音色 + mel 谱图 + 轻量卷积编码器（判据门重跑）。

对照 M1.0 的两个根因改动：
  ①音频合成：正弦纯音 → **带通噪声**（中心频率随数字 x 位置移动，300-900 / 900-1500Hz）
    ——位置变化 → 频谱形状连续变化 → 拷贝基线不再最优；
  ②编码器：线性频带能量 → mel 式三角滤波谱图（8 子窗 × 64 带）→ 轻量 CNN → 64 维，
    **端到端与动力学联合训练**（编码目标 = 利于状态转移预测的表示，next-state 语义）。

预注册（判据门同 M1.0，新增分模态独立验收条款）：
  红线：撞墙事件窗/平稳窗潜码距离比 > 1.5；
  G1 a 可测（有限）；G2 margin 非饱和且有视界 CI 下界>0；G3 b=4 首崩 < b=12 首崩；
  新增 M2 条款（M1.0 教训）：音频判据独立验收，量级预期不跨模态。
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
SPF = int(SR / FPS)                          # 1920 样本/帧
N_MEL, N_SUB = 64, 8                         # 64 带 × 8 子窗
SUB = SPF // N_SUB                           # 240 样本/子窗
DIM = 64
H_ROLL, COS_THR = 18, 0.9
EPOCHS, BATCH = 300, 128
N_SEQ = 64
SEEDS = (0, 1, 2)
OUT_JSON = os.path.join(ROOT, "logs", "m1b_audio_mel.json")
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


def broadband_tone(seg_len, center, rng):
    """带通噪声：白噪 → 频域高斯窗（中心 center，带宽 250Hz）→ irfft。宽带音色。"""
    w = rng.standard_normal(seg_len)
    spec = np.fft.rfft(w)
    freqs = np.fft.rfftfreq(seg_len, 1 / SR)
    bw = 250.0
    gain = np.exp(-((freqs - center) ** 2) / (2 * bw ** 2))
    return np.fft.irfft(spec * gain, n=seg_len)


def synth_audio_broadband(raw, bounces):
    """宽带音色：数字 1 → 300-900Hz 带通噪声（中心随 x），数字 2 → 900-1500Hz（中心随 x）。
    撞墙帧叠加冲击。返回 (T, SPF)。"""
    T = raw.shape[0]
    rng = np.random.default_rng(7)
    audio = np.zeros((T, SPF), dtype=np.float64)
    tt = np.arange(SPF) / SR
    for t in range(T):
        c = detect(raw[t])
        seg = np.zeros(SPF)
        rng_t = np.random.default_rng(hash((t)) % (2 ** 32))
        if len(c) >= 1:
            f1 = 300 + (c[0]["cx"] / 64.0) * 600
            seg += 0.5 * broadband_tone(SPF, f1, rng_t)
        if len(c) >= 2:
            f2 = 900 + (c[1]["cx"] / 64.0) * 600
            seg += 0.5 * broadband_tone(SPF, f2, rng_t)
        for b in bounces:
            if b["frame"] == t:
                n_imp = int(SR * 0.06)
                ti = np.arange(n_imp) / SR
                rr = np.random.default_rng(1000 + b["frame"])
                imp = rr.standard_normal(n_imp) * np.exp(-ti * 60) * 0.9
                imp += np.sin(2 * np.pi * 90 * ti) * np.exp(-ti * 40) * 0.8
                seg[:n_imp] += imp / max(np.abs(imp).max(), 1e-9)
        audio[t] = seg / max(np.abs(seg).max(), 1e-9) * 0.5
    return audio


# mel 式三角滤波器组（线性频率间隔三角带，0-12kHz，64 带）——固定，零训练
def build_filterbank():
    fb = np.zeros((N_MEL, SUB // 2 + 1), dtype=np.float32)   # 子窗长度（240 样本 → 121 频点）
    freqs = np.fft.rfftfreq(SUB, 1 / SR)
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


FB = build_filterbank()


def frame_to_melspec(audio_seq):
    """(T, SPF) → (T, N_SUB, N_MEL) log 谱图（8 子窗 × 64 三角带）。"""
    T = audio_seq.shape[0]
    out = np.zeros((T, N_SUB, N_MEL), dtype=np.float32)
    for t in range(T):
        for s in range(N_SUB):
            seg = audio_seq[t, s * SUB:(s + 1) * SUB]
            spec = np.abs(np.fft.rfft(seg)) ** 2
            out[t, s] = np.log1p(FB @ spec)
    return out


class CNNEnc(nn.Module):
    """(1,8,64) 谱图 → 64 维潜码。"""

    def __init__(self, dim=DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.AvgPool2d((2, 2)),                              # 4×32
            nn.Conv2d(32, 64, (3, 5), padding=(1, 2)), nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Linear(64, dim))

    def forward(self, x):
        return self.net(x)


class Koop(nn.Module):
    def __init__(self, dim=DIM, s=0.5):
        super().__init__()
        self.K = nn.Parameter(torch.eye(dim) * 0.9 + 0.01 * torch.randn(dim, dim))
        self.r = nn.Sequential(nn.Linear(2 * dim, 128), nn.SiLU(),
                               nn.Linear(128, 128), nn.SiLU(),
                               nn.Linear(128, dim))
        self.s = float(s)

    def forward(self, sp, st):
        return st @ self.K.T + self.s * torch.tanh(self.r(torch.cat([sp, st], dim=1)))


def main():
    ds = MovingMNIST(n=N_SEQ, seed=999, train=False)
    specs, raws, bounces_all = [], [], []
    for i in range(N_SEQ):
        raw = ds[i].numpy()
        cents = [detect(raw[t]) for t in range(raw.shape[0])]
        bounces = []
        for di in range(2):
            xs = [c[di]["cx"] for c in cents if len(c) > di]   # 重叠帧（1 域）跳过
            if len(xs) < 3:
                continue
            for t in range(1, len(xs) - 1):
                if (xs[t] - xs[t - 1]) * (xs[t + 1] - xs[t]) < 0 and abs(xs[t]) > 8:
                    bounces.append(dict(digit=di, frame=t))
                    break
        audio = synth_audio_broadband(raw, bounces)
        specs.append(frame_to_melspec(audio))
        raws.append(raw); bounces_all.append(bounces)
    specs = np.array(specs)                       # (N, T, 8, 64)
    N = len(specs)
    n_train = int(N * 0.75)
    print(f"谱图 {N} 条（train {n_train} / val {N - n_train}）｜mel {N_MEL}×{N_SUB}", flush=True)

    spec_t = torch.tensor(specs, dtype=torch.float32).unsqueeze(2)   # (N,T,1,8,64)
    mu = spec_t[:n_train].mean()
    sd = spec_t[:n_train].std().clamp_min(1e-6)
    spec_t = (spec_t - mu) / sd

    enc = CNNEnc().to(DEV)
    dyn = Koop().to(DEV)
    params = list(enc.parameters()) + list(dyn.parameters())
    opt = torch.optim.Adam(params, lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    # 训练对索引（train 序列相邻帧）
    pairs = []
    for k in range(n_train):
        for t in range(1, specs.shape[1]):
            pairs.append((k, t - 1, t))
    print(f"训练对 {len(pairs)}", flush=True)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(len(pairs))[:BATCH]
        loss_v = 0.0
        for (k, ta, tb) in [(pairs[j]) for j in idx]:
            za = enc(spec_t[k, ta].unsqueeze(0).to(DEV))
            zb = enc(spec_t[k, tb].unsqueeze(0).to(DEV))
            p = dyn(za, zb)
            loss_v = loss_v + F.mse_loss(p, zb)
        loss = loss_v / len(idx)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan ep{ep_}")
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    enc.eval(); dyn.eval()
    print("端到端训练完成", flush=True)

    # 潜码缓存（全序列）
    lat = []
    with torch.no_grad():
        for k in range(N):
            z = enc(spec_t[k].to(DEV))             # (T,64)
            lat.append(z.cpu().numpy())
    lat = np.array(lat)

    # ---- 红线 ----
    ratios = []
    for k in range(n_train, N):
        tr = (lat[k] - mu.numpy()) / sd.numpy()
        for b in bounces_all[k]:
            if b["frame"] < 2 or b["frame"] >= tr.shape[0] - 1:
                continue
            d_event = float(np.linalg.norm(tr[b["frame"]] - tr[b["frame"] - 1]))
            d_calm = float(np.linalg.norm(tr[b["frame"] - 1] - tr[b["frame"] - 2]))
            if d_calm > 1e-9:
                ratios.append(d_event / max(d_calm, 1e-9))
    ratio = float(np.median(ratios)) if ratios else float("nan")
    redline = bool(ratios and ratio > 1.5)
    print(f"红线：事件/平稳距离比中位 = {ratio:.2f} → {'PASS' if redline else 'FAIL'}")

    # ---- G1 a ----
    val_lat = [(lat[k] - mu.numpy()) / sd.numpy() for k in range(n_train, N)]
    lams = []
    g = torch.Generator(device="cpu").manual_seed(5000)
    with torch.no_grad():
        for _ in range(16):
            tr = val_lat[int(torch.randint(0, len(val_lat), (1,), generator=g))]
            t0i = 4
            za = torch.tensor(tr[t0i - 1]).double().to(DEV)
            zb = torch.tensor(tr[t0i]).double().to(DEV)
            enc64 = copy.deepcopy(enc).double().eval()
            dyn64 = copy.deepcopy(dyn).double().eval()
            dv = torch.randn(zb.shape, generator=g).double().to(DEV)
            zb2 = zb + dv / dv.norm() * 1e-4
            sep = [1e-4]
            for t in range(30):
                # 扰动传播：zb2 编码同一谱图扰动后的表示——近似用潜码直接演化
                p1 = dyn64(za.unsqueeze(0), zb.unsqueeze(0))[0]
                p2 = dyn64(za.unsqueeze(0), zb2.unsqueeze(0))[0]
                sep.append(float((p2 - p1).norm()))
                za, zb, zb2 = zb, p1, p2
            arr = np.array(sep)
            if np.isfinite(arr).all() and arr[8] > 1e-15 and arr[20] > 0:
                lams.append(math.log(arr[20] / arr[8]) / 12)
            del enc64, dyn64
    a_mean = float(np.median(lams)) if lams else float("nan")
    a_se = float(np.std(lams, ddof=1) / math.sqrt(len(lams))) if len(lams) > 1 else float("nan")
    g1 = bool(np.isfinite(a_mean))
    print(f"G1 a = {a_mean:+.4f} ± {a_se:.4f}（n={len(lams)}）→ {'可测' if g1 else '不可测'}")

    # ---- G2 margin ----
    cos_p = np.full((len(val_lat), H_ROLL), np.nan)
    cos_c = np.full((len(val_lat), H_ROLL), np.nan)
    with torch.no_grad():
        for k, tr in enumerate(val_lat):
            if tr.shape[0] < H_ROLL + 2:
                continue
            sp = torch.tensor(tr[0]).float().to(DEV)
            st = torch.tensor(tr[1]).float().to(DEV)
            for t in range(H_ROLL):
                p = dyn(sp.unsqueeze(0), st.unsqueeze(0))[0]
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
    non_sat = not (np.allclose(margin, 1.0) or np.allclose(margin, 0.0))
    g2 = bool(non_sat and h_star > 0)
    print(f"G2 margin: " + " ".join(f"{v:+.3f}" for v in margin))
    print(f"   h* = {h_star}｜非饱和 = {non_sat} → {'可测' if g2 else '不可测'}")

    # ---- G3 量化轴 ----
    firsts = {}
    with torch.no_grad():
        for b in (4, 12):
            firsts[b] = []
            for tr in val_lat[:12]:
                if tr.shape[0] < H_ROLL + 2:
                    continue
                sp = torch.tensor(tr[0]).float().to(DEV)
                st = torch.tensor(tr[1]).float().to(DEV)
                first = None
                for t in range(H_ROLL):
                    p = dyn(sp.unsqueeze(0), st.unsqueeze(0))[0]
                    pr = p.cpu().numpy()
                    dd = (lat.reshape(-1, DIM).max(0) - lat.reshape(-1, DIM).min(0)) / (2 ** b - 1)
                    pr = np.clip(np.round(pr / dd) * dd,
                                 lat.reshape(-1, DIM).min(0), lat.reshape(-1, DIM).max(0))
                    p = torch.tensor(((pr - mu.numpy()) / sd.numpy()).astype(np.float32)).to(DEV)
                    zt = torch.tensor(tr[t + 2]).float().to(DEV)
                    c = float(F.cosine_similarity(p[None].float(), zt[None].float()).item())
                    if first is None and c < COS_THR:
                        first = t + 1
                    sp, st = st, p
                firsts[b].append(first if first is not None else H_ROLL)
    f4 = float(np.median(firsts[4])); f12 = float(np.median(firsts[12]))
    g3 = bool(f4 < f12)
    print(f"G3 量化轴：b=4 首崩 {f4:.0f} vs b=12 首崩 {f12:.0f} → {'有梯度' if g3 else '无梯度'}")

    gate = bool(redline and g1 and g2 and g3)
    verdict = ("PASS：判据门通过 ⇒ 进 M2（音频三判据全套）" if gate else
               "FAIL：判据门未过 ⇒ 如实记录（M1.1 方案不足以支撑音频判据）")
    print(f"\n判据门判定：{'PASS' if gate else 'FAIL'}——{verdict}")
    print(f"  明细：红线 {ratio:.2f}｜G1 a={a_mean:+.4f}｜G2 h*={h_star} 非饱和={non_sat}｜G3 {f4:.0f}<{f12:.0f}={g3}")

    out = dict(redline=dict(ratio=ratio, n=len(ratios), passed=redline),
               g1=dict(a_mean=a_mean, a_se=a_se, n=len(lams), passed=g1),
               g2=dict(margin=margin, h_star=h_star, passed=g2),
               g3=dict(f4=f4, f12=f12, passed=g3),
               gate=gate, verdict=verdict, n_seqs=N,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "m1b_audio_mel.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/m1b_audio_mel.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
