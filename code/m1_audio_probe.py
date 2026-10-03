# -*- coding: utf-8 -*-
"""M1.0 · 音频潜码零训练探针（判据门）。

管线：MM 序列 → 程序化连续音效（数字 x 位置→音高，撞墙→冲击叠加）→ 逐帧 FFT 频带能量
     （32 带 × 2 声道 = 64 维，零训练）→ 音频潜码序列 → 同构 koop 动力学 → 三判据初测。

预注册（跑前写死）：
  红线（瞬态保留）：撞墙事件窗 vs 平稳窗的音频潜码距离比 > 1.5。
  判据门（全部满足 ⇒ PASS 进 M2；任一不过 ⇒ 上 M1.1 mel 卷积）：
    G1 a 可测（有限非 nan，SE 报告）；
    G2 margin 曲线非饱和（不全 1.0/不全 0，且有视界 CI 下界>0）；
    G3 量化轴有梯度（b=4 的首崩 < b=12 的首崩）。
  音频潜码口径：每视觉帧 80ms=1920 样本/声道 → FFT 幅谱 → 32 线性频带能量 → log1p → 64 维。
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
SAMPLES_PER_FRAME = int(SR / FPS)          # 1920
N_BANDS = 32
DIM = N_BANDS * 2                          # 立体声 64 维
H_ROLL, COS_THR = 18, 0.9
EPOCHS, BATCH = 400, 256
N_SEQ = 64                                 # 音频序列数（train 48 / val 16）
SEEDS = (0, 1, 2)
OUT_JSON = os.path.join(ROOT, "logs", "m1_audio_probe.json")
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
    """程序化连续音效：数字 x 位置→音高（数字1: 200-800Hz，数字2: 800-1400Hz），
    撞墙帧叠加冲击。返回 (T, SAMPLES_PER_FRAME) float32（单声道，后续复制双声道）。"""
    T = raw.shape[0]
    cents = [detect(raw[t]) for t in range(T)]
    audio = np.zeros((T, SAMPLES_PER_FRAME), dtype=np.float64)
    tt = np.arange(SAMPLES_PER_FRAME) / SR
    # 撞墙事件检测（单数字 vx 符号翻转）
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
            f1 = 200 + (c[0]["cx"] / 64.0) * 600
            seg += 0.25 * np.sin(2 * np.pi * f1 * tt)
        if len(c) >= 2:
            f2 = 800 + (c[1]["cx"] / 64.0) * 600
            seg += 0.25 * np.sin(2 * np.pi * f2 * tt)
        for b in bounces:
            if b["frame"] == t:
                n_imp = int(SR * 0.06)
                ti = np.arange(n_imp) / SR
                rng = np.random.default_rng(1000 + b["frame"])
                imp = rng.standard_normal(n_imp) * np.exp(-ti * 60) * 0.9
                imp += np.sin(2 * np.pi * 90 * ti) * np.exp(-ti * 40) * 0.8
                seg[:n_imp] += imp / max(np.abs(imp).max(), 1e-9)
        audio[t] = seg
    return audio, bounces


def frame_to_latent(audio_seq):
    """(T, 1920) → (T, 64)：FFT 幅谱 32 线性频带能量，log1p，双声道拼接。"""
    T = audio_seq.shape[0]
    out = np.zeros((T, DIM), dtype=np.float32)
    for t in range(T):
        seg = audio_seq[t]
        spec = np.abs(np.fft.rfft(seg))[: SAMPLES_PER_FRAME // 2]
        bands = np.array_split(spec, N_BANDS)
        feats = np.array([np.sqrt((b ** 2).sum()) for b in bands])
        out[t, :N_BANDS] = np.log1p(feats)
        out[t, N_BANDS:] = np.log1p(feats)          # 单声道复制为双（对称）
    return out


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
    # 音频合成 + 潜码（全部序列）
    lat_seqs, bounce_info = [], []
    for i in range(N_SEQ):
        raw = ds[i].numpy()
        # 修正：去掉「==2 连通域」过滤（重叠帧检测合并导致 64 条只剩 4 条——采样 bug，
        # 非编码器本质问题；音频合成只需质心映射，重叠帧单音高合法）
        audio, bounces = synth_audio(raw)
        lat_seqs.append(frame_to_latent(audio))
        bounce_info.append(dict(seq=i, bounces=bounces))
    lat_seqs = np.array(lat_seqs)                    # (N, T, 64)
    N = len(lat_seqs)
    n_train = int(N * 0.75)
    print(f"音频序列 {N} 条（train {n_train} / val {N - n_train}）｜潜码 {DIM} 维", flush=True)

    mu = lat_seqs[:n_train].reshape(-1, DIM).mean(0)
    sd = np.maximum(lat_seqs[:n_train].reshape(-1, DIM).std(0), 1e-6)
    nrm = lambda x: (x - mu) / sd

    # 训练对（train 序列的相邻帧）
    Atr, Btr = [], []
    for k in range(n_train):
        tr = lat_seqs[k]
        for t in range(1, tr.shape[0]):
            Atr.append(tr[t - 1]); Btr.append(tr[t])
    Atr = torch.tensor(nrm(np.array(Atr)), dtype=torch.float32)
    Btr = torch.tensor(nrm(np.array(Btr)), dtype=torch.float32)

    torch.manual_seed(0); np.random.seed(0)
    m = Koop().to(DEV)
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
    print("音频动力学训练完成", flush=True)

    # ---- 红线：瞬态保留 ----
    ratios = []
    for bi, binfo in enumerate(bounce_info):
        if not binfo["bounces"]:
            continue
        k = bi
        tr = nrm(lat_seqs[k])
        for b in binfo["bounces"]:
            if b["frame"] < 2 or b["frame"] >= tr.shape[0] - 1:
                continue
            d_event = float(np.linalg.norm(tr[b["frame"]] - tr[b["frame"] - 1]))
            calm = tr[max(0, b["frame"] - 4):b["frame"] - 1]
            d_calm = float(np.linalg.norm(tr[b["frame"] - 1] - tr[b["frame"] - 2])) if len(calm) else 0.0
            if d_calm > 1e-9:
                ratios.append(d_event / max(d_calm, 1e-9))
    ratio = float(np.median(ratios)) if ratios else float("nan")
    redline = bool(ratios and ratio > 1.5)
    print(f"红线（瞬态保留）：事件/平稳潜码距离比中位 = {ratio:.2f} → {'PASS >1.5' if redline else 'FAIL'}")

    # ---- G1 a ----
    val_seqs = [nrm(lat_seqs[k]) for k in range(n_train, N)]
    lams = []
    g = torch.Generator(device="cpu").manual_seed(5000)
    with torch.no_grad():
        for _ in range(16):
            tr = val_seqs[int(torch.randint(0, len(val_seqs), (1,), generator=g))]
            if tr.shape[0] < 24:
                continue
            t0i = 4
            za = torch.tensor(tr[t0i - 1]).double().to(DEV)
            zb = torch.tensor(tr[t0i]).double().to(DEV)
            m64 = copy.deepcopy(m).double().eval()
            dv = torch.randn(zb.shape, generator=g).double().to(DEV)
            zb2 = zb + dv / dv.norm() * 1e-4
            sep = [1e-4]
            for t in range(30):
                p1 = m64(za.unsqueeze(0), zb.unsqueeze(0))[0]
                p2 = m64(za.unsqueeze(0), zb2.unsqueeze(0))[0]
                sep.append(float((p2 - p1).norm()))
                za, zb, zb2 = zb, p1, p2
            arr = np.array(sep)
            if np.isfinite(arr).all() and arr[8] > 1e-15 and arr[20] > 0:
                lams.append(math.log(arr[20] / arr[8]) / 12)
            del m64
    a_mean = float(np.median(lams)) if lams else float("nan")
    a_se = float(np.std(lams, ddof=1) / math.sqrt(len(lams))) if len(lams) > 1 else float("nan")
    g1 = bool(np.isfinite(a_mean))
    print(f"G1 a = {a_mean:+.4f} ± {a_se:.4f}（n={len(lams)}）→ {'可测' if g1 else '不可测'}")

    # ---- G2 margin 曲线 ----
    cos_p = np.full((len(val_seqs), H_ROLL), np.nan)
    cos_c = np.full((len(val_seqs), H_ROLL), np.nan)
    with torch.no_grad():
        for k, tr in enumerate(val_seqs):
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
    non_sat = not (np.allclose(margin, 1.0) or np.allclose(margin, 0.0))
    g2 = bool(non_sat and h_star > 0)
    print(f"G2 margin 曲线: " + " ".join(f"{v:+.3f}" for v in margin))
    print(f"   h* = {h_star}｜非饱和 = {non_sat} → {'可测' if g2 else '不可测'}")

    # ---- G3 量化轴梯度（音频潜码 η 量化）----
    firsts = {}
    with torch.no_grad():
        for b in (4, 12):
            firsts[b] = []
            for tr in val_seqs[:12]:
                if tr.shape[0] < H_ROLL + 2:
                    continue
                sp = torch.tensor(tr[0]).float().to(DEV)
                st = torch.tensor(tr[1]).float().to(DEV)
                first = None
                for t in range(H_ROLL):
                    p = m(sp.unsqueeze(0), st.unsqueeze(0))[0]
                    # 输出量化（音频潜码 per-dim range）
                    pr = p.cpu().numpy() * sd + mu
                    dd = (lat_seqs.reshape(-1, DIM).max(0) - lat_seqs.reshape(-1, DIM).min(0)) / (2 ** b - 1)
                    pr = np.clip(np.round(pr / dd) * dd,
                                 lat_seqs.reshape(-1, DIM).min(0), lat_seqs.reshape(-1, DIM).max(0))
                    p = torch.tensor(nrm(pr.astype(np.float32))).to(DEV)
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
               "FAIL：判据门未过 ⇒ 上 M1.1（mel 谱 + 轻量卷积编码器）")
    print(f"\n判据门判定：{'PASS' if gate else 'FAIL'}——{verdict}")
    print(f"  明细：红线 {ratio:.2f}｜G1 a={a_mean:+.4f}｜G2 h*={h_star} 非饱和={non_sat}｜G3 {f4:.0f}<{f12:.0f}={g3}")

    out = dict(prereg=dict(redline="事件/平稳距离比>1.5", g1="a 有限", g2="margin 非饱和+h*>0",
                           g3="b4 首崩 < b12 首崩"),
               redline=dict(ratio=ratio, n=len(ratios), passed=redline),
               g1=dict(a_mean=a_mean, a_se=a_se, passed=g1),
               g2=dict(margin=margin, h_star=h_star, passed=g2),
               g3=dict(f4=f4, f12=f12, passed=g3),
               gate=gate, verdict=verdict, n_seqs=N,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(os.path.join(ROOT, "logs", "m1_audio_probe.json"), "w",
                        encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/m1_audio_probe.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
