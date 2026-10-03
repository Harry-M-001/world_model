# -*- coding: utf-8 -*-
"""M1.3 · 深化验证：多参数调制音频 + 非线性动力学对照（判据门第五次，本质可行性判定）。

改动（对照 M1.2b）：
  ①音频设计多参数调制：基频=x 位置、谐波亮度=y 位置（高谐波权重）、颤音速率=|速度|
    ——音频帧间变化携带 x/y/速度三个状态量（信息量×3）；
  ②动力学双臂：koop 线性核（对照）vs MLP 深网络（256×2 非线性）——区分容量问题 vs 信号问题。

预注册判定：
  D1 若 MLP margin>0 且 koop≈0 ⇒ 容量问题是主因 ⇒ 音频判据可行，M2 开工；
  D2 若两臂都 ≈0 ⇒ 信号信息不足是主因 ⇒ 音频设计需根本重想（如实记录，M 线再评估）；
  D3 若两臂都负 ⇒ 信号反预测性（如实记录）。
  红线/量化轴同前。
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
SPF = int(SR / FPS)
N_MEL, DIM = 64, 64
H_ROLL, COS_THR = 18, 0.9
EPOCHS, BATCH = 400, 256
N_SEQ = 64
OUT_JSON = os.path.join(ROOT, "logs", "m1e_deep.json")
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


def synth_audio_multi(raw, bounces):
    """多参数调制：x→基频、y→亮度（高谐波权重）、|速度|→颤音。谐波 8 个。"""
    T = raw.shape[0]
    audio = np.zeros((T, SPF), dtype=np.float64)
    cents = [detect(raw[t]) for t in range(T)]
    prev = None
    for t in range(T):
        c = cents[t]
        seg = np.zeros(SPF)
        tt = np.arange(SPF) / SR
        base_f = (150, 500)
        rng_f = (300, 400)
        for di in range(2):
            if len(c) <= di:
                continue
            x, y = c[di]["cx"], c[di]["cy"]
            f0 = base_f[di] + (x / 64.0) * rng_f[di]
            bright = 0.3 + 0.7 * (y / 64.0)              # y 越大音色越亮
            vib = 0.0
            if prev is not None and len(prev) > di:
                speed = np.hypot(c[di]["cx"] - prev[di]["cx"], c[di]["cy"] - prev[di]["cy"])
                vib = min(speed * 1.5, 6.0)              # 颤音速率 Hz
            for k in range(1, 9):
                fk = f0 * k
                if fk > SR / 2 - 100:
                    break
                amp = (1.0 / k) * (1 - bright) ** max(0, k - 2) if k > 1 else 1.0
                vib_mod = 1 + 0.05 * np.sin(2 * np.pi * vib * tt) if vib > 0.1 else 1.0
                seg += amp * np.sin(2 * np.pi * fk * vib_mod * tt) * 0.3
        prev = c
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


def build_filterbank_full():
    fb = np.zeros((N_MEL, SPF // 2 + 1), dtype=np.float32)
    freqs = np.fft.rfftfreq(SPF, 1 / SR)
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
    T = audio_seq.shape[0]
    out = np.zeros((T, N_MEL), dtype=np.float32)
    for t in range(T):
        spec = np.abs(np.fft.rfft(audio_seq[t])) ** 2
        out[t] = np.log1p(FB_FULL @ spec)
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


class MLPDeep(nn.Module):
    """非线性动力学：256×2 深网络。"""

    def __init__(self, dim=DIM, s=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, dim))
        self.s = float(s)

    def forward(self, sp, st):
        return st + self.s * self.net(torch.cat([sp, st], dim=1))


def main():
    ds = MovingMNIST(n=N_SEQ, seed=999, train=False)
    lat, bounces_all = [], []
    for i in range(N_SEQ):
        raw = ds[i].numpy()
        cents = [detect(raw[t]) for t in range(raw.shape[0])]
        bounces = []
        for di in range(2):
            xs = [c[di]["cx"] for c in cents if len(c) > di]
            if len(xs) < 3:
                continue
            for t in range(1, len(xs) - 1):
                if (xs[t] - xs[t - 1]) * (xs[t + 1] - xs[t]) < 0 and abs(xs[t]) > 8:
                    bounces.append(dict(digit=di, frame=t))
                    break
        audio = synth_audio_multi(raw, bounces)
        lat.append(frame_to_latent(audio))
        bounces_all.append(bounces)
    lat = np.array(lat)
    N = len(lat)
    n_train = int(N * 0.75)
    flat = lat.reshape(-1, DIM)
    print(f"[诊断] mel 潜码逐维 std: min {flat.std(0).min():.4f} max {flat.std(0).max():.4f}", flush=True)

    mu = lat[:n_train].reshape(-1, DIM).mean(0)
    sd = np.maximum(lat[:n_train].reshape(-1, DIM).std(0), 1e-6)
    nrm = lambda x: (x - mu) / sd

    # 红线
    ratios = []
    for k in range(n_train, N):
        tr = nrm(lat[k])
        for b in bounces_all[k]:
            if b["frame"] < 2 or b["frame"] >= tr.shape[0] - 1:
                continue
            d_e = float(np.linalg.norm(tr[b["frame"]] - tr[b["frame"] - 1]))
            d_c = float(np.linalg.norm(tr[b["frame"] - 1] - tr[b["frame"] - 2]))
            if d_c > 1e-9:
                ratios.append(d_e / max(d_c, 1e-9))
    ratio = float(np.median(ratios)) if ratios else float("nan")
    redline = bool(ratios and ratio > 1.5)
    print(f"红线：{ratio:.2f} → {'PASS' if redline else 'FAIL'}")

    Atr, Btr = [], []
    for k in range(n_train):
        tr = nrm(lat[k])
        for t in range(1, tr.shape[0]):
            Atr.append(tr[t - 1]); Btr.append(tr[t])
    Atr = torch.tensor(np.array(Atr), dtype=torch.float32)
    Btr = torch.tensor(np.array(Btr), dtype=torch.float32)

    results = {}
    for nm, mk in (("koop", Koop), ("mlp_deep", MLPDeep)):
        torch.manual_seed(0); np.random.seed(0)
        m = mk().to(DEV)
        opt = torch.optim.Adam(m.parameters(), lr=1e-3)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
        for ep_ in range(EPOCHS):
            idx = torch.randperm(len(Atr))[:BATCH]
            p = m(Atr[idx].to(DEV), Btr[idx].to(DEV))
            loss = F.mse_loss(p, Btr[idx].to(DEV))
            if not torch.isfinite(loss):
                raise RuntimeError(f"nan {nm} ep{ep_}")
            opt.zero_grad(); loss.backward(); opt.step(); sch.step()
        m.eval()

        val_lat = [nrm(lat[k]) for k in range(n_train, N)]
        # G1 a
        lams = []
        g = torch.Generator(device="cpu").manual_seed(5000)
        m64 = copy.deepcopy(m).double().eval()
        with torch.no_grad():
            for _ in range(16):
                tr = val_lat[int(torch.randint(0, len(val_lat), (1,), generator=g))]
                t0i = 4
                za = torch.tensor(tr[t0i - 1]).double().to(DEV)
                zb = torch.tensor(tr[t0i]).double().to(DEV)
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

        # G2 margin
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
        non_sat = not (np.allclose(margin, 1.0) or np.allclose(margin, 0.0))
        g2 = bool(non_sat and h_star > 0)
        results[nm] = dict(a_mean=a_mean, a_se=a_se, margin=margin, h_star=h_star,
                           g1=g1, g2=g2)
        print(f"[{nm}] a={a_mean:+.4f}±{a_se:.4f}｜margin: " +
              " ".join(f"{v:+.3f}" for v in margin) + f"｜h*={h_star} G2={g2}", flush=True)
        del m
        torch.cuda.empty_cache()

    # G3（用 mlp_deep）
    torch.manual_seed(0); np.random.seed(0)
    m = MLPDeep().to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(len(Atr))[:BATCH]
        p = m(Atr[idx].to(DEV), Btr[idx].to(DEV))
        loss = F.mse_loss(p, Btr[idx].to(DEV))
        opt.zero_grad(); loss.backward(); opt.step(); sch.step()
    m.eval()
    firsts = {}
    with torch.no_grad():
        for b in (4, 12):
            firsts[b] = []
            for tr in [nrm(lat[k]) for k in range(n_train, N)][:12]:
                if tr.shape[0] < H_ROLL + 2:
                    continue
                sp = torch.tensor(tr[0]).float().to(DEV)
                st = torch.tensor(tr[1]).float().to(DEV)
                first = None
                for t in range(H_ROLL):
                    p = m(sp.unsqueeze(0), st.unsqueeze(0))[0]
                    pr = p.cpu().numpy()
                    dd = (lat.reshape(-1, DIM).max(0) - lat.reshape(-1, DIM).min(0)) / (2 ** b - 1)
                    pr = np.clip(np.round(pr / dd) * dd,
                                 lat.reshape(-1, DIM).min(0), lat.reshape(-1, DIM).max(0))
                    p = torch.tensor(nrm(pr.astype(np.float32))).to(DEV)
                    zt = torch.tensor(tr[t + 2]).float().to(DEV)
                    c = float(F.cosine_similarity(p[None].float(), zt[None].float()).item())
                    if first is None and c < COS_THR:
                        first = t + 1
                    sp, st = st, p
                firsts[b].append(first if first is not None else H_ROLL)
    f4 = float(np.median(firsts[4])); f12 = float(np.median(firsts[12]))
    g3 = bool(f4 < f12)
    print(f"G3（mlp_deep）：b=4 首崩 {f4:.0f} vs b=12 首崩 {f12:.0f} → {'有梯度' if g3 else '无梯度'}")

    # D 判定
    mlp_m = float(np.mean(results["mlp_deep"]["margin"]))
    koop_m = float(np.mean(results["koop"]["margin"]))
    d1 = bool(mlp_m > 0 and koop_m <= 0.01)
    d2 = bool(abs(mlp_m) < 0.005 and abs(koop_m) < 0.005)
    print(f"\n判定：mlp margin={mlp_m:+.4f} vs koop {koop_m:+.4f}")
    print(f"  D1 容量问题主因（mlp>0, koop≈0）→ {'是 ⇒ 音频判据可行，M2 开工' if d1 else '否'}")
    print(f"  D2 信号信息不足（两臂≈0）→ {'是 ⇒ 音频设计需根本重想' if d2 else '否'}")

    out = dict(redline=dict(ratio=ratio, passed=redline),
               koop=results["koop"], mlp_deep=results["mlp_deep"],
               quant=dict(f4=f4, f12=f12, g3=g3),
               d1_capacity=bool(d1), d2_signal=bool(d2),
               verdict=("D1 容量问题主因 ⇒ 音频判据可行" if d1 else
                        "D2 信号信息不足" if d2 else "D3 信号反预测性（如实记录）"),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/m1e_deep.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
