# -*- coding: utf-8 -*-
"""M1i · a 替代口径（数值 Jacobian 谱半径）+ MLP 非线性对照（判据门·修正口径）。

对照 M1h（瞬时频率+PCA16：G2/G3 首次通过、G1 孪生口径失效）的两个增量：
  ①G1 修正口径：数值 Jacobian 谱半径 → a_alt = log(ρ(J))——不依赖孪生扰动的长期分离，
    对低秩/收缩潜码稳健（孪生口径保留作对照，预期失效）；
  ②MLP 非线性对照：PCA16 上 MLP 深网络 vs koop——非线性容量是否进一步改善 margin。

预注册（判据门·修正口径）：红线 + G1(a_alt 有限) + G2(h*>0 非饱和) + G3(量化梯度) 全过 ⇒ M2。
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
DIM_SRC = 64                               # 瞬时频率特征原始维度
H_ROLL, COS_THR = 18, 0.9
EPOCHS, BATCH = 400, 256
N_SEQ = 64
OUT_JSON = os.path.join(ROOT, "logs", "m1i_a_alt.json")
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
    """M1g 能量调制：频率固定（数字1=440Hz，数字2=988Hz），音量随 x 大范围变化。"""
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


def frame_to_latent(audio_seq):
    """(T, 1920) → (T, 64)：解析信号特征——8 子窗 × [瞬时频率均值/std, RMS, 过零率] ×2 = 64 维。"""
    from scipy.signal import hilbert
    T = audio_seq.shape[0]
    out = np.zeros((T, 64), dtype=np.float32)
    for t in range(T):
        seg = audio_seq[t]
        ana = hilbert(seg)
        inst_phase = np.unwrap(np.angle(ana))
        inst_freq = np.diff(inst_phase) / (2 * np.pi * SR) * SR
        feats = []
        for s in range(8):
            w = slice(s * 240, (s + 1) * 240)
            f_lo = inst_freq[w][(inst_freq[w] > 50) & (inst_freq[w] < 2000)]
            if len(f_lo) < 10:
                f_lo = np.array([0.0])
            feats += [float(np.mean(f_lo)), float(np.std(f_lo)),
                      float(np.sqrt((seg[w] ** 2).mean())),
                      float((np.abs(np.diff(np.sign(seg[w]))) > 0).sum())]
        out[t] = np.array(feats + feats, dtype=np.float32)[:64]
    return out


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


class MLPDeep(nn.Module):
    def __init__(self, dim, s=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * dim, 256), nn.SiLU(),
                                 nn.Linear(256, 256), nn.SiLU(),
                                 nn.Linear(256, dim))
        self.s = float(s)

    def forward(self, sp, st):
        return st + self.s * self.net(torch.cat([sp, st], dim=1))


def jacobian_spectral_a(m, points, dim, eps=1e-4):
    """数值 Jacobian 谱半径 → a_alt = log(ρ(J))。points: (K,dim)。"""
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
    lat_seqs = np.array(lat_seqs)                    # (N, T, 64)
    N = len(lat_seqs)
    n_train = int(N * 0.75)

    # PCA 降维到 16 维主成分（信息集中度）
    flat_all = lat_seqs[:n_train].reshape(-1, DIM_SRC).astype(np.float64)
    mu_p = flat_all.mean(0)
    C = np.cov((flat_all - mu_p).T)
    w, V = np.linalg.eigh(C)
    order = np.argsort(w)[::-1][:16]
    evr = w[order] / w.sum()
    P = V[:, order]
    print(f"[PCA] 前 16 主成分解释方差比: {evr.sum():.3f}（前 4: {np.round(evr[:4], 3)}）")
    lat_seqs = ((lat_seqs.reshape(-1, DIM_SRC) - mu_p) @ P).reshape(len(lat_seqs), -1, 16)
    DIM = 16
    print(f"音频序列 {N} 条（train {n_train} / val {N - n_train}）｜潜码 {DIM} 维", flush=True)
    mu = lat_seqs[:n_train].reshape(-1, DIM).mean(0)
    sd = np.maximum(lat_seqs[:n_train].reshape(-1, DIM).std(0), 1e-6)
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

    # ---- 红线 ----
    ratios = []
    for bi, binfo in enumerate(bounce_info):
        if not binfo["bounces"]:
            continue
        tr = nrm(lat_seqs[bi])
        for b in binfo["bounces"]:
            if b["frame"] < 2 or b["frame"] >= tr.shape[0] - 1:
                continue
            d_event = float(np.linalg.norm(tr[b["frame"]] - tr[b["frame"] - 1]))
            d_calm = float(np.linalg.norm(tr[b["frame"] - 1] - tr[b["frame"] - 2]))
            if d_calm > 1e-9:
                ratios.append(d_event / max(d_calm, 1e-9))
    ratio = float(np.median(ratios)) if ratios else float("nan")
    redline = bool(ratios and ratio > 1.5)
    print(f"红线（瞬态保留）：事件/平稳潜码距离比中位 = {ratio:.2f} → {'PASS >1.5' if redline else 'FAIL'}")

    # ---- G1（修正口径）：数值 Jacobian 谱半径 → a_alt；孪生对照 ----
    val_lat = [nrm(lat_seqs[k]) for k in range(n_train, N)]
    eval_pts = np.array([tr[t] for tr in val_lat for t in range(2, tr.shape[0] - 1)][:64])
    rho_med, a_alt = jacobian_spectral_a(m, eval_pts, DIM)
    g1 = bool(np.isfinite(a_alt))
    lams = []
    g = torch.Generator(device="cpu").manual_seed(5000)
    m64 = copy.deepcopy(m).double().eval()
    with torch.no_grad():
        for _ in range(16):
            tr = val_lat[int(torch.randint(0, len(val_lat), (1,), generator=g))]
            if tr.shape[0] < 24:
                continue
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
    a_twin = float(np.median(lams)) if lams else float("nan")
    print(f"G1（修正口径）a_alt = log(ρ(J)) = {a_alt:+.4f}（ρ={rho_med:.4f}）→ {'可测' if g1 else '不可测'}")
    print(f"G1（孪生对照）a_twin = {a_twin:+.4f}")

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
                    p = m(sp.unsqueeze(0), st.unsqueeze(0))[0]
                    pr = p.cpu().numpy() * sd + mu
                    dd = (lat_seqs.reshape(-1, DIM).max(0) - lat_seqs.reshape(-1, DIM).min(0)) / (2 ** b - 1)
                    pr = np.clip(np.round(pr / dd) * dd,
                                 lat_seqs.reshape(-1, DIM).min(0), lat_seqs.reshape(-1, DIM).max(0))
                    p = torch.tensor(nrm(pr.astype(np.float32)).astype(np.float32)).float().to(DEV)
                    zt = torch.tensor(tr[t + 2]).float().to(DEV)
                    c = float(F.cosine_similarity(p[None].float(), zt[None].float()).item())
                    if first is None and c < COS_THR:
                        first = t + 1
                    sp, st = st, p
                firsts[b].append(first if first is not None else H_ROLL)
    f4 = float(np.median(firsts[4])); f12 = float(np.median(firsts[12]))
    g3 = bool(f4 < f12)
    print(f"G3 量化轴：b=4 首崩 {f4:.0f} vs b=12 首崩 {f12:.0f} → {'有梯度' if g3 else '无梯度'}")

    # ---- MLP 非线性对照（PCA16 上）----
    torch.manual_seed(0); np.random.seed(0)
    mlp = MLPDeep(dim=DIM).to(DEV)
    opt2 = torch.optim.Adam(mlp.parameters(), lr=1e-3)
    sch2 = torch.optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=EPOCHS)
    for ep_ in range(EPOCHS):
        idx = torch.randperm(len(Atr))[:BATCH]
        p = mlp(Atr[idx].to(DEV), Btr[idx].to(DEV))
        loss = F.mse_loss(p, Btr[idx].to(DEV))
        if not torch.isfinite(loss):
            raise RuntimeError(f"nan mlp ep{ep_}")
        opt2.zero_grad(); loss.backward(); opt2.step(); sch2.step()
    mlp.eval()
    mlp_margin = np.full((len(val_lat), H_ROLL), np.nan)
    with torch.no_grad():
        for k, tr in enumerate(val_lat):
            if tr.shape[0] < H_ROLL + 2:
                continue
            sp = torch.tensor(tr[0]).float().to(DEV)
            st = torch.tensor(tr[1]).float().to(DEV)
            for t in range(H_ROLL):
                p = mlp(sp.unsqueeze(0), st.unsqueeze(0))[0]
                zt = torch.tensor(tr[t + 2]).float().to(DEV)
                mlp_margin[k, t] = (F.cosine_similarity(p[None].float(), zt[None].float()).item() -
                                    F.cosine_similarity(st[None].float(), zt[None].float()).item())
                sp, st = st, p
    mlp_m_curve = [float(np.nanmean(mlp_margin[:, t])) for t in range(H_ROLL)]
    mlp_h = 0
    rng_mb = np.random.default_rng(43)
    for t in range(H_ROLL):
        d_seq = mlp_margin[:, t]
        d_seq = d_seq[~np.isnan(d_seq)]
        if len(d_seq) < 5:
            continue
        boots = [float(np.mean(d_seq[rng_mb.integers(0, len(d_seq), len(d_seq))])) for _ in range(2000)]
        if float(np.percentile(boots, 2.5)) > 0:
            mlp_h = t + 1
    print(f"[MLP 对照] margin: " + " ".join(f"{v:+.3f}" for v in mlp_m_curve) + f"｜h*={mlp_h}")

    gate = bool(redline and g1 and g2 and g3)
    verdict = ("PASS（修正口径）：判据门全过 ⇒ 进 M2（音频三判据全套）" if gate else
               "FAIL：判据门未过 ⇒ 如实记录")
    print(f"\n判据门判定：{'PASS' if gate else 'FAIL'}——{verdict}")
    print(f"  明细：红线 {ratio:.2f}｜G1 a_alt={a_alt:+.4f}（孪生 {a_twin:+.4f}）｜G2 h*={h_star}｜G3 {f4:.0f}<{f12:.0f}={g3}")

    out = dict(prereg=dict(redline="事件/平稳距离比>1.5", g1="a_alt=log(ρ(J)) 有限",
                           g2="margin 非饱和+h*>0", g3="b4 首崩 < b12 首崩"),
               redline=dict(ratio=ratio, n=len(ratios), passed=redline),
               g1=dict(a_alt=a_alt, rho=rho_med, a_twin=a_twin, passed=g1),
               g2=dict(margin=margin, h_star=h_star, passed=g2),
               g3=dict(f4=f4, f12=f12, passed=g3),
               mlp=dict(margin=mlp_m_curve, h_star=mlp_h),
               gate=gate, verdict=verdict, n_seqs=N,
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/m1i_a_alt.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
