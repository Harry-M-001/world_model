# -*- coding: utf-8 -*-
"""N2' · 四架构对照 + **涌现专业化 vs 真实模态边界**的对齐度（想法 1 落地）。

动机：ERNIE 5.0 用「模态无关路由 + 涌现式专业化」，AVWM 用「人为模态专家」，两派 2026 正面对立，
无人裁决——因为真实数据没有「模态分工」的 ground truth。
**我们的受控合成域有**：视觉=运动学 / 音频=碰撞事件（带衰减）/ 触觉=瞬时接触力（无衰减）。
⇒ 量化「路由涌现的专业化簇」与「真实模态边界」的 ARI —— 这是只有我们能做的裁决实验。

三模态（同源生成 ⇒ 帧级严格对齐）：
  vis    ∈ R^8   2 个圆盘的 [x,y,vx,vy]
  aud    ∈ R^8   8 个频带能量：碰撞时按冲击强度注入，之后指数衰减（τ=3 帧）
  hap    ∈ R^4   [f1,f2,接触力] ：碰撞瞬间的法向力脉冲，**无衰减**（与音频的差别就在这里）
真实模态分工标签：token 属于 vis / aud / hap 中的哪一个（ground truth，只用于事后对齐度评估，不进训练）。

四架构（输入均为两帧 (prev,cur) 拼接）：
  ① concat      三模态拼一起 → 共享 trunk
  ② orca        公共潜状态 + 每模态读出头（Orca 式）
  ③ expert      每模态独立专家分支 + 融合（AVWM 式）
  ④ route       共享专家池 + **按 token 特征路由（不告诉它模态）**（ERNIE 式）

预注册判据：
  C1 四架构均收敛（末轮 loss < 首轮 0.5×）
  C2 路由架构的预测误差不劣于 concat（共享专家池可行）
  C3 路由涌现专业化：ARI(路由簇, 随机) > 0（即路由分配非均匀）
  C4 核心：ARI(路由簇, **真实模态**) —— 报告数值并与 0（随机）/ 1（完美）对照；
     无论高低都是结论：高 ⇒ 涌现专业化对应真实模态；低 ⇒ ERNIE 式的涌现分工是假象。
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

from multiphys import step_state, rand_state, LIM, N_DIGIT   # noqa: E402

DEV = "cuda" if torch.cuda.is_available() else "cpu"
OUT_JSON = os.path.join(ROOT, "logs", "n2_arch_probe.json")

D_VIS, D_AUD, D_HAP = 8, 8, 4
D_ALL = D_VIS + D_AUD + D_HAP          # 20
N_BAND = D_AUD
TAU_AUD = 3.0                          # 音频混响衰减（帧）
T_STEPS = 32
N_TRAIN, N_VAL = 6000, 1000
EPOCHS, BATCH = 200, 256
SEEDS = (0, 1, 2)
N_EXPERT = 8
LB_COEF = 0.01               # 负载均衡系数（Switch Transformer 常用量级）
TOK_PER_MOD = 4                        # 每模态切成 4 个 token（共 12 个）

t0 = time.time()


# ---------------------------------------------------------------- 数据
def gen_trimodal(n_seq, seed):
    """同源生成三模态；返回 (vis, aud, hap) 各 (n_seq, T, dim)。"""
    rng = np.random.default_rng(seed)
    V, A, Hp = [], [], []
    for _ in range(n_seq):
        st = rand_state(rng, N_DIGIT)                     # (2,4)
        vs, aud_st = [], np.zeros(N_BAND)
        for t in range(T_STEPS):
            prev = st.copy()
            st, nc = step_state(st, "scatter", boundary="reflect")
            vs.append(st.reshape(-1).copy())
            # 碰撞事件 → 冲击强度（相对速度）+ 位置决定的音色（频带重心）
            if nc > 0:
                rel = float(np.linalg.norm(st[:, 2:4] - prev[:, 2:4]))
                impact = float(np.clip(rel, 0.0, 6.0)) / 6.0
                cx = float(np.mean(st[:, 0])) / max(LIM, 1.0)      # 0..1：撞击点位置→音色
                band_c = 1.0 + cx * (N_BAND - 2.0)
                inj = impact * np.exp(-0.5 * ((np.arange(N_BAND) - band_c) ** 2) / 2.0)
            else:
                inj = np.zeros(N_BAND)
            aud_st = aud_st * math.exp(-1.0 / TAU_AUD) + inj       # 衰减 + 注入
            force = np.array([impact if nc > 0 else 0.0] if nc > 0 else [0.0])
            A.append(aud_st.copy())
            Hp.append(np.array([force[0], force[0] * 0.7,
                                1.0 if nc > 0 else 0.0,
                                float(np.mean(st[:, 0])) / max(LIM, 1.0)]))
        V.append(np.array(vs))
        # A/Hp 长度对齐
    A = np.array(A).reshape(n_seq, T_STEPS, N_BAND)
    Hp = np.array(Hp).reshape(n_seq, T_STEPS, D_HAP)
    return np.array(V), A, Hp


def make_pairs(V, A, Hp, k_lo=1, k_hi=11):
    Z = np.concatenate([V, A, Hp], -1)                    # (n,T,20)
    P, C, Y = [], [], []
    for t in range(k_lo, k_hi + 1):
        P.append(Z[:, t - 1]); C.append(Z[:, t]); Y.append(Z[:, t + 1])
    return (np.concatenate(P), np.concatenate(C), np.concatenate(Y))


# ---------------------------------------------------------------- 架构
class Concat(nn.Module):
    def __init__(self, d=D_ALL, h=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(2 * d, h), nn.SiLU(),
                                 nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d))

    def forward(self, xp, xc):
        return self.net(torch.cat([xp, xc], -1))


class Orca(nn.Module):
    """公共潜状态 + 每模态读出头。"""
    def __init__(self, dims=(D_VIS, D_AUD, D_HAP), z=64, h=128):
        super().__init__()
        self.dims = dims
        d = sum(dims)
        self.enc = nn.Sequential(nn.Linear(2 * d, h), nn.SiLU(), nn.Linear(h, z))
        self.heads = nn.ModuleList([nn.Sequential(nn.Linear(z, 64), nn.SiLU(), nn.Linear(64, m))
                                    for m in dims])

    def forward(self, xp, xc):
        z = self.enc(torch.cat([xp, xc], -1))
        return torch.cat([h(z) for h in self.heads], -1)


class Expert(nn.Module):
    """每模态独立专家 + 融合（AVWM 式）。"""
    def __init__(self, dims=(D_VIS, D_AUD, D_HAP), h=64):
        super().__init__()
        self.dims = dims
        self.arms = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * m, h), nn.SiLU(), nn.Linear(h, h)) for m in dims])
        self.fuse = nn.Sequential(nn.Linear(h * len(dims), 128), nn.SiLU(),
                                  nn.Linear(128, sum(dims)))

    def forward(self, xp, xc):
        feats = []
        off = 0
        for m, arm in zip(self.dims, self.arms):
            feats.append(arm(torch.cat([xp[:, off:off + m], xc[:, off:off + m]], -1)))
            off += m
        return self.fuse(torch.cat(feats, -1))


class Route(nn.Module):
    """共享专家池 + **按 token 特征路由**（不预设模态，ERNIE 式）。

    token 化：每模态切成 TOK_PER_MOD 段 → 共 12 个 token；路由只看 token 自身的特征。
    side='route_logits' 供事后算 ARI；不进损失（模态无关！）。
    """
    def __init__(self, dims=(D_VIS, D_AUD, D_HAP), tok_per=TOK_PER_MOD,
                 n_exp=N_EXPERT, d_tok=16, h=64):
        super().__init__()
        self.dims, self.tok_per, self.n_exp = dims, tok_per, n_exp
        self.slices, self.labels = [], []
        off = 0
        for mi, m in enumerate(dims):
            step = m / tok_per
            for j in range(tok_per):
                self.slices.append((off + int(round(j * step)), off + int(round((j + 1) * step))))
                self.labels.append(mi)
            off += m
        self.n_tok = len(self.slices)
        self.emb = nn.Linear(2 * (dims[0] // tok_per + 2), d_tok)     # 统一 token 宽度
        self.tok_w = max(sum(hi - lo for lo, hi in
                             [(s[0], s[1]) for s in self.slices]) // self.n_tok, 2)
        self.emb = nn.Linear(2 * self.tok_w, d_tok)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(d_tok, h), nn.SiLU(),
                                                    nn.Linear(h, d_tok)) for _ in range(n_exp)])
        self.router = nn.Linear(d_tok, n_exp)
        self.out = nn.Linear(self.n_tok * d_tok, sum(dims))

    def _tokens(self, xp, xc):
        """把两帧的每段切出来并 pad/cut 到统一宽度 tok_w。返回 (B, n_tok, 2*tok_w)。"""
        toks = []
        for lo, hi in self.slices:
            seg = torch.cat([xp[:, lo:hi], xc[:, lo:hi]], -1)
            w = seg.shape[-1]
            if w < 2 * self.tok_w:
                seg = F.pad(seg, (0, 2 * self.tok_w - w))
            elif w > 2 * self.tok_w:
                seg = seg[:, :2 * self.tok_w]
            toks.append(seg)
        return torch.stack(toks, 1)

    def forward(self, xp, xc):
        t = self._tokens(xp, xc)                       # (B, n_tok, 2*tok_w)
        e = self.emb(t)                                # (B, n_tok, d_tok)
        logits = self.router(e)                        # (B, n_tok, n_exp)
        prob = torch.softmax(logits, -1)
        idx = logits.argmax(-1)                        # top-1 硬路由（不预设模态）
        # ★ 负载均衡（Switch Transformer）：不加的话路由会坍缩到单一专家，
        #   「涌现专业化」就成了坍缩的伪信号（第一版实测 12/12 token 全走专家 7）。
        f = F.one_hot(idx, self.n_exp).float().mean(dim=(0, 1))   # 实际分配比例
        P = prob.mean(dim=(0, 1))                                 # 平均路由概率
        self.lb = float(self.n_exp) * (f * P).sum()
        out = torch.zeros_like(e)
        for k, exp in enumerate(self.experts):
            m = (idx == k)
            if m.any():
                out[m] = exp(e[m])
        self.last_idx = idx.detach()
        self.last_prob = prob.detach()
        return self.out(out.flatten(1))


def build(arch):
    return {"concat": Concat, "orca": Orca, "expert": Expert, "route": Route}[arch]()


# ---------------------------------------------------------------- ARI
def ari(labels_pred, labels_true):
    """Adjusted Rand Index（手写；sklearn 不可用）。"""
    a = np.asarray(labels_pred).ravel()
    b = np.asarray(labels_true).ravel()
    n = a.size
    if n == 0:
        return float("nan")

    def comb2(x):
        return x * (x - 1) / 2.0
    ct = {}
    for i, j in zip(a, b):
        ct[(i, j)] = ct.get((i, j), 0) + 1
    sum_ij = sum(comb2(v) for v in ct.values())
    ai = np.array([np.sum(a == x) for x in np.unique(a)], dtype=float)
    bj = np.array([np.sum(b == x) for x in np.unique(b)], dtype=float)
    sum_a, sum_b = float(np.sum(comb2(ai))), float(np.sum(comb2(bj)))
    tot = comb2(float(n))
    exp = sum_a * sum_b / tot if tot > 0 else 0.0
    mx = 0.5 * (sum_a + sum_b)
    denom = (mx - exp)
    return float((sum_ij - exp) / denom) if abs(denom) > 1e-12 else float("nan")


# ---------------------------------------------------------------- 主流程
def main():
    Vtr, Atr, Htr = gen_trimodal(N_TRAIN, seed=0)
    Vva, Ava, Hva = gen_trimodal(N_VAL, seed=777)
    Xtr_p, Xtr_c, Ytr = make_pairs(Vtr, Atr, Htr)
    Xva_p, Xva_c, Yva = make_pairs(Vva, Ava, Hva)
    mu = Xtr_c.mean(0); sd = np.maximum(np.concatenate([Xtr_p, Xtr_c]).std(0), 1e-6)
    nrm = lambda x: (x - mu) / sd

    tp = torch.tensor(nrm(Xtr_p), dtype=torch.float32)
    tc = torch.tensor(nrm(Xtr_c), dtype=torch.float32)
    ty = torch.tensor(nrm(Ytr), dtype=torch.float32)
    vp = torch.tensor(nrm(Xva_p), dtype=torch.float32)
    vc = torch.tensor(nrm(Xva_c), dtype=torch.float32)
    vy = torch.tensor(nrm(Yva), dtype=torch.float32)

    res = {}
    for arch in ("concat", "orca", "expert", "route"):
        mse_all, ari_all, ari_rand_all = [], [], []
        collapse_all, ent_all = [], []
        for sd_ in SEEDS:
            torch.manual_seed(sd_); np.random.seed(sd_)
            m = build(arch).to(DEV)
            opt = torch.optim.Adam(m.parameters(), lr=1e-3)
            sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
            N = len(tp); first = None; last = None
            for ep in range(EPOCHS):
                idx = torch.randperm(N)[:BATCH]
                p = m(tp[idx].to(DEV), tc[idx].to(DEV))
                loss = F.mse_loss(p, ty[idx].to(DEV))
                if arch == "route":
                    loss = loss + LB_COEF * m.lb          # 负载均衡：防路由坍缩
                if first is None:
                    first = float(loss)
                last = float(loss)
                opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            with torch.no_grad():
                pred = m(vp.to(DEV), vc.to(DEV)).cpu().numpy()
            mse_all.append(float(((pred - nrm(Yva)) ** 2).mean()))
            if arch == "route":
                idx = m.last_idx.cpu().numpy()                  # (B, n_tok)
                ntok = m.n_tok
                flat = idx.ravel()
                lab_true = np.tile(np.array(m.labels), idx.shape[0])
                ari_all.append(ari(flat, lab_true))
                rng = np.random.default_rng(0)
                ari_rand_all.append(ari(rng.integers(0, N_EXPERT, flat.size), lab_true))
                # ★ 坍缩守卫：专家使用的归一化熵；max 占比 > 0.9 判为坍缩（ARI 无意义）
                u = np.bincount(idx.ravel(), minlength=N_EXPERT) / idx.size
                ent = float(-(u[u > 0] * np.log(u[u > 0])).sum() / math.log(N_EXPERT))
                collapse_all.append(float(u.max()) > 0.9)
                ent_all.append(ent)
                if sd_ == SEEDS[0]:
                    # 路由画像：每个 token 的专家众数 + 与「数值动态」的对照
                    mode_exp = [int(np.bincount(idx[:, j], minlength=N_EXPERT).argmax())
                                for j in range(ntok)]
                    # 候选分组②：段语义（vis: 位置/速度/位置/速度；aud: 频带低→高；hap: 力/力/标志/位置）
                    sem = [0, 1, 0, 1, 2, 2, 2, 2, 3, 3, 4, 5][:ntok]
                    # 候选分组③：该 token 在验证集上的标准差分位（高动态 vs 低动态）
                    col_std = np.concatenate([Xva_p, Xva_c], 0).std(0)
                    seg_std = np.array([col_std[lo:hi].mean() for lo, hi in m.slices])
                    dyn = (seg_std > np.median(seg_std)).astype(int)
                    route_profile = dict(mode_expert=mode_exp, labels=m.labels,
                                         ari_vs_modality=ari(flat, lab_true),
                                         ari_vs_segment_semantics=ari(
                                             np.tile(np.array(sem), idx.shape[0]), lab_true)
                                         if False else
                                         ari(flat, np.tile(np.array(sem), idx.shape[0])),
                                         ari_vs_dynamics=ari(
                                             flat, np.tile(dyn, idx.shape[0])),
                                         modality_purity=None,
                                         per_token_expert_hist=[
                                             np.bincount(idx[:, j], minlength=N_EXPERT).tolist()
                                             for j in range(ntok)])
                    # 模态纯度：同一模态的 token 被分到同一专家的比例（主导专家占比）
                    labs = np.array(m.labels)
                    pur = {}
                    for mi, nm in zip(range(3), ("vis", "aud", "hap")):
                        js = np.where(labs == mi)[0]
                        cnt = np.sum([np.bincount(idx[:, j], minlength=N_EXPERT)
                                      for j in js], axis=0)
                        pur[nm] = float(cnt.max() / cnt.sum())
                    route_profile["modality_purity"] = pur
            del m
            torch.cuda.empty_cache()
        res[arch] = dict(mse=float(np.mean(mse_all)),
                         mse_se=float(np.std(mse_all, ddof=1) / math.sqrt(len(mse_all))),
                         loss_first=first, loss_last=last)
        if arch == "route":
            res[arch].update(ari_vs_modality=float(np.mean(ari_all)),
                             ari_se=float(np.std(ari_all, ddof=1) / math.sqrt(len(ari_all))),
                             ari_random_baseline=float(np.mean(ari_rand_all)),
                             route_profile=route_profile,
                             collapsed=bool(any(collapse_all)),
                             expert_usage_entropy=float(np.mean(ent_all)))
        print(f"  {arch:<7} MSE={res[arch]['mse']:.4f}±{res[arch]['mse_se']:.4f}"
              f"   loss {first:.4f}→{last:.4f}"
              + (f"   ARI(路由,真实模态)={res[arch]['ari_vs_modality']:+.4f}"
                 f"（随机基线 {res[arch]['ari_random_baseline']:+.4f}）"
                 if arch == "route" else ""))

    # 判据
    c1 = all(res[a]["loss_last"] < 0.5 * res[a]["loss_first"] for a in res)
    c2 = res["route"]["mse"] <= res["concat"]["mse"] * 1.05
    ar = res["route"]["ari_vs_modality"]; ar0 = res["route"]["ari_random_baseline"]
    collapsed = res["route"]["collapsed"]
    ent = res["route"]["expert_usage_entropy"]
    c3 = (not collapsed) and (ar > ar0 + 0.02)
    print(f"  C0 非坍缩守卫（专家使用熵={ent:.3f}，坍缩={collapsed}）: {not collapsed}"
          f"   —— 坍缩时 ARI 无意义（第一版 12/12 token 全走同一专家即为坍缩）")
    print("\n判据：")
    print(f"  C1 四架构均收敛: {c1}")
    print(f"  C2 路由架构不劣于 concat（×1.05 容差）: {c2}   "
          f"route {res['route']['mse']:.4f} vs concat {res['concat']['mse']:.4f}")
    print(f"  C3 路由分配非随机（ARI > 随机基线+0.02）: {c3}   {ar:+.4f} vs {ar0:+.4f}")
    print(f"  C4 核心结论：涌现专业化 vs 真实模态边界 ARI = {ar:+.4f}"
          f"（0=无对齐，1=完美对齐）")
    rp = res["route"].get("route_profile")
    if rp:
        print(f"     对照：ARI(路由,段语义) = {rp['ari_vs_segment_semantics']:+.4f}；"
              f"ARI(路由,动态高低) = {rp['ari_vs_dynamics']:+.4f}")
        print(f"     每个 token 的专家众数：{rp['mode_expert']}   （模态标签 {rp['labels']}）")
    out = dict(config=dict(D_VIS=D_VIS, D_AUD=D_AUD, D_HAP=D_HAP, T_STEPS=T_STEPS,
                           N_TRAIN=N_TRAIN, N_VAL=N_VAL, EPOCHS=EPOCHS, SEEDS=list(SEEDS),
                           N_EXPERT=N_EXPERT, TOK_PER_MOD=TOK_PER_MOD),
               results=res, c1=bool(c1), c2=bool(c2), c3=bool(c3),
               verdict=("涌现专业化**对应**真实模态边界" if ar > 0.3 else
                        ("涌现专业化**不对应**真实模态边界（ERNIE 式涌现分工是假象）"
                         if ar < 0.1 else "部分对齐，迹象级")),
               elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"  判定：{out['verdict']}")
    print(f"\n产物 → logs/n2_arch_probe.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    main()
