"""
T2b 本体 · 块级路由（MoE）在世界模型 rollout 中的物理语义分工

规格：
  · 数据：`multiphys` 三档（匀速 / 重力 / 盘-盘弹性碰撞），**每条序列内只切换一次**
    （t_switch=10），切换方向随机（防止模型利用时间位置识别过程）
  · **标签只用于事后评测**：`labels` 不入模型输入、不进损失（唯一例外是 oracle 臂，
    且它与涌现臂**严格分开报告**）
  · 模型：把 `dyn.LatentDynamics` 的 MLP 换成 **B 个专家 + top-1 路由器**，其余同构
    （匀速外推基线 + 残差），专家总容量与稠密基线相当

评测口径：
  ① I(R;P) + **置换检验**（打乱标签）
  ② **条件互信息** I(R;P|S)（S = 分箱的瞬时速度/方向），零假设用**组内置换**
     —— 用来区分「路由分工物理过程」与「路由只是在检测运动状态」
  ③ 切换点命中率（路由是否在已知 t_switch 换专家）+ chance level
  ④ 单步潜余弦（含按过程分解）与稠密基线对照

守卫：
  G-t2b1 标签不得进入输入/损失 —— 非 oracle 臂在构造上就没有标签入口（断言 oracle 开关）
  G-t2b2 **路由必须真的变化**：全落在同一专家 ⇒ 判据退化，直接断言失败
  G-t2b3 记录潜向量缓存与产出模型的 mtime（防止静默使用旧版本文件）
  G-t2b4 专家**冷启动陷阱**：若专家末层零初始化 ⇒ 所有专家完全相同 ⇒ 路由梯度恒 0 ⇒
         路由永远学不到。本脚本改用小随机初始化，并在训练中打印**路由熵**确认它在动。
  G-t2b5 互信息估计**向量化**且必须报告「可用分箱数」——分箱太细时空箱让条件互信息
         无法估计（首版用 Python 逐样本循环，既慢又掩盖了空箱问题）。

用法：
  python t2b_route.py --arm dense     # 同容量稠密基线
  python t2b_route.py --arm moe       # 主臂（标签不可见）
  python t2b_route.py --arm oracle    # 上界臂（喂标签，分开报告）
  python t2b_route.py --damped ...    # 第二轮 (a)：数据换成三条含阻尼的过程对
                                      #（uniform/gravity/scatter × damped），
                                      # 独立缓存与 `_dmp` 输出后缀，round-1 产物不动
产物：logs/t2b_route_{tag}.json（round-1 tag={arm}[_lbX]；round-2 追加 _dmp）
"""
import argparse
import datetime
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

from vae import LATENT, CKPT, load_vae                        # noqa: E402
from multiphys import (PROCESSES, build_dataset,             # noqa: E402
                       T_FRAMES, H, W)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX = 2
N_PROC = len(PROCESSES)
PAIRS_R1 = [("uniform", "gravity"), ("uniform", "scatter"), ("gravity", "scatter")]
PAIRS_R2 = [("uniform", "damped"), ("gravity", "damped"), ("scatter", "damped")]
DAMPED = "--damped" in sys.argv            # T2b 第二轮 (a)：阻尼档（索引 3，round-1 不受影响）
assert not ("--act" in sys.argv and DAMPED), "act 轮固定用 round-1 过程对（隔离「+动作」单轴）"
ACT = "--act" in sys.argv                    # T2b 第三轮：动作条件化（round-1 过程对）
ACT_DIGITS = 2
ACT_DIM = ACT_DIGITS + 2                     # [digit one-hot(2), ux, uy]
PAIRS = PAIRS_R1 if ACT else (PAIRS_R2 if DAMPED else PAIRS_R1)
DATA_CK = os.path.join(CKPT, "t2b_route_data%s.pt"
                       % ("_act" if ACT else ("_damped" if DAMPED else "")))
GUARD_FR = 2


def mtime(p):
    return datetime.datetime.fromtimestamp(
        os.path.getmtime(p)).strftime("%Y-%m-%d %H:%M:%S")


# ================================================================ 数据
def get_data(n_train, n_val, seed=0):
    DSUF = "dmp" if DAMPED else "r1"
    key = f"n{n_train}_v{n_val}_s{seed}_{'act' if ACT else DSUF}"
    if os.path.exists(DATA_CK):
        d = torch.load(DATA_CK, map_location="cpu", weights_only=False)
        if d.get("key") == key:
            print(f"    复用缓存（{os.path.basename(DATA_CK)}  mtime {mtime(DATA_CK)}）")
            return d
        print("    缓存键不符 → 重算")
    vae = load_vae()
    vae.load_state_dict(torch.load(os.path.join(CKPT, "vae_gray_s3.pt"),
                                   map_location=DEV, weights_only=True)["state"])
    vae.eval()
    out = {"key": key}
    for tag, n, tr in (("tr", n_train, True), ("va", n_val, False)):
        t0 = time.time()
        frames, labels, metas = build_dataset(n, seed=seed, train=tr, pairs=PAIRS,
                                              verbose=False, with_actions=ACT)
        with torch.no_grad():
            x = torch.from_numpy(np.ascontiguousarray(frames)).float().div_(255.0)
            x = x.reshape(-1, H, W).unsqueeze(1)
            zs = []
            for i in range(0, x.shape[0], 512):
                zs.append(vae.encode_det(x[i:i + 512].to(DEV)).cpu())
            z = torch.cat(zs).view(n, T_FRAMES, LATENT)
        out[f"z_{tag}"] = z
        out[f"lab_{tag}"] = torch.from_numpy(labels.astype(np.int64))
        out[f"state_{tag}"] = torch.from_numpy(
            np.stack([m["states"] for m in metas]).astype(np.float32))
        out[f"tsw_{tag}"] = torch.tensor([m["t_switch"] for m in metas])
        if ACT:
            out[f"act_{tag}"] = torch.from_numpy(
                np.stack([m["actions"] for m in metas]).astype(np.float32))
        pc = {}
        for mm in metas:
            pc[(mm["proc_a"], mm["proc_b"])] = pc.get((mm["proc_a"], mm["proc_b"]), 0) + 1
        print(f"    {tag}: {n} 条 × {T_FRAMES} 帧  编码 {time.time()-t0:.0f}s  "
              f"过程对 {pc}")
    torch.save(out, DATA_CK)
    print(f"    已缓存 -> {os.path.basename(DATA_CK)}  mtime {mtime(DATA_CK)}")
    return out


# ================================================================ 模型
class Expert(nn.Module):
    def __init__(self, d_in, hidden, d_out):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, d_out))
        # G-t2b4：**不能零初始化**（否则专家完全相同 ⇒ 路由梯度恒 0）
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return self.net(x)


class MoEDynamics(nn.Module):
    def __init__(self, latent=LATENT, ctx=CTX, n_expert=8, hidden=128,
                 topk=1, oracle=False, act_dim=0):
        super().__init__()
        self.ctx, self.latent = ctx, latent
        self.n_expert, self.topk, self.oracle = n_expert, topk, oracle
        self.act_dim = act_dim
        d_in = latent * ctx + act_dim + (N_PROC if oracle else 0)
        self.router = nn.Linear(d_in, n_expert)
        self.experts = nn.ModuleList([Expert(d_in, hidden, latent)
                                      for _ in range(n_expert)])

    def forward(self, zc, proc=None, act=None):
        zt = zc[:, -1]
        base = zt + (zt - zc[:, -2])
        x = zc.flatten(1)
        if self.act_dim:
            assert act is not None, "act 模式需要动作输入"
            x = torch.cat([x, act], dim=1)
        if self.oracle:
            assert proc is not None, "oracle 臂需要标签（但**只在这一臂**用）"
            x = torch.cat([x, proc], dim=1)
        logits = self.router(x)
        outs = torch.stack([e(x) for e in self.experts], dim=1)
        if self.topk == 1:
            idx = logits.argmax(1)
            sel = outs.gather(1, idx.view(-1, 1, 1).expand(-1, 1, self.latent))
            out = base + sel.squeeze(1)
        else:
            top = logits.topk(self.topk, dim=1)
            w = top.values.softmax(1)
            sel = (outs.gather(1, top.indices.unsqueeze(-1)
                               .expand(-1, -1, self.latent)) * w.unsqueeze(-1)).sum(1)
            out = base + sel
            idx = top.indices[:, 0]
        return out, dict(route=idx, logits=logits)


class DenseDynamics(nn.Module):
    def __init__(self, latent=LATENT, ctx=CTX, hidden=362, act_dim=0):
        super().__init__()
        self.act_dim = act_dim
        self.net = nn.Sequential(nn.Linear(latent * ctx + act_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, latent))
        nn.init.normal_(self.net[-1].weight, std=1e-3)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, zc, proc=None, act=None):
        zt = zc[:, -1]
        x = zc.flatten(1)
        if self.act_dim:
            assert act is not None, "act 模式需要动作输入"
            x = torch.cat([x, act], dim=1)
        return zt + (zt - zc[:, -2]) + self.net(x), dict(route=None, logits=None)


def build_model(arm, B, hidden, topk, act_dim=0):
    if arm == "dense":
        return DenseDynamics(hidden=hidden, act_dim=act_dim)
    return MoEDynamics(n_expert=B, hidden=hidden, topk=topk,
                       oracle=(arm == "oracle"), act_dim=act_dim)


# ================================================================ 训练
def train(m, ztr, ltr, epoch, batch, seed, arm, lb, verbose=True, afeats=None):
    torch.manual_seed(seed)
    n, T = ztr.shape[0], ztr.shape[1]
    mean = ztr.mean(dim=(0, 1)).to(DEV)
    std = ztr.std(dim=(0, 1)).clamp_min(1e-6).to(DEV)
    zn = (ztr.to(DEV) - mean) / std
    lab1h = F.one_hot(ltr.to(DEV), N_PROC).float()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epoch)
    ii, tt = torch.meshgrid(torch.arange(n), torch.arange(CTX, T - 1), indexing="ij")
    idx_all = torch.stack([ii.reshape(-1), tt.reshape(-1)], dim=1)
    t0 = time.time()
    hist = []
    for ep in range(epoch):
        perm = torch.randperm(idx_all.shape[0])[:min(idx_all.shape[0], 200 * batch)]
        tot, nb = 0.0, 0
        for i0 in range(0, perm.numel(), batch):
            sel = idx_all[perm[i0:i0 + batch]]
            bi, bt = sel[:, 0], sel[:, 1]
            # 输入 = [z_{bt−1}, z_bt]，目标 z_{bt+1}，
            #   标签/动作对应转移 bt（不得使用 [bt−2, bt−1]，那是隔帧两步预测）
            zc = torch.stack([zn[bi, bt - 1], zn[bi, bt]], dim=1)
            tgt = zn[bi, bt + 1]
            proc = lab1h[bi, bt] if arm == "oracle" else None
            act_in = afeats[bi, bt].to(DEV) if afeats is not None else None
            pred, info = m(zc, proc, act_in)
            loss = F.mse_loss(pred, tgt)
            if info["route"] is not None and lb > 0:
                p = info["logits"].softmax(1)
                f = F.one_hot(info["route"], m.n_expert).float().mean(0)
                loss = loss + lb * m.n_expert * (f * p.mean(0)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0)
            opt.step()
            tot += float(loss.detach())
            nb += 1
        sch.step()
        hist.append(tot / max(nb, 1))
        if verbose and (ep % max(1, epoch // 6) == 0 or ep == epoch - 1):
            ent = None
            if arm != "dense":
                with torch.no_grad():
                    sel = idx_all[torch.randint(0, idx_all.shape[0], (4096,))]
                    bi, bt = sel[:, 0], sel[:, 1]
                    proc = lab1h[bi, bt] if arm == "oracle" else None
                    _, inf = m(torch.stack([zn[bi, bt - 1], zn[bi, bt]], dim=1), proc,
                               afeats[bi, bt].to(DEV) if afeats is not None else None)
                    c = F.one_hot(inf["route"], m.n_expert).float().mean(0)
                    ent = float(-(c * (c + 1e-12).log()).sum())
            print(f"      ep{ep:>4}  loss {hist[-1]:.4f}"
                  + (f"   路由熵 {ent:.3f}/{math.log(m.n_expert):.3f}"
                     f"（{np.exp(ent):.1f} 个等效专家）" if ent is not None else ""))
    if verbose:
        print(f"    训练完成 {time.time()-t0:.0f}s")
    return m, mean, std, hist


# ================================================================ 互信息（向量化）
def _mi_from_counts(cnt, alpha=0.5):
    J = cnt.astype(np.float64) + alpha
    p = J / J.sum()
    pa = p.sum(1, keepdims=True)
    pb = p.sum(0, keepdims=True)
    return float((p * np.log(p / (pa * pb))).sum())


def _joint(R, P, a, b):
    return np.bincount(R * b + P, minlength=a * b).reshape(a, b)


def mi_perm_test(R, P, n_perm=200, seed=0):
    rng = np.random.default_rng(seed)
    a, b = int(R.max()) + 1, int(P.max()) + 1
    real = _mi_from_counts(_joint(R, P, a, b))
    null = np.array([_mi_from_counts(_joint(R, rng.permutation(P), a, b))
                     for _ in range(n_perm)])
    return dict(mi=real, excess=real - float(null.mean()),
                null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)),
                p_value=float((null >= real).mean()), n_perm=n_perm)


def cond_mi(R, P, S, n_perm=200, seed=0, min_bin=20):
    """I(R;P|S)，组内置换零分布。返回可用分箱数与样本量（G-t2b5）。"""
    rng = np.random.default_rng(seed)
    a, b = int(R.max()) + 1, int(P.max()) + 1
    inv = np.unique(S, return_inverse=True)[1]
    N = len(R)
    grp = [np.where(inv == g)[0] for g in range(int(inv.max()) + 1)]
    grp = [g for g in grp if len(g) >= min_bin]
    used = int(sum(len(g) for g in grp))

    def cmi(Parr):
        tot = 0.0
        for g in grp:
            c = np.bincount(R[g] * b + Parr[g], minlength=a * b).reshape(a, b)
            tot += (len(g) / N) * _mi_from_counts(c)
        return tot

    real = cmi(P)
    null = []
    for _ in range(n_perm):
        Pp = np.empty_like(P)
        for g in grp:
            Pp[g] = P[g][rng.permutation(len(g))]
        null.append(cmi(Pp))
    null = np.array(null)
    return dict(cmi=real, excess=real - float(null.mean()),
                null_mean=float(null.mean()),
                null_p95=float(np.percentile(null, 95)),
                p_value=float((null >= real).mean()), n_perm=n_perm,
                n_bins_total=int(inv.max()) + 1, n_bins_used=len(grp),
                n_samples_used=used, n_samples=N)


def state_bins(state, digit, t, speed_edges, dir_bins):
    v = state[:, t, digit, 2:4]
    sp = v.norm(dim=-1)
    sb = torch.bucketize(sp, torch.tensor(speed_edges, dtype=torch.float32))
    ang = torch.atan2(v[:, 1], v[:, 0])
    db = ((ang + math.pi) / (2 * math.pi) * dir_bins).long().clamp(0, dir_bins - 1)
    return sb * dir_bins + db


def make_state_bins(state, speed_edges, dir_bins):
    per = (len(speed_edges) + 1) * dir_bins
    b = [state_bins(state, 0, t, speed_edges, dir_bins) * per
         + state_bins(state, 1, t, speed_edges, dir_bins)
         for t in range(CTX, T_FRAMES - 1)]
    return torch.stack(b, dim=1)


# ================================================================ 评测
def collect_routes(m, zva, lva, arm, afeats=None):
    mean = zva.mean(dim=(0, 1)).to(DEV)
    std = zva.std(dim=(0, 1)).clamp_min(1e-6).to(DEV)
    zn = (zva.to(DEV) - mean) / std
    n, T = zva.shape[0], zva.shape[1]
    R, P = [], []
    with torch.no_grad():
        for t in range(CTX, T - 1):
            proc = (F.one_hot(lva[:, t].to(DEV), N_PROC).float()
                    if arm == "oracle" else None)
            _, info = m(zn[:, t - 1:t + 1], proc,
                        afeats[:, t].to(DEV) if afeats is not None else None)   # 输入对齐：z_{t−1},z_t → z_{t+1}
            R.append(info["route"].cpu())
            P.append(lva[:, t])
    return torch.stack(R, dim=1), torch.stack(P, dim=1)


def switch_hit_v2(R, tsw, T, W=5, n_shift=200, seed=0, pre_len=4):
    """路由是否在已知切换点换专家 —— **带随机循环移位零分布**的版本。

    注意：初版定义（「切换后第一帧 route != 切换前众数」）会平凡通过：
      路由近似均匀时 P(不等) = 1 − 1/B = 0.875 ⇒ 随机路由也拿 0.98。
    现在：比较「切换前 pre_len 帧的众数」与「切换后窗口的众数」是否不同；
    零分布用**随机循环移位**（打散与 t_switch 的对齐，保留路由统计与熵）。
    跳过切换后第 1 帧 —— 帧 t=tsw 的输入对 [z_{tsw-1}, z_{tsw}] 由旧过程产生，
    最早能在 t=tsw+1 看出新过程。
    """
    rng = np.random.default_rng(seed)

    def stat(Rs):
        hits, n = 0, 0
        for i in range(Rs.shape[0]):
            t = int(tsw[i]) - CTX
            if t < pre_len or t + 1 + W > Rs.shape[1]:
                continue
            pm = int(torch.mode(Rs[i, t - pre_len:t]).values)
            qm = int(torch.mode(Rs[i, t + 1:t + 1 + W]).values)
            n += 1
            hits += int(pm != qm)
        return (hits / n if n else None), n

    real, n = stat(R)
    nulls = []
    for _ in range(n_shift):
        Rs = torch.stack([torch.roll(R[i], int(rng.integers(1, R.shape[1])))
                          for i in range(R.shape[0])])
        v, _ = stat(Rs)
        if v is not None:
            nulls.append(v)
    nulls = np.array(nulls)
    return dict(hit_rate=real, n=n,
                null_mean=float(nulls.mean()) if nulls.size else None,
                null_p95=float(np.percentile(nulls, 95)) if nulls.size else None,
                p_value=float((nulls >= real).mean()) if nulls.size else None,
                n_shift=n_shift)


def single_step_cos(m, zva, lva, arm, afeats=None):
    mean = zva.mean(dim=(0, 1)).to(DEV)
    std = zva.std(dim=(0, 1)).clamp_min(1e-6).to(DEV)
    zn = (zva.to(DEV) - mean) / std
    n, T = zva.shape[0], zva.shape[1]
    per = {p: [] for p in PROCESSES}
    with torch.no_grad():
        for t in range(CTX, T - 1):
            proc = (F.one_hot(lva[:, t].to(DEV), N_PROC).float()
                    if arm == "oracle" else None)
            pred, _ = m(zn[:, t - 1:t + 1], proc,
                afeats[:, t].to(DEV) if afeats is not None else None)       # 输入对齐：z_{t−1},z_t → z_{t+1}
            c = F.cosine_similarity(pred * std + mean, zva[:, t + 1].to(DEV), dim=-1)
            for k, nm in enumerate(PROCESSES):
                sel = lva[:, t] == k
                if sel.any():
                    per[nm].append(float(c[sel].mean()))
    return dict(all=float(np.mean([np.mean(v) for v in per.values() if v])),
                per_proc={k: (float(np.mean(v)) if v else None)
                          for k, v in per.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="moe", choices=["moe", "dense", "oracle"])
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=2048)
    ap.add_argument("--lb", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-perm", type=int, default=200)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--damped", action="store_true",
                    help="T2b 第二轮：用含阻尼档的过程对（独立缓存与输出后缀 _dmp）")
    ap.add_argument("--act", action="store_true",
                    help="T2b 第三轮：动作条件化（round-1 过程对，后缀 _act）")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.n_train, a.n_val, a.n_perm = 30, 600, 256, 40
    hidden = a.hidden          # 稠密臂也可指定；容量对照时用 --hidden 480（≈MoE 的 331k）
    t0 = time.time()
    print("=" * 104)
    print(f"  T2b · 块级路由（arm={a.arm}  B={a.B}  topk={a.topk}  hidden={hidden}  "
          f"epochs={a.epochs}  n_train={a.n_train}  n_val={a.n_val}）")
    print("=" * 104)
    assert not (a.arm == "dense" and a.topk != 1), "稠密臂无 topk 概念"

    D = get_data(a.n_train, a.n_val, seed=a.seed)
    ztr, ltr, zva, lva, tsw, sta = (D["z_tr"], D["lab_tr"], D["z_va"], D["lab_va"],
                                    D["tsw_va"], D["state_va"])
    afeats_tr = afeats_va = None
    if ACT:
        atr, ava = D["act_tr"], D["act_va"]            # (n, T, 3) = [digit, ux, uy]
        afeats_tr = torch.cat([F.one_hot(atr[:, :, 0].long(), ACT_DIGITS).float(),
                               atr[:, :, 1:3]], dim=-1).to(DEV)
        afeats_va = torch.cat([F.one_hot(ava[:, :, 0].long(), ACT_DIGITS).float(),
                               ava[:, :, 1:3]], dim=-1).to(DEV)
        print(f"    动作特征 {tuple(afeats_tr.shape)}（digit one-hot + ux,uy）")
    print(f"    ztr {tuple(ztr.shape)}  zva {tuple(zva.shape)}  "
          f"标签取值 {sorted(set(ltr.flatten().tolist()))}")

    m = build_model(a.arm, a.B, hidden, a.topk,
                    act_dim=ACT_DIM if ACT else 0).to(DEV)
    n_par = sum(p.numel() for p in m.parameters())
    print(f"    参数量 {n_par:,}")
    m, mean, std, hist = train(m, ztr, ltr, a.epochs, a.batch, a.seed, a.arm, a.lb,
                               afeats=afeats_tr)

    # ---- G-t2b6 动作响应度（T15 判据①）：翻转施加数字 ⇒ 预测必须改变
    act_response = None
    if ACT:
        with torch.no_grad():
            zv = (zva.to(DEV) - mean) / std
            tg = CTX + 3
            zc_g = zv[:128, tg - 1:tg + 1]
            ag = afeats_va[:128, tg].clone()
            proc_g = (F.one_hot(lva[:128, tg].to(DEV), N_PROC).float()
                      if a.arm == "oracle" else None)
            p_a, _ = m(zc_g, proc_g, ag)
            a_flip = ag.clone()
            a_flip[:, 0] = 1 - a_flip[:, 0]            # 翻转施加的数字
            p_b, _ = m(zc_g, proc_g, a_flip)
            act_response = float((p_a - p_b).norm(dim=-1).mean()
                                 / (p_a.norm(dim=-1).mean() + 1e-9))
        print(f"    动作响应度（翻转施加数字）: {act_response:.4f}（须 > 0.01）")
        assert act_response > 0.01, (
            f"模型不响应动作（响应度 {act_response:.4f}）⇒ 动作条件化失败，拒绝出结论")

    # 文件名必须含 lb：同 arm 不同 lb 会互相覆盖
    tag = (f"dense_h{hidden}" if a.arm == "dense"
           else f"{a.arm}_lb{a.lb:g}") + ("_dmp" if DAMPED else "") + ("_act" if ACT else "")
    # tag 不含 B/topk 时，扫描结果会互相覆盖并覆盖 round-1 主结果
    if (a.B, a.topk) != (8, 1):
        tag += f"_B{a.B}k{a.topk}"
    out = os.path.join(CKPT, f"t2b_route_{tag}_B{a.B}_ep{a.epochs}.pt")
    torch.save(dict(state=m.state_dict(), arm=a.arm, B=a.B, topk=a.topk,
                    hidden=hidden, mean=mean.cpu(), std=std.cpu()), out)
    print(f"    已保存 {os.path.basename(out)}  mtime {mtime(out)}")

    cos = single_step_cos(m, zva, lva, a.arm, afeats=afeats_va)
    print(f"\n    单步潜余弦 {cos['all']:.5f}   "
          + "  ".join(f"{k}={v:.4f}" for k, v in cos["per_proc"].items() if v))

    Q = dict(config=dict(arm=a.arm, B=a.B, topk=a.topk, hidden=hidden,
                         epochs=a.epochs, n_train=a.n_train, n_val=a.n_val,
                         lb=a.lb, seed=a.seed, n_perm=a.n_perm, act=ACT),
             n_params=n_par, cos=cos, act_response=act_response, loss_tail=hist[-5:],
             data_ckpt_mtime=mtime(DATA_CK), model_mtime=mtime(out),
             elapsed_s=round(time.time() - t0, 1))

    if a.arm == "dense":
        print("    稠密臂无路由 → 跳过路由分析")
    else:
        R, P = collect_routes(m, zva, lva, a.arm, afeats=afeats_va)
        keep = torch.ones_like(P, dtype=torch.bool)
        for i in range(P.shape[0]):
            base = int(tsw[i]) - CTX
            for d in range(-GUARD_FR, GUARD_FR + 2):
                j = base + d
                if 0 <= j < keep.shape[1]:
                    keep[i, j] = False
        Rn, Pn = R[keep].numpy().astype(np.int64), P[keep].numpy().astype(np.int64)
        cnt = np.bincount(Rn, minlength=a.B)
        usage = cnt / cnt.sum()
        ent = float(-(usage * np.log(usage + 1e-12)).sum())
        # ---- G-t2b2：路由必须真的变化
        assert (usage > 0).sum() > 1, (
            f"路由完全不变化（全部落在专家 {int(usage.argmax())}）⇒ 判据退化，拒绝出结论")
        print(f"\n    路由使用 {np.round(usage, 4).tolist()}")
        print(f"    路由熵 {ent:.3f}/{math.log(a.B):.3f}（{np.exp(ent):.2f} 个等效专家）"
              f"  非空专家 {(usage>0).sum()}/{a.B}  保留帧 {len(Rn)}")

        S_c = make_state_bins(sta, [1.5, 2.5], 4)[keep].numpy()
        S_f = make_state_bins(sta, [1.0, 2.0], 6)[keep].numpy()
        mi = mi_perm_test(Rn, Pn, n_perm=a.n_perm)
        c1 = cond_mi(Rn, Pn, S_c, n_perm=a.n_perm)
        c2 = cond_mi(Rn, Pn, S_f, n_perm=a.n_perm, seed=1)
        sh = switch_hit_v2(R, tsw, T_FRAMES)
        print(f"\n    ① I(R;P)          = {mi['mi']:.4f} nats  零分布 {mi['null_mean']:.4f}"
              f"  **超出 {mi['excess']:+.4f}**  p={mi['p_value']:.3f}"
              f"  {'✓ 显著' if mi['p_value'] < 0.05 else '✗ 不显著'}")
        for nm, c in (("粗分箱", c1), ("细分箱", c2)):
            print(f"    ② I(R;P|S) {nm} = {c['cmi']:.4f} nats  零分布 "
                  f"{c['null_mean']:.4f}  **超出 {c['excess']:+.4f}**  p={c['p_value']:.3f}  "
                  f"{'✓ 显著' if c['p_value'] < 0.05 else '✗ 不显著'}"
                  f"   （可用箱 {c['n_bins_used']}/{c['n_bins_total']}，覆盖样本 "
                  f"{c['n_samples_used']}/{c['n_samples']}）")
        print(f"    ③ 切换点命中率 {sh['hit_rate']:.3f} vs 移位零分布 "
              f"{sh['null_mean']:.3f}（p={sh['p_value']:.3f}）"
              f"  {'✓ 显著' if (sh['p_value'] or 1) < 0.05 else '✗ 不显著'}  n={sh['n']}")
        Q.update(route=dict(usage=usage.tolist(), entropy=ent, n_kept=int(len(Rn)),
                            mi=mi, cmi_coarse=c1, cmi_fine=c2, switch=sh))

    jp = os.path.join(ROOT, "logs", f"t2b_route_{tag}.json")
    with open(jp, "w", encoding="utf-8") as f:
        json.dump(Q, f, ensure_ascii=False, indent=2)
    print(f"\n  → logs/t2b_route_{tag}.json（{Q['elapsed_s']}s）")


if __name__ == "__main__":
    main()
