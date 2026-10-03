"""
T2 · VAE（把 Moving MNIST 的每一帧压成潜向量）
=====================================================================
结构：4 层 stride-2 卷积编码 → 线性到 (μ, logσ²) → 线性+4 层反卷积解码。
损失：BCEWithLogits（重建）+ β·KL（先验约束），β 前 2 个 epoch 线性 warmup。
用 GroupNorm 而不是 BatchNorm：batch 小的时候 BN 的统计量太吵，
而潜空间一旦被 BN 的噪声污染，后面动力学模型的单步预测误差就分不清是
「动力学学得不好」还是「VAE 潜空间本身不光滑」。

判据相关：潜空间必须**光滑**——z 在时间上接近连续，动力学才可能学。
所以除了重建，还要监控 KL 与潜维度的活跃度。

用法：
    python vae.py --epochs 40              # 训练 + 存 ckpt + 出重建对比图
    python vae.py --eval-only              # 只出图与指标
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mvmnist import MovingMNIST, H, W  # noqa: E402
from guards import G  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CKPT = os.path.join(ROOT, "ckpt")
FIGS = os.path.join(ROOT, "figs")
LOGS = os.path.join(ROOT, "logs")

LATENT = 64
N_TRAIN_SEQ = 4000
VAL_SEQ = 128


class VAE(nn.Module):
    def __init__(self, latent=LATENT, ch=32):
        super().__init__()
        self.latent = latent
        c1, c2, c3, c4 = ch, ch * 2, ch * 4, ch * 8          # 32/64/128/256
        self.enc = nn.Sequential(
            nn.Conv2d(1, c1, 4, 2, 1), nn.GroupNorm(4, c1), nn.SiLU(),      # 64→32
            nn.Conv2d(c1, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.SiLU(),     # 32→16
            nn.Conv2d(c2, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),     # 16→8
            nn.Conv2d(c3, c4, 4, 2, 1), nn.GroupNorm(8, c4), nn.SiLU(),     # 8→4
        )
        self.feat = c4 * 4 * 4
        self.fc_mu = nn.Linear(self.feat, latent)
        self.fc_lv = nn.Linear(self.feat, latent)
        self.fc_dec = nn.Linear(latent, self.feat)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(c4, c3, 4, 2, 1), nn.GroupNorm(8, c3), nn.SiLU(),   # 4→8
            nn.ConvTranspose2d(c3, c2, 4, 2, 1), nn.GroupNorm(8, c2), nn.SiLU(),   # 8→16
            nn.ConvTranspose2d(c2, c1, 4, 2, 1), nn.GroupNorm(4, c1), nn.SiLU(),   # 16→32
            nn.ConvTranspose2d(c1, 1, 4, 2, 1),                                    # 32→64
        )

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, self.feat // 16, 4, 4)
        return self.dec(h)                     # 返回 logits

    def forward(self, x):
        mu, lv = self.encode(x)
        std = torch.exp(0.5 * lv)
        z = mu + std * torch.randn_like(std)
        return self.decode(z), mu, lv, z

    @torch.no_grad()
    def encode_det(self, x):
        return self.encode(x)[0]               # 推理时用 μ（确定性编码）


def kl_per_dim(mu, lv):
    return 0.5 * (mu ** 2 + lv.exp() - lv - 1.0)     # (B,L)


def rec_term(logits, x):
    """ELBO 的重建项：对**全部像素求和**后再对 batch 取平均（单位 nats）。

    这里必须 sum 到底，不能只 sum(1)。sum(1) 只把通道维（size=1）求和，
    得到的其实是「逐像素平均 BCE」（~0.05），比 KL（~3）小两个数量级，
    于是 loss 被 KL 独占 → 后验塌缩（z 恒定、重建却「看着还行」）。
    这种 bug 不报错，只会让判据被无意义地满足。
    """
    return F.binary_cross_entropy_with_logits(logits, x, reduction="none").flatten(1).sum(1).mean()


def flatten_batch(seq):
    """(B,T,1,H,W) -> (B*T,1,H,W)"""
    return seq.flatten(0, 1)


def build_loaders(batch=32):
    tr = MovingMNIST(N_TRAIN_SEQ, seed=0, train=True)
    va = MovingMNIST(VAL_SEQ, seed=999, train=False)
    return (DataLoader(tr, batch_size=batch, shuffle=True, num_workers=0, drop_last=True),
            DataLoader(va, batch_size=batch, shuffle=False, num_workers=0))


def to_img(seq_uint8):
    """(B,T,H,W) uint8（值为 0/1 的二值墨迹）-> (B,T,1,H,W) float in [0,1]。

    注意：mvmnist 生成的是 **0/1** 的 uint8，不是 0/255。
    这里绝不能再除 255——那样会得到一张「几乎全黑」的图，
    BCE 会瞬间降到 ~0、KL 直接塌缩，而 loss 曲线看上去还很漂亮。
    """
    return seq_uint8.float().unsqueeze(2)


def guard_only(batch=32, device="cuda"):
    """只跑守卫块然后退出，不训练、不保存。

    存在的理由：G1/G2/G3 都在训练入口，而训练要 8 分钟、还会覆盖 ckpt。
    需要一个「只验证守卫」的快速入口（也能直接进 CI）。
    """
    torch.manual_seed(0)
    trl, _ = build_loaders(batch)
    model = VAE().to(device)
    xb = flatten_batch(to_img(next(iter(trl))))
    G.input_stats(xb, "VAE 输入", kind="binary", expect_shape=(None, 1, H, W))
    x = xb.to(device)
    logits = model(x)[0]
    rec = float(rec_term(logits, x).detach())
    ref = float(F.binary_cross_entropy_with_logits(logits, x, reduction="sum").detach() / x.size(0))
    G.loss_reduction(rec, ref, "重建项")
    mu, lv = model.encode(x)
    kd = kl_per_dim(mu, lv).mean(0)
    G.shape(kd, (LATENT,), "逐维平均 KL")
    # 未训练时 rec 与 kl 的量级本来就不可比（解码器还没学会输出），
    # 所以这里只打印不判定；真正的 G3 判定放在训练结束之后。
    print(f"  [G3] 未训练时的分项（量级不可比，仅供参考）: rec={rec:.2f}  "
          f"kl={float(kd.sum().detach()):.4f}")
    return G.summary()


def train(epochs=40, batch=32, lr=2e-3, beta=1.0, warm=5, device="cuda"):
    torch.manual_seed(0)
    np.random.seed(0)
    trl, val = build_loaders(batch)
    model = VAE().to(device)
    nparam = sum(p.numel() for p in model.parameters())
    # G1：数据表示守卫。这类「数据表示错了但 loss 曲线很漂亮」的 bug 必须在这里拦住
    xb = flatten_batch(to_img(next(iter(trl))))
    G.input_stats(xb, "VAE 输入", kind="binary", expect_shape=(None, 1, H, W))
    print(f"VAE 参数量 {nparam/1e6:.3f} M  latent={LATENT}  epochs={epochs} batch={batch}")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    # G2：归约口径守卫。用一个**独立重算**的参考值对拍重建项——
    # `.sum(1)` 只归约通道维这种写法必然对不上（当初正是它把 KL 变成主导项）。
    _x = flatten_batch(to_img(next(iter(trl)))).to(device)
    _logits = model(_x)[0]
    _rec = float(rec_term(_logits, _x).detach())
    _ref = float(F.binary_cross_entropy_with_logits(_logits, _x, reduction="sum").detach()
                 / _x.size(0))
    G.loss_reduction(_rec, _ref, "重建项")
    hist = []
    step = 0
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        b_used = beta * min(1.0, (ep + 1) / max(1, warm))     # β warmup
        acc = dict(rec=0.0, kl=0.0, n=0)
        for seq in trl:
            x = flatten_batch(to_img(seq)).to(device, non_blocking=True)
            logits, mu, lv, z = model(x)
            rec = rec_term(logits, x)
            kl = kl_per_dim(mu, lv).sum(1).mean()
            loss = rec + b_used * kl
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            acc["rec"] += float(rec.detach()) * x.size(0)
            acc["kl"] += float(kl.detach()) * x.size(0)
            acc["n"] += x.size(0)
            step += 1
        sched.step()
        tr_rec, tr_kl = acc["rec"] / acc["n"], acc["kl"] / acc["n"]
        if ep % 5 == 0 or ep == epochs - 1:
            m = evaluate(model, val, device, n_batches=4)
            print(f"  ep {ep:3d}  train rec {tr_rec:8.2f} kl {tr_kl:7.2f} | "
                  f"val rec {m['rec']:8.2f} kl {m['kl']:7.2f} | "
                  f"活跃潜维 {m['active']}/{m['nlat']}  zstd {m['zstd']:.3f} | "
                  f"潜交换重建 {m['rec_swap']:.1f}（越大说明 z 越被真正使用）| "
                  f"{time.time()-t0:.0f}s", flush=True)
            hist.append(dict(epoch=ep, train_rec=tr_rec, train_kl=tr_kl, **{
                k: v for k, v in m.items() if k != "kl_per_dim"}))
    os.makedirs(CKPT, exist_ok=True)
    path = os.path.join(CKPT, "vae_mvmnist.pt")
    # G3：训练结束后的分项量级才是有意义的判定 ——
    # 若 KL 比 rec 小两个数量级，说明 β 压太狠、后验要塌；反之说明 KL 项形同虚设。
    m = evaluate(model, val, device, n_batches=8)
    G.loss_parts({"rec": m["rec"], "kl": m["kl"]}, "训练结束时的分项（β=%.2f）" % beta)
    torch.save(dict(state=model.state_dict(), latent=LATENT, nparam=nparam), path)
    print(f"ckpt -> {path}  ({time.time()-t0:.0f}s, {step} steps)")
    return model, hist


@torch.no_grad()
def evaluate(model, val_loader, device="cuda", n_batches=None):
    model.eval()
    recs, kls, allmu, kld = [], [], [], []
    rec_swap = None
    for i, seq in enumerate(val_loader):
        if n_batches is not None and i >= n_batches:
            break
        x = flatten_batch(to_img(seq)).to(device)
        logits, mu, lv, z = model(x)
        recs.append(float(rec_term(logits, x)))
        kd = kl_per_dim(mu, lv)
        kls.append(float(kd.sum(1).mean()))
        kld.append(kd.mean(0).cpu())
        allmu.append(mu.cpu())
        if rec_swap is None and mu.size(0) > 1:
            # 潜空间信息量检验：把 z 在 batch 内打乱后重建。
            # 若重建质量几乎不变，说明解码器根本没用 z —— 后验塌缩了。
            perm = torch.randperm(mu.size(0), device=device)
            zs = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
            logits_s = model.decode(zs[perm])
            rec_swap = float(rec_term(logits_s, x))
    mu = torch.cat(allmu)
    # G5：诊断量形状守卫。多个 batch 的 (L,) 直接 cat 会得到 (n_batch·L,)，
    # 于是 64 维被说成 256 维。这里先断言形状，再按 batch 平均。
    G.shape(torch.stack(kld).mean(0), (model.latent,), "逐维平均 KL")
    kd = torch.stack(kld).mean(0)             # (L,) 每维平均 KL —— 最直接的「这维有没有被用」
    active = int((kd > 0.01).sum())
    return dict(rec=float(np.mean(recs)), kl=float(np.mean(kls)), active=active,
                nlat=len(kd), zstd=float(mu.std()), rec_swap=rec_swap,
                kl_per_dim=kd.tolist())


def load_vae(device="cuda"):
    path = os.path.join(CKPT, "vae_mvmnist.pt")
    d = torch.load(path, map_location=device, weights_only=True)
    model = VAE(latent=d["latent"]).to(device)
    model.load_state_dict(d["state"])
    model.eval()
    return model


@torch.no_grad()
def encode_sequences(model, n_seq, seed, train, device="cuda", batch=64, cache=None):
    """把 n_seq 个序列逐帧编码为 μ，返回 (n_seq, T, L) float32。

    用 μ（不是采样值 z）——动力学建模需要一个**确定性**的潜轨迹，
    否则每帧都带随机噪声，单步预测误差里会混进 VAE 自己的采样噪声。
    结果落盘复用：T3/T5 的误差曲线必须基于同一批潜向量。
    """
    if cache and os.path.exists(cache):
        return torch.load(cache, map_location="cpu", weights_only=True)["z"]
    ds = MovingMNIST(n_seq, seed=seed, train=train)
    zs = []
    for i in range(0, n_seq, batch):
        idx = list(range(i, min(i + batch, n_seq)))
        seq = torch.stack([ds[j] for j in idx])
        x = to_img(seq).to(device)
        z = model.encode_det(x.flatten(0, 1))
        zs.append(z.view(len(idx), -1, model.latent).float().cpu())
    z = torch.cat(zs)
    if cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        torch.save(dict(z=z, n_seq=n_seq, seed=seed, split="train" if train else "val"), cache)
    return z


# ------------------------------------------------------------------ 出图
def recon_figure(model, device="cuda", n_seq=4, n_frame=8, seed=999):
    from PIL import Image, ImageDraw
    from figstyle import pil_font
    os.makedirs(FIGS, exist_ok=True)
    ds = MovingMNIST(n_seq, seed=seed, train=False)
    with torch.no_grad():
        seq = torch.stack([ds[i] for i in range(n_seq)])            # (N,T,H,W)
        x = to_img(seq).to(device)
        logits = model(x.flatten(0, 1))[0]
        rec = torch.sigmoid(logits).view(n_seq, -1, 1, H, W).cpu()

    orig = (seq > 0).float()              # (N,T,H,W) 二值原图（数据是 0/1）
    recb = (rec > 0.5).float()[:, :, 0]   # (N,T,H,W) —— 必须挤掉通道维，
    #                                       否则 orig(4d) 与 recb(5d) 广播会在 dim1 上撞车
    acc = float((orig == recb).float().mean())
    pad, lbl = 4, 18
    # 每条序列占「原图行 + 重建行」两行；画布高度必须按 n_seq 展开，
    # 否则 4 条序列会叠印在同一个位置（只看得见最后一条）。
    rows = 2 * n_seq
    Wc = pad * (n_frame + 1) + n_frame * W
    Hc = lbl + rows * H + (rows + 1) * pad
    canvas = Image.new("L", (Wc, Hc), 30)
    dr = ImageDraw.Draw(canvas)
    dr.text((4, 2), f"每两条一组：上=原图，下=VAE 重建（latent {LATENT} 维）"
                    f"  逐像素一致率 {acc*100:.2f}%",
            fill=210, font=pil_font(13))
    for r in range(n_seq):
        y0 = lbl + pad + (2 * r) * (H + pad)
        y1 = y0 + H + pad
        for c in range(n_frame):
            cx = pad + c * (W + 1)
            canvas.paste(Image.fromarray((255 - orig[r, c].numpy() * 255).astype(np.uint8)),
                         (cx, y0))
            canvas.paste(Image.fromarray((255 - recb[r, c].numpy() * 255).astype(np.uint8)),
                         (cx, y1))
    fp = os.path.join(FIGS, "t2_recon.png")
    canvas.save(fp)
    acc = float((orig == recb).float().mean())
    return fp, acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--guard-only", action="store_true",
                    help="只跑守卫块然后退出：不训练、不保存、不覆盖 ckpt")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    if a.guard_only:
        return guard_only(batch=a.batch, device=a.device)
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    if not a.eval_only:
        model, hist = train(a.epochs, a.batch, device=a.device)
    else:
        model = load_vae(a.device)
        hist = []
    fp, acc = recon_figure(model, a.device)
    print(f"重建对比图 -> {fp}   逐像素一致率 {acc*100:.2f}%")
    with open(os.path.join(LOGS, "t2_vae.json"), "w", encoding="utf-8") as f:
        json.dump(dict(latent=LATENT, pix_acc=acc, hist=hist), f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
