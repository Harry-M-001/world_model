"""
T2 · Moving MNIST 数据（合成式，完全可复现）
=====================================================================
不下载 800MB 的 moving-mnist npz，而是用 torchvision 的静态 MNIST
按固定规则合成——好处是**每个序列由「索引 → 种子」唯一确定**，
不用落盘几百 MB，任何人在任何机器上都能复现同一个样本。

规则（与经典 Moving MNIST 一致）：
  · 画布 64×64，数字保持原生 28×28（不缩放，保留 MNIST 的笔画细节）
  · 每个序列放 2 个数字，初始位置均匀随机
  · 速度每轴从 {±1,±2,±3} px/帧 均匀取；碰壁即反向（弹跳）
  · 20 帧；两数字重叠处取 max（避免相加溢出成纯白）
  · 序列内数字不重复，避免两个一样的数字叠在一起无法分辨

指数：H=64, W=64, T=20, 2 个数字、28×28、速度 ≤3 px/帧。
"""
import os

import numpy as np
import torch
from torch.utils.data import Dataset

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(ROOT, "data")
MNIST_RAW = os.path.join(DATA_DIR, "mnist_raw")

T_FRAMES, H, W = 20, 64, 64
DIGIT = 28
N_DIGIT = 2
SPEEDS = (1, 2, 3)


def load_digits(train=True):
    """静态 MNIST → 二值 uint8 (N,28,28)。第一次会从 S3 镜像下载约 10MB。"""
    from torchvision.datasets import MNIST
    os.makedirs(MNIST_RAW, exist_ok=True)
    ds = MNIST(root=MNIST_RAW, train=train, download=True)
    raw = ds.data.numpy()
    return (raw > 127).astype(np.uint8)


def make_sequence(digits, rng, n_frames=T_FRAMES, n_digit=N_DIGIT):
    """按给定 rng 合成一个 (n_frames, 64, 64) uint8 序列。"""
    frames = np.zeros((n_frames, H, W), dtype=np.uint8)
    order = rng.permutation(len(digits))[:n_digit]   # 序列内数字不重复
    for k in order:
        d = digits[k]
        x = int(rng.integers(0, W - DIGIT + 1))
        y = int(rng.integers(0, H - DIGIT + 1))
        vx = int(rng.choice(SPEEDS)) * int(rng.choice((-1, 1)))
        vy = int(rng.choice(SPEEDS)) * int(rng.choice((-1, 1)))
        for t in range(n_frames):
            patch = frames[t, y:y + DIGIT, x:x + DIGIT]
            np.maximum(patch, d, out=patch)          # 重叠取 max
            nx, ny = x + vx, y + vy
            if nx < 0 or nx > W - DIGIT:             # 碰壁反向
                vx = -vx
                nx = x + vx
            if ny < 0 or ny > H - DIGIT:
                vy = -vy
                ny = y + vy
            x, y = nx, ny
    return frames


class MovingMNIST(Dataset):
    """索引确定性的 Moving MNIST。__getitem__ 返回 (T,64,64) uint8 张量。"""

    def __init__(self, n, seed=0, train=True, digits=None):
        self.n = int(n)
        self.seed = int(seed)
        self.digits = load_digits(train) if digits is None else digits

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rng = np.random.default_rng((self.seed * 1_000_003 + int(i)) % (2 ** 32))
        return torch.from_numpy(make_sequence(self.digits, rng))


def save_val_set(n=256, seed=999, path=None):
    """把验证集落盘成 .npy，让后续（T3/T5 的误差曲线）用的是同一批样本。"""
    path = path or os.path.join(DATA_DIR, "moving_mnist_val_64x20.npy")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        return path
    digits = load_digits(train=False)
    arr = np.stack([make_sequence(
        digits, np.random.default_rng((seed * 1_000_003 + i) % (2 ** 32))
    ) for i in range(n)])
    np.save(path, arr)
    return path


if __name__ == "__main__":
    import time
    os.makedirs(DATA_DIR, exist_ok=True)
    t0 = time.time()
    tr = MovingMNIST(64, seed=0, train=True)
    x = tr[0]
    print(f"训练集样本: {tuple(x.shape)} dtype={x.dtype} "
          f"前景占比={x.float().mean():.4f}  （64 个序列 {time.time()-t0:.1f}s）")

    t0 = time.time()
    p = save_val_set(256)
    print(f"验证集落盘: {p}  {os.path.getsize(p)/1e6:.1f} MB  ({time.time()-t0:.1f}s)")

    # 确定性检查：同一索引两次取到的必须逐位相同
    a, b = tr[7].numpy(), tr[7].numpy()
    print("确定性检查:", "PASS" if np.array_equal(a, b) else "FAIL")
    c = MovingMNIST(64, seed=1, train=True)[7].numpy()
    print("不同 seed 应不同:", "PASS" if not np.array_equal(a, c) else "FAIL")

    # 预览图：4 个序列 × 前 8 帧
    from PIL import Image
    tile = 64
    scale = 4          # 论文里图宽 13.5 cm：2048 px ⇒ 约 385 dpi（规范要求 ≥300 dpi）
    canvas = Image.new("L", (tile * 8 * scale, tile * 4 * scale), 255)
    for r in range(4):
        seq = tr[r].numpy()
        for cidx in range(8):
            img = Image.fromarray(255 - seq[cidx] * 255).resize(
                (tile * scale, tile * scale), Image.NEAREST)
            canvas.paste(img, (cidx * tile * scale, r * tile * scale))
    out = os.path.join(ROOT, "figs")
    os.makedirs(out, exist_ok=True)
    fp = os.path.join(out, "t2_dataset_preview.png")
    canvas.save(fp)
    print("预览图:", fp)
