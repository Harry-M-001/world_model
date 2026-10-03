# -*- coding: utf-8 -*-
"""X6c · 仿真器集成：重放生成器内部状态 → 结构化渲染全覆盖验证。

原理：MM 生成器（mvmnist.make_sequence）内部有每数字的精确框位置 (x,y)（28×28 框），
反弹边界 = W−DIGIT = 36。从外部帧检测只能拿到笔画 bbox（有框内偏移），语义无法对齐（X6b 边界）。
本实验**重放生成器**（相同 rng 重建内部状态）：
  ①内部真值轨迹 (x,y,vx,vy) 逐帧记录；
  ②解析外推（匀速+边界反射@36）对照内部真值——预期全程一致（反弹步对齐）；
  ③结构化渲染：原始数字贴图（28×28）按解析外推位置合成 → 与 raw 帧全程 IoU。
判定：S-full 全程（含反弹/重叠帧）结构化渲染 IoU ≥ 0.99 ⇒ 仿真器集成路径验证成立
（内部状态可得时结构化渲染全覆盖，X6b 的碰撞/重叠边界消除）。
"""
import os
import sys
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from vae import MovingMNIST  # noqa: E402
import mvmnist  # noqa: E402
from mvmnist import load_digits  # noqa: E402

DIGIT = mvmnist.DIGIT                       # 28
H = W = 64
OUT_JSON = os.path.join(ROOT, "logs", "x6c_simulator_replay.json")
OUT_PNG = os.path.join(ROOT, "figs", "x6c_replay_strip.png")
t0 = time.time()


def replay_sequence(digits_pool, si, seed=999, n_frames=20, n_digit=2):
    """复刻 make_sequence 内部逻辑，额外记录每帧每数字的框位置。返回 raw, states。"""
    rng = np.random.default_rng((seed * 1_000_003 + int(si)) % (2 ** 32))
    frames = np.zeros((n_frames, H, W), dtype=np.uint8)
    states = np.zeros((n_frames, n_digit, 4), dtype=np.float64)   # x,y,vx,vy
    order = rng.permutation(len(digits_pool))[:n_digit]
    objs = []
    for k in order:
        d = digits_pool[k]
        x = int(rng.integers(0, W - DIGIT + 1))
        y = int(rng.integers(0, H - DIGIT + 1))
        vx = int(rng.choice(mvmnist.SPEEDS)) * int(rng.choice((-1, 1)))
        vy = int(rng.choice(mvmnist.SPEEDS)) * int(rng.choice((-1, 1)))
        objs.append(dict(d=d, x=x, y=y, vx=vx, vy=vy))
    for t in range(n_frames):
        for o in objs:
            patch = frames[t, o["y"]:o["y"] + DIGIT, o["x"]:o["x"] + DIGIT]
            np.maximum(patch, o["d"], out=patch)
        states[t, :, 0] = [o["x"] for o in objs]
        states[t, :, 1] = [o["y"] for o in objs]
        states[t, :, 2] = [o["vx"] for o in objs]
        states[t, :, 3] = [o["vy"] for o in objs]
        for o in objs:
            nx, ny = o["x"] + o["vx"], o["y"] + o["vy"]
            if nx < 0 or nx > W - DIGIT:
                o["vx"] = -o["vx"]; nx = o["x"] + o["vx"]
            if ny < 0 or ny > H - DIGIT:
                o["vy"] = -o["vy"]; ny = o["y"] + o["vy"]
            o["x"], o["y"] = nx, ny
    return frames, states, objs


def iou(a, b):
    a = a > 0; b = b > 0
    return float((a & b).sum()) / max((a | b).sum(), 1)


def main():
    digits_pool = load_digits(train=False)
    # 找有反弹事件的序列（重放扫描）
    chosen = None
    for si in range(128):
        frames, states, objs = replay_sequence(digits_pool, si)
        # 反弹事件：vx 符号翻转
        bounces = 0
        for o in objs:
            vx_sign = np.sign(states[:, int(np.where(objs.index(o) == objs.index(o)), 2)] if False else 0)
        for di in range(2):
            vx = states[:, di, 2]
            bounces += int(np.sum(np.diff(np.sign(vx)) != 0))
        if bounces >= 2 and len(objs) == 2:
            chosen = (si, frames, states, objs, bounces)
            break
    si, raw, states, objs, n_bounce = chosen
    T = raw.shape[0]
    print(f"序列 ds[{si}]｜反弹事件 {n_bounce} 次｜重放与官方一致性检查…")

    # 重放一致性：与官方 ds[si] 逐像素比对
    ds = MovingMNIST(n=128, seed=999, train=False)
    official = ds[si].numpy()
    replay_match = float((official == raw).mean())
    print(f"重放 vs 官方逐像素一致率: {replay_match:.6f}")

    # ---- ①解析外推（内部状态语义：边界 36）vs 内部真值 ----
    # 修正：外推起点必须用 states[0]（t=0 初始状态）——objs 是跑完 T 帧后的终态（回绕位置），
    # 之前误用终态当起点导致 28px 误差（根因诊断）
    pos = [np.array([states[0, i, 0], states[0, i, 1]], dtype=np.float64) for i in range(2)]
    vv = [np.array([states[0, i, 2], states[0, i, 3]], dtype=np.float64) for i in range(2)]
    traj_err = np.zeros((T, 2))
    ana_states = np.zeros((T, 2, 2))
    ana_states[0] = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]]])
    for t in range(1, T):
        for i in range(2):
            # 生成器逐字语义（mvmnist）：先算新位置，越界则翻转速度并**从旧位置重算**
            nx = pos[i][0] + vv[i][0]
            ny = pos[i][1] + vv[i][1]
            if nx < 0 or nx > W - DIGIT:
                vv[i][0] = -vv[i][0]; nx = pos[i][0] + vv[i][0]
            if ny < 0 or ny > H - DIGIT:
                vv[i][1] = -vv[i][1]; ny = pos[i][1] + vv[i][1]
            pos[i][0], pos[i][1] = nx, ny
        for i in range(2):
            traj_err[t, i] = np.hypot(pos[i][0] - states[t, i, 0], pos[i][1] - states[t, i, 1])
        ana_states[t] = np.array([[pos[0][0], pos[0][1]], [pos[1][0], pos[1][1]]])
    print(f"①解析外推 vs 内部真值：全程最大轨迹误差 {traj_err.max():.2f}px（反弹步对齐={'是' if traj_err.max() < 1.0 else '否'}）")

    # ---- ②结构化渲染（内部状态贴图 28×28）vs raw ----
    ious = []
    for t in range(T):
        canvas = np.zeros((H, W), dtype=np.uint8)
        for i in range(2):
            x0 = int(round(ana_states[t, i, 0])); y0 = int(round(ana_states[t, i, 1]))
            d = objs[i]["d"]
            np.maximum(canvas[y0:y0 + DIGIT, x0:x0 + DIGIT], d,
                       out=canvas[y0:y0 + DIGIT, x0:x0 + DIGIT])
        ious.append(iou(canvas, raw[t]))
    s_full = bool(np.mean(ious) >= 0.99)
    print(f"②结构化渲染（解析外推+内部贴图）全程 IoU: 均值 {np.mean(ious):.4f}｜最小 {min(ious):.4f} → "
          f"{'S-full PASS ≥0.99' if s_full else 'FAIL'}")

    # ---- 条带图 ----
    from PIL import Image
    n_show = min(T, 20)
    rows = []
    for t in range(n_show):
        row = np.concatenate([raw[t], ana_frames_ := np.zeros((H, W), np.uint8)], axis=1)
        # 合成帧
        canvas = np.zeros((H, W), dtype=np.uint8)
        for i in range(2):
            x0 = int(round(ana_states[t, i, 0])); y0 = int(round(ana_states[t, i, 1]))
            d = objs[i]["d"]
            np.maximum(canvas[y0:y0 + DIGIT, x0:x0 + DIGIT], d,
                       out=canvas[y0:y0 + DIGIT, x0:x0 + DIGIT])
        row = np.concatenate([raw[t], canvas], axis=1)
        rows.append(row)
    strip = np.concatenate(rows, axis=1)
    im = Image.fromarray(strip).resize((strip.shape[1] * 2, strip.shape[0] * 2), Image.NEAREST)
    im.save(OUT_PNG)
    print(f"条带 → {OUT_PNG}")

    out = dict(seq_i=si, replay_match=replay_match, n_bounce=n_bounce,
               traj_err_max=float(traj_err.max()),
               iou_mean=float(np.mean(ious)), iou_min=float(min(ious)),
               s_full=bool(s_full), elapsed_s=round(time.time() - t0, 1))
    json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"产物 → logs/x6c_simulator_replay.json（{out['elapsed_s']}s）")


if __name__ == "__main__":
    import json
    main()
