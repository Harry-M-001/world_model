# -*- coding: utf-8 -*-
"""M0 · 程序化音效扩全序列：撞墙+盘-盘碰撞事件全检测 → 立体声冲击音效 → MP4 + 全事件对齐断言。"""
import json
import os
import subprocess
import sys
import time
import wave

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from vae import MovingMNIST  # noqa: E402

FPS, SR = 12.5, 24000
VIEW = 256
FFMPEG = r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
OUT_DIR = os.path.join(ROOT, "data", "m0_av_frames")
OUT_MP4 = os.path.join(ROOT, "figs", "m0_full_av.mp4")
OUT_JSON = os.path.join(ROOT, "logs", "m0_events.json")
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


def impact_sound(seed, dur_s=0.09, amp=1.0):
    rng = np.random.default_rng(seed)
    n = int(SR * dur_s)
    t = np.arange(n) / SR
    noise = rng.standard_normal(n) * np.exp(-t * 60)
    thump = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 40) * 0.8
    s = noise * 0.5 + thump
    return (s / max(np.abs(s).max(), 1e-9) * 0.9 * amp).astype(np.float32)


def main():
    ds = MovingMNIST(n=128, seed=999, train=False)
    os.makedirs(OUT_DIR, exist_ok=True)
    for old in os.listdir(OUT_DIR):
        os.remove(os.path.join(OUT_DIR, old))

    events, gframe = [], 0
    audio = np.zeros((int(SR * 8), 2), dtype=np.float32)
    for si in range(128):
        if len(events) >= 8 or gframe > 240:      # 8 事件或 3 分钟封顶
            break
        raw = ds[si].numpy()
        T = raw.shape[0]
        cents = [detect(raw[t]) for t in range(T)]
        if any(len(c) != 2 for c in cents):
            continue
        # 事件检测：撞墙（单数字 vx 符号翻转且贴边界）+ 碰撞（两质心距离骤降后弹开）
        seq_events = []
        for di in range(2):
            xs = [c[di]["cx"] for c in cents]
            for t in range(1, len(xs) - 1):
                if (xs[t] - xs[t - 1]) * (xs[t + 1] - xs[t]) < 0 and abs(xs[t] - 32) > 12:
                    seq_events.append(dict(kind="wall", digit=di, frame=t, x=xs[t], y=cents[t][di]["cy"]))
                    break
        dists = [np.hypot(c[0]["cx"] - c[1]["cx"], c[0]["cy"] - c[1]["cy"]) for c in cents]
        for t in range(1, len(dists) - 1):
            if dists[t] < 12 and dists[t - 1] >= 12:
                if not any(e["kind"] == "collide" and abs(e["frame"] - t) < 3 for e in seq_events):
                    seq_events.append(dict(kind="collide", digit=-1, frame=t,
                                           x=(cents[t][0]["cx"] + cents[t][1]["cx"]) / 2,
                                           y=(cents[t][0]["cy"] + cents[t][1]["cy"]) / 2))
        if not seq_events:
            continue
        # 音效+帧导出
        for ev in seq_events:
            pan = np.clip(ev["x"] / 64.0, 0, 1)
            burst = impact_sound(seed=si * 100 + ev["frame"], amp=0.7 if ev["kind"] == "collide" else 1.0)
            start = int(round((gframe + ev["frame"]) / FPS * SR))
            end = min(start + len(burst), audio.shape[0])
            audio[start:end, 0] += burst[:end - start] * (1 - pan)
            audio[start:end, 1] += burst[:end - start] * pan
            ev.update(gframe=gframe + ev["frame"], sample=start, pan=round(float(pan), 2))
            events.append(ev)
        for t in range(T):
            f = (raw[t] * 255).astype(np.uint8)
            im = np.stack([f] * 3, axis=-1)
            hit = [e for e in seq_events if e["frame"] == t]
            if hit:
                im[:3, :] = 255, 60, 60; im[-3:, :] = 255, 60, 60
                im[:, :3] = 255, 60, 60; im[:, -3:] = 255, 60, 60
            from PIL import Image as PILImage
            PILImage.fromarray(im).resize((VIEW, VIEW)).save(
                os.path.join(OUT_DIR, f"frame_{gframe + t:03d}.png"))
        gframe += T
    print(f"事件 {len(events)} 个（撞墙 {sum(1 for e in events if e['kind']=='wall')} / "
          f"碰撞 {sum(1 for e in events if e['kind']=='collide')}）｜总帧 {gframe}", flush=True)

    audio = (audio / max(np.abs(audio).max(), 1e-9) * 0.9).astype(np.float32)
    wav_path = os.path.join(OUT_DIR, "audio.wav")
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())

    r = subprocess.run([FFMPEG, "-y", "-framerate", str(FPS),
                        "-i", os.path.join(OUT_DIR, "frame_%03d.png"),
                        "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", OUT_MP4],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"ffmpeg 失败: {r.stderr[-300:]}"

    # 全事件对齐断言
    for ev in events:
        expected = int(round(ev["gframe"] / FPS * SR))
        assert abs(ev["sample"] - expected) <= 240, f"音画错位: {ev}"
    print(f"全事件音画对齐断言 ✓（{len(events)} 事件）")
    json.dump(dict(n_events=len(events), events=events, gframe=gframe,
                   mp4_kb=os.path.getsize(OUT_MP4) // 1024),
              open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"MP4 → {OUT_MP4}（{os.path.getsize(OUT_MP4)//1024} KB，{time.time()-t0:.0f}s）")


if __name__ == "__main__":
    main()
