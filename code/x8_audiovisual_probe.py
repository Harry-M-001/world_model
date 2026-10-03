# -*- coding: utf-8 -*-
"""X8 · 音画探针：清晰 + 有声 + 正确的三合一演示。

原理：MM 域数字撞墙的物理时刻/位置完全已知（边界反射公式）。
  视频轨：原始清晰帧（256px）——撞墙帧数字反弹可见；
  音频轨：撞墙时刻合成冲击音（白噪 burst + 指数衰减 60ms），**立体声 pan = 撞点 x 位置**；
  合成：ffmpeg → MP4（12.5fps，24000Hz 立体声）。
音画对齐断言：撞墙帧号 t ⇒ 音效起始样本 = round(t / 12.5 * 24000)，容差 ±240 样本。
"""
import json
import os
import sys
import subprocess
import time

import numpy as np
from scipy import ndimage

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

from vae import MovingMNIST  # noqa: E402

FPS = 12.5
SR = 24000
VIEW = 256
OUT_DIR = os.path.join(ROOT, "data", "x8_av_probe")
OUT_MP4 = os.path.join(ROOT, "figs", "x8_audiovisual_probe.mp4")
FFMPEG = r"C:\Users\Lenovo\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg.Essentials_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.1.1-essentials_build\bin\ffmpeg.exe"
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


def find_sequences_with_bounce(n_scan=128, want=3):
    """扫描：找 want 条有明确撞墙事件（vx 符号翻转）且连通域完整的序列。"""
    ds = MovingMNIST(n=n_scan, seed=999, train=False)
    found = []
    for i in range(n_scan):
        raw = ds[i].numpy()
        cents = [detect(raw[t]) for t in range(raw.shape[0])]
        if any(len(c) != 2 for c in cents):
            continue
        for di in range(2):
            xs = [c[di]["cx"] for c in cents]
            for t in range(1, len(xs) - 1):
                if (xs[t] - xs[t - 1]) * (xs[t + 1] - xs[t]) < 0 and abs(xs[t + 1] - xs[t]) > 0.5:
                    found.append((i, raw, dict(digit=di, frame=t, x=cents[t][di]["cx"], y=cents[t][di]["cy"])))
                    break
            else:
                continue
            break
        if len(found) >= want:
            break
    return found


def impact_sound(sr=SR, dur_s=0.09, rng=None):
    """冲击音：白噪 burst × 指数衰减 + 低频正弦敲击。返回 (n,) float32。"""
    rng = rng or np.random.default_rng(0)
    n = int(sr * dur_s)
    t = np.arange(n) / sr
    noise = rng.standard_normal(n) * np.exp(-t * 60)
    thump = np.sin(2 * np.pi * 90 * t) * np.exp(-t * 40) * 0.8
    s = noise * 0.5 + thump
    return (s / max(np.abs(s).max(), 1e-9) * 0.9).astype(np.float32)


def main():
    found = find_sequences_with_bounce(want=3)
    assert len(found) >= 2, f"只找到 {len(found)} 条撞墙序列"
    audio = np.zeros((int(SR * 6), 2), dtype=np.float32)
    events = []
    gframe = 0
    os.makedirs(OUT_DIR, exist_ok=True)
    for seq_i, raw, bounce in found:
        T = raw.shape[0]
        pan = np.clip(bounce["x"] / 64.0, 0, 1)
        burst = impact_sound(rng=np.random.default_rng(seq_i))
        start_sample = int(round((gframe + bounce["frame"]) / FPS * SR))
        end = min(start_sample + len(burst), audio.shape[0])
        audio[start_sample:end, 0] += burst[:end - start_sample] * (1 - pan)
        audio[start_sample:end, 1] += burst[:end - start_sample] * pan
        events.append(dict(seq=seq_i, gframe=gframe + bounce["frame"], sample=start_sample, pan=round(pan, 2)))
        for t in range(T):
            f = (raw[t] * 255).astype(np.uint8)
            im = np.stack([f] * 3, axis=-1)
            if t == bounce["frame"]:
                im[:3, :] = 255, 60, 60; im[-3:, :] = 255, 60, 60
                im[:, :3] = 255, 60, 60; im[:, -3:] = 255, 60, 60
            from PIL import Image as PILImage
            PILImage.fromarray(im).resize((VIEW, VIEW)).save(
                os.path.join(OUT_DIR, f"frame_{gframe + t:03d}.png"))
        gframe += T
    audio = (audio / max(np.abs(audio).max(), 1e-9) * 0.9).astype(np.float32)
    wav_path = os.path.join(OUT_DIR, "audio.wav")
    import wave
    with wave.open(wav_path, "wb") as w:
        w.setnchannels(2); w.setsampwidth(2); w.setframerate(SR)
        w.writeframes((audio * 32767).astype(np.int16).tobytes())
    print(f"{len(found)} 条序列拼接｜撞墙事件: {events}")

    r = subprocess.run([FFMPEG, "-y", "-framerate", str(FPS),
                        "-i", os.path.join(OUT_DIR, "frame_%03d.png"),
                        "-i", wav_path, "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-c:a", "aac", "-b:a", "128k", "-shortest", OUT_MP4],
                       capture_output=True, text=True, timeout=300)
    assert r.returncode == 0, f"ffmpeg 失败: {r.stderr[-300:]}"
    # 音画对齐断言：每事件的音效起始样本 vs 预期帧号换算
    for ev in events:
        expected = int(round(ev["gframe"] / FPS * SR))
        assert abs(ev["sample"] - expected) <= 240, f"音画错位: {ev}"
    assert os.path.getsize(OUT_MP4) > 10_000, "MP4 过小"
    print("音画对齐断言 ✓（全部事件帧号↔样本位置一致）")
    print(f"MP4 → {OUT_MP4}（{os.path.getsize(OUT_MP4)//1024} KB，{time.time()-t0:.0f}s）")
    json.dump(dict(events=events, fps=FPS, sr=SR, mp4_kb=os.path.getsize(OUT_MP4) // 1024),
              open(os.path.join(ROOT, "logs", "x8_audiovisual_probe.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
