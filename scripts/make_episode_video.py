#!/usr/bin/env python3
"""Make an MP4 of one VLA training episode (any domain: blender / isaac / isaac_moved).

Left: RGB with the phone label (green circle = visible, red = hidden/outside), right: depth (0.15-2 m).
Top text: frame, current skill(s), suction state, phone xy; bottom: episode + domain.
    python scripts/make_episode_video.py data/isaac/Find_and_Pick_001_isaac
    python scripts/make_episode_video.py data/isaac_moved/Find_and_Pick_001_moved_1_00 --fps 15
Output: <episode>/episode_preview.mp4 (or --out).
"""
from pathlib import Path
import argparse
import json
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla.dataset import EncodedFrames  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("episode", type=Path)
    ap.add_argument("--fps", type=float, default=None, help="default: 30 / stride (real speed)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    ep = args.episode
    if (ep / "rgb.npy").exists():
        rgb, dep = np.load(ep / "rgb.npy", mmap_mode="r"), np.load(ep / "depth.npy", mmap_mode="r")
    else:
        rgb = EncodedFrames(ep / "rgb_jpg.npy", ep / "rgb_off.npy", "rgb")
        dep = EncodedFrames(ep / "depth_png.npy", ep / "depth_off.npy", "depth")
    n = len(rgb)
    frames = np.load(ep / "frames.npy") if (ep / "frames.npy").exists() else np.arange(n)
    lab = np.load(ep / "vla_labels.npz")
    segs = json.loads((ep / "segments.json").read_text())["segments"]
    man = json.loads((ep / "manifest.json").read_text()) if (ep / "manifest.json").exists() else {}
    stride = int(frames[1] - frames[0]) if len(frames) > 1 else 1
    fps = args.fps or 30.0 / stride
    out = args.out or ep / "episode_preview.mp4"
    W, H = 336 * 2, 252
    vw = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W * 2, H * 2))
    for i in range(n):
        t = int(frames[i])
        im = cv2.cvtColor(np.asarray(rgb[i]), cv2.COLOR_RGB2BGR)
        d = np.asarray(dep[i], np.float32)
        dn = np.clip((np.nan_to_num(d, nan=2.0) - 0.15) / 1.85, 0, 1)
        dc = cv2.applyColorMap((255 * (1 - dn)).astype(np.uint8), cv2.COLORMAP_TURBO)
        u, v = lab["phone_uv"][t]
        vis = int(lab["phone_visible"][t])
        if u >= 0:
            p = (int(u * 336 / 960), int(v * 252 / 720))
            cv2.circle(im, p, 9, (0, 255, 0) if vis == 1 else (0, 0, 255), 2)
            cv2.circle(dc, p, 9, (255, 255, 255), 2)
        frame = np.hstack([im, dc])
        frame = cv2.resize(frame, (W * 2, H * 2), interpolation=cv2.INTER_LINEAR)
        skills = [s["name"] for s in segs if s["start"] <= t <= s["end"] and s["name"] != "FULL_TASK"]
        suction = "ON" if lab["suction"][t] > 0.5 else "off"
        x, y = lab["phone_pose"][t][:2]
        cv2.rectangle(frame, (0, 0), (W * 2, 34), (0, 0, 0), -1)
        cv2.putText(frame, f"frame {t:3d}  {' + '.join(skills) or '-'}  suction {suction}  phone x={x:.3f} y={y:.3f}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"{ep.name}  [{man.get('domain', '?')}]", (10, H * 2 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        vw.write(frame)
    vw.release()
    print(f"[video] {n} frames at {fps:.0f} fps -> {out}")


if __name__ == "__main__":
    main()
