"""Side-by-side video: recorded DATASET demo (top) vs ISAAC run (bottom).

    ~/isaac_env/bin/python tools/make_compare_video.py --run /tmp/tomo_idp3_isaac_runs/<run>

<run> is an isaac_backend.py output folder (closed loop or --teacher-forced);
its isaac_rgbd_rollout.mp4 holds Isaac RGB | depth per step. Writes
compare_dataset_vs_isaac.mp4 and compare_contact_sheet.png into <run>.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import converter_module, episode_paths, load_config, resolve  # noqa: E402


def depth_colormap(d, lo=0.15, hi=3.0):
    v = np.isfinite(d) & (d >= lo) & (d <= hi)
    g = np.zeros(d.shape, np.uint8)
    g[v] = (np.clip((d[v] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    h = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    h[~v] = 0
    return h


def label(img, text, y=24, scale=0.65):
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 4)
    cv2.putText(img, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--sheet-frames", default="0,15,30,50,100,150,200,300")
    args = ap.parse_args()
    run = Path(args.run)
    summ = json.loads((run / "summary.json").read_text())
    episode = summ.get("episode_start") or summ.get("episode")
    mode = "REPLAY of recorded joints" if summ.get("mode") == "teacher_forced" else \
        f"POLICY closed loop (replan every {summ.get('replan_every')})"
    cfg, base = load_config(args.config)
    paths = episode_paths(resolve(cfg["isaac_mode"]["dataset_start"]["dataset_dir"], base), episode)
    tools = converter_module(resolve(cfg["idp3_policy"]["training_bundle"], base))

    ds_cap = cv2.VideoCapture(str(paths["mp4"]))
    is_cap = cv2.VideoCapture(str(run / "isaac_rgbd_rollout.mp4"))
    out = cv2.VideoWriter(str(run / "compare_dataset_vs_isaac.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                          30.0, (960, 720))
    sheet_at = {int(x) for x in args.sheet_frames.split(",")}
    sheet = []
    t = 0
    while True:
        ok_d, ds_bgr = ds_cap.read()
        ok_i, is_tile = is_cap.read()
        if not ok_i:
            break
        if not ok_d:
            ds_bgr = np.zeros((720, 960, 3), np.uint8)
            ds_depth = np.full((720, 960), np.inf, np.float32)
        else:
            exr = paths["depth_dir"] / f"depth_{t:04d}.exr"
            ds_depth = tools.read_depth(exr, "Depth.V")[0] if exr.is_file() else np.full((720, 960), np.inf, np.float32)
        top = cv2.resize(cv2.hconcat([ds_bgr, depth_colormap(ds_depth)]), (960, 360), interpolation=cv2.INTER_AREA)
        label(top, f"DATASET {episode}  t={t}")
        bot = is_tile.copy()
        cv2.rectangle(bot, (0, 0), (960, 34), (0, 0, 0), -1)
        label(bot, f"ISAAC  {mode}  t={t}")
        frame = cv2.vconcat([top, bot])
        out.write(frame)
        if t in sheet_at:
            sheet.append(cv2.resize(frame, (480, 360)))
        t += 1
    out.release()
    if sheet:
        while len(sheet) % 4:
            sheet.append(np.zeros_like(sheet[0]))
        rows = [cv2.hconcat(sheet[i:i + 4]) for i in range(0, len(sheet), 4)]
        cv2.imwrite(str(run / "compare_contact_sheet.png"), cv2.vconcat(rows))
    print(f"[compare] {t} frames -> {run / 'compare_dataset_vs_isaac.mp4'}")


if __name__ == "__main__":
    main()
