"""Shared helpers (no omni / torch imports): config paths, dataset frames, IPC.

Imported by the Isaac side (Isaac Sim python 3.12) AND by the policy server
(idp3_rgb conda env, python 3.11), so keep it dependency-light.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent


def load_config(path: str | Path) -> tuple[dict, Path]:
    path = Path(path).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    return cfg, path.parent


def resolve(value: str | Path, base: Path) -> Path:
    p = Path(str(value)).expanduser()
    return (p if p.is_absolute() else base / p).resolve()


def episode_paths(dataset_dir: Path, episode: str) -> dict:
    """Find_and_Pick_001 -> {yaml, mp4, depth_dir}."""
    ep = dataset_dir / episode
    out = {"dir": ep, "yaml": ep / f"{episode}.mmrt.yaml", "mp4": ep / f"{episode}.mp4",
           "depth_dir": ep / "depth"}
    for k in ("yaml",):               # mp4/depth only needed by the dataset-comparison tools
        if not out[k].is_file():
            raise FileNotFoundError(f"Episode file missing: {out[k]}")
    return out


def converter_module(training_bundle: Path):
    """The exact point-cloud converter used to build the training set."""
    training_bundle = Path(training_bundle)
    if str(training_bundle) not in sys.path:
        sys.path.insert(0, str(training_bundle))
    import tomo_rgbd_pointcloud_review as cloud_tools  # noqa: E402
    return cloud_tools


def read_dataset_frames(mp4: Path, depth_dir: Path, frames, cloud_tools) -> dict:
    """{frame: (rgb uint8 HxWx3, depth float32 HxW metres)} decoded like training."""
    import cv2
    frames = sorted(set(int(f) for f in frames))
    want = set(frames)
    out, i = {}, 0
    cap = cv2.VideoCapture(str(mp4))
    try:
        while want:
            ok, bgr = cap.read()
            if not ok:
                break
            if i in want:
                d, _ = cloud_tools.read_depth(Path(depth_dir) / f"depth_{i:04d}.exr", "Depth.V")
                out[i] = (cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), np.asarray(d, np.float32))
                want.discard(i)
            i += 1
    finally:
        cap.release()
    if want:
        raise RuntimeError(f"Frames not found in {mp4}: {sorted(want)}")
    return out


# ---------------------------------------------------------------- IPC
# Plain dicts of bytes / str / int / float / lists only -- never numpy objects,
# so the Isaac python (numpy 2.5) and the policy env (numpy 2.4) never have to
# unpickle each other's numpy internals.

def pack(a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.tobytes()}


def unpack(d: dict) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(d["shape"]).copy()
