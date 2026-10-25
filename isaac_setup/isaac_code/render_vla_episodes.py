#!/usr/bin/env python3
"""Render VLA training episodes in the dataset-matched Isaac scene (headless, PhysX off).

  --mode original : the recorded demos (joints + recorded phone pose per frame)  -> data/isaac/<ep>_isaac
  --mode moved    : moved-phone IK demos from data/moved_demos.npz               -> data/isaac_moved/<key>
Phone pose of a moved demo (no physics, exact kinematics of the recorded carry):
    O(t) = P_moved_start                          t < suction_on (155)
    O(t) = H_new(t) @ inv(H_src(t)) @ P_src(t)    t >= 155, H = FK of the suction link
so the phone follows the new hand exactly like the recorded phone followed the recorded hand
(= P_src(t) for t >= 245 where the IK demo is back on the recorded arm path).
Per episode folder (compressed, ~60 KB per rendered frame):
    q.npy [T,46] full rate | frames.npy rendered frame ids | rgb_jpg.npy + rgb_off.npy (JPEG q95, 336x252)
    depth_png.npy + depth_off.npy (16-bit PNG, mm, no-hit = 8914 mm like the Blender far plane / ACT adapter)
    depth_valid.npy | vla_labels.npz + segments.json (vla/annotations.py) | manifest.json
Labels: phone TOP-FACE centre projected with the live camera; visible = inside the image and the Isaac depth
agrees within 2 cm. Moved demos render only frames < 245 (later frames are identical to the original render).
Resumable: finished episodes (manifest "complete": true) are skipped.

Run with Isaac Sim python:
  ~/isaac-sim/python.sh isaac_setup/isaac_code/render_vla_episodes.py --mode original --stride 2
  ~/isaac-sim/python.sh isaac_setup/isaac_code/render_vla_episodes.py --mode moved --per-source 5 --stride 2
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import shutil
import sys
import time
from pathlib import Path

import cv2
import numpy as np

IC = Path(__file__).resolve().parent
ROOT = IC.parents[1]
sys.path.insert(0, str(IC))
from common import episode_paths, load_config, resolve  # noqa: E402

TOP_FACE = np.array([0.0, 0.0, 0.0145, 1.0])
FAR_M = 8.914
BYTES_PER_FRAME = 70_000            # JPEG q95 + PNG16 estimate, for the disk check
MOVED_LAST_FRAME = 244


def load_annotations():
    spec = importlib.util.spec_from_file_location("vla_annotations", ROOT / "vla/annotations.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def yaw_of(M):
    return math.atan2(M[1, 0], M[0, 0])


class Writer:
    def __init__(self):
        self.rgb, self.dep, self.frames = [], [], []

    def add(self, t, rgb, depth):
        rgb_s = cv2.resize(rgb, (336, 252), interpolation=cv2.INTER_AREA)
        dep = np.where(np.isposinf(depth) | ~np.isfinite(depth), FAR_M, depth).astype(np.float32)
        dep_s = cv2.resize(dep, (336, 252), interpolation=cv2.INTER_NEAREST)
        ok1, j = cv2.imencode(".jpg", cv2.cvtColor(rgb_s, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
        ok2, p = cv2.imencode(".png", np.clip(np.rint(dep_s * 1000.0), 0, 65535).astype(np.uint16))
        if not (ok1 and ok2):
            raise RuntimeError("image encoding failed")
        self.rgb.append(j.ravel()); self.dep.append(p.ravel()); self.frames.append(int(t))

    def save(self, d: Path):
        for name, chunks in (("rgb", self.rgb), ("depth", self.dep)):
            off = np.zeros(len(chunks) + 1, np.int64)
            off[1:] = np.cumsum([len(c) for c in chunks])
            np.save(d / f"{name}_{'jpg' if name == 'rgb' else 'png'}.npy", np.concatenate(chunks).astype(np.uint8))
            np.save(d / f"{name}_off.npy", off)
        np.save(d / "frames.npy", np.asarray(self.frames, np.int64))
        np.save(d / "depth_valid.npy", np.ones(len(self.frames), bool))


def project(rt, O, depth, K, W, H, tol):
    Cinv = np.linalg.inv(rt.camera_world())
    pc = Cinv @ O @ TOP_FACE
    z = -pc[2]
    if z <= 0.05:
        return (-1.0, -1.0), 0
    u, v = K[0, 0] * pc[0] / z + K[0, 2], K[1, 2] - K[1, 1] * pc[1] / z
    if not (0 <= u < W and 0 <= v < H):
        return (u, v), 0
    d = float(depth[int(v), int(u)])
    d = FAR_M if not math.isfinite(d) else d
    return (u, v), (1 if abs(d - z) < tol else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(IC / "config.yaml"))
    ap.add_argument("--mode", choices=["original", "moved"], required=True)
    ap.add_argument("--episodes", default="", help="comma list of Find_and_Pick_XXX (default: all)")
    ap.add_argument("--moved-npz", default=str(ROOT / "data/moved_demos.npz"))
    ap.add_argument("--per-source", type=int, default=5, help="moved demos rendered per source episode")
    ap.add_argument("--stride", type=int, default=2, help="render every N-th frame (q stays full rate)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--depth-tol", type=float, default=0.02)
    args = ap.parse_args()
    A = load_annotations()
    out_root = Path(args.out) if args.out else ROOT / ("data/isaac" if args.mode == "original" else "data/isaac_moved")
    out_root.mkdir(parents=True, exist_ok=True)

    cfg, base = load_config(args.config)
    cfg["isaac_mode"]["headless"], cfg["isaac_mode"]["gui"] = True, False
    cfg.setdefault("visualization", {}).setdefault("observer", {})["enabled"] = False
    ds_dir = resolve(cfg["isaac_mode"]["dataset_start"]["dataset_dir"], base)
    names = [x.strip() for x in args.episodes.split(",") if x.strip()] or \
        sorted(p.name for p in ds_dir.glob("Find_and_Pick_*") if (p / f"{p.name}.mmrt.yaml").exists())

    jobs = []                                       # (out_name, source, demo_key or None)
    if args.mode == "original":
        jobs = [(f"{n}_isaac", n, None) for n in names]
    else:
        z = np.load(args.moved_npz)
        meta = json.loads(str(z["meta"]))
        if not all(m.get("version") == 2 for m in meta):
            raise SystemExit("moved_demos.npz is not version 2 (run data_prep/make_moved_demos.sh)")
        by_src = {}
        for m in meta:
            by_src.setdefault(f"Find_and_Pick_{int(m['source']):03d}", []).append(m["key"])
        for n in names:
            for key in sorted(by_src.get(n, []))[: args.per_source]:
                jobs.append((f"{n}_moved_{key}", n, key))
    todo = [j for j in jobs if not (out_root / j[0] / "manifest.json").exists()
            or not json.loads((out_root / j[0] / "manifest.json").read_text()).get("complete")]
    frames_per = math.ceil(307 / args.stride) if args.mode == "original" else math.ceil((MOVED_LAST_FRAME + 1) / args.stride)
    need = len(todo) * frames_per * BYTES_PER_FRAME
    free = shutil.disk_usage(out_root).free
    print(f"[render] {args.mode}: {len(jobs)} episodes, {len(todo)} to do, ~{frames_per} frames each, "
          f"~{need / 1e9:.1f} GB needed, {free / 1e9:.1f} GB free", flush=True)
    if need + 2e9 > free:
        raise SystemExit("not enough disk space (keep 2 GB spare): lower --per-source or raise --stride")
    if not todo:
        return

    from isaac_backend import IsaacBackend
    from runtime.isaac_runtime import _pose_matrix
    backend = IsaacBackend(config=cfg, config_dir=base)
    rt = backend.runtime
    K, W, H = None, 960, 720
    t_start = time.time()
    try:
        for ji, (out_name, src, key) in enumerate(todo):
            backend.load_episode(src, 0, None)
            ep = backend.episode
            T = int(ep.n_frames)
            parsed = A.parse_mmrt_full(backend.episode_paths["yaml"])
            on, off = parsed["suction_on"], parsed["suction_off"]
            if key is None:
                q = np.stack([ep.q46(t) for t in range(T)]).astype(np.float64)
                P = [None] * T
            else:
                q = z[f"q_{key}"].astype(np.float64)
                ph = z[f"phone_{key}"][0]
                P0 = _pose_matrix([ph[0], ph[1], ph[2], 0.0, 0.0, ph[3]])
                P = []
                for t in range(T):
                    if t < on:
                        P.append(P0)
                    else:
                        H_src = rt.fk_world(ep.q46(t), rt.suction_prim)
                        H_new = rt.fk_world(q[t], rt.suction_prim)
                        P.append(H_new @ np.linalg.inv(H_src) @ _pose_matrix(ep.phone_pose(t)))
            last = T - 1 if key is None else MOVED_LAST_FRAME
            wr = Writer()
            uv = np.full((T, 2), -1.0, np.float32)
            vis = np.full(T, -1, np.int8)
            phone6 = np.zeros((T, 6), np.float32)
            for t in range(T):                                    # phone pose labels for every frame
                M = _pose_matrix(ep.phone_pose(t)) if P[t] is None else P[t]
                phone6[t] = [M[0, 3], M[1, 3], M[2, 3], 0.0, 0.0, yaw_of(M)]
            t0 = time.time()
            for t in range(0, last + 1, args.stride):
                if P[t] is None:
                    rt.set_phone_pose(ep.phone_pose(t), yaw_offset_rad=backend.phone_yaw_offset)
                else:
                    rt._set_phone_world(P[t] @ rt._obj_to_prim)
                rt.drive(q[t])
                rgb, depth = backend.read_camera()
                if K is None:
                    K = rt.camera_K(); H, W = depth.shape[:2]
                O = rt.phone_object_pose()
                (u, v), vv = project(rt, O, depth, K, W, H, args.depth_tol)
                uv[t], vis[t] = (u, v), vv
                wr.add(t, rgb, depth)
            tmp = out_root / (out_name + ".tmp")
            if tmp.exists():
                shutil.rmtree(tmp)
            tmp.mkdir()
            np.save(tmp / "q.npy", q.astype(np.float32))
            wr.save(tmp)
            A.write_episode_labels(tmp, T, on, off, phone6, uv, vis)
            (tmp / "manifest.json").write_text(json.dumps({
                "episode": out_name, "source_episode": src, "domain": "isaac" if key is None else "isaac_moved",
                "moved_key": key, "frames": T, "rendered_frames": len(wr.frames), "stride": args.stride,
                "complete": True}, indent=2))
            dst = out_root / out_name
            if dst.exists():
                shutil.rmtree(dst)
            tmp.rename(dst)
            el = time.time() - t_start
            print(f"[render] {ji + 1}/{len(todo)} {out_name}: {len(wr.frames)} frames in {time.time() - t0:.0f} s, "
                  f"phone visible {int((vis == 1).sum())}, ETA {el / (ji + 1) * (len(todo) - ji - 1) / 60:.0f} min",
                  flush=True)
    finally:
        backend.close()


if __name__ == "__main__":
    main()
