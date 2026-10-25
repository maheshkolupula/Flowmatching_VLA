"""No-policy check: does Isaac see what the training data saw?

Replays the RECORDED joints of one demonstration (default Find_and_Pick_001)
in Isaac, renders RGB + distance_to_image_plane at chosen frames and compares
them with the dataset MP4 + Depth.V EXR for the same frames.

Run with Isaac Sim python:
    ~/isaac-sim/python.sh tools/check_scene_vs_dataset.py --config config.yaml
Outputs (default /tmp/tomo_idp3_isaac_check/):
    compare_fXXX.png   dataset RGB | Isaac RGB | dataset depth | Isaac depth | |diff|
    report.json        per-frame depth agreement numbers
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import converter_module, episode_paths, load_config, read_dataset_frames, resolve  # noqa: E402


def depth_vis(d, lo=0.15, hi=3.0):
    import cv2
    v = np.isfinite(d) & (d >= lo) & (d <= hi)
    g = np.zeros(d.shape, np.uint8)
    g[v] = (np.clip((d[v] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    h = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    h[~v] = 0
    return h


def compare(ds_rgb, ds_d, is_rgb, is_d, lo=0.15, hi=3.0):
    vd = np.isfinite(ds_d) & (ds_d >= lo) & (ds_d <= hi)
    vi = np.isfinite(is_d) & (is_d >= lo) & (is_d <= hi)
    both = vd & vi
    diff = np.abs(ds_d - is_d)
    inter, union = int((vd & vi).sum()), int((vd | vi).sum())
    return {
        "dataset_valid_fraction": float(vd.mean()),
        "isaac_valid_fraction": float(vi.mean()),
        "valid_mask_iou": inter / union if union else 1.0,
        "median_abs_depth_err_m": float(np.median(diff[both])) if both.any() else None,
        "frac_within_1cm": float((diff[both] < 0.01).mean()) if both.any() else None,
        "frac_within_3cm": float((diff[both] < 0.03).mean()) if both.any() else None,
        "rgb_mean_abs_err_0_255": float(np.abs(ds_rgb.astype(float) - is_rgb.astype(float)).mean()),
        "dataset_rgb_mean": float(ds_rgb.mean()),
        "isaac_rgb_mean": float(is_rgb.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--episode", default=None)
    ap.add_argument("--frames", default="0,10,20,30,50,100,155,200,290,306")
    ap.add_argument("--out", default="/tmp/tomo_idp3_isaac_check")
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--no-fixes", action="store_true", help="render the raw USD (no scene_fixes)")
    ap.add_argument("--sweep-dome", default="", help="e.g. 300,600,1000,1500: RGB error per dome intensity")
    ap.add_argument("--sweep-iso", default="", help="e.g. 25,50,100: manual exposure sweep (auto-exposure off)")
    ap.add_argument("--latency-test", action="store_true",
                    help="measure how many app updates the render lags a pose change")
    args = ap.parse_args()

    cfg, base = load_config(args.config)
    icfg = cfg["isaac_mode"]
    ds = icfg["dataset_start"]
    episode = args.episode or ds["episode"]
    dataset_dir = resolve(ds["dataset_dir"], base)
    paths = episode_paths(dataset_dir, episode)
    pcfg = cfg["idp3_policy"]
    cloud_tools = converter_module(resolve(pcfg["training_bundle"], base))
    frames = [int(x) for x in args.frames.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    from isaacsim.simulation_app import SimulationApp
    app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720})

    import cv2
    from runtime.isaac_runtime import CANONICAL_46, EpisodeReference, IsaacRuntime

    rt = IsaacRuntime(
        map_joint_names=list(CANONICAL_46), sim_app=app,
        usd_path=resolve(icfg["usd_path"], base), robot_root=icfg["robot_root"],
        camera_prim=icfg["camera_prim"], box_prim=icfg["box_prim"],
        cam_wh=tuple(icfg["cam_wh"]), physics=False,
        container_prim=icfg["container_prim"], phone_mesh_prim=icfg["phone_mesh_prim"],
        suction_prim=icfg["suction_prim"], camera_intrinsics=icfg["camera_intrinsics"],
        render_updates_per_frame=int(icfg.get("render_updates_per_frame", 3)),
        scene_fixes=None if args.no_fixes else icfg.get("scene_fixes"),
    )
    ep = EpisodeReference(paths["yaml"])
    yaw_off = math.radians(float(ds.get("phone_yaw_offset_deg", 0.0)))
    if not args.no_fixes:
        rt.apply_appearance(icfg.get("appearance"), ep.phone_material, base, ep.materials)
    for _ in range(int(icfg.get("warmup_ticks", 30))):
        app.update()

    print(f"[check] decoding dataset frames {frames} of {episode}")
    ref = read_dataset_frames(paths["mp4"], paths["depth_dir"], frames, cloud_tools)

    report = {"episode": episode, "frames": {}, "runtime": None}
    for f in frames:
        rt.set_phone_pose(ep.phone_pose(f), yaw_offset_rad=yaw_off)
        rt.drive(ep.q46(f))
        rt.render(6)
        rgb, depth = rt.read_camera()
        ds_rgb, ds_d = ref[f]
        m = compare(ds_rgb, ds_d, rgb, depth)
        m["camera"] = {k: v for k, v in rt.diagnostics().items() if k.startswith("camera_w") or k == "camera_pitch_deg"}
        report["frames"][str(f)] = m
        both = np.isfinite(ds_d) & np.isfinite(depth)
        err = np.zeros(depth.shape, np.float32)
        err[both] = np.abs(ds_d - depth)[both]
        err_vis = cv2.applyColorMap((np.clip(err / 0.10, 0, 1) * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
        row1 = cv2.hconcat([cv2.cvtColor(ds_rgb, cv2.COLOR_RGB2BGR), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)])
        row2 = cv2.hconcat([depth_vis(ds_d), depth_vis(depth)])
        tile = cv2.vconcat([row1, row2])
        tile = cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2), interpolation=cv2.INTER_AREA)
        tile = cv2.hconcat([tile, cv2.resize(err_vis, (tile.shape[1] // 2, tile.shape[0]))])
        for (x, txt) in [(10, "dataset RGB / depth"), (490, "ISAAC RGB / depth"), (970, "|depth diff| 0..10cm")]:
            cv2.putText(tile, txt, (x, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(tile, f"frame {f}  head=({ep.q46(f)[0]:+.3f},{ep.q46(f)[1]:+.3f})  "
                    f"IoU={m['valid_mask_iou']:.3f}  med|dz|={m['median_abs_depth_err_m'] or float('nan'):.4f} m",
                    (10, tile.shape[0] - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imwrite(str(out / f"compare_f{f:03d}.png"), tile)
        np.savez_compressed(out / f"arrays_f{f:03d}.npz", isaac_rgb=rgb, isaac_depth=depth,
                            dataset_rgb=ds_rgb, dataset_depth=ds_d,
                            camera_world=rt.camera_world(), K=rt.camera_K())
        print(f"[check] frame {f:3d}: IoU={m['valid_mask_iou']:.3f} "
              f"med|dz|={m['median_abs_depth_err_m']} within3cm={m['frac_within_3cm']} "
              f"rgbMAE={m['rgb_mean_abs_err_0_255']:.1f} pitch={m['camera']['camera_pitch_deg']}")

    if args.sweep_dome:
        sweep = {}
        for inten in [float(x) for x in args.sweep_dome.split(",")]:
            rt.set_dome_intensity(inten)
            errs = []
            for f in frames:
                rt.set_phone_pose(ep.phone_pose(f), yaw_offset_rad=yaw_off)
                rt.drive(ep.q46(f))
                rgb, _ = rt.read_camera()
                errs.append(float(np.abs(rgb.astype(float) - ref[f][0].astype(float)).mean()))
                if f == 50:
                    cv2.imwrite(str(out / f"dome_{int(inten)}_f050.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            sweep[inten] = round(float(np.mean(errs)), 2)
            print(f"[check] dome {inten:7.1f}: mean RGB MAE {sweep[inten]:.2f}  per-frame {np.round(errs, 1).tolist()}")
        report["dome_sweep_rgb_mae"] = sweep

    if args.sweep_iso:
        # region masks from the DATASET depth back-projected to world (frame 50)
        f = 50 if 50 in ref else frames[0]
        ds_rgb, ds_d = ref[f]
        rt.set_phone_pose(ep.phone_pose(f), yaw_offset_rad=yaw_off)
        rt.drive(ep.q46(f))
        C, K = rt.camera_world(), rt.camera_K()
        v, u = np.indices(ds_d.shape)
        z = ds_d
        P = np.stack([(u - K[0, 2]) / K[0, 0] * z, -(v - K[1, 2]) / K[1, 1] * z, -z, np.ones_like(z)], -1)
        W = (P.reshape(-1, 4) @ C.T)[:, :3].reshape(z.shape + (3,))
        ok = np.isfinite(z) & (z > 0.15) & (z < 3.0)
        masks = {
            "table": ok & (np.abs(W[..., 2]) < 0.003) & (W[..., 0] > 0.2) & (W[..., 0] < 0.7)
                     & (np.abs(W[..., 1]) > 0.18) & (np.abs(W[..., 1]) < 0.4),
            "box": ok & (W[..., 2] > 0.04) & (W[..., 2] < 0.25) & (np.abs(W[..., 1]) < 0.12)
                   & (W[..., 0] > 0.3) & (W[..., 0] < 0.5),
            "floor": ok & (W[..., 2] < -0.5),
        }
        tgt = {k: ds_rgb[m].reshape(-1, 3).mean(0) for k, m in masks.items()}
        print("[check] dataset region means:", {k: np.round(v, 1).tolist() for k, v in tgt.items()})
        sweep = {}
        for iso in [float(x) for x in args.sweep_iso.split(",")]:
            rt.set_exposure(auto=False, film_iso=iso)
            rt.render(4)
            rgb, _ = rt.read_camera()
            rgb, _ = rt.read_camera()
            got = {k: rgb[m].reshape(-1, 3).mean(0) for k, m in masks.items()}
            mae = float(np.abs(rgb.astype(float) - ds_rgb.astype(float)).mean())
            sweep[iso] = {"mae": round(mae, 2), **{k: np.round(v, 1).tolist() for k, v in got.items()}}
            print(f"[check] filmIso {iso:6.1f}: MAE {mae:6.2f} " +
                  " ".join(f"{k}={np.round(v, 0).astype(int).tolist()}" for k, v in got.items()))
            cv2.imwrite(str(out / f"iso_{int(iso)}_f{f:03d}.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        report["iso_sweep"] = sweep

    if args.latency_test:
        # A read right after a pose change must already show the new pose.
        def clean(d):
            return np.nan_to_num(d, nan=50.0, posinf=50.0, neginf=50.0)
        rt._set_joint_positions_raw(ep.q46(0), render=False)
        rt.read_camera()
        rt._set_joint_positions_raw(ep.q46(50), render=False)
        first = clean(rt.read_camera()[1])
        rt.render(8)
        settled = clean(rt.read_camera()[1])
        lag = float(np.median(np.abs(first - settled)))
        print(f"[check] latency: median |first read - settled| after pose change = {lag:.6f} m "
              f"({'OK, no lag' if lag < 1e-3 else 'LAG - increase render_updates_per_frame'})")
        report["latency_first_read_vs_settled_median_m"] = lag

    report["runtime"] = rt.diagnostics()
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(f"[check] wrote {out}/report.json and compare_f*.png")
    rt.close()


if __name__ == "__main__":
    main()
