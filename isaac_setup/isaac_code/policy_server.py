"""TOMO XYZRGB iDP3 policy server (run in the idp3_rgb conda env).

Isaac Sim ships its own python/numpy, while the policy was trained in the
idp3_rgb env (diffusers, hydra, the audited iDP3 source).  So the policy runs
here, Isaac talks to it over a localhost socket (multiprocessing.connection).

Per Isaac frame:
    RGB (960x720 uint8) + distance_to_image_plane depth (m) + commanded q46
      -> EXACT training converter (tomo_rgbd_pointcloud_review.py):
         336x252 floor-grid of native pixels, native K, optical-axis Z,
         0.15..3.0 m, 4096 rows sampled with SeedSequence([42, frame_id])
      -> obs history [t-1, t]  (duplicated at t = 0, like the offline rollout)
      -> FutureOnlyRGBDPolicy.predict_action -> action_pred [16, 46],
         row 0 = q[t+1]

    ~/miniconda3/envs/idp3_rgb/bin/python idp3_policy_server.py --config config.yaml
    ~/miniconda3/envs/idp3_rgb/bin/python idp3_policy_server.py --config config.yaml \
        --selftest Find_and_Pick_001      # converter == prepared training clouds?
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from multiprocessing.connection import Listener
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import converter_module, episode_paths, load_config, pack, read_dataset_frames, resolve, unpack  # noqa: E402

JOINT_NAMES = (
    ["Head_joint1", "Head_joint2"]
    + [f"Left_arm_joint{i}" for i in range(1, 8)]
    + [f"Right_arm_joint{i}" for i in range(1, 8)]
    + [f"RF_joint_{f}{s}" for f in range(1, 6) for s in range(1, 4)]
    + [f"LF_joint_{f}{s}" for f in range(1, 6) for s in range(1, 4)]
)


class CloudBuilder:
    """Bit-for-bit the training-set conversion of one RGB-D frame."""

    def __init__(self, cfg: dict, base: Path):
        pc = cfg["idp3_policy"]["point_cloud"]
        k = cfg["isaac_mode"]["camera_intrinsics"]
        self.tools = converter_module(resolve(cfg["idp3_policy"]["training_bundle"], base))
        self.K = self.tools.scale_intrinsics(dict(width=int(k["width"]), height=int(k["height"]),
                                                  fx=float(k["fx"]), fy=float(k["fy"]),
                                                  cx=float(k["cx"]), cy=float(k["cy"])),
                                             int(k["width"]), int(k["height"]))
        self.uv = self.tools.grid_uv(int(k["width"]), int(k["height"]), int(pc["grid_w"]), int(pc["grid_h"]))
        self.kind = str(pc["depth_kind"])
        self.lo, self.hi = float(pc["min_depth_m"]), float(pc["max_depth_m"])
        self.n = int(pc["num_points"])
        self.seed = int(pc["point_seed"])

    def __call__(self, rgb: np.ndarray, depth: np.ndarray, frame_id: int):
        depth = np.asarray(depth, np.float32)
        cloud, valid_uv, valid = self.tools.make_cloud(depth, rgb, self.K, self.uv, self.kind, self.lo, self.hi)
        sampled, _, _ = self.tools.sample_rows(cloud, valid_uv, self.n,
                                               np.random.SeedSequence([self.seed, int(frame_id)]))
        stats = {"valid_grid_points": int(len(cloud)), "grid_points": int(valid.size),
                 "padded": bool(len(cloud) < self.n),
                 "median_z_m": float(np.median(sampled[:, 2])),
                 "rgb_mean": float(sampled[:, 3:].mean())}
        return sampled.astype(np.float32), stats


class Policy:
    def __init__(self, cfg: dict, base: Path):
        import torch
        pcfg = cfg["idp3_policy"]
        bundle = resolve(pcfg["training_bundle"], base)
        project = resolve(pcfg["idp3_project_root"], base)
        ckpt = resolve(pcfg["checkpoint"], base)
        for p, what in ((bundle, "training bundle"), (project, "iDP3 project root"), (ckpt, "checkpoint")):
            if not p.exists():
                raise FileNotFoundError(f"{what} missing: {p}")
        sys.path.insert(0, str(bundle))
        import train_tomo_rgbd_one_episode as base_mod

        self.torch = torch
        self.device = torch.device(pcfg.get("device", "cuda:0"))
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available in this env")
        base_mod.verify_project(project)
        policy, _, _ = base_mod.make_policy(project, np.zeros((2, 46), np.float32), self.device)
        payload = torch.load(ckpt, map_location="cpu", weights_only=True)
        if payload.get("schema") != "tomo_idp3_rgbd_301_augmented_v3":
            raise RuntimeError(f"Wrong checkpoint schema: {payload.get('schema')}")
        if payload.get("joint_names") != JOINT_NAMES:
            raise RuntimeError("Checkpoint 46-joint order differs from the Isaac order")
        if payload.get("model_meta", {}).get("first_action") != "q[t+1], index 0":
            raise RuntimeError("Checkpoint is not the future-only q[t+1] contract")
        policy.load_state_dict(payload["policy_state_dict"], strict=True)
        policy.to(self.device).eval()
        self.policy = policy
        stats = policy.normalizer["action"].get_input_stats()
        self.train_min = stats["min"].detach().float().cpu().numpy().reshape(-1)
        self.train_max = stats["max"].detach().float().cpu().numpy().reshape(-1)
        self.meta = {"checkpoint": str(ckpt), "step": payload.get("step"),
                     "equivalent_epoch": payload.get("equivalent_epoch"),
                     "evaluated_weights": payload.get("evaluated_weights"),
                     "suction_trained": payload.get("suction_trained"),
                     "device": str(self.device)}
        print(f"[policy] loaded {ckpt} step={payload.get('step')} "
              f"epoch~{payload.get('equivalent_epoch'):.1f} weights={payload.get('evaluated_weights')} on {self.device}")

    def infer(self, prev_cloud, cur_cloud, prev_q, cur_q, seed: int) -> np.ndarray:
        torch = self.torch
        obs = {
            "point_cloud": torch.from_numpy(np.stack([prev_cloud, cur_cloud])[None].astype(np.float32)).to(self.device),
            "agent_pos": torch.from_numpy(np.stack([prev_q, cur_q])[None].astype(np.float32)).to(self.device),
        }
        gpu = [self.device.index or 0] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=gpu):
            torch.manual_seed(int(seed))
            if gpu:
                torch.cuda.manual_seed_all(int(seed))
            with torch.inference_mode():
                out = self.policy.predict_action(obs)
        full = out["action_pred"].detach().float().cpu().numpy()
        if full.shape != (1, 16, 46) or not np.isfinite(full).all():
            raise RuntimeError(f"Bad policy output {full.shape}")
        return full[0].astype(np.float32)


class XYZRGBModel:
    """idp3colourrgbtraining/model/best_val.pt: XYZRGB cloud from RGB + Z depth."""
    n_obs = 2

    def __init__(self, cfg, base):
        self.clouds = CloudBuilder(cfg, base)
        self.policy = Policy(cfg, base)
        self.meta = {**self.policy.meta, "model": "XYZRGB iDP3 (idp3colourrgbtraining)"}
        self.train_min, self.train_max = self.policy.train_min, self.policy.train_max

    def cloud(self, rgb, depth, fid):
        return self.clouds(rgb, depth, fid)

    def infer(self, hist, seed):
        return self.policy.infer(hist[0][1], hist[1][1], hist[0][2], hist[1][2], seed)


class ActAdapterModel:
    """RGB-D ACT V2 (DINOv2) through its verified adapter (build_obs / load_policy / predict)."""

    def __init__(self, cfg, base, key="act_rgbd_policy"):
        import importlib.util
        acfg = cfg[key]
        path = resolve(acfg["adapter"], base)
        spec = importlib.util.spec_from_file_location("tomo_act_adapter", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod      # dataclasses need the module registered
        spec.loader.exec_module(mod)
        self.mod = mod
        ckpt = str(resolve(acfg["checkpoint"], base)) if acfg.get("checkpoint") else mod.CHECKPOINT
        self.policy = mod.load_policy(acfg.get("device", "cuda:0"), checkpoint=ckpt)
        self.n_obs = int(mod.N_OBS)
        if list(mod.JOINT_NAMES) != JOINT_NAMES:
            raise RuntimeError("ACT adapter joint order differs from the Isaac 46-D order")
        self.meta = {"checkpoint": ckpt, "step": None, "equivalent_epoch": None, "evaluated_weights": "see adapter",
                     "suction_trained": False, "device": str(acfg.get("device", "cuda:0")),
                     "model": key, "horizon": int(mod.HORIZON)}
        self.train_min = np.full(46, -1e9)
        self.train_max = np.full(46, 1e9)
        print(f"[policy] ACT adapter {path.name}: {ckpt} n_obs={self.n_obs} horizon={mod.HORIZON}", flush=True)

    def cloud(self, rgb, depth, fid):
        return self.mod.build_obs(np.asarray(rgb, np.uint8), np.asarray(depth, np.float32), int(fid)), \
            {"valid_grid_points": None, "padded": False, "median_z_m": float(np.nanmedian(depth)), "rgb_mean": float(np.mean(rgb))}

    def infer(self, hist, seed):
        out = np.asarray(self.policy.predict([h[1] for h in hist], np.stack([h[2] for h in hist]).astype(np.float32),
                                             int(seed)), np.float32)
        if out.ndim != 2 or out.shape[1] != 46 or not np.isfinite(out).all():
            raise RuntimeError(f"ACT adapter returned {out.shape}")
        return out


class PhoneAdapterModel(ActAdapterModel):
    """Phone-conditioned diffusion policy (training/phone_conditioned) + YOLO; needs the per-frame
    extras (camera pose, K, true phone start pose, phone_source) sent by the Isaac runner."""
    wants_extras = True

    def __init__(self, cfg, base):
        import importlib.util
        pcfg = cfg["phone_policy"]
        path = resolve(pcfg["adapter"], base)
        spec = importlib.util.spec_from_file_location("tomo_phone_adapter", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        if pcfg.get("yolo_weights"):
            mod.YOLO_WEIGHTS = str(Path(pcfg["yolo_weights"]).expanduser())
        self.mod = mod
        ckpt = str(Path(pcfg["checkpoint"]).expanduser())
        self.policy = mod.PhoneConditionedPolicy(pcfg.get("device", "cuda:0"), ckpt, mod.YOLO_WEIGHTS)
        self.n_obs = int(mod.N_OBS)
        self.meta = {"checkpoint": ckpt, "step": None, "equivalent_epoch": None, "evaluated_weights": "ema",
                     "suction_trained": False, "device": str(pcfg.get("device", "cuda:0")),
                     "model": "phone-conditioned diffusion policy", "horizon": int(mod.HORIZON)}
        self.train_min = np.full(46, -1e9)
        self.train_max = np.full(46, 1e9)
        print(f"[policy] phone-conditioned adapter: {ckpt} (YOLO {mod.YOLO_WEIGHTS})", flush=True)

    def cloud(self, rgb, depth, fid, extras=None):
        return self.mod.build_obs(np.asarray(rgb, np.uint8), np.asarray(depth, np.float32), int(fid), extras), \
            {"valid_grid_points": None, "padded": False, "median_z_m": None, "rgb_mean": None}

    def infer(self, hist, seed):
        out = super().infer(hist, seed)
        return out

    def reset(self):
        self.policy.reset()


class DepthAdapterModel:
    """~/Desktop/tomo_idp3 depth-only iDP3 through its verified adapter module
    (build_cloud / load_policy / predict; see tomo_idp3_depth_adapter.py)."""

    def __init__(self, cfg, base):
        import importlib.util
        dcfg = cfg["depth_idp3_policy"]
        path = resolve(dcfg["adapter"], base)
        spec = importlib.util.spec_from_file_location("tomo_idp3_depth_adapter", str(path))
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod      # dataclasses need the module registered
        spec.loader.exec_module(mod)
        self.mod = mod
        ckpt = str(Path(dcfg["checkpoint"]).expanduser()) if dcfg.get("checkpoint") else mod.CHECKPOINT
        self.policy = mod.load_policy(dcfg.get("device", "cuda:0"), checkpoint=ckpt)
        mod.CHECKPOINT = ckpt
        self.n_obs = int(mod.N_OBS)
        if list(mod.JOINT_NAMES) != JOINT_NAMES:
            raise RuntimeError("depth adapter joint order differs from the Isaac 46-D order")
        self.meta = {"checkpoint": str(mod.CHECKPOINT), "step": getattr(mod, "STEP", None),
                     "equivalent_epoch": getattr(mod, "EPOCH", None),
                     "evaluated_weights": getattr(mod, "WEIGHTS", "see adapter"),
                     "suction_trained": False, "device": str(dcfg.get("device", "cuda:0")),
                     "model": "tomo_idp3 depth-only iDP3", "horizon": int(mod.HORIZON),
                     "n_action_steps": int(mod.N_ACTION_STEPS)}
        self.train_min = np.full(46, -1e9)
        self.train_max = np.full(46, 1e9)
        print(f"[policy] depth-only adapter {path.name}: {mod.CHECKPOINT} n_obs={self.n_obs} "
              f"horizon={mod.HORIZON} n_action_steps={mod.N_ACTION_STEPS}", flush=True)

    def cloud(self, rgb, depth, fid):
        c = np.asarray(self.mod.build_cloud(np.asarray(depth, np.float32), int(fid)), np.float32)
        return c, {"valid_grid_points": int(c.shape[0]), "padded": False,
                   "median_z_m": float(np.median(c[:, 2])), "rgb_mean": None}

    def infer(self, hist, seed):
        clouds = np.stack([h[1] for h in hist]).astype(np.float32)
        qs = np.stack([h[2] for h in hist]).astype(np.float32)
        out = np.asarray(self.policy.predict(clouds, qs, int(seed)), np.float32)
        if out.ndim != 2 or out.shape[1] != 46 or not np.isfinite(out).all():
            raise RuntimeError(f"depth adapter returned {out.shape}")
        return out


def section(cfg: dict, model: str) -> dict:
    extra = cfg[f"{model}_policy"]
    return {**cfg["idp3_policy"], **extra}


def serve(cfg: dict, base: Path, model: str = "xyzrgb"):
    pcfg = section(cfg, model)
    net = ActAdapterModel(cfg, base, key=f"{model}_policy")
    seed0 = int(pcfg.get("inference_seed", 12345))
    addr = (pcfg.get("host", "127.0.0.1"), int(pcfg.get("port", 6010)))
    listener = Listener(addr, authkey=str(pcfg.get("authkey", "tomo")).encode())
    print(f"[policy] READY on {addr[0]}:{addr[1]}", flush=True)
    while True:
        try:
            conn = listener.accept()
        except Exception as exc:   # port scan / wrong authkey: keep serving
            print(f"[policy] rejected a connection ({type(exc).__name__}: {exc})", flush=True)
            continue
        print(f"[policy] client connected: {listener.last_accepted}", flush=True)
        history = []   # [(frame_id, cloud, q)] last two observations
        try:
            while True:
                try:
                    msg = conn.recv()
                except EOFError:
                    break
                cmd = msg.get("cmd")
                if cmd == "hello":
                    conn.send({"ok": True, "joint_names": JOINT_NAMES, **net.meta, "n_obs": net.n_obs,
                               "train_min": net.train_min.tolist(), "train_max": net.train_max.tolist()})
                elif cmd == "reset":
                    history.clear()
                    if hasattr(net, "reset"):
                        net.reset()
                    conn.send({"ok": True})
                elif cmd in ("observe", "infer"):
                    t0 = time.perf_counter()
                    reply = {"ok": True}
                    if "rgb" in msg:
                        fid = int(msg["frame_id"])
                        if getattr(net, "wants_extras", False):
                            cloud, cstats = net.cloud(unpack(msg["rgb"]), unpack(msg["depth"]), fid, msg.get("extras"))
                        else:
                            cloud, cstats = net.cloud(unpack(msg["rgb"]), unpack(msg["depth"]), fid)
                        q = np.asarray(msg["q"], np.float32)
                        history.append((fid, cloud, q))
                        del history[:-net.n_obs]
                        reply["cloud_stats"] = cstats
                        if msg.get("return_cloud"):
                            reply["cloud"] = pack(cloud)
                    if cmd == "infer":
                        if not history:
                            raise RuntimeError("infer before any observation")
                        hist = [history[0]] * (net.n_obs - len(history)) + history   # pad at t=0 like training
                        t1 = time.perf_counter()
                        horizon = net.infer(hist, seed0 + hist[-1][0])
                        reply["horizon"] = pack(horizon)
                        reply["obs_frames"] = [h[0] for h in hist]
                        pu = getattr(getattr(net, "policy", None), "phone_used", None)
                        if pu is not None:
                            reply["phone_used"] = pu
                        reply["infer_ms"] = round((time.perf_counter() - t1) * 1e3, 2)
                    reply["total_ms"] = round((time.perf_counter() - t0) * 1e3, 2)
                    conn.send(reply)
                elif cmd == "shutdown":
                    conn.send({"ok": True})
                    conn.close()
                    listener.close()
                    print("[policy] shutdown requested", flush=True)
                    return
                else:
                    conn.send({"ok": False, "error": f"unknown cmd {cmd!r}"})
        except Exception as exc:  # report to client, keep serving
            import traceback
            traceback.print_exc()
            try:
                conn.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
            except Exception:
                pass
        finally:
            try:
                conn.close()
            except Exception:
                pass
            print("[policy] client disconnected; waiting for next", flush=True)


def selftest(cfg: dict, base: Path, episode: str, frames):
    """Converter on raw dataset frames must equal the prepared training clouds."""
    ds = cfg["isaac_mode"]["dataset_start"]
    paths = episode_paths(resolve(ds["dataset_dir"], base), episode)
    clouds = CloudBuilder(cfg, base)
    ep_no = int(episode.split("_")[-1])
    manifest = resolve("../../../dataset/manifests/dataset_manifest.json", base)
    prepared = None
    if manifest.is_file():
        for e in json.loads(manifest.read_text())["episodes"]:
            if int(e["episode"]) == ep_no and Path(e["path"]).is_dir():
                prepared = Path(e["path"])
    ref = read_dataset_frames(paths["mp4"], paths["depth_dir"], frames, clouds.tools)
    pc = np.load(prepared / "point_cloud.npy", mmap_mode="r") if prepared else None
    ok = True
    for f in frames:
        cloud, st = clouds(ref[f][0], ref[f][1], f)
        same = None if pc is None else bool(np.array_equal(cloud, pc[f]))
        ok &= same is not False
        print(f"[selftest] {episode} frame {f}: valid={st['valid_grid_points']} "
              f"medianZ={st['median_z_m']:.3f} identical_to_training_cloud={same}")
    print(f"[selftest] prepared episode dir: {prepared}")
    print("[selftest] PASS" if ok else "[selftest] FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--selftest", default=None, metavar="EPISODE")
    ap.add_argument("--frames", default="0,20,50,155,306")
    ap.add_argument("--ping", action="store_true", help="exit 0 if a server is already answering")
    ap.add_argument("--shutdown", action="store_true", help="ask a running server to exit")
    ap.add_argument("--model", choices=["act_rgbd"], default="act_rgbd",
                    help="act_rgbd = RGB-D DINOv2 ACT v2; act_rgb = RGB-only ResNet-18 ACT")
    args = ap.parse_args()
    cfg, base = load_config(args.config)
    if args.ping or args.shutdown:
        from multiprocessing.connection import Client
        p = section(cfg, args.model)
        try:
            c = Client((p.get("host", "127.0.0.1"), int(p.get("port", 6010))),
                       authkey=str(p.get("authkey", "tomo")).encode())
            c.send({"cmd": "shutdown" if args.shutdown else "hello"})
            ok = bool(c.recv().get("ok"))
            c.close()
        except Exception:
            ok = False
        print("[policy] server " + ("answering" if ok else "not running"))
        sys.exit(0 if ok else 1)
    if args.selftest:
        frames = [int(x) for x in args.frames.split(",")]
        sys.exit(0 if selftest(cfg, base, args.selftest, frames) else 1)
    serve(cfg, base, args.model)


if __name__ == "__main__":
    main()
