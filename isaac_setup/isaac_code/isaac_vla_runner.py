#!/usr/bin/env python3
"""Closed-loop TOMO VLA runner using the existing PhysX-OFF IsaacBackend."""
from __future__ import annotations
import argparse
import json
import time
from multiprocessing.connection import Client
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import load_config, pack, unpack
from isaac_backend import IsaacBackend


class VLAClient:
    def __init__(self, cfg):
        p = cfg["vla_policy"]
        self.conn = Client(
            (p.get("host", "127.0.0.1"), int(p.get("port", 6031))),
            authkey=str(p.get("authkey", "tomo-vla")).encode(),
        )
        self.info = self.call({"cmd": "hello"})

    def call(self, msg):
        self.conn.send(msg)
        r = self.conn.recv()
        if not r.get("ok"):
            raise RuntimeError(r.get("error"))
        return r

    def reset(self):
        self.call({"cmd": "reset"})

    def step(self, rgb, depth, q, frame_id, instruction, infer):
        r = self.call({
            "cmd": "infer" if infer else "observe",
            "frame_id": int(frame_id),
            "rgb": pack(rgb),
            "depth": pack(np.asarray(depth, np.float32)),
            "q": np.asarray(q, float).tolist(),
            "instruction": instruction,
        })
        for k in ("horizon", "suction", "done"):
            if k in r:
                r[k] = unpack(r[k])
        return r


def grasp_gate(backend: IsaacBackend):
    scfg = backend.cfg.get("suction", {})
    if backend._grasp_offset is None:
        return False, None
    rt = backend.runtime
    target = (rt.hand_world() @ backend._grasp_offset)[:3, 3]
    phone = rt.phone_object_pose()[:3, 3]
    dxy = float(np.linalg.norm(target[:2] - phone[:2]))
    dz = float(target[2] - phone[2])
    ok = (
        dxy <= float(scfg.get("attach_xy_tolerance_m", 0.03))
        and dz <= float(scfg.get("attach_z_tolerance_m", 0.006))
    )
    return ok, {"dxy": dxy, "dz": dz}


def apply_learned_suction(backend, probability, step):
    scfg = backend.cfg.get("suction", {})
    on_th = float(scfg.get("learned_on_threshold", 0.60))
    off_th = float(scfg.get("learned_off_threshold", 0.40))
    safety = bool(scfg.get("learned_safety_gate", True))

    if not backend._vacuum and not backend._released and probability >= on_th:
        ok, geo = grasp_gate(backend)
        if (not safety) or ok:
            backend.fire_vacuum(
                reason=f"VLA suction p={probability:.3f}" + ("" if geo is None else f" gate={geo}"),
                step=step,
            )
    elif backend._vacuum and probability <= off_th:
        # The model has no "am I holding the phone" input, so after a late grasp it can predict
        # release too early. Only let go once the phone is over the box (release_only_over_box).
        if bool(scfg.get("release_only_over_box", True)):
            p = backend.runtime.phone_object_pose()[:3, 3]
            bx, by = scfg.get("box_xy", [0.41, 0.0])
            r = float(scfg.get("box_release_radius_m", 0.10))
            if (p[0] - bx) ** 2 + (p[1] - by) ** 2 > r * r:
                if step % 10 == 0:
                    print(f"[suction] release p={probability:.2f} ignored at step {step}: phone at "
                          f"({p[0]:.2f}, {p[1]:.2f}) is not over the box", flush=True)
                return
        backend.release_vacuum(reason=f"VLA release p={probability:.3f}", step=step)


def run_instruction(backend, client, cfg, instruction, max_steps, stop_condition="done"):
    p = cfg["vla_policy"]
    replan = int(p.get("replan_every", 8))
    done_th = float(p.get("done_threshold", 0.60))
    control_hz = float(p.get("control_hz", 30.0))  # 30 Hz matches training dataset frame_rate
    dt_target = 1.0 / control_hz
    max_joint_step_rad = float(cfg.get("motion", {}).get("delta_clamp_rad", 0.15))

    client.reset()
    q = backend._current_joint_targets.copy()
    horizon = suction = done = None
    k = replan
    logs = []

    print(f"\n[VLA] Starting task: {instruction!r} (Target rate: {control_hz:.1f} Hz, replan_every: {replan})")

    for t in range(max_steps):
        t0 = time.perf_counter()
        rgb, depth = backend.read_camera()
        infer = horizon is None or k >= replan
        r = client.step(rgb, depth, q, t, instruction, infer)
        if infer:
            horizon = np.asarray(r["horizon"], np.float32)
            suction = np.asarray(r["suction"], np.float32)
            done = np.asarray(r["done"], np.float32)
            k = 0

        q_target = horizon[k].astype(np.float64)
        if max_joint_step_rad > 0:
            dq = np.clip(q_target - q, -max_joint_step_rad, max_joint_step_rad)
            q_next = q + dq
        else:
            q_next = q_target

        backend.apply_joint_targets(q_next, step=t + 1)
        # backend suction mode is "learned", so apply_joint_targets does not auto-fire.
        apply_learned_suction(backend, float(suction[k]), step=t + 1)
        q = q_next
        done_p = float(done[k])
        phone_xyz = backend.runtime.phone_object_pose()[:3, 3].tolist()
        logs.append({
            "t": t + 1,
            "instruction": instruction,
            "suction_p": float(suction[k]),
            "done_p": done_p,
            "vacuum": bool(backend._vacuum),
            "phone_xyz": phone_xyz,
        })
        k += 1

        if (t + 1) % 15 == 0 or t == 0:
            print(f"[VLA loop] t={t+1:3d}/{max_steps} | suction_p={float(suction[k-1]):.3f} | vac={int(backend._vacuum)} | phone=[{phone_xyz[0]:.2f}, {phone_xyz[1]:.2f}, {phone_xyz[2]:.2f}] | done_p={done_p:.3f}")

        if done_p >= done_th:
            print(f"[VLA] command done at t={t+1}, p={done_p:.3f}: {instruction!r}")
            break
            
        if stop_condition == "touched" and backend._vacuum:
            print(f"[VLA] Stopping early: Target reached (Suction ON)")
            break
            
        if stop_condition == "lifted" and phone_xyz[2] > 0.10:
            print(f"[VLA] Stopping early: Object lifted (Z > 0.10m)")
            break

        # Regulate playback speed to real-time 30 Hz
        elapsed = time.perf_counter() - t0
        if elapsed < dt_target:
            time.sleep(dt_target - elapsed)

    return logs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config_vla.yaml"))
    ap.add_argument("--instruction", default=None)
    ap.add_argument(
        "--sequence",
        default=None,
        help='Pipe-separated commands, e.g. "reach the phone|grasp the phone|lift the phone|put the phone in the box"',
    )
    ap.add_argument("--max-steps", type=int, default=307)
    ap.add_argument("--out", type=Path, default=Path("/tmp/tomo_vla_run"))
    args = ap.parse_args()

    cfg, base = load_config(args.config)
    cfg.setdefault("suction", {})["mode"] = "learned"
    client = None
    pcfg = cfg.get("vla_policy", {})
    host = pcfg.get("host", "127.0.0.1")
    port = int(pcfg.get("port", 6031))
    print(f"⏳ Connecting to VLA Policy Server ({host}:{port})...")
    print("   👉 Make sure Terminal 1 is running:")
    print("      cd /home/emagesoft/Downloads/vlaflow/isaac_setup/isaac_code")
    print("      python3 vla_policy_server.py --config config_vla.yaml\n")
    for attempt in range(1, 121):
        try:
            client = VLAClient(cfg)
            print("✅ Connected to VLA Policy Server successfully!\n")
            break
        except (ConnectionRefusedError, OSError):
            if attempt % 5 == 0:
                print(f"   Waiting for policy server to start... ({attempt}/120s)")
            time.sleep(1.0)
    if client is None:
        raise SystemExit("\n❌ Error: VLA policy server not reachable after 120s.\nPlease start Terminal 1 first:\n  python3 vla_policy_server.py --config config_vla.yaml")
    backend = IsaacBackend(config=cfg, config_dir=base)

    commands = (
        [x.strip() for x in args.sequence.split("|") if x.strip()]
        if args.sequence
        else [args.instruction or cfg["vla_policy"].get("default_instruction", "pick up the phone and put it in the box")]
    )
    all_logs = []
    try:
        if not args.sequence and not args.instruction:
            client.conn.send({"cmd": "get_user_input"})
            reply = client.conn.recv()
            if reply.get("command") != "exit":
                vla_prompt = reply.get("vla_prompt", cfg["vla_policy"].get("default_instruction", ""))
                stop_condition = reply.get("stop_condition", "done")
                
                logs = run_instruction(backend, client, cfg, vla_prompt, args.max_steps, stop_condition)
                all_logs.extend(logs)
        else:
            for command in commands:
                all_logs.extend(run_instruction(backend, client, cfg, command, args.max_steps))
        
        args.out.mkdir(parents=True, exist_ok=True)
        with (args.out / "vla_log.jsonl").open("w") as f:
            for x in all_logs:
                f.write(json.dumps(x) + "\n")
    finally:
        try:
            client.conn.close()
        except Exception:
            pass
        backend.close()


if __name__ == "__main__":
    main()
