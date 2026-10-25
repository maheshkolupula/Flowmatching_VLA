"""Isaac backend -- TOMO XYZRGB iDP3 (model/best_val.pt), PhysX OFF.

Closed loop (run with Isaac Sim python, policy server already up):

    Isaac RGB 960x720 + distance_to_image_plane depth + commanded q46
        --(localhost)--> idp3_policy_server.py  (idp3_rgb env)
        <-- 16 x 46 future joint positions (row 0 = q[t+1])
    execute `replan_every` rows kinematically -> re-render -> repeat

Start state = recorded demonstration frame 0: head joints ~0, camera level
along robot +X ("looking straight").  From there the POLICY moves the head;
in the demos it pitches the view down to the table over frames ~10..30.

    ~/isaac-sim/python.sh isaac_backend.py --config config.yaml
    (or ./run_isaac_idp3.sh, which also starts/stops the policy server)

The IsaacBackend class keeps the tomo_brain_inference backend API
(read_camera / get_joints / drive_next_steps / fire_vacuum / ...), so this file
is also a drop-in for that project; the `seat` argument is optional here.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from common import episode_paths, load_config, pack, resolve, unpack  # noqa: E402


def _resolver(config_dir: Path | None):
    if config_dir is not None:
        return lambda v: resolve(v, config_dir)
    try:  # inside tomo_brain_inference
        from configs import resolve as tbi_resolve
        return tbi_resolve
    except Exception:
        return lambda v: resolve(v, ROOT)


class IsaacBackend:
    def __init__(self, seat=None, config: dict | None = None, config_dir: Path | None = None,
                 sim_app=None):
        self.seat = seat
        self.cfg = config
        self._res = _resolver(config_dir)
        icfg = config["isaac_mode"]
        headless = bool(icfg.get("headless", True)) and not bool(icfg.get("gui", False))

        if sim_app is None:
            print(f"[isaac-backend] launching SimulationApp(headless={headless}) ...")
            from isaacsim.simulation_app import SimulationApp
            sim_app = SimulationApp({"headless": headless, "width": 1280, "height": 720})
        self.sim_app = sim_app
        self.headless = headless

        from runtime.isaac_runtime import CANONICAL_46, EpisodeReference, IsaacRuntime
        self.joint_names = list(seat.map_joint_names) if seat is not None else list(CANONICAL_46)
        self.runtime = IsaacRuntime(
            map_joint_names=self.joint_names,
            sim_app=self.sim_app,
            usd_path=self._res(icfg["usd_path"]),
            robot_root=icfg["robot_root"],
            camera_prim=icfg["camera_prim"],
            box_prim=icfg["box_prim"],
            cam_wh=tuple(icfg["cam_wh"]),
            physics=bool(icfg.get("physics", False)),
            container_prim=icfg.get("container_prim", "/tomo_re_uhand/Box_1_v1___open"),
            phone_mesh_prim=icfg.get("phone_mesh_prim", "/Iphone/geo/iPhone_14_Pro_Max"),
            suction_prim=icfg.get("suction_prim"),
            camera_intrinsics=icfg.get("camera_intrinsics"),
            render_updates_per_frame=int(icfg.get("render_updates_per_frame", 3)),
            scene_fixes=icfg.get("scene_fixes"),
        )

        # ---------------------------------------------------- start state
        ds = icfg.get("dataset_start", {}) or {}
        self.episode = None
        self._vacuum, self._released, self.suction_events = False, False, []
        if ds.get("enabled", False):
            self.load_episode(ds["episode"], int(ds.get("start_frame", 0)), ds.get("phone_override"))
        elif icfg.get("home_yaml"):
            self.runtime.apply_home_pose(home_yaml=self._res(icfg["home_yaml"]))
            self._grasp_offset = None
        else:
            raise RuntimeError("Set isaac_mode.dataset_start.enabled or isaac_mode.home_yaml")

        obs = (config.get("visualization") or {}).get("observer") or {}
        if obs.get("enabled", False):
            self.runtime.create_observer(obs.get("eye", [1.25, -1.05, 0.95]), obs.get("target", [0.38, -0.18, 0.08]),
                                         tuple(obs.get("wh", [960, 720])), float(obs.get("hfov_deg", 55.0)))
        for _ in range(int(icfg.get("warmup_ticks", 30))):
            self.runtime.step()
        self._set_viewport_view()

        self._current_joint_targets = self.runtime.home_pose.copy()
        self._current_hand_xyz = self._hand_xyz(self._current_joint_targets)
        self._vacuum = False
        self._released = False
        self.suction_events: list[dict] = []
        self._last_frame: tuple[np.ndarray, np.ndarray] | None = None
        print("[isaac-backend] diagnostics:\n" + json.dumps(self.runtime.diagnostics(), indent=2))

    # ------------------------------------------------------------ episodes
    def load_episode(self, name: str, start_frame: int = 0, phone_override: dict | None = None) -> None:
        """(Re)start from a recorded demonstration: frame-`start_frame` joints,
        that episode's phone pose, phone box design and table look."""
        from runtime.isaac_runtime import EpisodeReference
        icfg = self.cfg["isaac_mode"]
        ds = icfg.get("dataset_start", {}) or {}
        paths = episode_paths(self._res(ds["dataset_dir"]), name)
        self.episode = EpisodeReference(paths["yaml"])
        self.episode_paths = paths
        f0 = int(start_frame)
        self.start_frame = f0
        self.phone_yaw_offset = math.radians(float(ds.get("phone_yaw_offset_deg", 0.0)))
        rt = self.runtime
        rt._held, rt._fall_active = None, False
        self._vacuum, self._released, self.suction_events = False, False, []
        rt.set_phone_pose(self.episode.phone_pose(f0), yaw_offset_rad=self.phone_yaw_offset)
        rt.apply_appearance(icfg.get("appearance"), self.episode.phone_material, self._res("."),
                            self.episode.materials)
        ov = phone_override or {}
        if ov.get("xy") is not None or ov.get("yaw_deg") is not None:
            pose = self.episode.phone_pose(f0).copy()
            if ov.get("xy") is not None:
                pose[0:2] = [float(v) for v in ov["xy"]]
            if ov.get("yaw_deg") is not None:
                pose[5] = math.radians(float(ov["yaw_deg"]))
            rt.set_phone_pose(pose, yaw_offset_rad=self.phone_yaw_offset)
            print(f"[isaac-backend] PHONE OVERRIDE: start pose xy={pose[:2].round(4).tolist()} "
                  f"yaw={math.degrees(pose[5]):.1f} deg (episode had "
                  f"{self.episode.phone_pose(f0)[:2].round(4).tolist()})")
        self._grasp_offset = self._grasp_offset_from_episode()
        rt.apply_home_pose(q46=self.episode.q46(f0))
        self._current_joint_targets = rt.home_pose.copy()
        for _ in range(3):
            rt.step()
        print(f"[isaac-backend] start = {self.episode.name} frame {f0}: "
              f"head=({self.episode.q46(f0)[0]:+.4f}, {self.episode.q46(f0)[1]:+.4f}) rad, "
              f"phone xyz={np.round(rt.phone_object_pose()[:3, 3], 4).tolist()} "
              f"({self.episode.phone_material}, table {self.episode.materials.get('Wooden_interact_table')})")

    # ------------------------------------------------------------ helpers
    def _set_viewport_view(self) -> None:
        if self.headless:
            return
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=[1.55, -1.25, 1.15], target=[0.35, -0.05, 0.15],
                            camera_prim_path="/OmniverseKit_Persp")
        except Exception as exc:
            print(f"[isaac-backend] viewport camera not set ({exc})")

    def _hand_xyz(self, q: np.ndarray) -> np.ndarray:
        if self.seat is not None:
            return np.asarray(self.seat.hand_in_base(q), float)
        return self.runtime.hand_world()[:3, 3].copy()

    def _grasp_offset_from_episode(self):
        """Phone box-centre pose in the suction-link frame at the demo's vacuum-on frame."""
        on, _ = self.episode.suction_frames()
        if on is None:
            return None
        from runtime.isaac_runtime import _pose_matrix
        self.runtime._set_joint_positions_raw(self.episode.q46(on), render=False)
        M = self.runtime.hand_world()
        off = np.linalg.inv(M) @ _pose_matrix(self.episode.phone_pose(on))
        print(f"[isaac-backend] suction grasp offset (demo frame {on}) in suction frame: "
              f"{np.round(off[:3, 3], 4).tolist()} m")
        return off

    # ------------------------------------------------------------ suction
    def _update_suction(self, step: int) -> None:
        scfg = self.cfg.get("suction", {}) or {}
        mode = str(scfg.get("mode", "proximity"))
        rt = self.runtime
        if mode == "schedule" and self.episode is not None:
            on, off = self.episode.suction_frames()
            if not self._vacuum and not self._released and on is not None and step >= on:
                self.fire_vacuum(reason=f"schedule frame {on}", step=step)
            elif self._vacuum and off is not None and step >= off:
                self.release_vacuum(reason=f"schedule frame {off}", step=step)
            return
        if mode != "proximity":
            return
        if not self._vacuum and not self._released and self._grasp_offset is not None:
            target = (rt.hand_world() @ self._grasp_offset)[:3, 3]  # where a grasped phone centre would be
            phone = rt.phone_object_pose()[:3, 3]
            dxy = float(np.linalg.norm(target[:2] - phone[:2]))
            dz = float(target[2] - phone[2])      # > 0: cup still above the phone top
            if (dxy <= float(scfg.get("attach_xy_tolerance_m", 0.03))
                    and dz <= float(scfg.get("attach_z_tolerance_m", 0.006))):
                self.fire_vacuum(reason=f"proximity dxy={dxy:.4f} dz={dz:+.4f}", step=step)
        elif self._vacuum:
            bb = rt.bbox_world(rt.container_prim)
            p = rt.phone_object_pose()[:3, 3]
            m = float(scfg.get("release_inside_margin_m", 0.0))
            inside = bb is not None and (bb[0][0] + m <= p[0] <= bb[1][0] - m) and (bb[0][1] + m <= p[1] <= bb[1][1] - m)
            if inside and p[2] <= float(scfg.get("release_max_z_m", 0.10)):
                self.release_vacuum(reason=f"over box, centre z={p[2]:.4f}", step=step)

    # ------------------------------------------------------------ iDP3 path
    def apply_joint_targets(self, q46: np.ndarray, step: int) -> None:
        q46 = np.asarray(q46, float)
        self.runtime.drive(q46)
        self._current_joint_targets = q46.copy()
        self._update_suction(step)
        self.runtime._step_fall_if_any()

    # ------------------------------------------------------------ Backend API
    def read_camera(self) -> tuple[np.ndarray, np.ndarray]:
        rgb, depth = self.runtime.read_camera()
        self._last_frame = (rgb, depth)
        return rgb, depth

    def get_camera_intrinsic(self):
        try:
            return self.runtime.camera_K()
        except Exception:
            from perception.intrinsics import render_K
            return render_K()

    def get_joints(self) -> dict[str, float]:
        return self.runtime.read_joints()

    def get_hand_xyz(self) -> np.ndarray:
        return self._hand_xyz(self._current_joint_targets).copy()

    def plan_trajectory(self, x: float, y: float, yaw: float) -> np.ndarray:
        if self.seat is None:
            raise RuntimeError("plan_trajectory needs a PerceptionSeat (tomo_brain_inference)")
        return self.seat.trajectory(x, y, yaw)

    def drive_next_steps(self, trajectory: np.ndarray, start: int, steps: int, delta_clamp: float):
        n = int(min(steps, trajectory.shape[0] - start))
        prev = self._current_joint_targets.copy()
        clamped = 0
        for k in range(n):
            target = np.asarray(trajectory[start + k], float)
            if float(np.abs(target - prev).max()) > delta_clamp:
                clamped += 1
            self.runtime.drive(target)
            prev = target
        self._current_joint_targets = prev
        self._current_hand_xyz = self._hand_xyz(prev)
        return n, clamped, self._current_hand_xyz.copy()

    def drive_and_grasp(self, trajectory, start, delta_clamp, grasp_target, tolerance_m) -> dict:
        n_total = int(trajectory.shape[0] - start)
        prev = self._current_joint_targets.copy()
        clamped, fired, min_d = 0, -1, float("inf")
        tgt = np.asarray(grasp_target, float)
        for k in range(max(0, n_total)):
            target = np.asarray(trajectory[start + k], float)
            if float(np.abs(target - prev).max()) > delta_clamp:
                clamped += 1
            self.runtime.drive(target)
            prev = target
            d = float(np.linalg.norm(self._hand_xyz(prev) - tgt))
            min_d = min(min_d, d)
            if not self._vacuum and d <= float(tolerance_m):
                self.fire_vacuum(reason="drive_and_grasp", step=start + k)
                fired = start + k
        self._current_joint_targets = prev
        self._current_hand_xyz = self._hand_xyz(prev)
        return {"steps_driven": max(0, n_total), "clamped_ticks": clamped, "grasped": self._vacuum,
                "fired_at_frame": fired,
                "min_real_distance_cm": None if min_d == float("inf") else round(min_d * 100, 3),
                "final_hand_xyz": self._current_hand_xyz.tolist(), "traj_index_final": start + max(0, n_total)}

    def fire_vacuum(self, reason: str = "manual", step: int | None = None) -> None:
        self.runtime.attach_box(self.runtime.hand_world())
        self._vacuum = True
        ev = {"event": "vacuum_on", "step": step, "reason": reason,
              "phone_xyz": self.runtime.phone_object_pose()[:3, 3].round(4).tolist()}
        self.suction_events.append(ev)
        print(f"[suction] ON  at step {step}: {reason}")

    def release_vacuum(self, reason: str = "manual", step: int | None = None) -> None:
        self._vacuum = False
        self._released = True
        rest_z = float((self.cfg.get("suction", {}) or {}).get("rest_z_in_box_m", 0.0524))
        self.runtime.start_fall(ground_z=rest_z)
        ev = {"event": "vacuum_off", "step": step, "reason": reason,
              "phone_xyz": self.runtime.phone_object_pose()[:3, 3].round(4).tolist()}
        self.suction_events.append(ev)
        print(f"[suction] OFF at step {step}: {reason}")

    def carry_object(self) -> None:
        if self._vacuum:
            self.runtime.carry_box(self.runtime.hand_world())

    def visualize_frame_for_qwen(self, rgb: np.ndarray):
        from PIL import Image
        pil = Image.fromarray(rgb.astype(np.uint8))
        pil.thumbnail((448, 448))
        return pil

    def pump(self) -> None:
        try:
            self.sim_app.update()
        except Exception:
            pass

    def get_latest_frame_nonblocking(self):
        return self._last_frame

    def close(self) -> None:
        try:
            self.runtime.close()
        except Exception:
            pass


# ============================================================================ policy client
class Idp3Client:
    def __init__(self, pcfg: dict, timeout_s: float = 120.0):
        from multiprocessing.connection import Client
        addr = (pcfg.get("host", "127.0.0.1"), int(pcfg.get("port", 6010)))
        key = str(pcfg.get("authkey", "tomo")).encode()
        t0 = time.time()
        while True:
            try:
                self.conn = Client(addr, authkey=key)
                break
            except (ConnectionRefusedError, OSError):
                if time.time() - t0 > timeout_s:
                    raise RuntimeError(
                        f"iDP3 policy server not reachable on {addr[0]}:{addr[1]}. Start it with\n"
                        f"  {pcfg.get('python', 'python')} idp3_policy_server.py --config config.yaml")
                time.sleep(1.0)
        self.info = self._call({"cmd": "hello"})
        print(f"[idp3] connected: {self.info['checkpoint']} step={self.info['step']} "
              f"weights={self.info['evaluated_weights']} device={self.info['device']}")

    def _call(self, msg: dict) -> dict:
        self.conn.send(msg)
        reply = self.conn.recv()
        if not reply.get("ok"):
            raise RuntimeError(f"policy server error: {reply.get('error')}")
        return reply

    def reset(self):
        self._call({"cmd": "reset"})

    def step(self, rgb, depth, q, frame_id: int, infer: bool, extras: dict | None = None) -> dict:
        msg = {"cmd": "infer" if infer else "observe", "frame_id": int(frame_id),
               "rgb": pack(rgb), "depth": pack(depth.astype(np.float32)), "q": [float(x) for x in q]}
        if extras is not None:
            msg["extras"] = extras
        r = self._call(msg)
        if "horizon" in r:
            r["horizon"] = unpack(r["horizon"])
        return r

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# ============================================================================ closed loop
def depth_colormap(d, lo=0.15, hi=3.0):
    import cv2
    v = np.isfinite(d) & (d >= lo) & (d <= hi)
    g = np.zeros(d.shape, np.uint8)
    g[v] = (np.clip((d[v] - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)
    h = cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    h[~v] = 0
    return h


class TrajectoryViz:
    """Outside view of the rollout with the policy's trajectory drawn in 3D.

    The tracked point is the GRASP POINT: where the phone centre sits when the
    suction cup holds it (demo frame-155 offset in the RF_link_52 frame). In the
    demo it lands exactly on the phone at frame 155 and then goes to the box.
      blue   = recorded demo path of the grasp point
      orange = path the policy actually executed so far
      green  = the policy's CURRENT 16-step plan (remaining rows)
      red    = phone centre
    """
    DEMO, EXEC, PLAN, PHONE = (255, 170, 40), (0, 140, 255), (60, 220, 60), (60, 60, 255)

    def __init__(self, backend: "IsaacBackend", out: Path, demo_q: np.ndarray | None):
        import cv2
        self.cv2, self.b, self.rt = cv2, backend, backend.runtime
        self.G = backend._grasp_offset if backend._grasp_offset is not None else np.eye(4)
        self.suction = self.rt.suction_prim or self.rt._HAND_PATH_DEFAULT
        self.demo = None if demo_q is None else np.stack([self.gp(q) for q in demo_q])
        self.exec_pts, self.video, self.out = [], None, out
        self.box = self.rt.bbox_world(self.rt.container_prim)

    def gp(self, q) -> np.ndarray:
        return (self.rt.fk_world(q, self.suction) @ self.G)[:3, 3]

    def _poly(self, img, P, color, thick):
        uv = self.rt.observer_project(P)
        ok = np.isfinite(uv).all(1)
        if ok.sum() >= 2:
            self.cv2.polylines(img, [uv[ok].astype(np.int32).reshape(-1, 1, 2)], False, color, thick,
                               self.cv2.LINE_AA)

    def frame(self, t: int, q, plan, head_rgb, vacuum: bool, phone_xyz):
        cv2 = self.cv2
        obs = self.rt.read_observer()
        gp_now = self.gp(q)
        self.exec_pts.append(gp_now)
        if obs is None:
            return
        img = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        if self.demo is not None:
            self._poly(img, self.demo, self.DEMO, 2)
        self._poly(img, np.array(self.exec_pts), self.EXEC, 3)
        if plan is not None and len(plan):
            P = np.stack([self.gp(r) for r in plan])
            for (u, v) in self.rt.observer_project(P):
                if np.isfinite(u):
                    cv2.circle(img, (int(u), int(v)), 4, self.PLAN, -1, cv2.LINE_AA)
        u, v = self.rt.observer_project(phone_xyz)[0]
        if np.isfinite(u):
            cv2.circle(img, (int(u), int(v)), 9, self.PHONE, 2, cv2.LINE_AA)
        u, v = self.rt.observer_project(gp_now)[0]
        if np.isfinite(u):
            cv2.circle(img, (int(u), int(v)), 6, self.EXEC, -1, cv2.LINE_AA)
        inset = cv2.resize(cv2.cvtColor(head_rgb, cv2.COLOR_RGB2BGR), (320, 240))
        cv2.rectangle(inset, (0, 0), (319, 239), (255, 255, 255), 2)
        img[10:250, img.shape[1] - 330:img.shape[1] - 10] = inset
        d = float(np.linalg.norm(gp_now - np.asarray(phone_xyz)))
        lines = [f"t={t:3d}  vacuum={'ON' if vacuum else 'off'}  grasp-point to phone = {100 * d:5.1f} cm",
                 "blue: demo path   orange: policy path   green: policy 16-step plan   red: phone",
                 "inset: head camera (what the model sees)"]
        for i, s in enumerate(lines):
            y = img.shape[0] - 70 + 24 * i
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(img, s, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 1, cv2.LINE_AA)
        if self.video is None:
            self.video = cv2.VideoWriter(str(self.out / "trajectory_view.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                         30.0, (img.shape[1], img.shape[0]))
        self.video.write(img)
        if t in (0, 30, 60, 100, 140, 155, 200, 250, 300):
            cv2.imwrite(str(self.out / "frames" / f"trajectory_t{t:03d}.png"), img)

    def finish(self, phone_start_xyz) -> dict:
        if self.video is not None:
            self.video.release()
        E = np.array(self.exec_pts)
        ph = np.asarray(phone_start_xyz, float)
        lo, hi = 100, min(len(E), 220)
        i_low = lo + int(np.argmin(E[lo:hi, 2])) if hi > lo else int(np.argmin(E[:, 2]))
        dist = np.linalg.norm(E[:, :2] - ph[:2], axis=1) + np.abs(E[:, 2] - ph[2])
        i_close = int(np.argmin(dist[: min(len(E), 230)]))
        stats = {"reach_xy_at_lowest_point": E[i_low, :2].round(4).tolist(), "reach_lowest_t": i_low,
                 "reach_lowest_dz_above_phone_m": round(float(E[i_low, 2] - ph[2]), 4),
                 "closest_t": i_close,
                 "closest_dxy_cm": round(100 * float(np.linalg.norm(E[i_close, :2] - ph[:2])), 2),
                 "closest_dz_cm": round(100 * float(E[i_close, 2] - ph[2]), 2)}
        self._plot(E, ph)
        return stats

    def _plot(self, E, ph):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.patches import Rectangle
        except Exception as exc:
            print(f"[plot] skipped: {exc}")
            return
        fig, ax = plt.subplots(1, 2, figsize=(14, 6))
        for a, (i, j, lab) in zip(ax, [(1, 0, "top view"), (0, 2, "side view")]):
            if self.demo is not None:
                a.plot(self.demo[:, i], self.demo[:, j], "-", color="tab:blue", lw=2, label="recorded demo")
                a.plot(self.demo[155, i], self.demo[155, j], "o", color="tab:blue", ms=8)
            a.plot(E[:, i], E[:, j], "-", color="tab:orange", lw=2, label="policy in Isaac")
            a.plot(ph[i], ph[j], "*", color="red", ms=16, label="phone start")
            if self.box is not None and lab == "top view":
                mn, mx = self.box
                a.add_patch(Rectangle((mn[1], mn[0]), mx[1] - mn[1], mx[0] - mn[0], fill=False,
                                      ec="saddlebrown", lw=2, label="cardboard box"))
            a.set_title(f"grasp-point path, {lab}")
            a.grid(alpha=0.3)
            a.legend(fontsize=8)
        ax[0].set_xlabel("y [m] (robot left +)")
        ax[0].set_ylabel("x [m] (forward)")
        ax[0].invert_xaxis()
        ax[0].set_aspect("equal")
        ax[1].set_xlabel("x [m] (forward)")
        ax[1].set_ylabel("z [m] (up)")
        fig.tight_layout()
        fig.savefig(self.out / "trajectory_plot.png", dpi=130)
        plt.close(fig)


def run_teacher_forced(backend: IsaacBackend, client: Idp3Client, cfg: dict, out: Path) -> dict:
    """Diagnostic: replay the RECORDED demo in Isaac (robot + phone) and ask the
    policy for its prediction at every frame, fed the recorded q.  Error vs the
    recorded next frames isolates the Isaac-vs-Blender visual gap from
    closed-loop drift.  Compare with the dataset numbers in best_val.pt
    (validation next-step L1 ~0.0005 rad head, ~0.005 rad right arm)."""
    ep = backend.episode
    if ep is None:
        raise RuntimeError("teacher-forced mode needs isaac_mode.dataset_start")
    out.mkdir(parents=True, exist_ok=True)
    steps = min(int(cfg["idp3_policy"].get("max_steps", 306)), ep.n_frames - 17)
    client.reset()
    rows, H = [], []
    yaw = backend.phone_yaw_offset
    import cv2
    video = None
    for t in range(steps):
        backend.runtime.set_phone_pose(ep.phone_pose(t), yaw_offset_rad=yaw)
        backend.runtime.drive(ep.q46(t))
        rgb, depth = backend.read_camera()
        tile = cv2.resize(cv2.hconcat([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_colormap(depth)]),
                          (960, 360), interpolation=cv2.INTER_AREA)
        if video is None:
            video = cv2.VideoWriter(str(out / "isaac_rgbd_rollout.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                    30.0, (960, 360))
        video.write(tile)
        if cfg.get("_save_tf_frames"):
            (out / "isaac_frames").mkdir(exist_ok=True)
            np.savez_compressed(out / "isaac_frames" / f"f{t:03d}.npz", rgb=rgb, depth=depth)
        r = client.step(rgb, depth, ep.q46(t), frame_id=t, infer=True)
        h = r["horizon"]
        H.append(h)
        fut = np.stack([ep.q46(t + 1 + i) for i in range(16)])
        e1, e16 = np.abs(h[0] - fut[0]), np.abs(h - fut).mean(0)
        rows.append({"t": t, "next_head": float(e1[:2].mean()), "next_right_arm": float(e1[9:16].mean()),
                     "next_both_arms": float(e1[2:16].mean()), "chunk_right_arm": float(e16[9:16].mean()),
                     "chunk_head": float(e16[:2].mean())})
        if t % 30 == 0:
            print(f"[teacher] t={t:3d} next-step L1 head={e1[:2].mean():.5f} right_arm={e1[9:16].mean():.5f} "
                  f"| 16-step chunk right_arm={e16[9:16].mean():.5f}")
    if video is not None:
        video.release()
    keys = [k for k in rows[0] if k != "t"]
    summary = {"mode": "teacher_forced", "episode_start": ep.name, "episode": ep.name, "frames": steps,
               "mean_L1_rad": {k: round(float(np.mean([r[k] for r in rows])), 6) for k in keys},
               "checkpoint_validation_macro_next_step_L1_rad": "head 0.00048, right_arm 0.0046, both_arms 0.0025 (dataset visuals)"}
    np.save(out / "teacher_forced_horizons.npy", np.stack(H))
    with (out / "teacher_forced_log.jsonl").open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("[done] " + json.dumps(summary, indent=2))
    return summary


def run_closed_loop(backend: IsaacBackend, client: Idp3Client, cfg: dict, out: Path) -> dict:
    import cv2
    pcfg = cfg["idp3_policy"]
    lcfg = cfg.get("logging", {}) or {}
    H_exec = int(pcfg.get("replan_every", 1))
    if not 1 <= H_exec <= 16:
        raise ValueError("idp3_policy.replan_every must be 1..16")
    max_steps = int(pcfg.get("max_steps", 306))
    jump_warn = float(pcfg.get("max_joint_step_rad", 0.20))
    save_every = int(lcfg.get("save_frames_every", 1))
    train_min = np.asarray(client.info["train_min"])
    train_max = np.asarray(client.info["train_max"])
    names = backend.joint_names
    out.mkdir(parents=True, exist_ok=True)
    (out / "frames").mkdir(exist_ok=True)

    ep = backend.episode
    f0 = backend.start_frame if ep is not None else 0
    demo_q = np.stack([ep.q46(min(f0 + t, ep.n_frames - 1)) for t in range(max_steps + 1)]) if ep else None

    viz = TrajectoryViz(backend, out, demo_q) if getattr(backend.runtime, "_obs_annot", None) is not None else None
    phone_start = backend.runtime.phone_object_pose()[:3, 3].copy()
    _R = backend.runtime.phone_object_pose()
    phone_start_pose = [float(v) for v in phone_start] + [float(math.atan2(_R[1, 0], _R[0, 0]))]
    phone_source = str(pcfg.get("phone_source", "gt"))
    last_phone_src = None

    q = backend._current_joint_targets.copy()
    executed = [q.copy()]
    horizons, replans, log_rows = [], [], []
    horizon, k = None, 16
    client.reset()
    video = None
    t_start = time.time()
    for t in range(max_steps):
        rgb, depth = backend.read_camera()
        do_infer = horizon is None or k >= H_exec
        extras = {"camera_world": backend.runtime.camera_world().reshape(-1).tolist(),
                  "K": backend.runtime.camera_K().reshape(-1).tolist(),
                  "phone_gt_start": phone_start_pose, "phone_source": phone_source}
        r = client.step(rgb, depth, q, frame_id=f0 + t, infer=do_infer, extras=extras)
        if r.get("phone_used") and (t == 0 or r["phone_used"][1] != last_phone_src):
            last_phone_src = r["phone_used"][1]
            print(f"[loop] t={t}: policy phone input {np.round(r['phone_used'][0], 3).tolist()} ({last_phone_src})")
        if do_infer:
            horizon, k = r["horizon"], 0
            horizons.append(horizon)
            replans.append(t)
        if viz is not None:
            viz.frame(t, q, horizon[k:], rgb, backend._vacuum, backend.runtime.phone_object_pose()[:3, 3])
        q_next = horizon[k].astype(np.float64)
        k += 1
        dq = float(np.abs(q_next - q).max())
        backend.apply_joint_targets(q_next, step=f0 + t + 1)
        q = q_next
        executed.append(q.copy())

        cs = r.get("cloud_stats", {})
        fwd_z = float(-backend.runtime.camera_world()[2, 2])
        head_pitch = round(math.degrees(math.asin(max(-1.0, min(1.0, fwd_z)))), 3)
        row = {"t": t + 1, "replanned": do_infer, "infer_ms": r.get("infer_ms"),
               "max_dq_rad": round(dq, 5), "jump": dq > jump_warn,
               "head_j1": round(float(q[0]), 5), "head_j2": round(float(q[1]), 5),
               "camera_pitch_deg": head_pitch,
               "cloud_valid_points": cs.get("valid_grid_points"), "cloud_median_z": cs.get("median_z_m"),
               "vacuum": backend._vacuum,
               "phone_xyz": backend.runtime.phone_object_pose()[:3, 3].round(4).tolist(),
               "outside_train_range": int(((q < train_min - 1e-4) | (q > train_max + 1e-4)).sum())}
        if demo_q is not None:
            row["abs_err_vs_demo_head"] = round(float(np.abs(q[:2] - demo_q[t + 1, :2]).mean()), 5)
            row["abs_err_vs_demo_arms"] = round(float(np.abs(q[2:16] - demo_q[t + 1, 2:16]).mean()), 5)
        log_rows.append(row)
        if dq > jump_warn:
            print(f"[WARN] t={t + 1}: max |dq| = {dq:.3f} rad")
        if t < 3 or (t + 1) % 15 == 0 or t == max_steps - 1:
            print(f"[loop] t={t + 1:3d}/{max_steps} head=({q[0]:+.3f},{q[1]:+.3f}) "
                  f"pitch={head_pitch:+6.1f}deg cloud={cs.get('valid_grid_points')} "
                  f"infer={str(r['infer_ms']) + 'ms' if r.get('infer_ms') else '-'} vac={int(backend._vacuum)} "
                  f"phone={row['phone_xyz']}")

        if save_every and (t % save_every == 0):
            tile = cv2.hconcat([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_colormap(depth)])
            tile = cv2.resize(tile, (960, 360), interpolation=cv2.INTER_AREA)
            cv2.putText(tile, f"t={t}  head=({executed[-2][0]:+.2f},{executed[-2][1]:+.2f})  "
                        f"vac={int(backend._vacuum)}", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if video is None:
                video = cv2.VideoWriter(str(out / "isaac_rgbd_rollout.mp4"),
                                        cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (tile.shape[1], tile.shape[0]))
            video.write(tile)
            if t in (0, 10, 20, 30, 50) or t % 50 == 0:
                cv2.imwrite(str(out / "frames" / f"t{t:03d}.png"), tile)
        if lcfg.get("save_depth_npy"):
            np.save(out / "frames" / f"depth_{t:03d}.npy", depth)

    # final frame
    rgb, depth = backend.read_camera()
    tile = cv2.resize(cv2.hconcat([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_colormap(depth)]), (960, 360))
    cv2.imwrite(str(out / "frames" / f"t{max_steps:03d}_final.png"), tile)
    if video is not None:
        video.write(tile)
        video.release()

    executed = np.stack(executed)
    np.save(out / "executed_q46.npy", executed)
    np.save(out / "horizons_16x46.npy", np.stack(horizons))
    np.save(out / "replan_steps.npy", np.asarray(replans))
    with (out / "executed_q46.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["t"] + names)
        for i, row in enumerate(executed):
            w.writerow([i] + [f"{v:.6f}" for v in row])
    with (out / "step_log.jsonl").open("w") as fh:
        for row in log_rows:
            fh.write(json.dumps(row) + "\n")

    rt = backend.runtime
    phone = rt.phone_object_pose()[:3, 3]
    bb = rt.bbox_world(rt.container_prim)
    in_box = bool(bb is not None and bb[0][0] <= phone[0] <= bb[1][0] and bb[0][1] <= phone[1] <= bb[1][1]
                  and phone[2] < 0.12 and not rt.holding)
    summary = {
        "episode_start": ep.name if ep else None, "start_frame": f0, "steps": max_steps,
        "replan_every": H_exec, "policy_calls": len(replans),
        "wall_time_s": round(time.time() - t_start, 1),
        "mean_infer_ms": round(float(np.mean([r["infer_ms"] for r in log_rows if r["infer_ms"]])), 2),
        "head_start_rad": executed[0, :2].round(4).tolist(),
        "head_min_joint2_rad": float(executed[:, 1].min().round(4)),
        "camera_pitch_first_last_deg": [log_rows[0]["camera_pitch_deg"], log_rows[-1]["camera_pitch_deg"]],
        "frames_head_below_minus_0p5rad": int((executed[:, 1] < -0.5).sum()),
        "joint_jumps_over_warn": int(sum(r["jump"] for r in log_rows)),
        "suction_events": backend.suction_events,
        "final_phone_xyz": phone.round(4).tolist(),
        "demo_final_phone_xyz": ep.phone_pose(ep.n_frames - 1)[:3].round(4).tolist() if ep else None,
        "phone_in_box": in_box,
        "checkpoint": client.info["checkpoint"],
        "phone_start_xyz": phone_start.round(4).tolist(),
        "trajectory": viz.finish(phone_start) if viz is not None else None,
        "physics": {"active_physics_scenes": rt.active_physics_scenes(),
                    "timeline_time_s": rt.diagnostics()["timeline_time_s"]},
    }
    if demo_q is not None:
        err = np.abs(executed - demo_q[: len(executed)])
        summary["mae_vs_demo_rad"] = {"head": float(err[:, :2].mean().round(5)),
                                      "left_arm": float(err[:, 2:9].mean().round(5)),
                                      "right_arm": float(err[:, 9:16].mean().round(5)),
                                      "fingers": float(err[:, 16:].mean().round(5))}
        summary["note_mae"] = ("closed-loop drift vs the recorded demo that shares this start state; "
                               "not a success metric by itself")
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    _plot(out, executed, demo_q, names)
    print("[done] " + json.dumps(summary, indent=2))
    return summary


def _plot(out: Path, executed, demo_q, names):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
        return
    t = np.arange(len(executed))
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (title, cols) in zip(axes, [("head", range(0, 2)), ("right arm", range(9, 16)),
                                        ("left arm", range(2, 9))]):
        for j in cols:
            line, = ax.plot(t, executed[:, j], label=names[j])
            if demo_q is not None:
                ax.plot(t, demo_q[: len(t), j], "--", color=line.get_color(), alpha=0.45)
        ax.set_title(f"{title}: Isaac closed loop (solid) vs recorded demo (dashed)")
        ax.set_ylabel("rad")
        ax.grid(alpha=0.25)
        ax.legend(ncol=4, fontsize=7)
    axes[-1].set_xlabel("step (30 Hz)")
    fig.tight_layout()
    fig.savefig(out / "joints_vs_demo.png", dpi=140)
    plt.close(fig)


def run_sweep(backend: IsaacBackend, client: Idp3Client, cfg: dict, out: Path, episodes: list[str],
              model: str, val_labels: set[int]) -> dict:
    """Closed loop on several recorded episodes (different phone positions /
    yaws / phone designs / tables) in ONE Isaac session."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in episodes:
        print(f"\n[sweep] ===== {name} =====")
        backend.load_episode(name, 0, None)
        s = run_closed_loop(backend, client, cfg, out / name)
        n = int(name.split("_")[-1])
        tr = s.get("trajectory") or {}
        rows.append({"episode": name, "split": "VAL" if n in val_labels else "train",
                     "phone_xy": s["phone_start_xyz"][:2],
                     "phone_yaw_deg": round(math.degrees(backend.episode.phone_pose(0)[5]), 1),
                     "phone_design": backend.episode.phone_material,
                     "table": backend.episode.materials.get("Wooden_interact_table"),
                     "reach_xy": tr.get("reach_xy_at_lowest_point"),
                     "closest_dxy_cm": tr.get("closest_dxy_cm"), "closest_dz_cm": tr.get("closest_dz_cm"),
                     "grasped": any(e["event"] == "vacuum_on" for e in s["suction_events"]),
                     "phone_in_box": bool(s["phone_in_box"]),
                     "final_phone_xyz": s["final_phone_xyz"],
                     "mae_vs_demo_rad": s.get("mae_vs_demo_rad")})
        r = rows[-1]
        print(f"[sweep] {name} ({r['split']}): phone {r['phone_xy']} reach {r['reach_xy']} "
              f"closest {r['closest_dxy_cm']} cm | grasped={r['grasped']} in_box={r['phone_in_box']}")
    P = np.array([r["phone_xy"] for r in rows], float)
    Rch = np.array([r["reach_xy"] for r in rows], float)
    # how much does the reach follow the phone? least-squares slope per axis
    def slope(a, b):
        a, b = a - a.mean(), b - b.mean()
        return float((a @ b) / (a @ a)) if (a @ a) > 1e-12 else float("nan")
    summary = {"model": model, "episodes": len(rows),
               "grasp_rate": round(float(np.mean([r["grasped"] for r in rows])), 3),
               "in_box_rate": round(float(np.mean([r["phone_in_box"] for r in rows])), 3),
               "median_closest_dxy_cm": round(float(np.median([r["closest_dxy_cm"] for r in rows])), 2),
               "phone_xy_std_cm": (100 * P.std(0)).round(2).tolist(),
               "reach_xy_std_cm": (100 * Rch.std(0)).round(2).tolist(),
               "reach_follows_phone_slope_xy": [round(slope(P[:, 0], Rch[:, 0]), 3), round(slope(P[:, 1], Rch[:, 1]), 3)],
               "note": "slope 1.0 = reach moves with the phone; 0.0 = reach ignores the phone",
               "runs": rows}
    (out / "sweep_summary.json").write_text(json.dumps(summary, indent=2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 7))
        for r in rows:
            c = "tab:green" if r["phone_in_box"] else ("tab:orange" if r["grasped"] else "tab:red")
            ax.plot(r["phone_xy"][1], r["phone_xy"][0], "*", color=c, ms=15)
            ax.plot(r["reach_xy"][1], r["reach_xy"][0], "o", color=c, ms=7, mfc="none", mew=2)
            ax.annotate("", xy=(r["reach_xy"][1], r["reach_xy"][0]), xytext=(r["phone_xy"][1], r["phone_xy"][0]),
                        arrowprops=dict(arrowstyle="->", color=c, alpha=0.6))
            ax.text(r["phone_xy"][1] + 0.004, r["phone_xy"][0] + 0.004,
                    r["episode"][-3:] + (" (val)" if r["split"] == "VAL" else ""), fontsize=8)
        ax.set_xlabel("y [m] (robot left +)")
        ax.set_ylabel("x [m] (forward)")
        ax.invert_xaxis()
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"{model}: phone start (star) -> where the hand came down (circle)\n"
                     f"green = placed in box, orange = grasped only, red = missed | "
                     f"in-box {summary['in_box_rate']:.0%}, slope x/y {summary['reach_follows_phone_slope_xy']}")
        fig.tight_layout()
        fig.savefig(out / "sweep_plot.png", dpi=130)
        plt.close(fig)
    except Exception as exc:
        print(f"[plot] skipped: {exc}")
    print("[sweep] " + json.dumps({k: v for k, v in summary.items() if k != "runs"}, indent=2))
    return summary


def run_view(backend: IsaacBackend, cfg: dict, out: Path, replay: bool) -> None:
    """No trained model needed. --view: open the scene and idle.
    --replay: play the recorded demo joints (robot + phone + kinematic suction)."""
    ep, rt = backend.episode, backend.runtime
    if not replay:
        print("[view] scene open - close the window to exit")
        while backend.sim_app.is_running():
            rt.step()
        return
    import cv2
    out.mkdir(parents=True, exist_ok=True)
    print(f"[replay] playing recorded {ep.name} ({ep.n_frames} frames, 30 Hz)")
    video = None
    for t in range(ep.n_frames):
        if not backend.sim_app.is_running():
            return
        backend.apply_joint_targets(ep.q46(t), step=t)
        rgb, depth = backend.read_camera()
        tile = cv2.resize(cv2.hconcat([cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), depth_colormap(depth)]), (960, 360))
        if video is None:
            video = cv2.VideoWriter(str(out / "replay_head_camera.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (960, 360))
        video.write(tile)
        time.sleep(1 / 30)
    if video is not None:
        video.release()
    print(f"[replay] done; suction events {backend.suction_events}; video {out / 'replay_head_camera.mp4'}")


def phone_positions(n: int, seed: int) -> list:
    """n phone poses spread over the dataset range: a 5-column grid (x 0.455-0.630,
    y -0.505..-0.325) with a random yaw 45-128 deg per spot."""
    rng = np.random.default_rng(seed)
    xs, ys = np.linspace(0.455, 0.630, 5), np.linspace(-0.505, -0.325, max(1, -(-n // 5)))
    grid = [(x, y) for y in ys for x in xs][:n]
    return [(float(x), float(y), float(round(rng.uniform(45.0, 128.0), 1))) for x, y in grid]


def run_positions(backend: IsaacBackend, client: Idp3Client, cfg: dict, out: Path, n: int, seed: int,
                  model: str, pause_s: float) -> None:
    """Show the model on n phone positions, one after another, in ONE Isaac window."""
    out.mkdir(parents=True, exist_ok=True)
    base_ep = backend.episode.name
    rows = []
    for i, (x, y, yaw) in enumerate(phone_positions(n, seed)):
        print(f"\n[positions] ===== {i + 1}/{n}: phone at x={x:.3f} y={y:.3f} yaw={yaw:.0f} deg =====")
        backend.load_episode(base_ep, 0, {"xy": [x, y], "yaw_deg": yaw})
        t_end = time.time() + pause_s          # let the viewer see the new phone spot
        while time.time() < t_end and backend.sim_app.is_running():
            backend.runtime.step()
        if not backend.sim_app.is_running():
            break
        s = run_closed_loop(backend, client, cfg, out / f"pos{i + 1:02d}")
        tr = s.get("trajectory") or {}
        rows.append({"n": i + 1, "phone_xy": [round(x, 4), round(y, 4)], "yaw_deg": yaw,
                     "hand_came_down_at": tr.get("reach_xy_at_lowest_point"),
                     "miss_cm": tr.get("closest_dxy_cm"),
                     "reached": bool(s["suction_events"]), "in_box": bool(s["phone_in_box"])})
        r = rows[-1]
        print(f"[positions] {i + 1}/{n}: phone {r['phone_xy']} -> hand came down at {r['hand_came_down_at']} "
              f"(miss {r['miss_cm']} cm) reached={r['reached']} in_box={r['in_box']}")
    (out / "positions_summary.json").write_text(json.dumps({"model": model, "runs": rows}, indent=2))
    print("\n[positions] SUMMARY")
    for r in rows:
        print(f"  {r['n']:2d}  phone {r['phone_xy']}  yaw {r['yaw_deg']:5.1f}  -> hand {r['hand_came_down_at']}  "
              f"miss {r['miss_cm']:5.1f} cm  {'REACHED' if r['reached'] else 'missed'}")
    print(f"[positions] reached {sum(r['reached'] for r in rows)}/{len(rows)}; videos + plots in {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--episode", default=None, help="override isaac_mode.dataset_start.episode")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--replan-every", type=int, default=None)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--keep-open", action="store_true", help="leave the GUI open after the rollout")
    ap.add_argument("--teacher-forced", action="store_true",
                    help="diagnostic: replay the recorded demo, only score the policy predictions")
    ap.add_argument("--save-frames", action="store_true", help="teacher-forced: save Isaac RGB-D per frame")
    ap.add_argument("--phone-xy", default=None, help="override phone start x,y [m] (e.g. where the policy reaches)")
    ap.add_argument("--phone-yaw", type=float, default=None, help="override phone start yaw [deg]")
    ap.add_argument("--random-phone", action="store_true",
                    help="random phone x,y,yaw inside the dataset range (x 0.445-0.639, y -0.514..-0.316, yaw 45-128 deg)")
    ap.add_argument("--seed", type=int, default=None, help="seed for --random-phone")
    ap.add_argument("--no-observer", action="store_true", help="skip the outside trajectory view")
    ap.add_argument("--model", choices=["act_rgbd"], default="act_rgbd",
                    help="xyzrgb = idp3colourrgbtraining best_val.pt; depth = ~/Desktop/tomo_idp3 depth-only")
    ap.add_argument("--episodes", default=None, help="comma list: closed loop on each (phone-position sweep)")
    ap.add_argument("--view", action="store_true", help="NO MODEL: just open the scene")
    ap.add_argument("--phone-source", choices=["gt", "yolo"], default=None,
                    help="--model phone: exact phone position from the sim (gt) or YOLO + depth (yolo)")
    ap.add_argument("--replay", action="store_true", help="NO MODEL: play the recorded demo in the scene")
    ap.add_argument("--positions", type=int, default=0, help="show N phone positions one by one (grid over dataset range)")
    ap.add_argument("--pause", type=float, default=3.0, help="--positions: seconds to show each new phone spot first")
    ap.add_argument("--val-episodes", default="4,10,11,36,43", help="episode numbers to label as validation")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg, cfg_dir = load_config(args.config)
    cfg["idp3_policy"] = {**cfg["idp3_policy"], **cfg[f"{args.model}_policy"]}
    if args.phone_source:
        cfg["idp3_policy"]["phone_source"] = args.phone_source
    if args.episode:
        cfg["isaac_mode"]["dataset_start"]["episode"] = args.episode
    if args.steps:
        cfg["idp3_policy"]["max_steps"] = args.steps
    if args.replan_every:
        cfg["idp3_policy"]["replan_every"] = args.replan_every
    if args.headless:
        cfg["isaac_mode"]["headless"], cfg["isaac_mode"]["gui"] = True, False
    cfg["_save_tf_frames"] = bool(args.save_frames)
    if args.random_phone:
        rng = np.random.default_rng(args.seed)
        args.phone_xy = f"{rng.uniform(0.445, 0.639):.4f},{rng.uniform(-0.514, -0.316):.4f}"
        args.phone_yaw = float(round(rng.uniform(45.0, 128.0), 1))
        print(f"[isaac-backend] RANDOM PHONE: xy={args.phone_xy} yaw={args.phone_yaw} deg")
    if args.phone_xy or args.phone_yaw is not None:
        cfg["isaac_mode"]["dataset_start"]["phone_override"] = {
            "xy": [float(v) for v in args.phone_xy.split(",")] if args.phone_xy else None,
            "yaw_deg": args.phone_yaw}
    if args.no_observer:
        cfg.setdefault("visualization", {}).setdefault("observer", {})["enabled"] = False

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ep_name = cfg["isaac_mode"]["dataset_start"]["episode"]
    out = Path(args.out) if args.out else Path(cfg["logging"]["run_dir"]) / f"{stamp}_{ep_name}_{args.model}"

    if args.view or args.replay:                  # no policy server needed
        backend = IsaacBackend(seat=None, config=cfg, config_dir=cfg_dir)
        try:
            run_view(backend, cfg, out, args.replay)
            if args.keep_open and args.replay and not backend.headless:
                print("[done] GUI left open -- close the window to exit")
                while backend.sim_app.is_running():
                    backend.runtime.step()
        finally:
            backend.close()
        return
    client = Idp3Client(cfg["idp3_policy"])       # fail fast before Isaac boots
    backend = IsaacBackend(seat=None, config=cfg, config_dir=cfg_dir)
    try:
        if args.teacher_forced:
            run_teacher_forced(backend, client, cfg, out)
        elif args.positions:
            run_positions(backend, client, cfg, out, args.positions, args.seed or 0, args.model, args.pause)
        elif args.episodes:
            run_sweep(backend, client, cfg, out, [e.strip() for e in args.episodes.split(",") if e.strip()],
                      args.model, {int(x) for x in args.val_episodes.split(",") if x.strip()})
        else:
            run_closed_loop(backend, client, cfg, out)
        print(f"[done] outputs in {out}")
        if args.keep_open and not backend.headless:
            print("[done] GUI left open -- close the window to exit")
            while backend.sim_app.is_running():
                backend.runtime.step()
    finally:
        client.close()
        backend.close()


if __name__ == "__main__":
    main()
