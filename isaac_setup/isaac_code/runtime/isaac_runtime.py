"""TOMO Isaac Sim runtime -- pure-USD kinematic playback, PhysX OFF.

Why pure USD
------------
The TOMO robot in tomo_packing_suction.usd is a NESTED link hierarchy: every
link prim is the child of its parent link, and every joint has
localPos1 = 0 / localRot1 = identity.  A joint angle q is therefore exactly

    child.local = T(localPos0) * R(localRot0) * R_axis(q)

This was checked against Find_and_Pick_001: forward kinematics built this way
reproduces the recorded left/right hand positions to ~9 mm (a constant
prim-origin offset) on every frame 0..306, and projects the recorded iPhone
location onto the phone in the rendered dataset image.  So we write link
transforms directly and never create a PhysX World/Articulation.  Nothing is
simulated, nothing can fall, drift or fight a drive.

Camera
------
The Blender dataset camera IS robot_view_cam under Head_camera_link, with
960x720, fx = fy = 606.3965, cx = 480, cy = 360 (77 deg HFOV).  The USD
authored lens is 18.147 mm / 20.955 mm aperture (60 deg HFOV) -- wrong for the
model.  apply_camera_intrinsics() fixes the focal length so Isaac renders the
same field of view as the training data.  Depth is `distance_to_image_plane`
(optical-axis Z), the same convention as the Blender Depth.V EXRs used for
training.

At dataset frame 0 all head joints are ~0 and the camera optical axis is
level along robot +X ("looking straight").  The head joints then follow the
policy (or the recorded trajectory in check mode) and pitch the view down to
the table exactly as in the demonstrations.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

CANONICAL_46 = (
    ["Head_joint1", "Head_joint2"]
    + [f"Left_arm_joint{i}" for i in range(1, 8)]
    + [f"Right_arm_joint{i}" for i in range(1, 8)]
    + [f"RF_joint_{f}{s}" for f in range(1, 6) for s in range(1, 4)]
    + [f"LF_joint_{f}{s}" for f in range(1, 6) for s in range(1, 4)]
)
assert len(CANONICAL_46) == 46 and len(set(CANONICAL_46)) == 46

PHONE_KEYS = [f"iPhone.location.{a}" for a in "xyz"] + [f"iPhone.rotation.{a}" for a in "xyz"]


# ============================================================================ math
def _gf_to_np(m) -> np.ndarray:
    """OpenUSD row-vector Gf.Matrix4d -> numpy column-vector 4x4."""
    return np.array(m, dtype=float).T


def _np_to_gf(M):
    from pxr import Gf
    return Gf.Matrix4d(np.asarray(M, dtype=float).T.tolist())


def _world(prim) -> np.ndarray:
    from pxr import Usd, UsdGeom
    if not prim or not prim.IsValid() or prim.IsPseudoRoot():
        return np.eye(4)
    return _gf_to_np(UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default()))


def _rot_xyz(rx: float, ry: float, rz: float) -> np.ndarray:
    """Blender 'XYZ' Euler (the mmrt rotation_format): R = Rz @ Ry @ Rx."""
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _pose_matrix(xyz_rpy: Sequence[float]) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = _rot_xyz(*xyz_rpy[3:6])
    M[:3, 3] = xyz_rpy[:3]
    return M


# ============================================================================ episode
class EpisodeReference:
    """Recorded Find_and_Pick_XXX.mmrt.yaml: q46, iPhone pose and io_events."""

    def __init__(self, path: Path):
        import yaml
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Episode YAML missing: {path}")
        d = yaml.safe_load(path.read_text())
        if d.get("metadata", {}).get("angle_units") != "RADIANS":
            raise RuntimeError(f"{path}: angle_units must be RADIANS")
        labels = []
        for group in d["labels"].values():
            labels.extend(str(x) for x in group)
        frames = np.asarray(d["frames"], dtype=np.float64)
        if frames.ndim != 2 or frames.shape[1] != len(labels):
            raise RuntimeError(f"{path}: frame width {frames.shape} != {len(labels)} labels")
        robot = [str(x) for x in d["labels"]["robots"]]
        col = {n: i for i, n in enumerate(labels)}
        missing = [n for n in CANONICAL_46 if n not in robot] + [k for k in PHONE_KEYS if k not in col]
        if missing:
            raise RuntimeError(f"{path}: missing fields {missing}")
        self.path = path
        self.name = path.name.split(".")[0]
        self.fps = float(d["metadata"]["frame_rate"])
        self.frames = frames
        self.n_frames = int(len(frames))
        self._q_cols = [col[n] for n in CANONICAL_46]
        self._phone_cols = [col[k] for k in PHONE_KEYS]
        self.io_events = [list(map(int, e)) for e in d.get("io_events", [])]
        # Blender render randomisation (metadata.py next to the YAML):
        # which iPhone box material / table materials this episode used.
        self.materials = {}
        meta = path.parent / "metadata.py"
        if meta.is_file():
            import ast
            txt = meta.read_text(encoding="utf-8")
            try:
                md = ast.literal_eval(txt.split("=", 1)[1].strip())
                for m in md.get("materials_swapped", []):
                    self.materials[str(m["object_name"])] = str(m["selected_material_name"])
            except Exception as exc:
                print(f"[episode] could not parse {meta}: {exc}")
        self.phone_material = self.materials.get("iPhone", "iPhone_14_pro_max")

    def q46(self, frame: int) -> np.ndarray:
        return self.frames[int(frame), self._q_cols].astype(np.float64)

    def phone_pose(self, frame: int) -> np.ndarray:
        return self.frames[int(frame), self._phone_cols].astype(np.float64)

    def suction_frames(self) -> Tuple[int | None, int | None]:
        """io_events rows are [frame, value, ...]; value != 0 -> vacuum on."""
        on = next((e[0] for e in self.io_events if len(e) > 1 and e[1] != 0), None)
        off = next((e[0] for e in self.io_events if len(e) > 1 and e[1] == 0
                    and (on is None or e[0] > on)), None)
        return on, off


# ============================================================================ runtime
class IsaacRuntime:
    def __init__(
        self,
        map_joint_names: List[str],
        sim_app,
        usd_path: Path,
        robot_root: str,
        camera_prim: str,
        box_prim: str,                      # kept for backward compatibility: the PHONE prim
        cam_wh: Tuple[int, int] = (960, 720),
        physics: bool = False,
        *,
        container_prim: str = "/tomo_re_uhand/Box_1_v1___open",
        phone_mesh_prim: str = "/Iphone/geo/iPhone_14_Pro_Max",
        suction_prim: str | None = None,
        camera_intrinsics: dict | None = None,
        render_updates_per_frame: int = 2,
        scene_fixes: dict | None = None,
    ):
        import omni.usd
        from pxr import Usd, UsdGeom

        if physics:
            raise RuntimeError("This runtime is PhysX-OFF by design. Set isaac_mode.physics: false.")

        self._sim_app = sim_app
        self.map_joint_names = list(map_joint_names)
        if self.map_joint_names != CANONICAL_46:
            raise RuntimeError("Policy joint order is not the 46-D training order:\n  "
                               + "\n  ".join(self.map_joint_names))
        self.physics = False
        self.robot_root = str(robot_root)
        self.camera_prim = str(camera_prim)
        self.phone_prim = str(box_prim)
        self.container_prim = str(container_prim)
        self.phone_mesh_prim = str(phone_mesh_prim)
        self.suction_prim = suction_prim
        self.cam_wh = (int(cam_wh[0]), int(cam_wh[1]))
        self.camera_intrinsics = dict(camera_intrinsics or {})
        self.render_updates_per_frame = max(1, int(render_updates_per_frame))
        self._held = None
        self._fall_active = False

        # ------------------------------------------------------------ open stage
        ctx = omni.usd.get_context()
        if ctx.open_stage(str(usd_path)) is False:
            raise RuntimeError(f"Could not open USD: {usd_path}")
        for _ in range(600):
            sim_app.update()
            if ctx.get_stage_loading_status()[2] == 0 and ctx.get_stage() is not None:
                break
        self.stage = ctx.get_stage()
        if self.stage is None:
            raise RuntimeError(f"Isaac returned no stage for: {usd_path}")
        if str(UsdGeom.GetStageUpAxis(self.stage)).upper() != "Z":
            raise RuntimeError("Expected a Z-up TOMO stage")
        self.mpu = float(UsdGeom.GetStageMetersPerUnit(self.stage))
        if abs(self.mpu - 1.0) > 1e-9:
            raise RuntimeError(f"Expected metersPerUnit 1.0, got {self.mpu}")
        # All runtime edits live in the in-memory session layer; the USD on
        # disk is never modified.
        self.stage.SetEditTarget(Usd.EditTarget(self.stage.GetSessionLayer()))

        for label, path in {"robot": self.robot_root, "camera": self.camera_prim,
                            "phone": self.phone_prim, "phone mesh": self.phone_mesh_prim,
                            "box": self.container_prim}.items():
            if not self.stage.GetPrimAtPath(path).IsValid():
                raise RuntimeError(f"Missing required {label} prim: {path}")

        # ------------------------------------------------------------ physics off
        self._disable_physics()

        # ------------------------------------------------------------ kinematics
        self._build_kinematic_tree()
        self._phone_frame0 = self._phone_object_frame()
        self._prepare_phone_xform()

        # ------------------------------------------------------------ dataset look
        self.scene_fixes = dict(scene_fixes or {})
        self.apply_scene_fixes(self.scene_fixes)

        # ------------------------------------------------------------ camera
        self.apply_camera_intrinsics()
        self._create_sensors()
        ex = self.scene_fixes.get("exposure") or {}
        if ex:   # replicator start-up can reset render settings -> apply again
            self.set_exposure(auto=bool(ex.get("auto", True)), film_iso=ex.get("film_iso"))
            for _ in range(3):
                self._capture()

        self.home_pose = np.zeros(46, dtype=np.float64)
        self._current_hold = self.home_pose.copy()

    # ======================================================================== physics
    def _disable_physics(self) -> None:
        """Deactivate every PhysicsScene so a timeline play can never simulate."""
        from pxr import UsdPhysics
        scenes = [p for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]
        for p in scenes:
            p.SetActive(False)
        print(f"[isaac] PhysX OFF: deactivated {len(scenes)} PhysicsScene prim(s); "
              f"robot is posed by USD forward kinematics only")

    # ======================================================================== scene fixes
    def _set_world_matrix(self, prim, W: np.ndarray) -> None:
        from pxr import UsdGeom
        xf = UsdGeom.Xformable(prim)
        reset = xf.GetResetXformStack()
        op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "fix")
        xf.SetXformOpOrder([op], reset)
        op.Set(_np_to_gf(np.linalg.inv(_world(prim.GetParent())) @ W))

    def _preview_material(self, name: str, spec: dict):
        from pxr import Gf, Sdf, UsdShade
        path = Sdf.Path(f"/DatasetLook/{name}")
        mat = UsdShade.Material.Define(self.stage, path)
        sh = UsdShade.Shader.Define(self.stage, path.AppendChild("Surface"))
        sh.CreateIdAttr("UsdPreviewSurface")
        sh.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*[float(c) for c in spec["color"]]))
        sh.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(spec.get("metallic", 0.0)))
        sh.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(spec.get("roughness", 0.6)))
        mat.CreateSurfaceOutput().ConnectToSource(sh.ConnectableAPI(), "surface")
        return mat

    def apply_scene_fixes(self, fx: dict) -> None:
        """Make the Isaac scene match the Blender scene the dataset was rendered in.

        Numbers in config.yaml were measured by back-projecting dataset depth
        and Isaac depth from the SAME camera pose (tools/check_scene_vs_dataset.py).
        Everything is authored in the session layer; the USD files are untouched.
        """
        from pxr import Gf, UsdGeom, UsdLux, UsdShade
        if not fx:
            return
        done = []
        for path, d in (fx.get("translate_world") or {}).items():
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"scene_fixes.translate_world: missing prim {path}")
            T = np.eye(4)
            T[:3, 3] = [float(v) for v in d]
            self._set_world_matrix(prim, T @ _world(prim))
            done.append(f"moved {path} by {list(map(float, d))} m")
        for path, box in (fx.get("fit_bbox_xy") or {}).items():
            # Stretch/move a prim so its world XY bounding box becomes the target
            # (axes not given keep their current extent). Height is untouched.
            prim = self.stage.GetPrimAtPath(path)
            bb = self.bbox_world(path)
            if not prim.IsValid() or bb is None:
                raise RuntimeError(f"scene_fixes.fit_bbox_xy: bad prim {path}")
            A = np.eye(4)
            for ax, key in ((0, "x"), (1, "y")):
                lo, hi = float(bb[0][ax]), float(bb[1][ax])
                n_lo, n_hi = (float(v) for v in box.get(key, (lo, hi)))
                A[ax, ax] = (n_hi - n_lo) / (hi - lo)
                A[ax, 3] = n_lo - A[ax, ax] * lo
            self._set_world_matrix(prim, A @ _world(prim))
            done.append(f"fit {path}: x[{bb[0][0]:.3f},{bb[1][0]:.3f}] y[{bb[0][1]:.3f},{bb[1][1]:.3f}] "
                        f"-> x{box.get('x', 'same')} y{box.get('y', 'same')}")
        for path in fx.get("hide") or []:
            prim = self.stage.GetPrimAtPath(path)
            if prim.IsValid():
                UsdGeom.Imageable(prim).MakeInvisible()
                done.append(f"hid {path}")
        for i, (path, spec) in enumerate((fx.get("materials") or {}).items()):
            prim = self.stage.GetPrimAtPath(path)
            if not prim.IsValid():
                raise RuntimeError(f"scene_fixes.materials: missing prim {path}")
            mat = self._preview_material(f"M{i}_{prim.GetName()}", spec)
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
            done.append(f"material {spec['color']} on {path}")
        lt = fx.get("lighting") or {}
        if lt.get("dome_intensity"):
            dome = UsdLux.DomeLight.Define(self.stage, "/DatasetLook/Dome")
            dome.CreateIntensityAttr(float(lt["dome_intensity"]))
            dome.CreateColorAttr(Gf.Vec3f(*[float(c) for c in lt.get("dome_color", [1, 1, 1])]))
            self._dome = dome
            done.append(f"dome light {lt['dome_intensity']}")
        if lt.get("key_light_intensity") is not None:
            key = self.stage.GetPrimAtPath(lt.get("key_light_prim", "/KeyLight"))
            if key.IsValid():
                key.GetAttribute("inputs:intensity").Set(float(lt["key_light_intensity"]))
                done.append(f"key light {lt['key_light_intensity']}")
        for path, inputs in (fx.get("shader_inputs") or {}).items():
            sh = UsdShade.Shader(self.stage.GetPrimAtPath(path))
            if not sh:
                raise RuntimeError(f"scene_fixes.shader_inputs: missing shader {path}")
            for name, value in inputs.items():
                inp = sh.GetInput(name)
                if not inp:
                    raise RuntimeError(f"scene_fixes.shader_inputs: {path} has no input {name}")
                cur = inp.Get()
                inp.Set(type(cur)(*value) if isinstance(value, (list, tuple)) else type(cur)(value))
                done.append(f"{path}.{name} = {value}")
        ex = fx.get("exposure") or {}
        if ex:
            self.set_exposure(auto=bool(ex.get("auto", True)), film_iso=ex.get("film_iso"))
            done.append(f"exposure auto={ex.get('auto', True)} filmIso={ex.get('film_iso')}")
        for line in done:
            print(f"[isaac] scene fix: {line}")

    def set_texture(self, shader_path: str, file: Path, input_name: str = "texture") -> None:
        from pxr import Sdf, UsdShade
        sh = UsdShade.Shader(self.stage.GetPrimAtPath(shader_path))
        if not sh or not sh.GetInput(input_name):
            raise RuntimeError(f"no texture input {input_name!r} on {shader_path}")
        file = Path(file).resolve()
        if not file.is_file():
            raise FileNotFoundError(f"texture missing: {file}")
        sh.GetInput(input_name).Set(Sdf.AssetPath(str(file)))
        print(f"[isaac] texture {shader_path} <- {file.name}")

    def apply_appearance(self, appearance: dict, phone_material: str | None, base: Path,
                         episode_materials: dict | None = None) -> None:
        """Per-episode dataset look.

        phone_textures maps the episode's Blender iPhone material
        (metadata.py: iPhone_14_plus / iPhone_14_pro_max) to a baked atlas
        (tools/bake_dataset_textures.py); `textures` are applied for every
        episode (e.g. the cardboard box); table_materials maps the episode's
        'Wooden_interact_table' material to a colour measured from the data."""
        if not appearance:
            return
        from pxr import UsdShade
        tm = appearance.get("table_materials") or {}
        name = (episode_materials or {}).get("Wooden_interact_table")
        if tm and name:
            spec = tm.get(name)
            prim = self.stage.GetPrimAtPath(appearance.get("table_prim", "/Env/geo/TM1001_00_610001_R00"))
            if spec is None:
                print(f"[isaac] no table colour for dataset material {name!r}; keeping scene_fixes colour")
            elif prim.IsValid():
                mat = self._preview_material(f"Table_{name.replace(' ', '_').replace('.', '_')}", spec)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(mat, UsdShade.Tokens.strongerThanDescendants)
                print(f"[isaac] table look = dataset '{name}' {spec['color']}")
        base = Path(base)
        res = lambda p: Path(p) if Path(p).is_absolute() else base / p  # noqa: E731
        for shader_path, file in (appearance.get("textures") or {}).items():
            self.set_texture(shader_path, res(file))
        phone = appearance.get("phone_textures") or {}
        if phone and phone_material:
            file = phone.get(phone_material)
            if file is None:
                print(f"[isaac] no baked phone texture for {phone_material}; keeping USD texture")
            else:
                self.set_texture(appearance["phone_texture_shader"], res(file))

    def set_exposure(self, auto: bool = True, film_iso: float | None = None) -> None:
        import carb
        s = carb.settings.get_settings()
        s.set("/rtx/post/histogram/enabled", bool(auto))
        if film_iso is not None:
            s.set("/rtx/post/tonemap/filmIso", float(film_iso))

    def set_dome_intensity(self, value: float) -> None:
        if getattr(self, "_dome", None) is not None:
            self._dome.GetIntensityAttr().Set(float(value))

    # ======================================================================== kinematics
    def _build_kinematic_tree(self) -> None:
        from pxr import Gf, UsdGeom, UsdPhysics

        axis_vec = {"X": Gf.Vec3d(1, 0, 0), "Y": Gf.Vec3d(0, 1, 0), "Z": Gf.Vec3d(0, 0, 1)}
        self._joints: Dict[str, dict] = {}
        for prim in self.stage.Traverse():
            if not (prim.IsA(UsdPhysics.RevoluteJoint) or prim.IsA(UsdPhysics.FixedJoint)):
                continue
            j = UsdPhysics.Joint(prim)
            b0 = [str(x) for x in j.GetBody0Rel().GetTargets()]
            b1 = [str(x) for x in j.GetBody1Rel().GetTargets()]
            if not b0 or not b1 or not b1[0].startswith(self.robot_root):
                continue
            child = self.stage.GetPrimAtPath(b1[0])
            if not child.IsValid() or child.GetParent().GetPath().pathString != b0[0]:
                continue  # root_joint etc. -- not part of the nested link chain
            lp1 = np.asarray(j.GetLocalPos1Attr().Get() or (0, 0, 0), float)
            lr1 = j.GetLocalRot1Attr().Get()
            if np.linalg.norm(lp1) > 1e-6 or (lr1 is not None and abs(lr1.GetReal() - 1.0) > 1e-6):
                raise RuntimeError(f"{prim.GetPath()}: localPos1/localRot1 not identity; "
                                   "pure-USD kinematics would be wrong")
            revolute = prim.IsA(UsdPhysics.RevoluteJoint)
            axis = str(prim.GetAttribute("physics:axis").Get() or "Z") if revolute else "Z"
            lr0 = j.GetLocalRot0Attr().Get()
            mimic_rel = prim.GetRelationship("newton:mimicJoint")
            mimic = None
            if mimic_rel and mimic_rel.GetTargets():
                mimic = (mimic_rel.GetTargets()[0].name,
                         float(prim.GetAttribute("newton:mimicCoef0").Get() or 0.0),
                         float(prim.GetAttribute("newton:mimicCoef1").Get() or 1.0))
            self._joints[prim.GetName()] = dict(
                child=child, revolute=revolute, axis=axis_vec[axis],
                lp0=Gf.Vec3d(*[float(x) for x in j.GetLocalPos0Attr().Get()]),
                lr0=Gf.Quatd(float(lr0.GetReal()), Gf.Vec3d(*[float(x) for x in lr0.GetImaginary()])),
                mimic=mimic,
            )

        missing = [n for n in CANONICAL_46 if n not in self._joints or not self._joints[n]["revolute"]]
        if missing:
            raise RuntimeError(f"STRICT MAP FAILED: policy joints absent from USD: {missing}")

        # Give every link its own translate/orient ops (suffix 'kin' so no
        # type clash with the authored ops coming from the referenced layer).
        for name, J in self._joints.items():
            xf = UsdGeom.Xformable(J["child"])
            reset = xf.GetResetXformStack()
            scale = None
            for op in xf.GetOrderedXformOps():
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    s = op.Get()
                    if s is not None and max(abs(float(v) - 1.0) for v in s) > 1e-6:
                        scale = Gf.Vec3d(*[float(v) for v in s])
            t_op = xf.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble, "kin")
            o_op = xf.AddOrientOp(UsdGeom.XformOp.PrecisionDouble, "kin")
            ops = [t_op, o_op]
            if scale is not None:
                s_op = xf.AddScaleOp(UsdGeom.XformOp.PrecisionDouble, "kin")
                s_op.Set(scale)
                ops.append(s_op)
            xf.SetXformOpOrder(ops, reset)
            t_op.Set(J["lp0"])
            J["t_op"], J["o_op"] = t_op, o_op

        n_mimic = sum(1 for J in self._joints.values() if J["mimic"])
        n_rev = sum(1 for J in self._joints.values() if J["revolute"])
        print(f"[isaac] STRICT JOINT MAP PASS: 46/46 policy joints | {n_rev} revolute "
              f"({n_mimic} mimic) + {len(self._joints) - n_rev} fixed links posed kinematically")

    # ------------------------------------------------------------ numpy FK
    def fk_world(self, joint_targets, path: str) -> np.ndarray:
        """World 4x4 of `path` for joint vector q46 WITHOUT touching the stage
        (used to draw the policy's planned 16-step future)."""
        if not hasattr(self, "_fk_cache"):
            self._fk_cache = {}
        chain = self._fk_cache.get(path)
        if chain is None:
            from pxr import UsdGeom
            by_child = {J["child"].GetPath().pathString: (n, J) for n, J in self._joints.items()}
            chain, prim = [], self.stage.GetPrimAtPath(path)
            while prim and prim.IsValid() and not prim.IsPseudoRoot():
                ps = prim.GetPath().pathString
                if ps in by_child:
                    name, J = by_child[ps]
                    q = J["lr0"]
                    w, (x, y, z) = q.GetReal(), q.GetImaginary()
                    R0 = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                                   [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                                   [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
                    chain.append(("joint", name, np.array(J["lp0"], float), R0,
                                  np.array(J["axis"], float), J["revolute"], J["mimic"]))
                elif ps == self.phone_prim or ps.startswith(self.phone_prim + "/"):
                    raise ValueError("fk_world is for robot prims only")
                else:
                    chain.append(("static", _gf_to_np(UsdGeom.Xformable(prim).GetLocalTransformation())))
                prim = prim.GetParent()
            chain.reverse()
            self._fk_cache[path] = chain
        values = dict(zip(CANONICAL_46, np.asarray(joint_targets, float).tolist()))
        M = np.eye(4)
        for item in chain:
            if item[0] == "static":
                M = M @ item[1]
                continue
            _, name, t, R0, axis, revolute, mimic = item
            if not revolute:
                a = 0.0
            elif name in values:
                a = values[name]
            elif mimic is not None:
                a = mimic[1] + mimic[2] * values.get(mimic[0], 0.0)
            else:
                a = 0.0
            k = axis / np.linalg.norm(axis)
            Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
            Ra = np.eye(3) + math.sin(a) * Kx + (1 - math.cos(a)) * Kx @ Kx
            L = np.eye(4)
            L[:3, :3] = R0 @ Ra
            L[:3, 3] = t
            M = M @ L
        return M

    def _set_joint_positions_raw(self, joint_targets, render: bool = True) -> None:
        from pxr import Gf, Sdf
        q = np.asarray(joint_targets, dtype=np.float64).reshape(-1)
        if q.shape != (46,) or not np.isfinite(q).all():
            raise ValueError(f"Bad joint target shape/data: {q.shape}")
        values = dict(zip(CANONICAL_46, q.tolist()))
        with Sdf.ChangeBlock():
            for name, J in self._joints.items():
                if J["revolute"]:
                    if name in values:
                        a = values[name]
                    elif J["mimic"] is not None:
                        leader, c0, c1 = J["mimic"]
                        a = c0 + c1 * values.get(leader, 0.0)
                    else:
                        a = 0.0
                    rot = Gf.Rotation(J["axis"], math.degrees(a)).GetQuat()
                    J["o_op"].Set(J["lr0"] * rot)
                else:
                    J["o_op"].Set(J["lr0"])
        self._current_hold = q.copy()
        if self._held is not None:
            self.carry_box(self.hand_world())
        if render:
            self.render()

    def drive(self, joint_targets: np.ndarray) -> None:
        self._set_joint_positions_raw(joint_targets, render=True)

    def hold_current(self) -> None:
        self.drive(self._current_hold)

    def apply_home_pose(self, home_yaml: Path | None = None, q46: np.ndarray | None = None,
                        n_settle: int = 1) -> None:
        if q46 is None:
            if home_yaml is None:
                raise ValueError("apply_home_pose needs q46 or home_yaml")
            import yaml
            d = yaml.safe_load(Path(home_yaml).read_text())
            labels = [str(x).split(".")[-1] for x in d["labels"]]
            values = dict(zip(labels, d["frames"][-1]))
            missing = [n for n in CANONICAL_46 if n not in values]
            if missing:
                raise RuntimeError(f"Home YAML missing joints: {missing}")
            q46 = np.asarray([values[n] for n in CANONICAL_46], float)
        self.home_pose = np.asarray(q46, float).copy()
        for _ in range(max(1, int(n_settle))):
            self.drive(self.home_pose)

    def step(self) -> None:
        self._step_fall_if_any()
        self.render()

    def render(self, n: int = 1) -> None:
        """Pump Kit (GUI viewport / events). Sensor reads capture on their own."""
        for _ in range(max(1, int(n))):
            self._sim_app.update()

    # ======================================================================== camera
    def apply_camera_intrinsics(self) -> None:
        from pxr import Gf, UsdGeom
        c = self.camera_intrinsics
        if not c:
            print("[isaac] WARNING: no camera_intrinsics given -- using authored lens")
            return
        w, h = int(c["width"]), int(c["height"])
        if (w, h) != self.cam_wh:
            raise RuntimeError(f"Camera resolution mismatch: cam_wh={self.cam_wh} intrinsics={(w, h)}")
        fx, fy, cx, cy = (float(c[k]) for k in ("fx", "fy", "cx", "cy"))
        if abs(cx - w / 2) > 1e-6 or abs(cy - h / 2) > 1e-6 or abs(fx - fy) > 1e-6:
            raise RuntimeError("Expected square pixels and a centred principal point")
        cam = UsdGeom.Camera(self.stage.GetPrimAtPath(self.camera_prim))
        old_f = cam.GetFocalLengthAttr().Get()
        hap = float(cam.GetHorizontalApertureAttr().Get() or 20.955)
        focal = hap * fx / w
        cam.GetFocalLengthAttr().Set(float(focal))
        cam.GetHorizontalApertureAttr().Set(float(hap))
        cam.GetVerticalApertureAttr().Set(float(hap * h / w))
        cam.GetHorizontalApertureOffsetAttr().Set(0.0)
        cam.GetVerticalApertureOffsetAttr().Set(0.0)
        cam.GetClippingRangeAttr().Set(Gf.Vec2f(0.01, 1000.0))
        print(f"[isaac] camera optics -> {w}x{h} fx=fy={fx:.4f} cx={cx} cy={cy} "
              f"(focal {old_f:.3f} -> {focal:.4f} mm, aperture {hap:.3f} mm, "
              f"HFOV {math.degrees(2 * math.atan(w / 2 / fx)):.2f} deg)")

    def camera_K(self) -> np.ndarray:
        c = self.camera_intrinsics
        return np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], float)

    def _create_sensors(self) -> None:
        import omni.replicator.core as rep
        self._rep = rep
        # Annotators only deliver data through the replicator orchestrator in
        # Isaac Sim 6; we never let the timeline advance (delta_time=0) and
        # PhysX is deactivated, so a capture step cannot move anything.
        rep.orchestrator.set_capture_on_play(False)
        self._rp = rep.create.render_product(self.camera_prim, resolution=self.cam_wh)
        rp_path = self._rp.path if hasattr(self._rp, "path") else self._rp
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._rgb_annot.attach([rp_path])
        self._depth_annot.attach([rp_path])
        ready = False
        for _ in range(30):
            self._capture()
            rgb = self._rgb_annot.get_data()
            dep = self._depth_annot.get_data()
            if rgb is not None and np.asarray(rgb).ndim == 3 and dep is not None and np.asarray(dep).ndim == 2:
                ready = True
                break
        if not ready:
            raise RuntimeError("RGB-D annotators produced no data after 30 capture steps")
        print(f"[isaac] RGB-D render product ready on {self.camera_prim} "
              f"({self.cam_wh[0]}x{self.cam_wh[1]}, depth=distance_to_image_plane, "
              f"rt_subframes={self.render_updates_per_frame})")

    # ------------------------------------------------------------ observer view
    def create_observer(self, eye, target, wh=(960, 720), hfov_deg: float = 55.0) -> None:
        """Third-person camera (visualisation only; the policy never sees it)."""
        import omni.replicator.core as rep
        from pxr import Gf, UsdGeom
        eye, target = np.asarray(eye, float), np.asarray(target, float)
        fwd = target - eye
        fwd /= np.linalg.norm(fwd)
        right = np.cross(fwd, [0.0, 0.0, 1.0])
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        M = np.eye(4)
        M[:3, 0], M[:3, 1], M[:3, 2], M[:3, 3] = right, up, -fwd, eye
        cam = UsdGeom.Camera.Define(self.stage, "/DatasetLook/ObserverCam")
        hap = 20.955
        focal = hap / (2 * math.tan(math.radians(hfov_deg) / 2))
        cam.CreateFocalLengthAttr(float(focal))
        cam.CreateHorizontalApertureAttr(hap)
        cam.CreateVerticalApertureAttr(hap * wh[1] / wh[0])
        cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 1000.0))
        xf = UsdGeom.Xformable(cam.GetPrim())
        xf.ClearXformOpOrder()
        xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble).Set(_np_to_gf(M))
        self._obs_M, self._obs_wh = M, (int(wh[0]), int(wh[1]))
        f_px = wh[0] * focal / hap
        self._obs_K = np.array([[f_px, 0, wh[0] / 2], [0, f_px, wh[1] / 2], [0, 0, 1]])
        self._obs_rp = rep.create.render_product("/DatasetLook/ObserverCam", resolution=self._obs_wh)
        self._obs_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._obs_annot.attach([self._obs_rp.path if hasattr(self._obs_rp, "path") else self._obs_rp])
        for _ in range(3):
            self._capture()
        print(f"[isaac] observer camera eye={eye.tolist()} target={target.tolist()} {self._obs_wh}")

    def read_observer(self) -> np.ndarray | None:
        """Observer RGB from the most recent capture (call after read_camera)."""
        if getattr(self, "_obs_annot", None) is None:
            return None
        a = np.asarray(self._obs_annot.get_data())
        if a.ndim != 3:
            return None
        rgb = a[:, :, :3]
        return np.ascontiguousarray(rgb if rgb.dtype == np.uint8 else np.clip(rgb * 255, 0, 255).astype(np.uint8))

    def observer_project(self, P) -> np.ndarray:
        """World points (N,3) -> observer pixel coords (N,2); NaN behind camera."""
        P = np.atleast_2d(np.asarray(P, float))
        Pc = (np.linalg.inv(self._obs_M) @ np.c_[P, np.ones(len(P))].T).T[:, :3]
        z = -Pc[:, 2]
        uv = np.full((len(P), 2), np.nan)
        ok = z > 1e-3
        uv[ok, 0] = self._obs_K[0, 0] * Pc[ok, 0] / z[ok] + self._obs_K[0, 2]
        uv[ok, 1] = self._obs_K[1, 1] * -Pc[ok, 1] / z[ok] + self._obs_K[1, 2]
        return uv

    def _capture(self) -> None:
        """Render the CURRENT stage state and wait until annotators hold it."""
        self._rep.orchestrator.step(rt_subframes=self.render_updates_per_frame,
                                    delta_time=0.0, pause_timeline=True, wait_for_render=True)

    def read_camera(self) -> Tuple[np.ndarray, np.ndarray]:
        self._capture()
        rgba = np.asarray(self._rgb_annot.get_data())
        depth = np.asarray(self._depth_annot.get_data(), dtype=np.float32).squeeze()
        W, H = self.cam_wh
        if rgba.ndim != 3 or rgba.shape[:2] != (H, W):
            raise RuntimeError(f"RGB annotator returned {rgba.shape}; expected ({H},{W},4). "
                               "Stale `kit` process holding the GPU?  ps -eo pid,etime,comm | grep kit")
        if depth.shape != (H, W):
            raise RuntimeError(f"Depth annotator returned {depth.shape}; expected ({H},{W})")
        rgb = rgba[:, :, :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb * 255.0, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(rgb), depth.copy()

    def camera_world(self) -> np.ndarray:
        return _world(self.stage.GetPrimAtPath(self.camera_prim))

    # ======================================================================== phone
    def _phone_object_frame(self) -> np.ndarray:
        """Box-centre frame of the phone mesh (x = long axis), in world."""
        from pxr import UsdGeom
        mesh = UsdGeom.Mesh(self.stage.GetPrimAtPath(self.phone_mesh_prim))
        pts = np.asarray(mesh.GetPointsAttr().Get(), float)
        Mw = _world(mesh.GetPrim())
        W = (Mw @ np.c_[pts, np.ones(len(pts))].T).T[:, :3]
        z_top, z_bot = W[:, 2].max(), W[:, 2].min()
        top = W[W[:, 2] > z_top - 0.002][:, :2]
        c = top.mean(0)
        _, V = np.linalg.eigh(np.cov((top - c).T))
        a = V[:, 1]
        yaw = math.atan2(a[1], a[0])
        ext = (W[:, :2] - c) @ V
        self.phone_size_m = (float(np.ptp(ext[:, 1])), float(np.ptp(ext[:, 0])), float(z_top - z_bot))
        F = np.eye(4)
        F[:3, :3] = _rot_xyz(0, 0, yaw)
        F[:3, 3] = [c[0], c[1], 0.5 * (z_top + z_bot)]
        return F

    def _prepare_phone_xform(self) -> None:
        from pxr import UsdGeom
        prim = self.stage.GetPrimAtPath(self.phone_prim)
        W = _world(prim)
        parent = prim.GetParent()
        PW = _world(parent)
        xf = UsdGeom.Xformable(prim)
        reset = xf.GetResetXformStack()
        self.box_op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, "kin")
        xf.SetXformOpOrder([self.box_op], reset)
        self.box_op.Set(_np_to_gf(np.linalg.inv(PW) @ W))
        self._phone_prim_M0 = W.copy()
        # phone prim world = object_frame(t) @ inv(object_frame0) @ prim_world0
        self._obj_to_prim = np.linalg.inv(self._phone_frame0) @ W
        self.box_M0 = W.copy()

    def phone_world(self) -> np.ndarray:
        return _world(self.stage.GetPrimAtPath(self.phone_prim))

    def _set_phone_world(self, W: np.ndarray) -> None:
        prim = self.stage.GetPrimAtPath(self.phone_prim)
        PW = _world(prim.GetParent())
        self.box_op.Set(_np_to_gf(np.linalg.inv(PW) @ np.asarray(W, float)))

    def set_phone_pose(self, xyz_rpy: Sequence[float], yaw_offset_rad: float = 0.0) -> None:
        """Place the phone so its box centre / long axis match the dataset pose."""
        obj = _pose_matrix(xyz_rpy)
        if yaw_offset_rad:
            R = np.eye(4)
            R[:3, :3] = _rot_xyz(0, 0, yaw_offset_rad)
            obj = obj @ R
        self._set_phone_world(obj @ self._obj_to_prim)

    def phone_object_pose(self) -> np.ndarray:
        """Current phone box-centre frame in world (inverse of set_phone_pose)."""
        return self.phone_world() @ np.linalg.inv(self._obj_to_prim)

    # ======================================================================== suction
    _HAND_PATH_DEFAULT = (
        "/tomo_re_uhand/Geometry/Body_base/Right_arm_link1/Right_arm_link2/"
        "Right_arm_link3/Right_arm_link4/Right_arm_link5/Right_arm_link6/"
        "Right_arm_endlink/Right_hand"
    )

    def hand_world(self, hand_path: str | None = None) -> np.ndarray:
        path = hand_path or self.suction_prim or self._HAND_PATH_DEFAULT
        prim = self.stage.GetPrimAtPath(path)
        if not prim.IsValid():
            raise RuntimeError(f"Suction/end-effector prim missing: {path}")
        return _world(prim)

    def attach_box(self, hand_M: np.ndarray) -> None:
        self._held = np.linalg.inv(np.asarray(hand_M, float)) @ self.phone_world()

    def carry_box(self, hand_M: np.ndarray) -> None:
        if self._held is not None:
            self._set_phone_world(np.asarray(hand_M, float) @ self._held)

    def detach_box(self) -> None:
        self._held = None

    @property
    def holding(self) -> bool:
        return self._held is not None

    def start_fall(self, ground_z: float = 0.0524) -> None:
        """Kinematic drop of the released phone (box-centre z -> ground_z)."""
        self._held = None
        self._fall_active = True
        self._fall_obj = self.phone_object_pose()
        self._fall_v = 0.0
        self._fall_dt = 1.0 / 30.0
        self._fall_ground_z = float(ground_z)

    def _step_fall_if_any(self) -> None:
        if not self._fall_active:
            return
        self._fall_v -= 9.81 * self._fall_dt
        self._fall_obj[2, 3] += self._fall_v * self._fall_dt
        if self._fall_obj[2, 3] <= self._fall_ground_z:
            self._fall_obj[2, 3] = self._fall_ground_z
            # settle flat: keep yaw only
            yaw = math.atan2(self._fall_obj[1, 0], self._fall_obj[0, 0])
            self._fall_obj[:3, :3] = _rot_xyz(0, 0, yaw)
            self._fall_active = False
            print(f"[fall] phone landed, centre z={self._fall_ground_z:.4f} m")
        self._set_phone_world(self._fall_obj @ self._obj_to_prim)

    @property
    def falling(self) -> bool:
        return self._fall_active

    # ======================================================================== reads
    def read_joints(self) -> Dict[str, float]:
        return {n: float(v) for n, v in zip(CANONICAL_46, self._current_hold)}

    def bbox_world(self, path: str):
        from pxr import Usd, UsdGeom
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(),
                                  [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
                                  useExtentsHint=False)
        rng = cache.ComputeWorldBound(self.stage.GetPrimAtPath(path)).ComputeAlignedRange()
        if rng.IsEmpty():
            return None
        return np.array(rng.GetMin(), float), np.array(rng.GetMax(), float)

    def active_physics_scenes(self) -> list:
        from pxr import UsdPhysics
        return [p.GetPath().pathString for p in self.stage.Traverse() if p.IsA(UsdPhysics.Scene)]

    def diagnostics(self) -> dict:
        C = self.camera_world()
        fwd = -C[:3, 2]
        try:
            import omni.timeline
            tl_time = float(omni.timeline.get_timeline_interface().get_current_time())
        except Exception:
            tl_time = None
        return {
            "physics_solver_stepping": False,
            "active_physics_scenes": self.active_physics_scenes(),
            "timeline_time_s": tl_time,
            "kinematics": "pure USD forward kinematics (nested links)",
            "joint_order_exact": True,
            "camera_prim": self.camera_prim,
            "camera_resolution": list(self.cam_wh),
            "camera_K": self.camera_K().tolist() if self.camera_intrinsics else None,
            "camera_world_position_m": C[:3, 3].round(5).tolist(),
            "camera_world_forward": fwd.round(5).tolist(),
            "camera_pitch_deg": round(math.degrees(math.asin(max(-1.0, min(1.0, fwd[2])))), 3),
            "depth": "distance_to_image_plane (optical-axis Z, metres)",
            "phone_prim": self.phone_prim,
            "phone_size_m": [round(x, 4) for x in self.phone_size_m],
            "phone_object_pose_xyz": self.phone_object_pose()[:3, 3].round(5).tolist(),
            "box_prim": self.container_prim,
            "holding_phone": self.holding,
        }

    def close(self) -> None:
        try:
            self._sim_app.close()
        except Exception:
            pass
