"""Bake the DATASET appearance of a mesh (phone, cardboard box) into its texture.

For every texel of the mesh's UV atlas we know the 3D point on the mesh. With
the robot, phone and box posed exactly as in a recorded frame (same USD forward
kinematics and scene_fixes as the Isaac runtime), that point is projected into
the Blender RGB frame; the colour is kept only where the dataset depth agrees
with the mesh depth (so hands / other objects in front are rejected).  Colours
are averaged over many frames, divided by Isaac's lighting gain so that the
Isaac render comes out at the dataset brightness, and written as a new PNG.
Texels never seen keep the original texture (optionally tinted).

Runs WITHOUT Isaac (pxr + numpy + cv2), e.g. with ~/isaac_env/bin/python:
    python tools/bake_dataset_textures.py --target phone --episode Find_and_Pick_001
    python tools/bake_dataset_textures.py --target box   --episode Find_and_Pick_001
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from common import converter_module, episode_paths, load_config, read_dataset_frames, resolve  # noqa: E402

TARGETS = {
    "phone": {"mesh": "/Iphone/geo/iPhone_14_Pro_Max",
              "shader": "/Iphone/geo/Looks/iPhone_14_pro_max/baseColorTex",
              "frames": list(range(30, 146, 4))},          # phone static & unoccluded before the grasp
    "box": {"mesh": "/tomo_re_uhand/Box_1_v1___open/Box_1_v1",
            "shader": "/tomo_re_uhand/Box_1_v1___open/Looks/Box1_v1/baseColorTex",
            "frames": list(range(30, 300, 6))},
}


def srgb_to_lin(c):
    c = np.asarray(c, np.float32) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin_to_srgb(c):
    c = np.clip(c, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * np.power(c, 1 / 2.4) - 0.055) * 255.0


def bilinear(img, u, v):
    h, w = img.shape[:2]
    u = np.clip(u, 0, w - 1.001)
    v = np.clip(v, 0, h - 1.001)
    u0, v0 = np.floor(u).astype(int), np.floor(v).astype(int)
    du, dv = (u - u0)[:, None], (v - v0)[:, None]
    return (img[v0, u0] * (1 - du) * (1 - dv) + img[v0, u0 + 1] * du * (1 - dv)
            + img[v0 + 1, u0] * (1 - du) * dv + img[v0 + 1, u0 + 1] * du * dv)


def offline_runtime(cfg, base):
    """IsaacRuntime kinematics/scene edits without Kit (pure pxr)."""
    from pxr import Usd
    from runtime import isaac_runtime as R
    icfg = cfg["isaac_mode"]
    stage = Usd.Stage.Open(str(resolve(icfg["usd_path"], base)))
    stage.SetEditTarget(Usd.EditTarget(stage.GetSessionLayer()))
    rt = R.IsaacRuntime.__new__(R.IsaacRuntime)
    rt.stage, rt.robot_root = stage, icfg["robot_root"]
    rt.phone_prim, rt.phone_mesh_prim = icfg["box_prim"], icfg["phone_mesh_prim"]
    rt.container_prim, rt.suction_prim = icfg["container_prim"], icfg["suction_prim"]
    rt.camera_prim = icfg["camera_prim"]
    rt._held, rt._fall_active = None, False
    fx = icfg.get("scene_fixes") or {}
    with contextlib.redirect_stdout(io.StringIO()):
        rt._build_kinematic_tree()
        rt._phone_frame0 = rt._phone_object_frame()
        rt._prepare_phone_xform()
        rt.apply_scene_fixes({k: fx[k] for k in ("translate_world", "fit_bbox_xy") if k in fx})
    return rt, R


def triangle_texels(uv_tri, W, H):
    """Texel coords (x, y int) inside one UV triangle + barycentrics."""
    tx, ty = uv_tri[:, 0] * W - 0.5, (1.0 - uv_tri[:, 1]) * H - 0.5
    x0, x1 = int(max(0, np.floor(tx.min()))), int(min(W - 1, np.ceil(tx.max())))
    y0, y1 = int(max(0, np.floor(ty.min()))), int(min(H - 1, np.ceil(ty.max())))
    if x1 < x0 or y1 < y0:
        return None
    gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
    gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)
    (ax, bx, cx_), (ay, by, cy_) = tx, ty
    den = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
    if abs(den) < 1e-12:
        return None
    l0 = ((by - cy_) * (gx - cx_) + (cx_ - bx) * (gy - cy_)) / den
    l1 = ((cy_ - ay) * (gx - cx_) + (ax - cx_) * (gy - cy_)) / den
    l2 = 1.0 - l0 - l1
    inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
    if not inside.any():
        return None
    return gx[inside].astype(int), gy[inside].astype(int), np.stack([l0[inside], l1[inside], l2[inside]], 1)


def refine(args, cfg, base):
    """One inverse-rendering step: texture *= dataset / isaac, per triangle.

    Uses Isaac renders made WITH the current texture (tools/check_scene_vs_dataset.py
    saves arrays_fXXX.npz: isaac_rgb/isaac_depth/dataset_rgb/dataset_depth/camera_world).
    A per-triangle colour ratio corrects Isaac-vs-Blender shading face by face
    while keeping the texture detail."""
    from pxr import Sdf, UsdGeom, UsdShade
    rt, R = offline_runtime(cfg, base)
    tgt = TARGETS[args.target]
    ep = R.EpisodeReference(episode_paths(resolve(cfg["isaac_mode"]["dataset_start"]["dataset_dir"], base),
                                          args.episode)["yaml"])
    K = cfg["isaac_mode"]["camera_intrinsics"]
    fx, fy, cx, cy = (float(K[k]) for k in ("fx", "fy", "cx", "cy"))
    mesh = UsdGeom.Mesh(rt.stage.GetPrimAtPath(tgt["mesh"]))
    pts_local = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
    fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    uv = np.asarray(UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st").Get(), np.float64).reshape(-1, 3, 2)
    cur = cv2.cvtColor(cv2.imread(str(args.texture), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = cur.shape[:2]
    frames = [int(x) for x in args.refine_frames.split(",")]
    views = []
    for f in frames:
        a = np.load(Path(args.refine_from) / f"arrays_f{f:03d}.npz")
        rt.set_phone_pose(ep.phone_pose(f))
        rt._set_joint_positions_raw(ep.q46(f), render=False)
        Mw = R._world(mesh.GetPrim())
        C = rt.camera_world()
        views.append((Mw, np.linalg.inv(C), C[:3, 3].copy(), srgb_to_lin(a["dataset_rgb"]),
                      srgb_to_lin(a["isaac_rgb"]), a["dataset_depth"], a["isaac_depth"]))
    ratios = np.full((len(fvi), 3), np.nan, np.float32)
    for ti in range(len(fvi)):
        r = triangle_texels(uv[ti], W, H)
        if r is None:
            continue
        _, _, L = r
        P_loc = L @ pts_local[fvi[ti]]
        sd, si = np.zeros(3), np.zeros(3)
        n_ok = 0
        for Mw, Cinv, cam_pos, dl, il, dd, idp in views:
            Pw = P_loc @ Mw[:3, :3].T + Mw[:3, 3]
            tri_w = pts_local[fvi[ti]] @ Mw[:3, :3].T + Mw[:3, 3]
            nrm = np.cross(tri_w[1] - tri_w[0], tri_w[2] - tri_w[0])
            nrm /= (np.linalg.norm(nrm) + 1e-12)
            Pc = Pw @ Cinv[:3, :3].T + Cinv[:3, 3]
            z = -Pc[:, 2]
            ok = z > 0.1
            u = fx * Pc[:, 0] / np.where(ok, z, 1) + cx
            v = fy * (-Pc[:, 1]) / np.where(ok, z, 1) + cy
            ok &= (u >= 1) & (u <= 958) & (v >= 1) & (v <= 718)
            ok &= ((cam_pos - Pw) @ nrm) > 0
            if not ok.any():
                continue
            ui, vi = np.round(u[ok]).astype(int), np.round(v[ok]).astype(int)
            good = (np.abs(dd[vi, ui] - z[ok]) < args.depth_tol) & (np.abs(idp[vi, ui] - z[ok]) < args.depth_tol)
            if not good.any():
                continue
            sd += dl[vi[good], ui[good]].sum(0)
            si += il[vi[good], ui[good]].sum(0)
            n_ok += int(good.sum())
        if n_ok >= 20 and si.min() > 1e-6:
            ratios[ti] = np.clip(sd / si, 0.3, 3.0)
    obs = ~np.isnan(ratios[:, 0])
    glob = np.nanmedian(ratios[obs], axis=0) if obs.any() else np.ones(3)
    ratios[~obs] = glob
    out_lin = srgb_to_lin(cur)
    done = np.zeros((H, W), bool)
    for ti in range(len(fvi)):
        r = triangle_texels(uv[ti], W, H)
        if r is None:
            continue
        gx, gy, _ = r
        m = ~done[gy, gx]
        out_lin[gy[m], gx[m]] *= ratios[ti]
        done[gy[m], gx[m]] = True
    out = np.clip(lin_to_srgb(out_lin), 0, 255).astype(np.uint8)
    out_path = Path(args.out) if args.out else Path(args.texture)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[refine] {args.target}: {int(obs.sum())}/{len(fvi)} triangles measured; "
          f"median ratio {np.round(glob, 3).tolist()}; wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--target", choices=sorted(TARGETS), required=True)
    ap.add_argument("--episode", default="Find_and_Pick_001")
    ap.add_argument("--frames", default=None, help="comma list; default depends on target")
    ap.add_argument("--light-gain", type=float, default=None,
                    help="Isaac rendered-linear / albedo for up-facing surfaces (default from config)")
    ap.add_argument("--depth-tol", type=float, default=0.004)
    ap.add_argument("--unseen-tint", default=None, help="r,g,b multiplier for texels never observed")
    ap.add_argument("--out", default=None)
    ap.add_argument("--refine-from", default=None,
                    help="folder with arrays_fXXX.npz from check_scene_vs_dataset.py (rendered WITH --texture)")
    ap.add_argument("--refine-frames", default="50,100,150,290")
    ap.add_argument("--texture", default=None, help="current texture to refine")
    args = ap.parse_args()

    cfg, base = load_config(args.config)
    if args.refine_from:
        if not args.texture:
            raise SystemExit("--refine-from needs --texture")
        refine(args, cfg, base)
        return
    tgt = TARGETS[args.target]
    frames = [int(x) for x in args.frames.split(",")] if args.frames else tgt["frames"]
    gain = args.light_gain or float((cfg["isaac_mode"].get("appearance") or {}).get("bake_light_gain", 0.68))
    K = cfg["isaac_mode"]["camera_intrinsics"]
    fx, fy, cx, cy = (float(K[k]) for k in ("fx", "fy", "cx", "cy"))

    rt, R = offline_runtime(cfg, base)
    from pxr import Sdf, UsdGeom, UsdShade
    ep = R.EpisodeReference(episode_paths(resolve(cfg["isaac_mode"]["dataset_start"]["dataset_dir"], base),
                                          args.episode)["yaml"])
    paths = episode_paths(resolve(cfg["isaac_mode"]["dataset_start"]["dataset_dir"], base), args.episode)
    tools = converter_module(resolve(cfg["idp3_policy"]["training_bundle"], base))
    print(f"[bake] {args.target} from {args.episode}: decoding {len(frames)} frames")
    ref = read_dataset_frames(paths["mp4"], paths["depth_dir"], frames, tools)

    # ----------------------------------------------------------- mesh + atlas
    mesh = UsdGeom.Mesh(rt.stage.GetPrimAtPath(tgt["mesh"]))
    pts_local = np.asarray(mesh.GetPointsAttr().Get(), np.float64)
    fvi = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), int).reshape(-1, 3)
    st_pv = UsdGeom.PrimvarsAPI(mesh).GetPrimvar("st")
    uv = np.asarray(st_pv.Get(), np.float64)
    if st_pv.GetInterpolation() != "faceVarying" or len(uv) != fvi.size:
        raise RuntimeError("expected un-indexed faceVarying st")
    uv = uv.reshape(-1, 3, 2)
    shader = UsdShade.Shader(rt.stage.GetPrimAtPath(tgt["shader"]))
    tex_path = Path(shader.GetInput("texture").Get().resolvedPath)
    atlas = cv2.cvtColor(cv2.imread(str(tex_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    H, W = atlas.shape[:2]
    print(f"[bake] mesh {tgt['mesh']}: {len(fvi)} triangles, atlas {tex_path.name} {W}x{H}")

    # USD st has v up; image row 0 is the top. (Verified below on the phone.)
    def texel_xy(t_uv):
        return t_uv[..., 0] * W - 0.5, (1.0 - t_uv[..., 1]) * H - 0.5

    acc = np.zeros((H, W, 3), np.float32)
    wsum = np.zeros((H, W), np.float32)

    poses = []
    for f in frames:
        rt.set_phone_pose(ep.phone_pose(f))
        rt._set_joint_positions_raw(ep.q46(f), render=False)
        Mw = R._world(mesh.GetPrim())
        C = rt.camera_world()
        poses.append((f, Mw, np.linalg.inv(C), C[:3, 3].copy()))
    img_lin = {f: srgb_to_lin(ref[f][0]) for f in frames}

    seen_tris = 0
    for ti in range(len(fvi)):
        tx, ty = texel_xy(uv[ti])
        x0, x1 = int(max(0, np.floor(tx.min()))), int(min(W - 1, np.ceil(tx.max())))
        y0, y1 = int(max(0, np.floor(ty.min()))), int(min(H - 1, np.ceil(ty.max())))
        if x1 < x0 or y1 < y0:
            continue
        gx, gy = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        gx, gy = gx.ravel().astype(np.float64), gy.ravel().astype(np.float64)
        (ax, bx, cx_), (ay, by, cy_) = tx, ty
        den = (by - cy_) * (ax - cx_) + (cx_ - bx) * (ay - cy_)
        if abs(den) < 1e-12:
            continue
        l0 = ((by - cy_) * (gx - cx_) + (cx_ - bx) * (gy - cy_)) / den
        l1 = ((cy_ - ay) * (gx - cx_) + (ax - cx_) * (gy - cy_)) / den
        l2 = 1.0 - l0 - l1
        inside = (l0 >= -1e-3) & (l1 >= -1e-3) & (l2 >= -1e-3)
        if not inside.any():
            continue
        gx, gy = gx[inside].astype(int), gy[inside].astype(int)
        L = np.stack([l0[inside], l1[inside], l2[inside]], 1)
        P_loc = L @ pts_local[fvi[ti]]
        got = False
        for f, Mw, Cinv, cam_pos in poses:
            Pw = P_loc @ Mw[:3, :3].T + Mw[:3, 3]
            tri_w = pts_local[fvi[ti]] @ Mw[:3, :3].T + Mw[:3, 3]
            n = np.cross(tri_w[1] - tri_w[0], tri_w[2] - tri_w[0])
            n /= (np.linalg.norm(n) + 1e-12)
            Pc = Pw @ Cinv[:3, :3].T + Cinv[:3, 3]
            z = -Pc[:, 2]
            ok = z > 0.1
            if not ok.any():
                continue
            u = fx * Pc[:, 0] / np.where(ok, z, 1) + cx
            v = fy * (-Pc[:, 1]) / np.where(ok, z, 1) + cy
            ok &= (u >= 0) & (u <= 959) & (v >= 0) & (v <= 719)
            view = cam_pos - Pw
            cosang = (view @ n) / (np.linalg.norm(view, axis=1) + 1e-12)
            ok &= cosang > 0.15                      # front-facing
            if not ok.any():
                continue
            d = ref[f][1][np.clip(np.round(v).astype(int), 0, 719), np.clip(np.round(u).astype(int), 0, 959)]
            ok &= np.abs(d - z) < args.depth_tol     # not occluded
            if not ok.any():
                continue
            col = bilinear(img_lin[f], u[ok], v[ok])
            w = (cosang[ok] / (z[ok] ** 2)).astype(np.float32)
            np.add.at(acc, (gy[ok], gx[ok]), col * w[:, None])
            np.add.at(wsum, (gy[ok], gx[ok]), w)
            got = True
        seen_tris += int(got)

    seen = wsum > 0
    baked_lin = np.zeros_like(acc)
    baked_lin[seen] = acc[seen] / wsum[seen][:, None] / gain
    out_lin = srgb_to_lin(atlas)
    if args.unseen_tint:
        out_lin = out_lin * np.array([float(x) for x in args.unseen_tint.split(",")], np.float32)
    out_lin[seen] = baked_lin[seen]
    # bleed baked colour 4 texels past triangle borders (hides UV seams)
    m = seen.astype(np.uint8)
    num, den_ = baked_lin * m[..., None], m.astype(np.float32)
    k = np.ones((3, 3), np.uint8)
    for _ in range(4):
        num_d = cv2.dilate(num, k)
        den_d = cv2.dilate(den_, k)
        fill = (den_d > 0) & (den_ == 0)
        num[fill], den_[fill] = num_d[fill], 1.0
        out_lin[fill] = num_d[fill]
    out = np.clip(lin_to_srgb(out_lin), 0, 255).astype(np.uint8)

    out_dir = ROOT / "scene_overrides"
    out_dir.mkdir(exist_ok=True)
    variant = ep.phone_material if args.target == "phone" else "box"
    out_path = Path(args.out) if args.out else out_dir / f"{args.target}_{variant}_from_{args.episode}.png"
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    rep = {"target": args.target, "episode": args.episode, "phone_material": ep.phone_material,
           "frames": frames, "triangles": int(len(fvi)), "triangles_observed": seen_tris,
           "texels_baked": int(seen.sum()), "atlas": [W, H], "light_gain": gain,
           "source_texture": str(tex_path), "output": str(out_path),
           "baked_mean_srgb": np.round(lin_to_srgb(baked_lin[seen].mean(0)), 1).tolist() if seen.any() else None}
    out_path.with_suffix(".json").write_text(json.dumps(rep, indent=2))
    print("[bake] " + json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
