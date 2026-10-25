#!/usr/bin/env python3
"""Policy adapter for the TOMO RGB-D DINOv2 ACT v2 checkpoint (DEPLOY_V2_FINAL.pt).

API (called by the deployer's server):
    N_OBS, HORIZON, CHECKPOINT, JOINT_NAMES
    build_obs(rgb_uint8[720,960,3], depth_z_m[720,960] float32, frame_id) -> Obs
    load_policy(device='cuda:0', checkpoint=CHECKPOINT) -> policy
    policy.predict(obs_hist: list[Obs] (len N_OBS, oldest first), qs: [N_OBS,46] rad, seed) -> [HORIZON,46] float32

Semantics (verified against the training code, bundle = tomo_act_bundle/):
  * Single current frame, no history (train_rgbd_dinov2_act_v2.py:72; rgbd_act_model_v2.py:76) -> N_OBS = 1.
  * Output rows are ABSOLUTE joint positions q[t+1] .. q[t+30] (target_offset=1, chunk 30;
    train:55-56 target(q,t)=q[t+1:t+31]). Native row 0 is already q[t+1]: nothing is dropped.
    The network predicts pred = q_norm[t] + delta (model:81); we denormalise pred*q_std+q_mean.
  * Inference: eval(), actions=None -> CVAE latent z = 0 (model:78-80) -> deterministic; seed unused
    except for torch.manual_seed (no stochastic op remains).

Preprocessing (reproduces prepare_rgbd_dataset.py:406-420 cache + train:57-58,72-74 loader):
  RGB  : cv2.resize(rgb,(336,252),INTER_AREA) uint8 -> float/255 CHW -> TF.resize(252x336) (identity)
         -> ImageNet mean/std.  (Training frames came from cv2-decoded MP4, BGR->RGB, prepare:257.)
  Depth: cv2.resize(d,(336,252),INTER_NEAREST) -> float16 -> float32 ->
         valid=isfinite&>0 ; clamp(nan_to_num(d),0.15,2.0) ; (d-0.15)/1.85 ; invalid->0.
         near/far = 0.15/2.0 from prepared_rgbd/depth_stats.json, which is what the trainer read
         (train:28). ckpt['depth_stats'] (0.18/8.91) is STALE and deliberately NOT used.
  Isaac-vs-Blender equivalence: Blender Depth.V has no inf; background is a finite far plane
  (~8.91 m) -> normalised 1.0 in training. Isaac distance_to_image_plane returns +inf for no-hit,
  which the training rule would send to 0.0 (never seen in training). With
  ISAAC_INF_AS_FAR=True (default) +inf is replaced by 8.914 m BEFORE the training rule, i.e. we
  convert the Isaac quantity into the Blender quantity; nothing in the training rule is changed.
  Joints: the 10 *_joint_11..51 joints had std==0 in training (floored to 1e-4, mean 0). With
  CLAMP_CONSTANT_JOINTS=True (default) their input is set to the training constant (0.0), so tiny
  sim drift cannot become a 10-sigma input. Set False for the literal training rule.

DINOv2 ViT-S/14: the torch hub cache used in training is gone; an offline, parameter-name-identical
re-implementation is embedded below and the real weights come from the checkpoint (strict load).
"""
from __future__ import annotations
import math, os, sys
from dataclasses import dataclass
from typing import List, Optional, Sequence

sys.dont_write_bytecode = True  # never write __pycache__ into the read-only bundle
os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from pathlib import Path as _P
_SHARE = _P(__file__).resolve().parents[2]      # share_act_dinov2/
BUNDLE = str(_SHARE / "code")                   # model code (was the ACT training bundle)
CHECKPOINT = str(_SHARE / "model/DEPLOY_V2_FINAL.pt")
N_OBS = 1
HORIZON = 30
IMG_H, IMG_W = 252, 336
SRC_H, SRC_W = 720, 960
DEPTH_NEAR_M, DEPTH_FAR_M, DEPTH_SCALE = 0.15, 2.0, 1.0   # prepared_rgbd/depth_stats.json
BLENDER_FAR_PLANE_M = 8.914                               # raw_summary p99.5 of Depth.V
ISAAC_INF_AS_FAR = True
CLAMP_CONSTANT_JOINTS = True

JOINT_NAMES = tuple(
    ["Head_joint1", "Head_joint2"]
    + [f"Left_arm_joint{i}" for i in range(1, 8)]
    + [f"Right_arm_joint{i}" for i in range(1, 8)]
    + [f"RF_joint_{f}{k}" for f in range(1, 6) for k in range(1, 4)]
    + [f"LF_joint_{f}{k}" for f in range(1, 6) for k in range(1, 4)]
)
assert len(JOINT_NAMES) == 46

_MEAN = torch.tensor([.485, .456, .406]).view(3, 1, 1)
_STD = torch.tensor([.229, .224, .225]).view(3, 1, 1)


# ----------------------------------------------------------------------------- preprocessing
@dataclass
class Obs:
    rgb: torch.Tensor     # (3,252,336) float32, ImageNet-normalised, RGB order (CPU)
    depth: torch.Tensor   # (1,252,336) float32, [0,1], invalid=0 (CPU)
    frame_id: int


def rgb_to_cache(rgb_uint8: np.ndarray) -> np.ndarray:
    """Exactly the cache step (prepare_rgbd_dataset.py:406-412): uint8 (252,336,3)."""
    rgb = np.ascontiguousarray(np.asarray(rgb_uint8)[..., :3], dtype=np.uint8)  # drop alpha if RGBA
    if rgb.shape[:2] != (IMG_H, IMG_W):
        rgb = cv2.resize(rgb, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA).astype(np.uint8, copy=False)
    return rgb


def depth_to_cache(depth_z_m: np.ndarray) -> np.ndarray:
    """Exactly the cache step (prepare_rgbd_dataset.py:414-420): float16 (252,336) metres."""
    d = np.asarray(depth_z_m, dtype=np.float32)
    if d.ndim == 3:
        d = d[..., 0]
    if ISAAC_INF_AS_FAR:
        d = np.where(np.isposinf(d), np.float32(BLENDER_FAR_PLANE_M), d)
    return cv2.resize(np.ascontiguousarray(d), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST).astype(np.float16)


def rgb_cache_to_tensor(rgb_cache_u8: np.ndarray) -> torch.Tensor:
    """train_rgbd_dinov2_act_v2.py:72-74 (no augmentation)."""
    rgb = torch.from_numpy(np.asarray(rgb_cache_u8).copy()).float().permute(2, 0, 1) / 255.
    rgb = TF.resize(rgb, [IMG_H, IMG_W], InterpolationMode.BILINEAR, antialias=True)
    return (rgb - _MEAN) / _STD


def depth_cache_to_tensor(depth_cache_f16: np.ndarray) -> torch.Tensor:
    """train_rgbd_dinov2_act_v2.py:57-58 norm_depth + :74 resize/mask (no augmentation)."""
    d = torch.from_numpy(np.asarray(depth_cache_f16, dtype=np.float32).copy()).unsqueeze(0)
    valid = torch.isfinite(d) & (d > 0)
    dm = torch.nan_to_num(d * DEPTH_SCALE, nan=DEPTH_FAR_M, posinf=DEPTH_FAR_M, neginf=DEPTH_NEAR_M).clamp(DEPTH_NEAR_M, DEPTH_FAR_M)
    dn = (dm - DEPTH_NEAR_M) / max(1e-6, DEPTH_FAR_M - DEPTH_NEAR_M)
    d = torch.where(valid, dn, torch.zeros_like(dn)); mask = valid.float()
    d = TF.resize(d, [IMG_H, IMG_W], InterpolationMode.BILINEAR, antialias=True)
    mask = TF.resize(mask, [IMG_H, IMG_W], InterpolationMode.NEAREST)
    return d * (mask > .5)


def build_obs(rgb_uint8: np.ndarray, depth_z_m: np.ndarray, frame_id: int) -> Obs:
    rgb_uint8 = np.asarray(rgb_uint8); depth_z_m = np.asarray(depth_z_m)
    if rgb_uint8.shape[:2] != (SRC_H, SRC_W) or depth_z_m.shape[:2] != (SRC_H, SRC_W):
        raise ValueError(f"expected 720x960 inputs, got rgb {rgb_uint8.shape} depth {depth_z_m.shape}")
    if rgb_uint8.dtype != np.uint8:
        raise ValueError(f"rgb must be uint8, got {rgb_uint8.dtype}")
    return Obs(rgb_cache_to_tensor(rgb_to_cache(rgb_uint8)),
               depth_cache_to_tensor(depth_to_cache(depth_z_m)), int(frame_id))


# ----------------------------------------------------------------------------- offline DINOv2 ViT-S/14
class _PatchEmbed(nn.Module):
    def __init__(s, p=14, d=384): super().__init__(); s.proj = nn.Conv2d(3, d, p, p)
    def forward(s, x): return s.proj(x).flatten(2).transpose(1, 2)
class _Attention(nn.Module):
    def __init__(s, d=384, h=6): super().__init__(); s.h = h; s.qkv = nn.Linear(d, 3 * d); s.proj = nn.Linear(d, d)
    def forward(s, x):
        B, N, C = x.shape; qkv = s.qkv(x).reshape(B, N, 3, s.h, C // s.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2]); return s.proj(x.transpose(1, 2).reshape(B, N, C))
class _LayerScale(nn.Module):
    def __init__(s, d=384): super().__init__(); s.gamma = nn.Parameter(torch.ones(d))
    def forward(s, x): return x * s.gamma
class _Mlp(nn.Module):
    def __init__(s, d=384, hd=1536): super().__init__(); s.fc1 = nn.Linear(d, hd); s.act = nn.GELU(); s.fc2 = nn.Linear(hd, d)
    def forward(s, x): return s.fc2(s.act(s.fc1(x)))
class _Block(nn.Module):
    def __init__(s, d=384):
        super().__init__(); s.norm1 = nn.LayerNorm(d, eps=1e-6); s.attn = _Attention(d); s.ls1 = _LayerScale(d)
        s.norm2 = nn.LayerNorm(d, eps=1e-6); s.mlp = _Mlp(d); s.ls2 = _LayerScale(d)
    def forward(s, x): x = x + s.ls1(s.attn(s.norm1(x))); return x + s.ls2(s.mlp(s.norm2(x)))
class DinoVisionTransformer(nn.Module):
    """facebookresearch/dinov2 hub 'dinov2_vits14' (no registers, img 518, interpolate_offset 0.1)."""
    def __init__(s, d=384, depth=12, p=14, img=518):
        super().__init__(); s.patch_size = p; s.patch_embed = _PatchEmbed(p, d)
        s.cls_token = nn.Parameter(torch.zeros(1, 1, d)); s.pos_embed = nn.Parameter(torch.zeros(1, (img // p) ** 2 + 1, d))
        s.mask_token = nn.Parameter(torch.zeros(1, d)); s.blocks = nn.ModuleList([_Block(d) for _ in range(depth)])
        s.norm = nn.LayerNorm(d, eps=1e-6); s.head = nn.Identity()
    def interpolate_pos_encoding(s, x, H, W):
        N = s.pos_embed.shape[1] - 1; npatch = x.shape[1] - 1
        if npatch == N and H == W: return s.pos_embed
        pe = s.pos_embed.float(); cls, pp = pe[:, 0], pe[:, 1:]; dim = x.shape[-1]; M = int(math.sqrt(N))
        h0, w0 = H // s.patch_size, W // s.patch_size
        pp = F.interpolate(pp.reshape(1, M, M, dim).permute(0, 3, 1, 2), mode="bicubic", antialias=False,
                           scale_factor=(float(h0 + 0.1) / M, float(w0 + 0.1) / M))
        assert pp.shape[-2:] == (h0, w0)
        return torch.cat((cls.unsqueeze(0), pp.permute(0, 2, 3, 1).view(1, -1, dim)), 1).to(x.dtype)
    def forward_features(s, x):
        B, _, H, W = x.shape; t = s.patch_embed(x)
        t = torch.cat((s.cls_token.expand(B, -1, -1), t), 1); t = t + s.interpolate_pos_encoding(t, H, W)
        for b in s.blocks: t = b(t)
        n = s.norm(t)
        return {"x_norm_clstoken": n[:, 0], "x_norm_regtokens": n[:, 1:1], "x_norm_patchtokens": n[:, 1:], "x_prenorm": t, "masks": None}


def _import_model_classes():
    if BUNDLE not in sys.path: sys.path.insert(0, BUNDLE)
    from rgbd_act_model_v2 import RGBDACT, RGBDACTConfig  # bundle/rgbd_act_model_v2.py (read-only)
    return RGBDACT, RGBDACTConfig


# ----------------------------------------------------------------------------- policy
class ACTPolicy:
    def __init__(self, device: str = "cuda:0", checkpoint: str = CHECKPOINT):
        RGBDACT, RGBDACTConfig = _import_model_classes()
        ck = torch.load(checkpoint, map_location="cpu", weights_only=False)
        names = tuple(ck["joint_names"])
        assert names == JOINT_NAMES, f"checkpoint joint order differs: {names}"
        cfg = RGBDACTConfig(**ck["config"])
        assert (cfg.image_h, cfg.image_w, cfg.chunk_size, cfg.num_joints) == (IMG_H, IMG_W, HORIZON, 46), cfg
        a = ck.get("args", {})
        assert int(a.get("target_offset", 1)) == 1, a.get("target_offset")
        orig = torch.hub.load
        torch.hub.load = lambda repo, model, *x, **k: DinoVisionTransformer() if ("dinov2" in str(repo) and model == "dinov2_vits14") else orig(repo, model, *x, **k)
        try:
            m = RGBDACT(cfg, freeze_dino=True)
        finally:
            torch.hub.load = orig
        m.load_state_dict(ck["model"], strict=True)
        self.device = torch.device(device)
        self.model = m.to(self.device).eval()
        self.q_mean = ck["q_mean"].float().to(self.device)
        self.q_std = ck["q_std"].float().to(self.device)
        self.const_joints = torch.nonzero(ck["q_std"] <= 1.0001e-4).flatten().to(self.device)
        self.checkpoint = checkpoint; self.epoch = ck.get("epoch")
        self.ckpt_depth_stats = ck.get("depth_stats")  # stale, kept for reference only

    @torch.no_grad()
    def predict(self, obs_hist: Sequence[Obs], qs: np.ndarray, seed: int = 0) -> np.ndarray:
        if len(obs_hist) != N_OBS:
            raise ValueError(f"obs_hist must have length {N_OBS}, got {len(obs_hist)}")
        qs = np.asarray(qs, dtype=np.float32).reshape(-1, 46)
        if qs.shape[0] != N_OBS:
            raise ValueError(f"qs must be [{N_OBS},46], got {qs.shape}")
        torch.manual_seed(int(seed))
        o = obs_hist[-1]
        rgb = o.rgb.unsqueeze(0).to(self.device); dep = o.depth.unsqueeze(0).to(self.device)
        q = torch.from_numpy(qs[-1].copy()).to(self.device)
        if CLAMP_CONSTANT_JOINTS and self.const_joints.numel():
            q[self.const_joints] = self.q_mean[self.const_joints]
        qn = ((q - self.q_mean) / self.q_std).unsqueeze(0)
        pred = self.model(rgb, dep, qn, actions=None)["pred"]          # (1,30,46) normalised absolute
        out = pred[0] * self.q_std + self.q_mean                          # rad, rows = q[t+1..t+30]
        return out.float().cpu().numpy()[:HORIZON].astype(np.float32)


def load_policy(device: str = "cuda:0", checkpoint: str = CHECKPOINT) -> ACTPolicy:
    return ACTPolicy(device, checkpoint)


# ----------------------------------------------------------------------------- self-test
RAW = "/home/emagesoft/Desktop/idp3colourrgbtraining/dataset/raw/findandpick_new"
PREP = BUNDLE + "/prepared_rgbd"


def _read_exr(path):
    import OpenEXR
    ch = OpenEXR.File(str(path)).channels()
    return np.asarray(ch["Depth.V"].pixels, dtype=np.float32)


def _read_video_frames(ep, idxs):
    import yaml
    doc = yaml.safe_load(open(f"{RAW}/{ep}/{ep}.mmrt.yaml"))
    start = int(doc.get("metadata", {}).get("start_frame", 0))
    cap = cv2.VideoCapture(f"{RAW}/{ep}/{ep}.mp4")
    out = {}
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)   # decode sequentially from start (as prepare:246-259)
    for i in range(max(idxs) + 1):
        ok, bgr = cap.read(); assert ok, (ep, i)
        if i in idxs: out[i] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    cap.release()
    return out


def _selftest(device="cuda:0"):
    import time
    print(f"torch {torch.__version__} cv2 {cv2.__version__} device {device}")
    pol = load_policy(device)
    print(f"loaded {os.path.basename(pol.checkpoint)} epoch={pol.epoch} N_OBS={N_OBS} HORIZON={HORIZON} "
          f"depth near/far used={DEPTH_NEAR_M}/{DEPTH_FAR_M} (ckpt stale {pol.ckpt_depth_stats.get('near_m'):.3f}/{pol.ckpt_depth_stats.get('far_m'):.3f})")
    ep = "Find_and_Pick_004"; idxs = [0, 100, 250]
    frames = _read_video_frames(ep, idxs)
    R = np.load(f"{PREP}/{ep}/rgb.npy", mmap_mode="r"); D = np.load(f"{PREP}/{ep}/depth.npy", mmap_mode="r"); Q = np.load(f"{PREP}/{ep}/q.npy")
    maxr = maxd = 0.0; n1 = []; ch = []
    for t in idxs:
        dep = _read_exr(f"{RAW}/{ep}/depth/depth_{t:04d}.exr")
        obs = build_obs(frames[t], dep, t)
        maxr = max(maxr, float((obs.rgb - rgb_cache_to_tensor(R[t])).abs().max()))
        maxd = max(maxd, float((obs.depth - depth_cache_to_tensor(D[t])).abs().max()))
        y = pol.predict([obs], Q[t:t + 1], seed=0)
        assert y.shape == (HORIZON, 46) and y.dtype == np.float32 and np.isfinite(y).all()
        n1.append(np.abs(y[0] - Q[t + 1]).mean()); ch.append(np.abs(y - Q[t + 1:t + 31]).mean())
    print(f"[{ep} frames {idxs}] build_obs vs training tensors: rgb max|d|={maxr:.3e} depth max|d|={maxd:.3e}")
    print(f"teacher-forced: next-step L1={np.mean(n1):.5f} rad  chunk L1={np.mean(ch):.5f} rad")
    # Isaac-style inf background
    d_inf = dep.copy(); d_inf[d_inf > 8.0] = np.inf
    o_inf = build_obs(frames[idxs[-1]], d_inf, 0)
    print(f"inf-background obs == Blender-background obs: max|d|={float((o_inf.depth - obs.depth).abs().max()):.3e}")
    y1 = pol.predict([obs], Q[idxs[-1]:idxs[-1] + 1], seed=1); y2 = pol.predict([obs], Q[idxs[-1]:idxs[-1] + 1], seed=2)
    print(f"determinism across seeds: max|d|={np.abs(y1 - y2).max():.2e}")
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(); ts = []
        for _ in range(30):
            t0 = time.perf_counter(); pol.predict([obs], Q[:1]); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        print(f"predict latency median {np.median(ts):.1f} ms (incl. H2D/D2H)")
    ok = maxr < 1e-6 and maxd < 1e-6 and np.mean(n1) < 0.02
    print("SELFTEST", "PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--selftest", action="store_true"); ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(0 if _selftest(a.device) else 1)
    ap.print_help()
