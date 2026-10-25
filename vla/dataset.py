"""Skill-chunk dataset for the TOMO DINO-ACT VLA.

One sample = (episode, skill segment, observed frame t):
    observation : RGB 252x336 + depth + 46 joints at frame t (exact ACT preprocessing)
    command     : one training paraphrase of the segment (language dropout -> empty = FULL_TASK only)
    targets     : 30 future joint poses q[t+1..t+30] (ACT target_offset=1). Steps after the segment's
                  terminal frame are HOLD targets (= q[end]) and ARE trained, so a sub-skill command
                  ("reach the phone") and the full command produce different trained actions.
                  done[j] = 1 from the step that reaches the terminal frame on; suction[j] = recorded
                  vacuum state at that step.
Augmentation (train only, all random draws from a per-call RNG, so DataLoader workers never repeat):
    virtual time-warp of the FUTURE chunk (speed 0.85-1.2, no disk copies; the observation is always a
    real frame), small shift/scale (RGB bilinear, depth NEAREST, phone label moved with it), colour
    jitter / gamma / noise / blur, depth noise + holes, small random erasing away from the phone.
Episode folder contract (all domains):
    q.npy [T,46] float32 (checkpoint joint order), rgb.npy [N,252,336,3] uint8,
    depth.npy [N,252,336] float16 metres (finite; no-hit = far plane), optional frames.npy [N] (q-frame
    of each image row, default arange(T)), optional depth_valid.npy [N], vla_labels.npz, segments.json,
    manifest.json {"source_episode", "domain"}.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
from typing import Iterable, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode

from .text_encoder import LanguageBank

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
IMG_H, IMG_W = 252, 336
NATIVE_W, NATIVE_H = 960.0, 720.0
GRID_H, GRID_W = 18, 24
NOT_VISIBLE = GRID_H * GRID_W          # heat-map class 432
DEPTH_NEAR, DEPTH_FAR = 0.15, 2.0      # exact ACT deployment normalisation


def norm_depth(d: torch.Tensor):
    """Same rule as the working ACT (train loader + act_rgbd_dinov2_adapter)."""
    valid = torch.isfinite(d) & (d > 0)
    dm = torch.nan_to_num(d, nan=DEPTH_FAR, posinf=DEPTH_FAR, neginf=DEPTH_NEAR).clamp(DEPTH_NEAR, DEPTH_FAR)
    dn = (dm - DEPTH_NEAR) / (DEPTH_FAR - DEPTH_NEAR)
    return torch.where(valid, dn, torch.zeros_like(dn)), valid.float()


def uv_to_patch(u336: float, v252: float) -> int:
    if not (0.0 <= u336 < IMG_W and 0.0 <= v252 < IMG_H):
        return NOT_VISIBLE
    return int(v252 // (IMG_H / GRID_H)) * GRID_W + int(u336 // (IMG_W / GRID_W))


def read_manifest(ep: Path) -> dict:
    p = ep / "manifest.json"
    doc = json.loads(p.read_text()) if p.exists() else {}
    doc.setdefault("source_episode", "_".join(ep.name.split("_")[:4]))
    doc.setdefault("domain", "blender")
    return doc


class EncodedFrames:
    """Rows of JPEG (RGB) or 16-bit PNG (depth in mm) stored as one flat uint8 array + offsets."""

    def __init__(self, data_path: Path, off_path: Path, kind: str):
        self.data = np.load(data_path, mmap_mode="r")
        self.off = np.load(off_path)
        self.kind = kind

    def __len__(self):
        return len(self.off) - 1

    def __getitem__(self, i):
        import cv2
        buf = np.asarray(self.data[self.off[i]:self.off[i + 1]])
        if self.kind == "rgb":
            return cv2.cvtColor(cv2.imdecode(buf, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        return cv2.imdecode(buf, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0     # mm -> m


class VLASkillDataset(Dataset):
    def __init__(
        self,
        roots: Sequence[tuple[str, Path]],          # [(domain_name, root_dir), ...]
        allowed_source_episodes: set[str],
        q_mean: np.ndarray,
        q_std: np.ndarray,
        language_bank: Path,
        chunk_size: int = 30,
        target_offset: int = 1,
        augment: bool = False,
        aug_prob: float = 0.8,
        warp_prob: float = 0.3,
        warp_range: tuple[float, float] = (0.85, 1.2),
        language_dropout: float = 0.10,
        seed: int = 42,
        use_test_instructions: bool = False,
        max_episodes_per_root: int | None = None,
    ):
        self.qm = np.asarray(q_mean, np.float32)
        self.qs = np.asarray(q_std, np.float32)
        self.H = int(chunk_size)
        self.offset = int(target_offset)
        self.augment = bool(augment)
        self.aug_prob = float(aug_prob)
        self.warp_prob = float(warp_prob)
        self.warp_range = tuple(float(x) for x in warp_range)
        self.lang_dropout = float(language_dropout)
        self.seed = int(seed)
        self.use_test = bool(use_test_instructions)
        self.bank_path = Path(language_bank)
        self.bank = LanguageBank(self.bank_path)
        self._calls = 0
        self._arr: dict[int, dict] = {}
        self.episodes: list[dict] = []
        self.samples: list[tuple[int, int, int, int]] = []    # (ep, seg, t, image_row)

        for domain, root in roots:
            root = Path(root)
            eps = sorted(p for p in root.iterdir() if p.is_dir() and (p / "segments.json").exists())
            if max_episodes_per_root:
                eps = eps[:max_episodes_per_root]
            for ep in eps:
                man = read_manifest(ep)
                if man["source_episode"] not in allowed_source_episodes:
                    continue
                q = np.load(ep / "q.npy", mmap_mode="r")
                T = len(q)
                frames = np.load(ep / "frames.npy") if (ep / "frames.npy").exists() else np.arange(T)
                row_of = np.full(T, -1, np.int64)
                row_of[frames] = np.arange(len(frames))
                if (ep / "depth_valid.npy").exists():
                    dv = np.load(ep / "depth_valid.npy").astype(bool)
                    row_of[frames[~dv]] = -1
                segdoc = json.loads((ep / "segments.json").read_text())
                ei = len(self.episodes)
                self.episodes.append({"dir": ep, "domain": domain, "source": man["source_episode"], "T": T,
                                      "segments": segdoc["segments"], "row_of": row_of})
                for si, seg in enumerate(segdoc["segments"]):
                    s, e = int(seg["start"]), int(seg["end"])
                    for t in range(max(0, s - self.offset), e + 1):
                        if t < T and row_of[t] >= 0:
                            self.samples.append((ei, si, t, int(row_of[t])))
        if not self.samples:
            raise RuntimeError("No VLA samples found: check the data roots, split and labels.")

    # ------------------------------------------------------------------ bookkeeping
    def __len__(self):
        return len(self.samples)

    def skill_of(self, idx: int) -> str:
        ei, si, _, _ = self.samples[idx]
        return self.episodes[ei]["segments"][si]["name"]

    def domain_of(self, idx: int) -> str:
        return self.episodes[self.samples[idx][0]]["domain"]

    def sample_weights(self, domain_weights: dict[str, float] | None = None) -> torch.Tensor:
        """w = domain weight / sqrt(#samples of this (domain, skill)): rare skills (GRASP, REACH) are
        drawn far more often than their raw 0.6-3 % share, without being repeated 100x."""
        domain_weights = domain_weights or {}
        counts: dict[tuple, int] = {}
        keys = [(self.domain_of(i), self.skill_of(i)) for i in range(len(self.samples))]
        for k in keys:
            counts[k] = counts.get(k, 0) + 1
        w = np.array([domain_weights.get(d, 1.0) / math.sqrt(counts[(d, s)]) for d, s in keys], np.float64)
        return torch.as_tensor(w / w.sum(), dtype=torch.double)

    def arrays(self, ei: int) -> dict:
        if ei not in self._arr:
            ep = self.episodes[ei]["dir"]
            lab = np.load(ep / "vla_labels.npz")
            if (ep / "rgb.npy").exists():                      # raw cache (Blender / ACT prepared_rgbd)
                rgb, depth = np.load(ep / "rgb.npy", mmap_mode="r"), np.load(ep / "depth.npy", mmap_mode="r")
            else:                                               # compressed Isaac render
                rgb = EncodedFrames(ep / "rgb_jpg.npy", ep / "rgb_off.npy", "rgb")
                depth = EncodedFrames(ep / "depth_png.npy", ep / "depth_off.npy", "depth")
            self._arr[ei] = {
                "rgb": rgb,
                "depth": depth,
                "q": np.asarray(np.load(ep / "q.npy", mmap_mode="r"), np.float32),
                "suction": lab["suction"].astype(np.float32),
                "phone_pose": lab["phone_pose"].astype(np.float32),
                "phone_uv": lab["phone_uv"].astype(np.float32),
                "phone_visible": lab["phone_visible"].astype(np.int64),
            }
        return self._arr[ei]

    def _rng(self, idx: int) -> np.random.Generator:
        """Unique stream per call and per worker (fixes the shared-seed copy of the original bundle)."""
        self._calls += 1
        wi = torch.utils.data.get_worker_info()
        wseed = (wi.seed if wi is not None else torch.initial_seed()) % (2 ** 32)
        return np.random.default_rng([self.seed, wseed, self._calls, idx])

    # ------------------------------------------------------------------ targets
    def _targets(self, a: dict, t: int, end: int, speed: float):
        H = self.H
        times = t + speed * (np.arange(1, H + 1, dtype=np.float64) + (self.offset - 1))
        hold = times > end
        tc = np.minimum(times, end)
        lo = np.floor(tc).astype(np.int64)
        hi = np.minimum(lo + 1, end)
        f = (tc - lo)[:, None].astype(np.float32)
        q = a["q"]
        tgt = (1 - f) * q[lo] + f * q[hi]
        suction = a["suction"][np.rint(tc).astype(np.int64)]
        done = (times >= end).astype(np.float32)
        tgt = (tgt - self.qm) / self.qs
        return (torch.from_numpy(tgt.astype(np.float32)), torch.from_numpy(hold),
                torch.from_numpy(suction.astype(np.float32)), torch.from_numpy(done))

    # ------------------------------------------------------------------ augmentation
    def _augment(self, rng, rgb, depth, dmask, uv):
        """rgb [3,H,W] 0..1, depth [1,H,W] normalised, dmask [1,H,W], uv (u336, v252) or None."""
        if rng.random() < 0.5:                                   # small camera shake / zoom
            tx = int(rng.integers(-10, 11))
            ty = int(rng.integers(-8, 9))
            s = float(rng.uniform(0.96, 1.04))
            kw = dict(angle=0.0, translate=[tx, ty], scale=s, shear=[0.0, 0.0])
            rgb = TF.affine(rgb, interpolation=InterpolationMode.BILINEAR, fill=0.0, **kw)
            depth = TF.affine(depth, interpolation=InterpolationMode.NEAREST, fill=0.0, **kw)
            dmask = TF.affine(dmask, interpolation=InterpolationMode.NEAREST, fill=0.0, **kw)
            depth = depth * (dmask > 0.5)
            if uv is not None:
                cx, cy = (IMG_W - 1) * 0.5, (IMG_H - 1) * 0.5
                uv = (cx + s * (uv[0] - cx) + tx, cy + s * (uv[1] - cy) + ty)
        if rng.random() < 0.8:
            rgb = TF.adjust_brightness(rgb, float(rng.uniform(0.85, 1.15)))
            rgb = TF.adjust_contrast(rgb, float(rng.uniform(0.85, 1.15)))
            rgb = TF.adjust_saturation(rgb, float(rng.uniform(0.9, 1.1)))
            rgb = TF.adjust_hue(rgb, float(rng.uniform(-0.02, 0.02)))
        if rng.random() < 0.3:
            rgb = TF.adjust_gamma(rgb.clamp(0, 1), float(rng.uniform(0.9, 1.1)))
        if rng.random() < 0.3:
            g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
            rgb = rgb + torch.randn(rgb.shape, generator=g) * float(rng.uniform(0.005, 0.02))
        if rng.random() < 0.1:
            rgb = TF.gaussian_blur(rgb, kernel_size=3, sigma=float(rng.uniform(0.3, 0.9)))
        if rng.random() < 0.15:                                  # small occluder away from the phone
            eh, ew = int(rng.integers(10, 40)), int(rng.integers(10, 50))
            y0, x0 = int(rng.integers(0, IMG_H - eh)), int(rng.integers(0, IMG_W - ew))
            far = uv is None or not (x0 - 25 <= uv[0] <= x0 + ew + 25 and y0 - 25 <= uv[1] <= y0 + eh + 25)
            if far:
                rgb[:, y0:y0 + eh, x0:x0 + ew] = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        rgb = rgb.clamp(0.0, 1.0)
        if rng.random() < 0.5:                                   # depth sensor noise (normalised units)
            g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
            noise = torch.randn(depth.shape, generator=g) * (0.002 / (DEPTH_FAR - DEPTH_NEAR))
            depth = torch.where(dmask > 0.5, (depth * float(rng.uniform(0.995, 1.005)) + noise).clamp(0, 1), depth)
        if rng.random() < 0.3:                                   # sparse depth holes
            g = torch.Generator().manual_seed(int(rng.integers(1 << 31)))
            holes = torch.rand(depth.shape, generator=g) < float(rng.uniform(0.005, 0.02))
            depth = depth.masked_fill(holes, 0.0)
            dmask = dmask.masked_fill(holes, 0.0)
        return rgb, depth, dmask, uv

    # ------------------------------------------------------------------ item
    def __getitem__(self, idx):
        rng = self._rng(idx)
        ei, si, t, row = self.samples[idx]
        epi = self.episodes[ei]
        seg = epi["segments"][si]
        a = self.arrays(ei)

        rgb = torch.from_numpy(np.asarray(a["rgb"][row]).copy()).float().permute(2, 0, 1) / 255.0
        depth, dmask = norm_depth(torch.from_numpy(np.asarray(a["depth"][row], np.float32).copy()).unsqueeze(0))
        q = torch.from_numpy((a["q"][t] - self.qm) / self.qs)

        speed = 1.0
        if self.augment and rng.random() < self.warp_prob:
            speed = float(rng.uniform(*self.warp_range))
        target, hold, suction_t, done_t = self._targets(a, t, int(seg["end"]), speed)

        vis = int(a["phone_visible"][t])
        uv = None
        if vis == 1:
            uv = (float(a["phone_uv"][t, 0]) * IMG_W / NATIVE_W, float(a["phone_uv"][t, 1]) * IMG_H / NATIVE_H)
        if self.augment and rng.random() < self.aug_prob:
            rgb, depth, dmask, uv = self._augment(rng, rgb, depth, dmask, uv)
        if vis == 1:
            patch = uv_to_patch(*uv)
            if patch == NOT_VISIBLE:
                vis = 0
        elif vis == 0:
            patch = NOT_VISIBLE
        else:
            patch = -1
        rgb = (rgb - IMAGENET_MEAN) / IMAGENET_STD

        pool = seg.get("test_instructions") if self.use_test and seg.get("test_instructions") else seg["instructions"]
        text = pool[int(rng.integers(len(pool)))]
        if seg["name"] == "FULL_TASK" and self.augment and rng.random() < self.lang_dropout:
            lang, text = self.bank.empty(), ""
        else:
            lang = self.bank.get(text)

        return {
            "rgb": rgb, "depth": depth, "q": q,
            "target": target, "hold_mask": hold, "suction_target": suction_t, "done_target": done_t,
            "text_tokens": lang["tokens"], "text_mask": lang["mask"], "sentence": lang["sentence"],
            "phone_pose": torch.from_numpy(a["phone_pose"][t].copy()),
            "phone_visible": torch.tensor(vis, dtype=torch.int64),
            "phone_patch": torch.tensor(patch, dtype=torch.int64),
            "instruction": text, "skill": seg["name"], "domain": epi["domain"],
            "episode": epi["dir"].name, "source_episode": epi["source"], "frame": int(t),
            "speed": float(speed),
        }
