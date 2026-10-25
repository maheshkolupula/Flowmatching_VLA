from __future__ import annotations
import math

from contextlib import contextmanager
from dataclasses import dataclass, asdict
from pathlib import Path
import sys
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from rgbd_act_model_v2 import (  # noqa: E402
    RGBDACT,
    RGBDACTConfig,
    CVAEStyleEncoder,
    kl_divergence,
)
from .offline_dinov2 import DinoVisionTransformer


@dataclass
class TOMOVLAConfig(RGBDACTConfig):
    # Frozen SigLIP text hidden dimension.
    text_dim: int = 768
    max_text_tokens: int = 16

    # Conditioning corruption is applied ONLY to q_token + q_query_mlp.
    # q_clean is still used for pred = q_clean + delta.
    joint_noise_std: float = 0.02       # normalized-q units
    joint_dropout_p: float = 0.10       # whole joint-conditioning vector (set to 0 while the q path is frozen)
    q_norm_clip: float = 6.0            # moved-phone IK demos reach |q_norm| ~6 (ACT data 4.6)

    # Auxiliary outputs.
    heatmap_classes: int = 433          # 18*24 patches + not-visible
    suction_init_bias: float = -4.0
    done_init_bias: float = -4.0


class MaskedCVAEStyleEncoder(CVAEStyleEncoder):
    """Same parameter names/shapes as the old CVAE, but supports ACT padding masks."""

    def forward(self, q, a, action_valid_mask: Optional[torch.Tensor] = None):
        b = q.shape[0]
        x = torch.cat(
            [self.cls.expand(b, -1, -1), self.q_proj(q).unsqueeze(1), self.a_proj(a)], 1
        ) + self.pos

        key_padding_mask = None
        if action_valid_mask is not None:
            valid = action_valid_mask.bool()
            if valid.shape != (b, self.cfg.chunk_size):
                raise ValueError(
                    f"action_valid_mask {tuple(valid.shape)} != {(b, self.cfg.chunk_size)}"
                )
            prefix = torch.zeros((b, 2), dtype=torch.bool, device=q.device)
            key_padding_mask = torch.cat([prefix, ~valid], dim=1)

        h = self.encoder(x, src_key_padding_mask=key_padding_mask)[:, 0]
        mu = self.mu(h)
        lv = self.logvar(h).clamp(-10, 10)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        return z, mu, lv


class IdentityFiLM(nn.Module):
    """x -> x*(1+gamma(lang)) + beta(lang), exactly identity at initialization."""

    def __init__(self, cond_dim: int, d_model: int):
        super().__init__()
        self.to_affine = nn.Linear(cond_dim, 2 * d_model)
        nn.init.zeros_(self.to_affine.weight)
        nn.init.zeros_(self.to_affine.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.to_affine(cond).chunk(2, dim=-1)
        while gamma.ndim < x.ndim:
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return x * (1.0 + gamma) + beta


class TOMOVLA(RGBDACT):
    """Checkpoint-compatible DINOv2 RGB-D ACT upgraded into a language-conditioned VLA.

    Old joint path is preserved:
      RGB + depth + q + z -> ACT decoder -> delta -> q_clean + delta

    New paths:
      sentence FiLM -> visual tokens
      word-token cross attention -> ACT memory (zero residual at initialization)
      sentence FiLM -> action queries
      decoder -> suction state + per-step done
      visual tokens -> phone pose + 18x24+not-visible heat-map
    """

    NEW_PREFIXES = (
        "vision_film.",
        "query_film.",
        "text_token_proj.",
        "lang_attn.",
        "lang_attn_gate",
        "suction_head.",
        "done_head.",
        "phone_pose_head.",
        "phone_heat_patch.",
        "phone_not_visible.",
        "phone_query_proj.",
        "flow_in.",
        "flow_time.",
    )

    def __init__(self, cfg: TOMOVLAConfig, freeze_dino: bool = True):
        super().__init__(cfg, freeze_dino=freeze_dino)
        self.cfg = cfg

        # Replace CVAE with a mask-aware implementation. Keys/shapes stay compatible.
        self.cvae = MaskedCVAEStyleEncoder(cfg)

        self.vision_film = IdentityFiLM(cfg.text_dim, cfg.d_model)
        self.query_film = IdentityFiLM(cfg.text_dim, cfg.d_model)

        self.text_token_proj = nn.Sequential(
            nn.Linear(cfg.text_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )
        self.lang_attn = nn.MultiheadAttention(
            cfg.d_model, cfg.nhead, dropout=cfg.dropout, batch_first=True
        )
        # Exact zero = old ACT memory is unchanged at migration.
        # The scalar gets a gradient on the first step; attention weights start
        # receiving gradients as soon as the gate leaves zero.
        self.lang_attn_gate = nn.Parameter(torch.zeros(()))

        self.suction_head = nn.Linear(cfg.d_model, 1)
        self.done_head = nn.Linear(cfg.d_model, 1)
        nn.init.zeros_(self.suction_head.weight)
        nn.init.constant_(self.suction_head.bias, cfg.suction_init_bias)
        nn.init.zeros_(self.done_head.weight)
        nn.init.constant_(self.done_head.bias, cfg.done_init_bias)

        self.phone_pose_head = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Linear(cfg.d_model, 4),  # x, y, sin(yaw), cos(yaw)
        )
        self.phone_heat_patch = nn.Linear(cfg.d_model, 1)
        self.phone_not_visible = nn.Sequential(
            nn.LayerNorm(cfg.d_model),
            nn.Linear(cfg.d_model, 1),
        )
        # "Phone feature" = visual tokens pooled with the heat-map attention. It feeds the pose head
        # (so pose is read where the phone is) and, through a zero-initialised projection, the 30
        # action queries (so the decoder gets an explicit "the phone is here" signal). Zero init
        # keeps the migrated model exactly equal to DEPLOY_V2_FINAL at step 0.
        self.phone_query_proj = nn.Linear(cfg.d_model, cfg.d_model)
        nn.init.zeros_(self.phone_query_proj.weight)
        nn.init.zeros_(self.phone_query_proj.bias)
        # ---- flow matching (vlaflow): noisy delta chunk + time -> queries; self.out predicts velocity
        self.flow_in = nn.Linear(cfg.num_joints, cfg.d_model)
        self.flow_time = nn.Sequential(nn.Linear(64, cfg.d_model), nn.GELU(), nn.Linear(cfg.d_model, cfg.d_model))
        nn.init.zeros_(self.flow_in.weight); nn.init.zeros_(self.flow_in.bias)
        nn.init.zeros_(self.flow_time[2].weight); nn.init.zeros_(self.flow_time[2].bias)
        self.flow_steps = 10

    @staticmethod
    def _time_emb(s: torch.Tensor) -> torch.Tensor:
        half = 32
        f = torch.exp(-math.log(10000.0) * torch.arange(half, device=s.device, dtype=torch.float32) / half)
        a = (s.float() * 100.0)[:, None] * f[None]
        return torch.cat([a.sin(), a.cos()], -1)

    def _flow_decode(self, base_q, mem, x, s):
        qs = base_q + self.flow_in(x.to(base_q.dtype)) + self.flow_time(self._time_emb(s).to(base_q.dtype)).unsqueeze(1)
        h = self.decoder(qs, mem)
        return self.out(h), h

    def corrupt_joint_condition(self, q: torch.Tensor) -> torch.Tensor:
        """Corrupt the *conditioning* copy of q, never the residual reference q_clean."""
        if not self.training:
            return q
        qc = q
        if self.cfg.joint_noise_std > 0:
            qc = qc + torch.randn_like(qc) * float(self.cfg.joint_noise_std)
        p = float(self.cfg.joint_dropout_p)
        if p > 0:
            # One dropout decision per sample; same corrupted q goes to BOTH q paths.
            keep = (torch.rand((q.shape[0], 1), device=q.device) >= p).to(q.dtype)
            qc = qc * keep
        return qc

    def _language_defaults(
        self,
        q: torch.Tensor,
        text_tokens: Optional[torch.Tensor],
        text_mask: Optional[torch.Tensor],
        sentence: Optional[torch.Tensor],
    ):
        b = q.shape[0]
        dev, dt = q.device, q.dtype
        if text_tokens is None:
            text_tokens = torch.zeros(
                b, self.cfg.max_text_tokens, self.cfg.text_dim, device=dev, dtype=dt
            )
        if text_mask is None:
            text_mask = torch.zeros(
                b, self.cfg.max_text_tokens, device=dev, dtype=torch.bool
            )
        else:
            text_mask = text_mask.to(device=dev, dtype=torch.bool)
        if sentence is None:
            sentence = torch.zeros(b, self.cfg.text_dim, device=dev, dtype=dt)

        text_tokens = text_tokens.to(device=dev, dtype=dt)
        sentence = sentence.to(device=dev, dtype=dt)

        # MultiheadAttention cannot safely receive a row with every key masked.
        # Give empty-language rows one zero key, then zero the residual afterward.
        present = text_mask.any(dim=1)
        safe_mask = text_mask.clone()
        if (~present).any():
            safe_mask[~present, 0] = True
            text_tokens = text_tokens.clone()
            text_tokens[~present, 0] = 0
        return text_tokens, safe_mask, sentence, present

    def encode_vla_observation(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        q_condition: torch.Tensor,
        z: torch.Tensor,
        text_tokens: torch.Tensor,
        text_mask: torch.Tensor,
        sentence: torch.Tensor,
        language_present: torch.Tensor,
    ):
        rt = self.rgb_encoder(rgb)
        dt = self.depth_encoder(depth)
        if rt.shape[:2] != dt.shape[:2]:
            raise RuntimeError(f"grid mismatch {rt.shape} {dt.shape}")

        # Base visual representation: this is what the training-only grounding
        # heads supervise, so phone geometry cannot be delegated to language.
        base_fused = self.fusion(torch.cat([rt, dt], -1)) + self._spatial_pos(rt.shape[0])

        # Language selects / modulates relevant visual content.
        fused = self.vision_film(base_fused, sentence)

        specials = torch.cat(
            [
                self.q_token(q_condition).unsqueeze(1),
                self.z_token(z).unsqueeze(1),
            ],
            1,
        ) + self.special_pos
        mem = torch.cat([fused, specials], 1)

        # Word-token language enters through a residual cross-attention adapter.
        lt = self.text_token_proj(text_tokens)
        cross, _ = self.lang_attn(
            query=mem,
            key=lt,
            value=lt,
            key_padding_mask=~text_mask,
            need_weights=False,
        )
        cross = cross * language_present.to(cross.dtype).view(-1, 1, 1)
        mem = mem + self.lang_attn_gate * cross

        pooled = base_fused.mean(dim=1)
        patch_logits = self.phone_heat_patch(base_fused).squeeze(-1)  # [B,432]
        invisible_logit = self.phone_not_visible(pooled)               # [B,1]
        heatmap_logits = torch.cat([patch_logits, invisible_logit], dim=1)
        attn = torch.softmax(patch_logits.float(), dim=1).to(base_fused.dtype)   # where is the phone
        phone_feat = torch.einsum("bn,bnd->bd", attn, base_fused)
        pose = self.phone_pose_head(phone_feat)

        shapes = {
            "rgb_tokens": tuple(rt.shape),
            "depth_tokens": tuple(dt.shape),
            "fused_tokens": tuple(fused.shape),
            "memory": tuple(mem.shape),
        }
        return mem, pose, heatmap_logits, phone_feat, shapes

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        q: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None,
        text_mask: Optional[torch.Tensor] = None,
        sentence: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        action_valid_mask: Optional[torch.Tensor] = None,
    ):
        b = q.shape[0]
        clip = float(getattr(self.cfg, "q_norm_clip", 0.0) or 0.0)
        q_clean = q.clamp(-clip, clip) if clip > 0 else q
        q_condition = self.corrupt_joint_condition(q_clean)

        # flow matching: no CVAE latent (z = 0)
        z = torch.zeros(b, self.cfg.latent_dim, device=q.device, dtype=q.dtype)
        mu = torch.zeros_like(z)
        lv = torch.zeros_like(z)

        text_tokens, text_mask, sentence, present = self._language_defaults(
            q_clean, text_tokens, text_mask, sentence
        )
        mem, phone_pose, heatmap_logits, phone_feat, shapes = self.encode_vla_observation(
            rgb,
            depth,
            q_condition,
            z,
            text_tokens,
            text_mask,
            sentence,
            present,
        )

        # SAME corrupted q is used in the second joint-conditioning route.
        qcond = self.q_query_mlp(q_condition)
        queries = self.action_queries.expand(b, -1, -1) + qcond.unsqueeze(1)
        queries = queries + self.phone_query_proj(phone_feat).unsqueeze(1)
        queries = self.query_film(queries, sentence)

        T = queries.shape[1]
        if actions is not None:
            # rectified flow: x_s = s*x0 + (1-s)*noise, target velocity = x0 - noise
            x0 = actions - q_clean.unsqueeze(1)
            noise = torch.randn_like(x0)
            s = torch.rand(b, device=q.device)
            xs = s[:, None, None] * x0 + (1 - s[:, None, None]) * noise
            v, h = self._flow_decode(queries, mem, xs, s)
            flow_target = x0 - noise
            delta = xs + (1 - s[:, None, None]) * v.float()           # one-step x0 estimate (logging)
        else:
            x = torch.randn(b, T, self.cfg.num_joints, device=q.device)
            n = int(self.flow_steps)
            for i in range(n):
                s = torch.full((b,), i / n, device=q.device)
                v, h = self._flow_decode(queries, mem, x, s)
                x = x + v.float() / n
            delta, v, flow_target = x, None, None
        pred = q_clean.unsqueeze(1) + delta

        suction_logits = self.suction_head(h).squeeze(-1)
        done_logits = self.done_head(h).squeeze(-1)

        return {
            "pred": pred,
            "delta": delta,
            "suction_logits": suction_logits,
            "done_logits": done_logits,
            "phone_pose": phone_pose,
            "phone_heatmap_logits": heatmap_logits,
            "mu": mu,
            "logvar": lv,
            "debug_shapes": shapes,
            "velocity": v,
            "flow_target": flow_target,
        }


@contextmanager
def _offline_dino_patch(enabled: bool = True):
    if not enabled:
        yield
        return
    original = torch.hub.load

    def loader(repo, model, *args, **kwargs):
        if "dinov2" in str(repo) and model == "dinov2_vits14":
            return DinoVisionTransformer()
        return original(repo, model, *args, **kwargs)

    torch.hub.load = loader
    try:
        yield
    finally:
        torch.hub.load = original


def build_vla_model(
    cfg: TOMOVLAConfig,
    freeze_dino: bool = True,
    offline_dino: bool = True,
) -> TOMOVLA:
    with _offline_dino_patch(offline_dino):
        return TOMOVLA(cfg, freeze_dino=freeze_dino)


def config_from_old(old_cfg: dict, **overrides) -> TOMOVLAConfig:
    base = dict(old_cfg)
    base.update(overrides)
    return TOMOVLAConfig(**base)
