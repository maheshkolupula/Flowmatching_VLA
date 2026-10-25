"""Minimal offline DINOv2 ViT-S/14 definition matching the ACT-V2 checkpoint keys.

This is intentionally architecture-compatible with facebookresearch/dinov2
`dinov2_vits14`.  It lets checkpoint migration / deployment construct the
backbone without downloading weights: the real DINO weights are then loaded
from DEPLOY_V2_FINAL.pt.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class _PatchEmbed(nn.Module):
    def __init__(self, p=14, d=384):
        super().__init__()
        self.proj = nn.Conv2d(3, d, p, p)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class _Attention(nn.Module):
    def __init__(self, d=384, h=6):
        super().__init__()
        self.h = h
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)
        x = F.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2])
        return self.proj(x.transpose(1, 2).reshape(B, N, C))


class _LayerScale(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(d))

    def forward(self, x):
        return x * self.gamma


class _Mlp(nn.Module):
    def __init__(self, d=384, hd=1536):
        super().__init__()
        self.fc1 = nn.Linear(d, hd)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hd, d)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, d=384):
        super().__init__()
        self.norm1 = nn.LayerNorm(d, eps=1e-6)
        self.attn = _Attention(d)
        self.ls1 = _LayerScale(d)
        self.norm2 = nn.LayerNorm(d, eps=1e-6)
        self.mlp = _Mlp(d)
        self.ls2 = _LayerScale(d)

    def forward(self, x):
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoVisionTransformer(nn.Module):
    """DINOv2 ViT-S/14, no registers, shape/key compatible with the uploaded checkpoint."""

    def __init__(self, d=384, depth=12, p=14, img=518):
        super().__init__()
        self.patch_size = p
        self.patch_embed = _PatchEmbed(p, d)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img // p) ** 2 + 1, d))
        self.mask_token = nn.Parameter(torch.zeros(1, d))
        self.blocks = nn.ModuleList([_Block(d) for _ in range(depth)])
        self.norm = nn.LayerNorm(d, eps=1e-6)
        self.head = nn.Identity()

    def interpolate_pos_encoding(self, x, H, W):
        N = self.pos_embed.shape[1] - 1
        npatch = x.shape[1] - 1
        if npatch == N and H == W:
            return self.pos_embed
        pe = self.pos_embed.float()
        cls, pp = pe[:, 0], pe[:, 1:]
        dim = x.shape[-1]
        M = int(math.sqrt(N))
        h0, w0 = H // self.patch_size, W // self.patch_size
        pp = F.interpolate(
            pp.reshape(1, M, M, dim).permute(0, 3, 1, 2),
            mode="bicubic",
            antialias=False,
            scale_factor=(float(h0 + 0.1) / M, float(w0 + 0.1) / M),
        )
        if pp.shape[-2:] != (h0, w0):
            raise RuntimeError(f"positional interpolation mismatch {pp.shape[-2:]} != {(h0, w0)}")
        return torch.cat(
            (cls.unsqueeze(0), pp.permute(0, 2, 3, 1).reshape(1, -1, dim)), 1
        ).to(x.dtype)

    def forward_features(self, x):
        B, _, H, W = x.shape
        t = self.patch_embed(x)
        t = torch.cat((self.cls_token.expand(B, -1, -1), t), 1)
        t = t + self.interpolate_pos_encoding(t, H, W)
        for block in self.blocks:
            t = block(t)
        n = self.norm(t)
        return {
            "x_norm_clstoken": n[:, 0],
            "x_norm_regtokens": n[:, 1:1],
            "x_norm_patchtokens": n[:, 1:],
            "x_prenorm": t,
            "masks": None,
        }
