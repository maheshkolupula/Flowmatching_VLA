from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
import torch

from .model import TOMOVLAConfig, build_vla_model, config_from_old


def migrate_act_checkpoint(
    old_path: str | Path,
    out_path: str | Path,
    text_dim: int = 768,
    max_text_tokens: int = 16,
    joint_noise_std: float = 0.02,
    joint_dropout_p: float = 0.10,
):
    old_path = Path(old_path)
    out_path = Path(out_path)
    old = torch.load(old_path, map_location="cpu", weights_only=False)
    cfg = config_from_old(
        old["config"],
        text_dim=text_dim,
        max_text_tokens=max_text_tokens,
        joint_noise_std=joint_noise_std,
        joint_dropout_p=joint_dropout_p,
    )
    model = build_vla_model(cfg, freeze_dino=True, offline_dino=True)
    missing, unexpected = model.load_state_dict(old["model"], strict=False)

    bad_missing = [
        k for k in missing
        if not any(k.startswith(prefix) or k == prefix for prefix in model.NEW_PREFIXES)
    ]
    if unexpected:
        raise RuntimeError(f"Unexpected old checkpoint keys: {unexpected[:20]}")
    if bad_missing:
        raise RuntimeError(f"Old weights failed to map into VLA: {bad_missing[:20]}")

    payload = {
        "schema": "tomo_dino_act_vla_v1",
        "model": model.state_dict(),
        "config": asdict(cfg),
        "q_mean": old["q_mean"],
        "q_std": old["q_std"],
        "joint_names": old["joint_names"],
        # DEPLOY_V2_FINAL carries a stale value (0.18/8.91); training and deployment use 0.15/2.0
        "depth_stats": {"near_m": 0.15, "far_m": 2.0, "scale_to_meters": 1.0, "isaac_inf_as_m": 8.914},
        "init_from": str(old_path),
        "init_from_epoch": old.get("epoch"),
        "training_stage": 0,
        "suction_trained": False,
        "done_trained": False,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return payload, missing


def load_vla_checkpoint(path: str | Path, device="cpu", freeze_dino=True):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("schema") not in ("tomo_dino_act_vla_v1", "tomo_dino_vla_flow_v1"):
        raise RuntimeError(f"Not a TOMO VLA checkpoint: {ck.get('schema')}")
    cfg = TOMOVLAConfig(**ck["config"])
    model = build_vla_model(cfg, freeze_dino=freeze_dino, offline_dino=True)
    res = model.load_state_dict(ck["model"], strict=False)
    bad = [k for k in res.missing_keys if not k.startswith("flow_")] + list(res.unexpected_keys)
    if bad:
        raise RuntimeError(f"checkpoint mismatch: {bad[:10]}")
    if res.missing_keys:
        print(f"[ckpt] new flow modules initialised: {len(res.missing_keys)} tensors")
    return model.to(device), ck
