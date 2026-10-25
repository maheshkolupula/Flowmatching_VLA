from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn.functional as F



@dataclass
class LossWeights:
    action: float = 1.0
    hold: float = 0.5          # weight of the HOLD steps after a skill's terminal frame (trained!)
    kl: float = 0.01
    suction: float = 0.4
    done: float = 0.3
    phone_pose: float = 1.0
    phone_heatmap: float = 0.5


def vla_losses(out, batch, w: LossWeights):
    hold = batch["hold_mask"].bool()
    step_w = torch.where(hold, torch.full_like(hold, w.hold, dtype=out["pred"].dtype),
                         torch.ones_like(hold, dtype=out["pred"].dtype))
    if out.get("velocity") is not None:     # flow matching loss (velocity MSE)
        err = (out["velocity"].float() - out["flow_target"].float()).pow(2).mean(-1)
    else:
        err = torch.abs(out["pred"].float() - batch["target"].float()).mean(-1)      # [B,30]
    action = (err * step_w).sum() / step_w.sum().clamp_min(1.0)
    kl = out["pred"].sum() * 0.0
    suction = F.binary_cross_entropy_with_logits(out["suction_logits"].float(), batch["suction_target"].float())
    done = F.binary_cross_entropy_with_logits(out["done_logits"].float(), batch["done_target"].float())

    vis = batch["phone_visible"] == 1                      # pose only where the image shows the phone
    if vis.any():
        p, y = out["phone_pose"].float()[vis], batch["phone_pose"].float()[vis]
        pose = (10.0 * torch.abs(p[:, :2] - y[:, :2]).mean(-1) + torch.abs(p[:, 2:] - y[:, 2:]).mean(-1)).mean()
    else:
        pose = out["phone_pose"].sum() * 0.0
    known = batch["phone_patch"] >= 0
    if known.any():
        heat = F.cross_entropy(out["phone_heatmap_logits"].float()[known], batch["phone_patch"][known])
    else:
        heat = out["phone_heatmap_logits"].sum() * 0.0

    total = (w.action * action + w.kl * kl + w.suction * suction + w.done * done
             + w.phone_pose * pose + w.phone_heatmap * heat)
    return {"total": total, "action_l1": action, "kl": kl, "suction": suction, "done": done,
            "phone_pose": pose, "phone_heatmap": heat}
