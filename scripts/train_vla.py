#!/usr/bin/env python3
"""Train the TOMO DINO-ACT VLA (resumable).

Stages (what is trainable; DINOv2 stays frozen unless stage 3):
  1  new VLA modules (lr_new) + visual fusion/depth/positional (lr_fusion). Joint dropout OFF
     (the q path is frozen here, so dropout would only teach the language adapters to patch it).
  2  + ACT decoder, output, q_token, q_query_mlp, z_token, specials, action queries, CVAE (lr_act).
     Joint dropout ON (cfg.joint_dropout_p).
  3  + last 2 DINOv2 blocks (lr_dino). Optional.
Budget in optimizer updates (batch 16 ~ 60 ms/update on the RTX 5050 -> 30k updates ~ 35-45 min).
Checkpoints in --out: last.pt (full state, resume with --resume), best.pt (deploy payload, lowest
val score = action MAE [rad, whole chunk incl. holds] + 0.01*(1-done F1) + 0.01*(1-suction F1)),
metrics.csv, train.log lines every --log-every updates.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import csv
import json
import math
import random
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla.checkpoint import load_vla_checkpoint  # noqa: E402
from vla.dataset import VLASkillDataset  # noqa: E402
from vla.losses import LossWeights, vla_losses  # noqa: E402

FUSION = ("fusion.", "depth_encoder.", "row_pos", "col_pos")
ACT = ("decoder.", "out.", "q_token.", "q_query_mlp.", "z_token.", "special_pos", "action_queries", "cvae.")


def group_of(model, name):
    if name.startswith("rgb_encoder.backbone."):
        return "dino"
    if any(name.startswith(x) for x in model.NEW_PREFIXES):
        return "new"
    if any(name.startswith(x) for x in FUSION):
        return "fusion"
    if any(name.startswith(x) for x in ACT):
        return "act"
    return "other"


def set_stage(model, stage):
    for p in model.parameters():
        p.requires_grad_(False)
    on = {"new", "fusion"} | ({"act"} if stage >= 2 else set())
    for name, p in model.named_parameters():
        if group_of(model, name) in on:
            p.requires_grad_(True)
    if stage >= 3:
        model.rgb_encoder.set_last_n_blocks_trainable(2)
    else:
        model.rgb_encoder.frozen = True
    if stage < 2:
        model.cfg.joint_dropout_p = 0.0


def make_optimizer(model, lrs, wd):
    groups = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            groups.setdefault(group_of(model, name), []).append(p)
    return torch.optim.AdamW([{"params": v, "lr": lrs[k], "name": k} for k, v in groups.items()], weight_decay=wd)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def parse_data(items):
    out = []
    for x in items:
        name, _, path = x.partition("=")
        if not path:
            raise SystemExit(f"--data must be name=path, got {x!r}")
        out.append((name, Path(path).expanduser()))
    return out


def to_dev(b, dev):
    keys = ["rgb", "depth", "q", "target", "hold_mask", "suction_target", "done_target", "text_tokens",
            "text_mask", "sentence", "phone_pose", "phone_visible", "phone_patch"]
    return {**b, **{k: b[k].to(dev, non_blocking=True) for k in keys}}


def f1(pred, true):
    tp = (pred & true).sum().item(); fp = (pred & ~true).sum().item(); fn = (~pred & true).sum().item()
    return 2 * tp / max(1, 2 * tp + fp + fn)


@torch.no_grad()
def evaluate(model, loader, dev, qm, qs, w, amp_dtype):
    model.eval()
    ae, ae_real, n = 0.0, 0.0, 0
    s_p, s_t, d_p, d_t, pose_e, heat_ok, heat_n, dom = [], [], [], [], [], 0, 0, {}
    for batch in loader:
        b = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            out = model(b["rgb"], b["depth"], b["q"], b["text_tokens"], b["text_mask"], b["sentence"])
        p = out["pred"].float() * qs + qm
        y = b["target"].float() * qs + qm
        e = torch.abs(p - y).mean(-1)                                  # [B,30] rad
        ae += e.mean(1).sum().item(); n += e.shape[0]
        real = ~b["hold_mask"]
        ae_real += ((e * real).sum(1) / real.sum(1).clamp_min(1)).sum().item()
        for d, v in zip(batch["domain"], e.mean(1).tolist()):
            dom.setdefault(d, []).append(v)
        s_p.append(out["suction_logits"] > 0); s_t.append(b["suction_target"] > 0.5)
        d_p.append(out["done_logits"] > 0); d_t.append(b["done_target"] > 0.5)
        vis = b["phone_visible"] == 1
        if vis.any():
            pose_e.append(torch.linalg.norm(out["phone_pose"][vis, :2].float() - b["phone_pose"][vis, :2], dim=-1))
        known = b["phone_patch"] >= 0
        if known.any():
            heat_ok += (out["phone_heatmap_logits"][known].argmax(-1) == b["phone_patch"][known]).sum().item()
            heat_n += int(known.sum().item())
    model.train()
    res = {
        "action_mae_rad": ae / max(n, 1), "action_mae_real_steps_rad": ae_real / max(n, 1),
        "suction_f1": f1(torch.cat(s_p), torch.cat(s_t)), "done_f1": f1(torch.cat(d_p), torch.cat(d_t)),
        "phone_xy_err_cm": float(torch.cat(pose_e).mean().item() * 100) if pose_e else float("nan"),
        "phone_patch_acc": heat_ok / heat_n if heat_n else float("nan"),
    }
    for d, v in dom.items():
        res[f"action_mae_rad_{d}"] = float(np.mean(v))
    res["score"] = res["action_mae_rad"] + 0.01 * (1 - res["done_f1"]) + 0.01 * (1 - res["suction_f1"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=Path, help="migrated VLA checkpoint (stage 1) or previous stage best.pt")
    ap.add_argument("--resume", action="store_true", help="continue from --out/last.pt")
    ap.add_argument("--language-bank", required=True, type=Path)
    ap.add_argument("--data", action="append", required=True, help="domain=path, e.g. blender=data/blender")
    ap.add_argument("--domain-weight", action="append", default=[], help="domain=weight, e.g. isaac_moved=2")
    ap.add_argument("--split-json", required=True, type=Path)
    ap.add_argument("--stage", type=int, choices=[1, 2, 3], default=1)
    ap.add_argument("--max-updates", type=int, default=30000)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--lr-new", type=float, default=1e-4)
    ap.add_argument("--lr-fusion", type=float, default=1e-5)
    ap.add_argument("--lr-act", type=float, default=1e-5)
    ap.add_argument("--lr-dino", type=float, default=1e-6)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hold-weight", type=float, default=0.5)
    ap.add_argument("--language-dropout", type=float, default=0.10)
    ap.add_argument("--aug-prob", type=float, default=0.6)
    ap.add_argument("--warp-prob", type=float, default=0.3)
    ap.add_argument("--eval-every", type=int, default=1500)
    ap.add_argument("--val-samples", type=int, default=2400, help="fixed random subset of the val set")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    seed_all(args.seed)
    dev = torch.device(args.device if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    args.out.mkdir(parents=True, exist_ok=True)
    state_path = args.out / "last.pt"

    start_ck = state_path if args.resume else args.init
    if start_ck is None or not Path(start_ck).is_file():
        raise SystemExit(f"checkpoint not found: {start_ck}")
    model, ck = load_vla_checkpoint(start_ck, device=dev, freeze_dino=True)
    set_stage(model, args.stage)
    lrs = {"new": args.lr_new, "fusion": args.lr_fusion, "act": args.lr_act, "dino": args.lr_dino, "other": args.lr_act}
    opt = make_optimizer(model, lrs, args.weight_decay)
    total, warm = args.max_updates, min(args.warmup, max(1, args.max_updates // 10))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total)))))
    scaler = torch.amp.GradScaler("cuda", enabled=(dev.type == "cuda" and amp_dtype == torch.float16))
    step, best = 0, math.inf
    if args.resume:
        st = ck["train_state"]
        opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"]); scaler.load_state_dict(st["scaler"])
        step, best = int(st["step"]), float(st["best"])
        print(f"[train] RESUMED {state_path} at update {step}/{total} (best score {best:.5f})", flush=True)

    split = json.loads(args.split_json.read_text())
    qm_np, qs_np = ck["q_mean"].float().numpy(), ck["q_std"].float().numpy()
    data = parse_data(args.data)
    common = dict(q_mean=qm_np, q_std=qs_np, language_bank=args.language_bank, chunk_size=model.cfg.chunk_size)
    tr = VLASkillDataset(data, set(split["train"]), augment=True, aug_prob=args.aug_prob, warp_prob=args.warp_prob,
                         language_dropout=args.language_dropout, seed=args.seed + step, **common)
    va = VLASkillDataset(data, set(split["val"]), augment=False, language_dropout=0.0, seed=args.seed + 1, **common)
    dw = {k: float(v) for k, v in (x.split("=") for x in args.domain_weight)}
    counts = {}
    for i in range(len(tr)):
        k = (tr.domain_of(i), tr.skill_of(i)); counts[k] = counts.get(k, 0) + 1
    print(f"[train] train samples {len(tr)}, val samples {len(va)}; per domain/skill:", flush=True)
    for k in sorted(counts):
        print(f"          {k[0]:>12s} {k[1]:<14s} {counts[k]}", flush=True)
    sampler = WeightedRandomSampler(tr.sample_weights(dw), num_samples=10 ** 8, replacement=True,
                                    generator=torch.Generator().manual_seed(args.seed + step))
    trl = DataLoader(tr, batch_size=args.batch_size, sampler=sampler, num_workers=args.workers,
                     pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    rng = np.random.default_rng(0)
    vidx = rng.choice(len(va), size=min(args.val_samples, len(va)), replace=False)
    val = DataLoader(Subset(va, sorted(vidx.tolist())), batch_size=args.batch_size, shuffle=False,
                     num_workers=min(4, args.workers))

    w = LossWeights(kl=model.cfg.kl_weight, hold=args.hold_weight)
    qm, qs = ck["q_mean"].float().to(dev), ck["q_std"].float().to(dev)
    ntrain = {g["name"]: sum(p.numel() for p in g["params"]) for g in opt.param_groups}
    print(f"[train] stage {args.stage}: trainable params per group {ntrain}; joint dropout "
          f"{model.cfg.joint_dropout_p}; AMP {amp_dtype}; batch {args.batch_size}; updates {step}->{total}", flush=True)
    mpath = args.out / "metrics.csv"
    if not mpath.exists():
        mpath.write_text("step,lr_new,train_total,train_action,train_suction,train_done,train_pose,train_heat,"
                         "val_score,val_action_mae_rad,val_action_mae_real_steps_rad,val_suction_f1,val_done_f1,"
                         "val_phone_xy_err_cm,val_phone_patch_acc\n")

    def payload(extra_state=None):
        p = {k: v for k, v in ck.items() if k != "train_state"}
        p.update({"model": model.state_dict(), "training_stage": args.stage, "step": step,
                  "suction_trained": True, "done_trained": True, "train_args": vars(args) | {"out": str(args.out)}})
        p["schema"] = "tomo_dino_vla_flow_v1"
        p["config"]["joint_dropout_p"] = p["config"].get("joint_dropout_p", 0.1)
        if extra_state is not None:
            p["train_state"] = extra_state
        return p

    def save_state():
        tmp = state_path.with_suffix(".tmp")
        torch.save(payload({"opt": opt.state_dict(), "sched": sched.state_dict(), "scaler": scaler.state_dict(),
                            "step": step, "best": best}), tmp)
        tmp.replace(state_path)

    model.train()
    hist = {k: [] for k in ["total", "action_l1", "suction", "done", "phone_pose", "phone_heatmap"]}
    t0, it = time.time(), iter(trl)
    step0 = step
    while step < total:
        batch = next(it)
        b = to_dev(batch, dev)
        with torch.autocast("cuda", dtype=amp_dtype, enabled=dev.type == "cuda"):
            out = model(b["rgb"], b["depth"], b["q"], b["text_tokens"], b["text_mask"], b["sentence"],
                        actions=b["target"], action_valid_mask=None)
        losses = vla_losses(out, b, w)
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        step += 1
        for k in hist:
            hist[k].append(float(losses[k].detach()))
        if step % args.log_every == 0 or step == step0 + 1:
            el = time.time() - t0
            eta = (total - step) * el / max(1, step - step0) / 60
            print(f"[stage {args.stage}] update {step}/{total} ({100 * step / total:.1f}%) "
                  f"loss {np.mean(hist['total'][-args.log_every:]):.4f} action {np.mean(hist['action_l1'][-args.log_every:]):.4f} "
                  f"suction {np.mean(hist['suction'][-args.log_every:]):.3f} done {np.mean(hist['done'][-args.log_every:]):.3f} "
                  f"pose {np.mean(hist['phone_pose'][-args.log_every:]):.3f} heat {np.mean(hist['phone_heatmap'][-args.log_every:]):.3f} "
                  f"gate {float(model.lang_attn_gate):+.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"{el / 60:.1f} min, ETA {eta:.0f} min", flush=True)
        if step % args.eval_every == 0 or step == total:
            print(f"[eval] update {step}: validating on {len(val.dataset)} fixed val samples ...", flush=True)
            ev = evaluate(model, val, dev, qm, qs, w, amp_dtype)
            tr_means = [np.mean(hist[k]) if hist[k] else float("nan") for k in hist]
            with mpath.open("a", newline="") as f:
                csv.writer(f).writerow([step, sched.get_last_lr()[0], *[f"{x:.6f}" for x in tr_means],
                                        *[f"{ev[k]:.6f}" for k in ["score", "action_mae_rad", "action_mae_real_steps_rad",
                                                                    "suction_f1", "done_f1", "phone_xy_err_cm",
                                                                    "phone_patch_acc"]]])
            hist = {k: [] for k in hist}
            tag = ""
            if ev["score"] < best:
                best = ev["score"]
                torch.save(payload(), args.out / "best.pt")
                tag = "  -> NEW BEST, saved best.pt"
            print("[eval] " + " ".join(f"{k}={v:.5f}" for k, v in ev.items()) + tag, flush=True)
            save_state()
    save_state()
    print(f"[train] done: stage {args.stage}, best val score {best:.5f} -> {args.out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
