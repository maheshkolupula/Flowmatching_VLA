#!/usr/bin/env python3
"""Evaluate a trained TOMO VLA on the validation episodes (never trained on).

1. Standard metrics on ALL val samples (every domain given with --data): action MAE [rad] (whole chunk and
   real steps), suction F1, done F1, phone xy error [cm], phone patch accuracy, per-domain action MAE.
2. Language test on val episodes: same image + joints, different command.
     at frame (skill end - 5): sub-skill command vs full-task command
       -> mean |difference| of the predicted chunk after the skill end [rad]  (large = language is used)
       -> done probability at the skill end step (high for the sub-skill, low for the full task)
     held-out paraphrases (never trained) vs trained wording -> similar done/hold behaviour expected.
"""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from vla.checkpoint import load_vla_checkpoint  # noqa: E402
from vla.dataset import VLASkillDataset, norm_depth, IMAGENET_MEAN, IMAGENET_STD  # noqa: E402
from vla.losses import LossWeights  # noqa: E402
from vla.text_encoder import LanguageBank  # noqa: E402
from train_vla import evaluate, parse_data  # noqa: E402

SKILLS = ["REACH_PHONE", "GRASP_PHONE", "PICK_UP_PHONE", "LIFT_PHONE", "CARRY_TO_BOX", "PLACE_IN_BOX"]


@torch.no_grad()
def language_test(model, ds, bank, qm, qs, dev):
    res = {s: {"hold_diff_rad": [], "done_sub": [], "done_full": [], "done_heldout": []} for s in SKILLS}
    full_cmd = "pick up the phone and put it in the box"
    for ei, epi in enumerate(ds.episodes):
        a = ds.arrays(ei)
        segs = {s["name"]: s for s in epi["segments"]}
        for name in SKILLS:
            seg = segs[name]
            t = max(int(seg["start"]), int(seg["end"]) - 5)
            row = int(epi["row_of"][t])
            if row < 0:
                continue
            rgb = torch.from_numpy(np.asarray(a["rgb"][row]).copy()).float().permute(2, 0, 1) / 255.0
            rgb = ((rgb - IMAGENET_MEAN) / IMAGENET_STD)[None].to(dev)
            depth = norm_depth(torch.from_numpy(np.asarray(a["depth"][row], np.float32).copy())[None])[0][None].to(dev)
            q = torch.from_numpy((a["q"][t] - ds.qm) / ds.qs)[None].to(dev)
            cmds = [seg["instructions"][0], full_cmd] + seg.get("test_instructions", [])[:1]
            outs = []
            for c in cmds:
                L = bank.get(c)
                o = model(rgb, depth, q, L["tokens"][None].to(dev), L["mask"][None].to(dev), L["sentence"][None].to(dev))
                outs.append(o)
            k = int(seg["end"]) - t                     # chunk index of the terminal step (target t+1+k-1)
            p_sub = outs[0]["pred"][0].float() * qs + qm
            p_full = outs[1]["pred"][0].float() * qs + qm
            res[name]["hold_diff_rad"].append(float(torch.abs(p_sub[k:] - p_full[k:]).mean()))
            res[name]["done_sub"].append(float(torch.sigmoid(outs[0]["done_logits"][0, k - 1])))
            res[name]["done_full"].append(float(torch.sigmoid(outs[1]["done_logits"][0, k - 1])))
            if len(outs) > 2:
                res[name]["done_heldout"].append(float(torch.sigmoid(outs[2]["done_logits"][0, k - 1])))
    return {s: {k: (float(np.mean(v)) if v else None) for k, v in d.items()} for s, d in res.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=str(ROOT / "runs/vlaflow/best.pt"), type=Path)
    ap.add_argument("--language-bank", default=str(ROOT / "model/siglip_language_bank.pt"), type=Path)
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--split-json", default=str(ROOT / "model/vla_split.json"), type=Path)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = torch.bfloat16 if dev.type == "cuda" and torch.cuda.is_bf16_supported() else torch.float16
    model, ck = load_vla_checkpoint(args.checkpoint, dev, freeze_dino=True)
    model.eval()
    split = json.loads(args.split_json.read_text())
    qm, qs = ck["q_mean"].float().to(dev), ck["q_std"].float().to(dev)
    ds = VLASkillDataset(parse_data(args.data), set(split["val"]), ck["q_mean"].float().numpy(),
                         ck["q_std"].float().numpy(), args.language_bank, chunk_size=model.cfg.chunk_size)
    metrics = evaluate(model, DataLoader(ds, batch_size=args.batch_size, num_workers=4), dev, qm, qs,
                       LossWeights(kl=model.cfg.kl_weight), amp)
    lang = language_test(model, ds, LanguageBank(args.language_bank), qm, qs, dev)
    result = {"checkpoint": str(args.checkpoint), "step": ck.get("step"), "val_samples": len(ds),
              "metrics": metrics, "language_test": lang}
    print(json.dumps(result, indent=2))
    out = args.out or args.checkpoint.parent / "evaluation.json"
    out.write_text(json.dumps(result, indent=2))
    print(f"[eval] written {out}")


if __name__ == "__main__":
    main()
