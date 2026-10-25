#!/usr/bin/env python3
"""Precompute frozen SigLIP embeddings for every training + held-out test instruction (runs once, CPU)."""
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vla.annotations import all_instruction_texts  # noqa: E402
from vla.text_encoder import SigLIPInstructionEncoder, save_language_bank  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "model/siglip_language_bank.pt"))
    ap.add_argument("--model", default="google/siglip-base-patch16-224")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--local-files-only", action="store_true")
    args = ap.parse_args()
    texts = all_instruction_texts(include_test=True)
    print(f"[language] {len(texts)} unique instructions")
    enc = SigLIPInstructionEncoder(args.model, device=args.device, max_tokens=args.max_tokens,
                                   local_files_only=args.local_files_only)
    if enc.hidden_size != 768:
        raise SystemExit(f"SigLIP hidden size {enc.hidden_size} != 768 (checkpoint text_dim)")
    save_language_bank(args.out, texts, enc)
    print(f"[language] done -> {args.out}")


if __name__ == "__main__":
    main()
