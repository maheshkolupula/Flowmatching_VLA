#!/usr/bin/env bash
# VLA-Flow: DINOv2 RGB-D VLA with a FLOW-MATCHING action head (instead of the ACT CVAE).
# Data (data/ -> dino_act_vla_model/data) and the SigLIP bank are reused, nothing to prepare.
# Starts from the trained DINO-ACT VLA (vision/language/phone heads already good); the new
# flow modules (flow_in, flow_time) start at zero, the decoder re-learns to predict velocity.
#   ./run_vlaflow.sh smoke    60-update test (~1 min)
#   ./run_vlaflow.sh train    flow training, 100k updates (~1 h 45 min)   RESUME=1 to continue
#   ./run_vlaflow.sh eval     metrics + language test (10 Euler steps)
set -euo pipefail
D="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; cd "$D"
PY="${PY:-python3}"
INIT="${INIT:-model/VLA_ACT_INIT.pt}"
BANK=model/siglip_language_bank.pt; SPLIT=model/vla_split.json
mkdir -p runs; exec > >(tee -a runs/pipeline.log) 2>&1
export PYTHONUNBUFFERED=1
say() { echo; echo "================ $* ($(date '+%F %T')) ================"; }
DATA=(--data blender=data/blender --data isaac=data/isaac --data isaac_moved=data/isaac_moved)
RES=(); [[ "${RESUME:-0}" == "1" ]] && RES=(--resume)
case "${1:-}" in
  smoke)
    say "SMOKE"; rm -rf runs/flow_smoke
    "$PY" scripts/train_vla.py --init "$INIT" --language-bank "$BANK" "${DATA[@]}" --split-json "$SPLIT" \
      --stage 2 --max-updates 60 --warmup 10 --eval-every 60 --val-samples 64 --log-every 10 --out runs/flow_smoke ;;
  train)
    say "TRAIN flow"
    "$PY" scripts/train_vla.py --init "$INIT" --language-bank "$BANK" "${DATA[@]}" --split-json "$SPLIT" \
      --domain-weight isaac_moved=1.5 --stage 2 --max-updates 300000 --warmup 1000 --eval-every 5000 \
      --lr-new 2e-4 --lr-act 5e-5 --batch-size 8 --out model_2 "${RES[@]}" ;;
  eval)
    say "EVAL"; "$PY" scripts/evaluate_vla.py --checkpoint runs/vlaflow/best.pt --batch-size 4 "${DATA[@]}" ;;
  *) sed -n 2,8p "$0"; exit 1 ;;
esac
