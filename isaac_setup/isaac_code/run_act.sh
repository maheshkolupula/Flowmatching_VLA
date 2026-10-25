#!/usr/bin/env bash
# Start the ACT policy server, run the Isaac Sim closed loop, stop the server again.
#   ./run_act.sh --model act_rgbd [isaac_backend.py options]      RGB-D DINOv2 ACT
#   ./run_act.sh --model act_rgb  [isaac_backend.py options]      RGB-only ResNet-18 ACT
#   ./run_act.sh --view [--episode Find_and_Pick_004]             still scene, no model
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CFG="${CFG:-$HERE/config.yaml}"
ISAAC_PY="${ISAAC_PY:-$HOME/isaac-sim/python.sh}"
ACT_PY="${ACT_PY:-python3}"             # any python with PyYAML (only reads config.yaml)
[[ -x "$ISAAC_PY" ]] || { echo "Isaac Sim python not found at $ISAAC_PY (set ISAAC_PY=/path/to/isaac-sim/python.sh)"; exit 1; }

for a in "$@"; do          # --view / --replay need no model and no server
  if [[ "$a" == "--view" || "$a" == "--replay" ]]; then
    exec "$ISAAC_PY" "$HERE/isaac_backend.py" --config "$CFG" "$@"
  fi
done

MODEL=act_rgbd; prev=""
for a in "$@"; do
  [[ "$prev" == "--model" ]] && MODEL="$a"
  [[ "$a" == --model=* ]] && MODEL="${a#--model=}"
  prev="$a"
done
[[ "$MODEL" == "act_rgbd" ]] || { echo "--model must be act_rgbd"; exit 1; }
SECTION="${MODEL}_policy"
PY_POLICY="${PY_POLICY:-$("$ACT_PY" -c "import yaml,os,sys;c=yaml.safe_load(open(sys.argv[1]));print(os.path.expanduser(c[sys.argv[2]]['python']))" "$CFG" "$SECTION")}"
LOG="${LOG:-/tmp/tomo_act_policy_server_${MODEL}.log}"

if "$PY_POLICY" "$HERE/policy_server.py" --config "$CFG" --model "$MODEL" --ping >/dev/null 2>&1; then
  echo "[run] reusing the $MODEL policy server that is already running"
else
  echo "[run] starting $MODEL policy server ($PY_POLICY) -> $LOG"
  "$PY_POLICY" "$HERE/policy_server.py" --config "$CFG" --model "$MODEL" > "$LOG" 2>&1 &
  SERVER_PID=$!
  trap 'kill "$SERVER_PID" 2>/dev/null || true' EXIT
  for _ in $(seq 1 240); do
    grep -q "READY" "$LOG" 2>/dev/null && break
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then echo "[run] policy server died:"; tail -30 "$LOG"; exit 1; fi
    sleep 1
  done
  grep -q "READY" "$LOG" || { echo "[run] policy server not ready after 240 s"; tail -30 "$LOG"; exit 1; }
  grep "\[policy\]\|\[act" "$LOG" || true
fi
"$ISAAC_PY" "$HERE/isaac_backend.py" --config "$CFG" "$@"
