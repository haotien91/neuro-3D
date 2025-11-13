#!/usr/bin/env bash
# Run reconstruction inference for a single subject (sub03 or sub11) interactively
# Usage examples:
#   bash scripts/infer_subject.sh --subject sub03 --gpu_ids "0 1" --data_path ./            # auto-pick latest ckpt
#   bash scripts/infer_subject.sh --subject sub11 --gpu_ids "0 1" --checkpoint_step 100000  # pick specific step
#
# Notes:
# - Strictly filters recon runs under model/point_generate/shape/<subject> by bc(22|32|64) and cls(12|72)
# - Matches classification dir by same run_name; otherwise fallback to latest bc/cls-matched dir for the same subject
# - Logs to <recon_run_dir>/inference_logs/infer_<step>_<timestamp>.log
# - Runs up to number of GPUs in parallel (e.g. 2 for H100x2)

set -euo pipefail

SUBJECT=""
GPU_IDS="0 1"
DATA_PATH="./"
CHECKPOINT_STEP="auto"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --subject) SUBJECT="$2"; shift 2;;
    --gpu_ids) GPU_IDS="$2"; shift 2;;                # e.g. "0 1" or "0,1"
    --data_path) DATA_PATH="$2"; shift 2;;
    --checkpoint_step) CHECKPOINT_STEP="$2"; shift 2;; # "auto" or integer
    -h|--help)
      echo "Usage: $0 --subject <sub03|sub11> [--gpu_ids '0 1'] [--data_path ./] [--checkpoint_step <auto|N>]";
      exit 0;;
    *) echo "Unknown arg: $1" >&2; exit 1;;
  esac
done

if [[ -z "$SUBJECT" ]]; then
  echo "Error: --subject is required (sub03 or sub11)" >&2
  exit 1
fi

# Normalize GPU list
GPU_IDS="${GPU_IDS//,/ }"
read -r -a GPU_ARR <<< "$GPU_IDS"
if [[ ${#GPU_ARR[@]} -lt 1 ]]; then
  echo "Error: no valid GPU IDs" >&2
  exit 1
fi
CONCURRENCY=${#GPU_ARR[@]}

RECON_BASE="model/point_generate/shape/${SUBJECT}"
RETRI_BASE="model/retraival/${SUBJECT}"

[[ -d "$RECON_BASE" ]] || { echo "No recon base: $RECON_BASE" >&2; exit 1; }
[[ -d "$RETRI_BASE" ]] || { echo "No classification base: $RETRI_BASE" >&2; exit 1; }

echo "=== Inference planner ==="
echo "Subject     : $SUBJECT"
echo "Data path   : $DATA_PATH"
echo "GPU IDs     : ${GPU_ARR[*]}"
echo "Checkpoint  : $CHECKPOINT_STEP"
printf '\n'

# Collect recon runs (strict filter to avoid unrelated dirs)
mapfile -t RUN_NAMES < <(find "$RECON_BASE" -mindepth 1 -maxdepth 1 -type d -printf "%f\n" \
  | grep -E '^sub[0-9]+-(12|72)class-1027.*_bc(22|32|64)_cls(12|72)_' \
  | sort)

if [[ ${#RUN_NAMES[@]} -eq 0 ]]; then
  echo "No matching recon runs found under $RECON_BASE"
  exit 0
fi

# Helper: choose checkpoint
choose_ckpt() {
  local recon_dir="$1"
  local step="$2"
  local ckpt=""
  if [[ "$step" == "auto" ]]; then
    if compgen -G "${recon_dir}/checkpoint-*.pth" > /dev/null; then
      local last_step
      last_step=$(ls -1 "${recon_dir}"/checkpoint-*.pth | sed -E 's/.*checkpoint-([0-9]+)\.pth/\1/' | sort -n | tail -1)
      ckpt="${recon_dir}/checkpoint-${last_step}.pth"
      echo "$ckpt"
      return 0
    else
      echo ""
      return 1
    fi
  else
    ckpt="${recon_dir}/checkpoint-${step}.pth"
    [[ -f "$ckpt" ]] && { echo "$ckpt"; return 0; } || { echo ""; return 1; }
  fi
}

# Helper: find classification dir for a run_name by (1) exact name, else (2) latest matching bc/cls
find_cls_dir() {
  local run_name="$1"
  local bc="" cls=""
  if [[ "$run_name" =~ bc([0-9]+) ]]; then bc="${BASH_REMATCH[1]}"; fi
  if [[ "$run_name" =~ cls([0-9]+) ]]; then cls="${BASH_REMATCH[1]}"; fi

  # 1) exact
  if [[ -d "${RETRI_BASE}/${run_name}" ]]; then
    echo "${RETRI_BASE}/${run_name}"
    return 0
  fi
  # 2) latest matching bc/cls within this subject (by mtime)
  local cand
  cand=$(find "$RETRI_BASE" -mindepth 1 -maxdepth 1 -type d -printf "%T@ %f\n" \
    | awk -v bc="$bc" -v cls="$cls" '$2 ~ ("_bc" bc "_cls" cls "_") {print $0}' \
    | sort -n \
    | tail -1 \
    | awk '{print $2}' || true)
  if [[ -n "$cand" && -d "${RETRI_BASE}/${cand}" ]]; then
    echo "${RETRI_BASE}/${cand}"
    return 0
  fi
  echo ""
  return 1
}

# Plan runs (resolve ckpt, cls_dir, args)
declare -a PLAN_RECON
declare -a PLAN_CKPT
declare -a PLAN_PRETRAIN
declare -a PLAN_PRETRAIN_NAME
declare -a PLAN_CLS
declare -a PLAN_BC
declare -a PLAN_INCH

for RUN_NAME in "${RUN_NAMES[@]}"; do
  RECON_DIR="${RECON_BASE}/${RUN_NAME}"

  CKPT="$(choose_ckpt "$RECON_DIR" "$CHECKPOINT_STEP" || true)"
  if [[ -z "$CKPT" ]]; then
    echo "[SKIP] $RUN_NAME : no checkpoint found (step=$CHECKPOINT_STEP)"
    continue
  fi

  PRETRAIN_DIR="$(find_cls_dir "$RUN_NAME" || true)"
  if [[ -z "$PRETRAIN_DIR" ]]; then
    echo "[SKIP] $RUN_NAME : classification dir not found (bc/cls mismatch?)"
    continue
  fi
  PRETRAIN_NAME="$(basename "$PRETRAIN_DIR")"

  NUM_CLASSES="" ; EEG_CHANNELS=""
  [[ "$RUN_NAME" =~ cls([0-9]+) ]] && NUM_CLASSES="${BASH_REMATCH[1]}"
  [[ "$RUN_NAME" =~ bc([0-9]+)  ]] && EEG_CHANNELS="${BASH_REMATCH[1]}"

  # in_channels from ckpt (fallback 1027)
  IN_CHANNELS=$(python - <<'PY'
import torch, sys, os
p = os.environ.get("CKPT")
try:
    ckpt = torch.load(p, map_location='cpu', weights_only=False)
    for k,v in ckpt['model'].items():
        if 'sa_layers.0.0.voxel_layers.0.weight' in k:
            print(v.shape[1]); sys.exit(0)
except Exception: pass
sys.exit(1)
PY
  CKPT="$CKPT") || true
  [[ -z "$IN_CHANNELS" ]] && IN_CHANNELS=1027

  PLAN_RECON+=("$RECON_DIR")
  PLAN_CKPT+=("$CKPT")
  PLAN_PRETRAIN+=("$PRETRAIN_DIR")
  PLAN_PRETRAIN_NAME+=("$PRETRAIN_NAME")
  PLAN_CLS+=("$NUM_CLASSES")
  PLAN_BC+=("$EEG_CHANNELS")
  PLAN_INCH+=("$IN_CHANNELS")

done

COUNT=${#PLAN_RECON[@]}
if [[ $COUNT -eq 0 ]]; then
  echo "No runnable plans (missing ckpt or pretrain match)."
  exit 0
fi

# Show plan for confirmation
echo "Planned runs for $SUBJECT: ($COUNT total)"
for i in $(seq 0 $((COUNT-1))); do
  rn="$(basename "${PLAN_RECON[$i]}")"
  echo "- [$((i+1))/$COUNT] $rn"
  echo "    ckpt     : ${PLAN_CKPT[$i]}"
  echo "    pretrain : ${PLAN_PRETRAIN[$i]}"
  echo "    pretrain_name: ${PLAN_PRETRAIN_NAME[$i]}"
  echo "    classes  : ${PLAN_CLS[$i]}"
  echo "    channels : ${PLAN_BC[$i]}"
  echo "    in_ch    : ${PLAN_INCH[$i]}"
done
read -p "Proceed? (y/N) " -r
[[ "$REPLY" =~ ^[Yy]$ ]] || { echo "Aborted."; exit 1; }

# Run one plan on a given GPU
run_one() {
  local GPU="$1" IDX="$2"
  local RECON_DIR="${PLAN_RECON[$IDX]}"
  local CKPT="${PLAN_CKPT[$IDX]}"
  local PRETRAIN_DIR="${PLAN_PRETRAIN[$IDX]}"
  local PRETRAIN_NAME="${PLAN_PRETRAIN_NAME[$IDX]}"
  local NUM_CLASSES="${PLAN_CLS[$IDX]}"
  local EEG_CHANNELS="${PLAN_BC[$IDX]}"
  local IN_CHANNELS="${PLAN_INCH[$IDX]}"
  local RUN_NAME ; RUN_NAME="$(basename "$RECON_DIR")"

  local LOG_DIR="${RECON_DIR}/inference_logs"
  mkdir -p "$LOG_DIR"
  local STEP ; STEP="$(basename "$CKPT" | sed -E 's/checkpoint-([0-9]+)\.pth/\1/')"
  local TS ; TS="$(date +%Y%m%d_%H%M%S)"
  local LOG_FILE="${LOG_DIR}/infer_${STEP}_${TS}.log"

  local CMD="python recon_main.py"
  CMD="$CMD --data_path $DATA_PATH --generation_type shape --task sample --sub $SUBJECT"
  CMD="$CMD --checkpoint_resume $CKPT --pretrain_model $PRETRAIN_NAME"
  CMD="$CMD --num_classes $NUM_CLASSES --eeg_channels $EEG_CHANNELS --in_channels $IN_CHANNELS"

  echo "[GPU $GPU] $RUN_NAME"
  echo "  -> $CMD"
  {
    echo "=== Inference ==="
    echo "Time       : $(date)"
    echo "GPU        : $GPU"
    echo "Subject    : $SUBJECT"
    echo "Run        : $RUN_NAME"
    echo "Checkpoint : $CKPT"
    echo "Pretrain_dir : $PRETRAIN_DIR"
    echo "Pretrain_name: $PRETRAIN_NAME"
    echo "num_classes: $NUM_CLASSES"
    echo "eeg_channels: $EEG_CHANNELS"
    echo "in_channels: $IN_CHANNELS"
    echo "------------- OUTPUT -------------"
  } | tee -a "$LOG_FILE"

  set +e
  CUDA_VISIBLE_DEVICES="$GPU" time $CMD 2>&1 | tee -a "$LOG_FILE"
  rc=${PIPESTATUS[0]}
  set -e
  if [[ $rc -ne 0 ]]; then
    echo "[GPU $GPU] ✗ Failed $RUN_NAME — $LOG_FILE"
  else
    echo "[GPU $GPU] ✓ Done   $RUN_NAME — $LOG_FILE"
  fi
}

# Execute in batches of CONCURRENCY
i=0 ; total="$COUNT"
while [[ $i -lt $total ]]; do
  declare -a PIDS=()
  for slot in $(seq 0 $((CONCURRENCY-1))); do
    [[ $i -ge $total ]] && break
    run_one "${GPU_ARR[$slot]}" "$i" &
    PIDS+=($!)
    i=$((i+1))
  done
  for pid in "${PIDS[@]}"; do wait "$pid"; done
done

echo "All done for $SUBJECT."
