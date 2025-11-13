#!/bin/bash
# Run inference for reconstruction models
# Usage: bash scripts/run_inference.sh --recon_path <path> --pretrain_model <name> [--gpu_id N]

RECON_PATH=""
PRETRAIN_MODEL=""
GPU_ID=""
CHECKPOINT_STEP="100000"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --recon_path) RECON_PATH="$2"; shift 2;;
    --pretrain_model) PRETRAIN_MODEL="$2"; shift 2;;
    --gpu_id) GPU_ID="$2"; shift 2;;
    --checkpoint_step) CHECKPOINT_STEP="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --recon_path <path> --pretrain_model <name> [--gpu_id N] [--checkpoint_step STEP]";
      echo "";
      echo "Example:";
      echo "  bash scripts/run_inference.sh \\";
      echo "    --recon_path model/point_generate/shape/sub03/sub03-12class-1027-gpu_id0_bc64_cls12_20251028_005520 \\";
      echo "    --pretrain_model sub03-12class-1027-gpu_id0_bc64_cls12_20251028_005520 \\";
      echo "    --gpu_id 0 \\";
      echo "    --checkpoint_step 100000";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$RECON_PATH" ]]; then
  echo "Error: --recon_path is required" >&2
  exit 1
fi

if [[ -z "$PRETRAIN_MODEL" ]]; then
  echo "Error: --pretrain_model is required" >&2
  exit 1
fi

# Remove trailing slash if present
RECON_PATH="${RECON_PATH%/}"

# Extract info from recon path
RUN_NAME=$(basename "$RECON_PATH")
SUB=$(basename $(dirname "$RECON_PATH"))
GEN_TYPE=$(basename $(dirname $(dirname "$RECON_PATH")))

# Find checkpoint file
CHECKPOINT_FILE="${RECON_PATH}/checkpoint-${CHECKPOINT_STEP}.pth"
if [[ ! -f "$CHECKPOINT_FILE" ]]; then
  echo "Error: Checkpoint not found: $CHECKPOINT_FILE" >&2
  echo "Available checkpoints:" >&2
  ls -1 "${RECON_PATH}"/checkpoint-*.pth 2>/dev/null || echo "  None found" >&2
  exit 1
fi

echo "=== Inference Configuration ==="
echo "Subject: $SUB"
echo "Run name: $RUN_NAME"
echo "Generation type: $GEN_TYPE"
echo "Checkpoint: $CHECKPOINT_FILE"
echo "Pretrain model: $PRETRAIN_MODEL"

# Extract parameters from run_name
NUM_CLASSES=""
EEG_CHANNELS=""
IN_CHANNELS=""

if [[ "$RUN_NAME" =~ cls([0-9]+) ]]; then
  NUM_CLASSES="${BASH_REMATCH[1]}"
fi
if [[ "$RUN_NAME" =~ bc([0-9]+) ]]; then
  EEG_CHANNELS="${BASH_REMATCH[1]}"
fi

# Extract in_channels from checkpoint
echo "Extracting in_channels from checkpoint..."
IN_CHANNELS=$(python -c "
import torch
import sys
try:
    ckpt = torch.load('$CHECKPOINT_FILE', map_location='cpu', weights_only=False)
    for k, v in ckpt['model'].items():
        if 'sa_layers.0.0.voxel_layers.0.weight' in k:
            print(v.shape[1])
            sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null)

echo ""
echo "Extracted parameters:"
echo "  - num_classes: ${NUM_CLASSES:-unknown}"
echo "  - eeg_channels: ${EEG_CHANNELS:-unknown}"
echo "  - in_channels: ${IN_CHANNELS:-unknown}"
echo ""

# Prompt for missing parameters
if [[ -z "$NUM_CLASSES" ]]; then
  read -p "Enter num_classes (12 or 72): " NUM_CLASSES
fi
if [[ -z "$EEG_CHANNELS" ]]; then
  read -p "Enter eeg_channels (22, 32, or 64): " EEG_CHANNELS
fi
if [[ -z "$IN_CHANNELS" ]]; then
  read -p "Enter in_channels (default 1027): " IN_CHANNELS
  IN_CHANNELS=${IN_CHANNELS:-1027}
fi

# Build command
CMD="python recon_main.py"
CMD="$CMD --data_path ./"
CMD="$CMD --generation_type $GEN_TYPE"
CMD="$CMD --task sample"
CMD="$CMD --sub $SUB"
CMD="$CMD --checkpoint_resume $CHECKPOINT_FILE"
CMD="$CMD --pretrain_model $PRETRAIN_MODEL"
CMD="$CMD --num_classes $NUM_CLASSES"
CMD="$CMD --eeg_channels $EEG_CHANNELS"
CMD="$CMD --in_channels $IN_CHANNELS"

# Add GPU if specified
if [[ -n "$GPU_ID" ]]; then
  echo "[INFO] Using GPU: $GPU_ID"
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $CMD"
fi

echo ""
echo "=== Running Inference ==="
echo "$CMD"
echo ""

# Execute
eval time $CMD
