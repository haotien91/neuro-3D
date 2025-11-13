#!/bin/bash
# Resume recon training from checkpoint
# Usage: bash scripts/resume_recon.sh --checkpoint_path <path> [--gpu_id N] [--use_wandb]

CHECKPOINT_PATH=""
GPU_ID=""
USE_WANDB=false
WANDB_PROJECT=""
WANDB_ENTITY=""
PRETRAIN_MODEL=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint_path) CHECKPOINT_PATH="$2"; shift 2;;
    --gpu_id) GPU_ID="$2"; shift 2;;
    --use_wandb) USE_WANDB=true; shift 1;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2;;
    --pretrain_model) PRETRAIN_MODEL="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --checkpoint_path <path> [--pretrain_model <name>] [--gpu_id N] [--use_wandb] [--wandb_project NAME] [--wandb_entity NAME]";
      echo "";
      echo "Example:";
      echo "  bash scripts/resume_recon.sh \\";
      echo "    --checkpoint_path model/point_generate/shape/sub03/my_run/checkpoint-40000.pth \\";
      echo "    --pretrain_model \"my_classification_run\" \\";
      echo "    --gpu_id 0 \\";
      echo "    --use_wandb \\";
      echo "    --wandb_project neuro-3d";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$CHECKPOINT_PATH" ]]; then
  echo "Error: --checkpoint_path is required" >&2
  exit 1
fi

if [[ ! -f "$CHECKPOINT_PATH" ]]; then
  echo "Error: Checkpoint file not found: $CHECKPOINT_PATH" >&2
  exit 1
fi

# Extract info from checkpoint path
# Expected format: model/point_generate/{type}/{sub}/{run_name}/checkpoint-{step}.pth
DIR=$(dirname "$CHECKPOINT_PATH")
RUN_NAME=$(basename "$DIR")
SUB=$(basename $(dirname "$DIR"))
GEN_TYPE=$(basename $(dirname $(dirname "$DIR")))

echo "=== Resume Training ==="
echo "Checkpoint: $CHECKPOINT_PATH"
echo "Subject: $SUB"
echo "Run name: $RUN_NAME"
echo "Generation type: $GEN_TYPE"

# Try to extract original parameters from multiple sources
LOG_FILE="$DIR/log.txt"
NUM_CLASSES=""
EEG_CHANNELS=""
MAX_STEPS=""
IN_CHANNELS=""

# 1. Try to extract from run_name (e.g., sub03-12class_bc64_cls12_...)
if [[ "$RUN_NAME" =~ cls([0-9]+) ]]; then
  NUM_CLASSES="${BASH_REMATCH[1]}"
fi
if [[ "$RUN_NAME" =~ bc([0-9]+) ]]; then
  EEG_CHANNELS="${BASH_REMATCH[1]}"
fi

# 2. Try to extract from log.txt
if [[ -f "$LOG_FILE" ]]; then
  echo "Found log file: $LOG_FILE"
  
  # Extract max_steps
  if [[ -z "$MAX_STEPS" ]]; then
    MAX_STEPS=$(grep -oP 'Max training steps\s*=\s*\K\d+' "$LOG_FILE" | head -1)
  fi
  
  # Extract from "Using X EEG channels and Y object classes" line
  if [[ -z "$NUM_CLASSES" || -z "$EEG_CHANNELS" ]]; then
    USING_LINE=$(grep "Using.*EEG channels.*object classes" "$LOG_FILE" | head -1)
    if [[ -n "$USING_LINE" ]]; then
      if [[ "$USING_LINE" =~ Using\ ([0-9]+)\ EEG\ channels\ and\ ([0-9]+)\ object\ classes ]]; then
        [[ -z "$EEG_CHANNELS" ]] && EEG_CHANNELS="${BASH_REMATCH[1]}"
        [[ -z "$NUM_CLASSES" ]] && NUM_CLASSES="${BASH_REMATCH[2]}"
      fi
    fi
  fi
fi

# 3. Extract in_channels from checkpoint
if [[ -z "$IN_CHANNELS" ]]; then
  echo "Extracting in_channels from checkpoint..."
  IN_CHANNELS=$(python -c "
import torch
import sys
try:
    ckpt = torch.load('$CHECKPOINT_PATH', map_location='cpu', weights_only=False)
    for k, v in ckpt['model'].items():
        if 'sa_layers.0.0.voxel_layers.0.weight' in k:
            print(v.shape[1])
            sys.exit(0)
except Exception as e:
    sys.exit(1)
" 2>/dev/null)
  if [[ -z "$IN_CHANNELS" ]]; then
    echo "Warning: Could not extract in_channels from checkpoint"
  fi
fi

echo "Extracted parameters:"
echo "  - num_classes: ${NUM_CLASSES:-unknown}"
echo "  - eeg_channels: ${EEG_CHANNELS:-unknown}"
echo "  - max_steps: ${MAX_STEPS:-unknown}"
echo "  - in_channels: ${IN_CHANNELS:-unknown (will use default)}"
echo "  - pretrain_model: ${PRETRAIN_MODEL:-not specified}"

# Prompt for missing parameters
if [[ -z "$NUM_CLASSES" ]]; then
  read -p "Enter num_classes (12 or 72): " NUM_CLASSES
fi
if [[ -z "$EEG_CHANNELS" ]]; then
  read -p "Enter eeg_channels (22, 32, or 64): " EEG_CHANNELS
fi
if [[ -z "$MAX_STEPS" ]]; then
  read -p "Enter max_steps (default 100000): " MAX_STEPS
  MAX_STEPS=${MAX_STEPS:-100000}
fi
if [[ -z "$PRETRAIN_MODEL" ]]; then
  echo ""
  echo "WARNING: --pretrain_model not specified!"
  echo "This is required to load the classification model weights."
  echo "Example: retri_color_shape_10-27_16-21_VideoImageEEGClassifyColor3_color_video_fea_time_len1"
  read -p "Enter pretrain_model name (or press Enter to skip): " PRETRAIN_MODEL
fi

# Ask for confirmation
read -p "Continue with resume? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 1
fi

# Build command
CMD="python recon_main.py"
CMD="$CMD --data_path ./"
CMD="$CMD --sub $SUB"
CMD="$CMD --generation_type $GEN_TYPE"
CMD="$CMD --run_name $RUN_NAME"
CMD="$CMD --checkpoint_resume $CHECKPOINT_PATH"

if [[ -n "$PRETRAIN_MODEL" ]]; then
  CMD="$CMD --pretrain_model $PRETRAIN_MODEL"
fi

if [[ -n "$NUM_CLASSES" ]]; then
  CMD="$CMD --num_classes $NUM_CLASSES"
fi

if [[ -n "$EEG_CHANNELS" ]]; then
  CMD="$CMD --eeg_channels $EEG_CHANNELS"
fi

if [[ -n "$MAX_STEPS" ]]; then
  CMD="$CMD --max_steps $MAX_STEPS"
fi

if [[ -n "$IN_CHANNELS" ]]; then
  CMD="$CMD --in_channels $IN_CHANNELS"
fi

if [[ "$USE_WANDB" == true ]]; then
  CMD="$CMD --use_wandb"
  if [[ -n "$WANDB_PROJECT" ]]; then
    CMD="$CMD --wandb_project $WANDB_PROJECT"
  fi
  if [[ -n "$WANDB_ENTITY" ]]; then
    CMD="$CMD --wandb_entity $WANDB_ENTITY"
  fi
  # Extract W&B run name from existing run
  WANDB_RUN_NAME="${RUN_NAME}_recon_shape"
  CMD="$CMD --wandb_run_name $WANDB_RUN_NAME"
fi

# Add GPU if specified
if [[ -n "$GPU_ID" ]]; then
  echo "[INFO] Using GPU: $GPU_ID"
  CMD="CUDA_VISIBLE_DEVICES=$GPU_ID $CMD"
fi

echo ""
echo "=== Running Command ==="
echo "$CMD"
echo ""

# Execute
eval $CMD
