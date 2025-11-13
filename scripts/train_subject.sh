#!/usr/bin/env bash
set -euo pipefail

# Train classification -> recon(shape) for a given subject, with controllable EEG channels and class count.
# Usage examples:
#   bash scripts/train_subject.sh \
#     --data_path ./ --sub sub03 --eeg_channels 32 --num_classes 12 --max_steps 100000 --run_name_prefix "sub03_bc32"
#   bash scripts/train_subject.sh \
#     --data_path ./ --sub sub03 --eeg_channels 32 --both_classes --max_steps 100000 --run_name_prefix "sub03_bc32"

DATA_PATH=""
SUB=""
EEG_CHANNELS=64
NUM_CLASSES=""
BOTH_CLASSES=false
MAX_STEPS=100000
RUN_NAME=""
RUN_NAME_PREFIX=""
IN_CHANNELS=1027
USE_WANDB=false
WANDB_PROJECT="neuro-3d"
WANDB_ENTITY=""
GPU_ID=""

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path) DATA_PATH="$2"; shift 2;;
    --sub) SUB="$2"; shift 2;;
    --eeg_channels) EEG_CHANNELS="$2"; shift 2;;
    --num_classes) NUM_CLASSES="$2"; shift 2;;
    --both_classes) BOTH_CLASSES=true; shift 1;;
    --max_steps) MAX_STEPS="$2"; shift 2;;
    --run_name) RUN_NAME="$2"; shift 2;;
    --run_name_prefix) RUN_NAME_PREFIX="$2"; shift 2;;
    --in_channels) IN_CHANNELS="$2"; shift 2;;
    --use_wandb) USE_WANDB=true; shift 1;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2;;
    --gpu_id) GPU_ID="$2"; shift 2;;
    -h|--help)
      echo "Usage: $0 --data_path <path> --sub <subXX> --eeg_channels {64|32|22} [--num_classes {72|12} | --both_classes] [--max_steps N] [--run_name NAME | --run_name_prefix PREFIX] [--in_channels N] [--use_wandb] [--wandb_project NAME] [--wandb_entity NAME] [--gpu_id N]";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$DATA_PATH" || -z "$SUB" ]]; then
  echo "Error: --data_path and --sub are required" >&2
  exit 1
fi

mkdir -p logs

make_run_name() {
  local cls="$1"
  if [[ -n "$RUN_NAME" ]]; then
    # If both_classes, append class suffix to avoid collision
    if [[ "$BOTH_CLASSES" == true ]]; then
      echo "${RUN_NAME}_cls${cls}"
    else
      echo "$RUN_NAME"
    fi
  else
    local ts
    ts=$(date +%Y%m%d_%H%M%S)
    local base
    if [[ -n "$RUN_NAME_PREFIX" ]]; then
      base="$RUN_NAME_PREFIX"
    else
      base="${SUB}_bc${EEG_CHANNELS}"
    fi
    echo "${base}_cls${cls}_${ts}"
  fi
}

run_one() {
  local cls="$1"
  local rn
  rn=$(make_run_name "$cls")

  # Build W&B args for classification
  local wandb_args_cls=""
  if [[ "$USE_WANDB" == true ]]; then
    wandb_args_cls="--use_wandb --wandb_project $WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args_cls="$wandb_args_cls --wandb_entity $WANDB_ENTITY"
    fi
    wandb_args_cls="$wandb_args_cls --wandb_run_name ${rn}_classification"
  fi

  # Build W&B args for recon
  local wandb_args_recon=""
  if [[ "$USE_WANDB" == true ]]; then
    wandb_args_recon="--use_wandb --wandb_project $WANDB_PROJECT"
    if [[ -n "$WANDB_ENTITY" ]]; then
      wandb_args_recon="$wandb_args_recon --wandb_entity $WANDB_ENTITY"
    fi
    wandb_args_recon="$wandb_args_recon --wandb_run_name ${rn}_recon_shape"
  fi

  # Set CUDA_VISIBLE_DEVICES if gpu_id is specified
  if [[ -n "$GPU_ID" ]]; then
    echo "[INFO] Using GPU: $GPU_ID"
  fi

  echo "[INFO] Starting classification: sub=${SUB}, eeg_channels=${EEG_CHANNELS}, classes=${cls}, run_name=${rn}"
  if [[ -n "$GPU_ID" ]]; then
    time CUDA_VISIBLE_DEVICES=$GPU_ID python ./classification/retri_shape_color.py \
      --root_path "$DATA_PATH" \
      --sub "$SUB" \
      --num_classes "$cls" \
      --eeg_channels "$EEG_CHANNELS" \
      --run_name "$rn" \
      $wandb_args_cls \
      2>&1 | tee "logs/${SUB}__${rn}__classification_$(date +%Y%m%d_%H%M%S).log"
  else
    time python ./classification/retri_shape_color.py \
      --root_path "$DATA_PATH" \
      --sub "$SUB" \
      --num_classes "$cls" \
      --eeg_channels "$EEG_CHANNELS" \
      --run_name "$rn" \
      $wandb_args_cls \
      2>&1 | tee "logs/${SUB}__${rn}__classification_$(date +%Y%m%d_%H%M%S).log"
  fi

  echo "[INFO] Starting recon(shape): sub=${SUB}, eeg_channels=${EEG_CHANNELS}, classes=${cls}, run_name=${rn}"
  if [[ -n "$GPU_ID" ]]; then
    time CUDA_VISIBLE_DEVICES=$GPU_ID python recon_main.py \
      --data_path "$DATA_PATH" \
      --sub "$SUB" \
      --in_channels "$IN_CHANNELS" \
      --generation_type 'shape' \
      --pretrain_model "$rn" \
      --num_classes "$cls" \
      --eeg_channels "$EEG_CHANNELS" \
      --run_name "$rn" \
      --max_steps "$MAX_STEPS" \
      $wandb_args_recon \
      2>&1 | tee "logs/${SUB}__${rn}__recon_shape_$(date +%Y%m%d_%H%M%S).log"
  else
    time python recon_main.py \
      --data_path "$DATA_PATH" \
      --sub "$SUB" \
      --in_channels "$IN_CHANNELS" \
      --generation_type 'shape' \
      --pretrain_model "$rn" \
      --num_classes "$cls" \
      --eeg_channels "$EEG_CHANNELS" \
      --run_name "$rn" \
      --max_steps "$MAX_STEPS" \
      $wandb_args_recon \
      2>&1 | tee "logs/${SUB}__${rn}__recon_shape_$(date +%Y%m%d_%H%M%S).log"
  fi
}

if [[ "$BOTH_CLASSES" == true ]]; then
  run_one 12
  run_one 72
else
  if [[ -z "$NUM_CLASSES" ]]; then
    echo "Error: specify --num_classes {72|12} or use --both_classes" >&2
    exit 1
  fi
  run_one "$NUM_CLASSES"
fi

echo "[DONE] Training pipeline completed for subject ${SUB}."
