#!/usr/bin/env bash
set -euo pipefail

# Run a full experiment sweep over EEG channels {64,32,22}
# Inputs: subjects and classes; channels are fixed to 64,32,22.
# Example:
#   bash scripts/run_experiments.sh \
#     --data_path ./ \
#     --subjects "sub01 sub02 sub03" \
#     --classes "12 72" \
#     --max_steps 100000 \
#     --in_channels 1027 \
#     --run_name_prefix "expA"

DATA_PATH=""
SUBJECTS=""
CLASSES=""
MAX_STEPS=100000
IN_CHANNELS=1027
RUN_NAME_PREFIX=""
USE_WANDB=false
WANDB_PROJECT="neuro-3d"
WANDB_ENTITY=""
GPU_IDS=""
PARALLEL_CLASSES=false

# CHANNELS=(64 32 22)
CHANNELS=(32 22)

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data_path) DATA_PATH="$2"; shift 2;;
    --subjects) SUBJECTS="$2"; shift 2;;
    --classes) CLASSES="$2"; shift 2;;
    --max_steps) MAX_STEPS="$2"; shift 2;;
    --in_channels) IN_CHANNELS="$2"; shift 2;;
    --run_name_prefix) RUN_NAME_PREFIX="$2"; shift 2;;
    --use_wandb) USE_WANDB=true; shift 1;;
    --wandb_project) WANDB_PROJECT="$2"; shift 2;;
    --wandb_entity) WANDB_ENTITY="$2"; shift 2;;
    --gpu_ids) GPU_IDS="$2"; shift 2;;
    --parallel_classes) PARALLEL_CLASSES=true; shift 1;;
    -h|--help)
      echo "Usage: $0 --data_path <path> --subjects \"sub01 sub02\" [--classes \"12 72\"] [--max_steps N] [--in_channels N] [--run_name_prefix PFX] [--use_wandb] [--wandb_project NAME] [--wandb_entity NAME] [--gpu_ids \"0 1\"] [--parallel_classes]";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

if [[ -z "$DATA_PATH" || -z "$SUBJECTS" ]]; then
  echo "Error: --data_path and --subjects are required" >&2
  exit 1
fi

# Default classes if not provided
if [[ -z "$CLASSES" ]]; then
  CLASSES="12 72"
fi

if [[ "$PARALLEL_CLASSES" == true && -n "$GPU_IDS" ]]; then
  # Parallel mode: run different classes on different GPUs
  GPU_ARRAY=($GPU_IDS)
  NUM_GPUS=${#GPU_ARRAY[@]}
  
  if [[ $NUM_GPUS -lt 2 ]]; then
    echo "Warning: --parallel_classes requires at least 2 GPUs in --gpu_ids" >&2
    echo "Falling back to sequential mode" >&2
    PARALLEL_CLASSES=false
  fi
fi

for SUB in $SUBJECTS; do
  for CH in "${CHANNELS[@]}"; do
    if [[ "$PARALLEL_CLASSES" == true ]]; then
      # Run classes in parallel on different GPUs
      CLS_ARRAY=($CLASSES)
      for i in "${!CLS_ARRAY[@]}"; do
        CLS="${CLS_ARRAY[$i]}"
        GPU_ID="${GPU_ARRAY[$i % $NUM_GPUS]}"
        
        RN_PREFIX_BASE=${RUN_NAME_PREFIX:-"${SUB}"}
        RN_PREFIX="${RN_PREFIX_BASE}_bc${CH}"
        echo "[RUN PARALLEL] sub=${SUB} ch=${CH} cls=${CLS} prefix=${RN_PREFIX} gpu=${GPU_ID}"
        
        # Build W&B args
        wandb_args=""
        if [[ "$USE_WANDB" == true ]]; then
          wandb_args="--use_wandb --wandb_project $WANDB_PROJECT"
          if [[ -n "$WANDB_ENTITY" ]]; then
            wandb_args="$wandb_args --wandb_entity $WANDB_ENTITY"
          fi
        fi
        
        # Run in background
        bash scripts/train_subject.sh \
          --data_path "$DATA_PATH" \
          --sub "$SUB" \
          --eeg_channels "$CH" \
          --num_classes "$CLS" \
          --max_steps "$MAX_STEPS" \
          --in_channels "$IN_CHANNELS" \
          --run_name_prefix "$RN_PREFIX" \
          --gpu_id "$GPU_ID" \
          $wandb_args &
      done
      
      # Wait for all parallel jobs to finish before moving to next channel
      wait
      echo "[DONE] Finished channel ${CH} for subject ${SUB}"
    else
      # Sequential mode (original behavior)
      for CLS in $CLASSES; do
        RN_PREFIX_BASE=${RUN_NAME_PREFIX:-"${SUB}"}
        RN_PREFIX="${RN_PREFIX_BASE}_bc${CH}"
        echo "[RUN] sub=${SUB} ch=${CH} cls=${CLS} prefix=${RN_PREFIX}"
        
        # Build W&B args
        wandb_args=""
        if [[ "$USE_WANDB" == true ]]; then
          wandb_args="--use_wandb --wandb_project $WANDB_PROJECT"
          if [[ -n "$WANDB_ENTITY" ]]; then
            wandb_args="$wandb_args --wandb_entity $WANDB_ENTITY"
          fi
        fi
        
        # Add GPU ID if specified
        gpu_arg=""
        if [[ -n "$GPU_IDS" ]]; then
          GPU_ARRAY=($GPU_IDS)
          gpu_arg="--gpu_id ${GPU_ARRAY[0]}"
        fi
        
        bash scripts/train_subject.sh \
          --data_path "$DATA_PATH" \
          --sub "$SUB" \
          --eeg_channels "$CH" \
          --num_classes "$CLS" \
          --max_steps "$MAX_STEPS" \
          --in_channels "$IN_CHANNELS" \
          --run_name_prefix "$RN_PREFIX" \
          $gpu_arg \
          $wandb_args
      done
    fi
  done
done

echo "[DONE] Experiment sweep finished."
