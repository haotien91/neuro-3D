#!/bin/bash
# Run inference for all trained models
# Usage: bash scripts/run_all_inference.sh [--gpu_id N] [--checkpoint_step STEP]

GPU_ID=""
CHECKPOINT_STEP="100000"
PARALLEL=false

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu_id) GPU_ID="$2"; shift 2;;
    --checkpoint_step) CHECKPOINT_STEP="$2"; shift 2;;
    --parallel) PARALLEL=true; shift 1;;
    -h|--help)
      echo "Usage: $0 [--gpu_id N] [--checkpoint_step STEP] [--parallel]";
      echo "";
      echo "Options:";
      echo "  --gpu_id N           GPU to use (optional)";
      echo "  --checkpoint_step N  Checkpoint step to use (default: 100000)";
      echo "  --parallel           Run inferences in parallel (use with caution)";
      echo "";
      echo "Example:";
      echo "  bash scripts/run_all_inference.sh --gpu_id 0 --checkpoint_step 100000";
      exit 0;;
    *) echo "Unknown arg: $1"; exit 1;;
  esac
done

# Define all model configurations
# Format: "recon_path|pretrain_model|description"
MODELS=(
  # sub03 - 12 class
  "model/point_generate/shape/sub03/sub03-12class-1027-gpu_id0_bc64_cls12_20251028_005520|sub03-12class-1027-gpu_id0_bc64_cls12_20251028_005520|sub03-12class-bc64"
  
  # sub03 - 72 class
  "model/point_generate/shape/sub03/sub03-72class-1027-gpu_id1_bc64_cls72_20251028_005359|sub03-72class-1027-gpu_id1_bc64_cls72_20251028_005359|sub03-72class-bc64"
  
  # sub11 - 12 class
  "model/point_generate/shape/sub11/sub11-12class-1027-gpu_id0_bc64_cls12_20251028_010104|sub11-12class-1027-gpu_id0_bc64_cls12_20251028_010104|sub11-12class-bc64"
  
  # sub11 - 72 class
  "model/point_generate/shape/sub11/sub11-72class-1027-gpu_id1_bc64_cls72_20251028_010053|sub11-72class-1027-gpu_id1_bc64_cls72_20251028_010053|sub11-72class-bc64"
)

echo "========================================="
echo "Running Inference for All Models"
echo "========================================="
echo "Total models: ${#MODELS[@]}"
echo "Checkpoint step: $CHECKPOINT_STEP"
if [[ -n "$GPU_ID" ]]; then
  echo "GPU: $GPU_ID"
else
  echo "GPU: Not specified (will use default)"
fi
echo "Parallel mode: $PARALLEL"
echo "========================================="
echo ""

# Function to run single inference
run_inference() {
  local config="$1"
  local index="$2"
  
  IFS='|' read -r recon_path pretrain_model desc <<< "$config"
  
  echo ""
  echo "[$index/${#MODELS[@]}] Running inference: $desc"
  echo "-------------------------------------------"
  
  local cmd="bash scripts/run_inference.sh"
  cmd="$cmd --recon_path $recon_path"
  cmd="$cmd --pretrain_model $pretrain_model"
  cmd="$cmd --checkpoint_step $CHECKPOINT_STEP"
  
  if [[ -n "$GPU_ID" ]]; then
    cmd="$cmd --gpu_id $GPU_ID"
  fi
  
  eval $cmd
  
  local exit_code=$?
  if [[ $exit_code -eq 0 ]]; then
    echo "[$index/${#MODELS[@]}] ✓ Completed: $desc"
  else
    echo "[$index/${#MODELS[@]}] ✗ Failed: $desc (exit code: $exit_code)"
  fi
  
  return $exit_code
}

# Run inferences
if [[ "$PARALLEL" == true ]]; then
  echo "Running in parallel mode..."
  echo "WARNING: This may cause OOM if models are large!"
  echo ""
  
  for i in "${!MODELS[@]}"; do
    run_inference "${MODELS[$i]}" "$((i+1))" &
  done
  
  wait
  echo ""
  echo "All parallel inferences completed"
else
  echo "Running in sequential mode..."
  echo ""
  
  failed_count=0
  for i in "${!MODELS[@]}"; do
    run_inference "${MODELS[$i]}" "$((i+1))"
    if [[ $? -ne 0 ]]; then
      ((failed_count++))
    fi
    echo ""
  done
  
  echo "========================================="
  echo "Summary"
  echo "========================================="
  echo "Total: ${#MODELS[@]}"
  echo "Failed: $failed_count"
  echo "Success: $((${#MODELS[@]} - failed_count))"
  echo "========================================="
fi
