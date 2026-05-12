#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12

set -euo pipefail

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

set +u
source activate jax
set -u

CONFIG_PATH="config/oc_extended/phase2"
LAYOUTS=(
  cramped_room2
  counter_circuit2
  coord_ring2
)

# Format: TRAINING_METHOD:CHECKPOINTS_PREFIX
# CHECKPOINTS_PREFIX should be the same method-specific prefix used during training.
METHODS=(
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp001/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp005/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp01/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp05/
  
  # ph2_v2:checkpoints/ph2v2/
  ph2_v2_ablate:checkpoints/ph2v2_ablate/
  # predz:checkpoints/predz/
  # privz:checkpoints/privz/
  # predz:checkpoints/predz/
  # predz:checkpoints/predz/
  # predz:checkpoints/predz/
  # predz:checkpoints/predz/
  # ph2_sp:checkpoints/sp/
  # ph2_v1:checkpoints/ph2v1/
)

for method_spec in "${METHODS[@]}"; do
  method="${method_spec%%:*}"
  prefix="${method_spec#*:}"
  result_name="$(basename "${prefix%/}")"

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating XP for method=${method}, layout=${layout}, prefix=${prefix}, result_name=${result_name}"
    python -m baselines.overcookedv2.eval_xp \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$method" \
      ++CHECKPOINTS_PREFIX="$prefix" \
      ++XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++XP_SAVE_DIR="xp_results"
  done
done
