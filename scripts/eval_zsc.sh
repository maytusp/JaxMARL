#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Running ZSC evaluation: within-method generalisation with SP diagonal and XP off-diagonal performance."

source activate jax

CONFIG_PATH="config/oc_extended/phase2"
LAYOUTS=(
  cramped_room5x5
  counter_circuit
  coord_ring
)

METHODS=(
  # sp:checkpoints/sp/
  # e3t:checkpoints/e3t/
  # ph2v5:checkpoints/ph2v5/
  # ph2v5_ablate:checkpoints/ph2v5_ablate/
  # ph2v4:checkpoints/ph2v4/
  # ph2v4_ablate:checkpoints/ph2v4_ablate/
  # lmpred_ema:checkpoints/lmpred_ema/
  # lmpred_ema_ablate:checkpoints/lmpred_ema_ablate/
  lmpred_ema_gamma0:checkpoints/lmpred_ema_gamma0/
  lmpred_ema_gamma09:checkpoints/lmpred_ema_gamma09/
  lmpred_ema_no_self_pred:checkpoints/lmpred_ema_no_self_pred/
)

ZSC_SAVE_DIR="${ZSC_SAVE_DIR:-zsc_results}"

# Uncomment to evaluate an exact training step for every seed.
# If undefined, the latest checkpoint per seed is used.
# XP_CHECKPOINT_STEP=1026

for method_spec in "${METHODS[@]}"; do
  method="${method_spec%%:*}"
  prefix="${method_spec#*:}"
  result_name="$(basename "${prefix%/}")"
  perspective_transform=true
  if [[ "$method" == *ablat* ]]; then
    perspective_transform=false
  fi

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating ZSC for method=${method}, layout=${layout}, prefix=${prefix}, result_name=${result_name}, perspective_transform=${perspective_transform}"
    extra_args=()
    if [[ -n "${XP_CHECKPOINT_STEP:-}" ]]; then
      extra_args+=(++XP_CHECKPOINT_STEP="$XP_CHECKPOINT_STEP")
    fi

    python -m baselines.overcookedv2.eval_zsc \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$method" \
      ++CHECKPOINTS_PREFIX="$prefix" \
      ++PERSPECTIVE_TRANSFORM=$perspective_transform \
      ++XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++XP_SAVE_DIR="$ZSC_SAVE_DIR" \
      ++XP_SAVE_VIDEOS=false \
      "${extra_args[@]}"
  done
done
