#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

CONFIG_PATH="config/oc_extended/phase2"
LAYOUTS=(
  cramped_room2
  counter_circuit2
  coord_ring2
)

# Format: TRAINING_METHOD:CHECKPOINTS_PREFIX
METHODS=(
  ph2v4:checkpoints/ph2v4/
  ph2v4_ablate:checkpoints/ph2v4_ablate/
)

# Optional overrides:
#   XP_CHECKPOINT_STEP=220 sbatch scripts/eval_alignment.sh
#   XP_SEEDS="[0,1]" sbatch scripts/eval_alignment.sh
#   EVAL_NUM_ENVS=64 EVAL_NUM_EPISODES=20 sbatch scripts/eval_alignment.sh
EVAL_NUM_ENVS="${EVAL_NUM_ENVS:-128}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-100}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-}"
ALIGN_RIDGE_LAMBDA="${ALIGN_RIDGE_LAMBDA:-0.001}"
ALIGN_TRAIN_FRACTION="${ALIGN_TRAIN_FRACTION:-0.7}"
ALIGN_SAVE_DIR="${ALIGN_SAVE_DIR:-alignment_results}"

for method_spec in "${METHODS[@]}"; do
  method="${method_spec%%:*}"
  prefix="${method_spec#*:}"
  result_name="$(basename "${prefix%/}")"
  perspective_transform=true
  if [[ "$method" == *ablat* ]]; then
    perspective_transform=false
  fi

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating alignment for method=${method}, layout=${layout}, prefix=${prefix}, result_name=${result_name}, perspective_transform=${perspective_transform}"
    extra_args=()
    if [[ -n "${XP_CHECKPOINT_STEP:-}" ]]; then
      extra_args+=(++XP_CHECKPOINT_STEP="$XP_CHECKPOINT_STEP")
    fi
    if [[ -n "${XP_SEEDS:-}" ]]; then
      extra_args+=(++XP_SEEDS="$XP_SEEDS")
    fi
    if [[ -n "$EVAL_NUM_STEPS" ]]; then
      extra_args+=(++EVAL_NUM_STEPS="$EVAL_NUM_STEPS")
    fi

    python -m baselines.overcookedv2.eval_alignment \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$method" \
      ++CHECKPOINTS_PREFIX="$prefix" \
      ++PERSPECTIVE_TRANSFORM=$perspective_transform \
      ++XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++EVAL_NUM_ENVS="$EVAL_NUM_ENVS" \
      ++EVAL_NUM_EPISODES="$EVAL_NUM_EPISODES" \
      ++ALIGN_RIDGE_LAMBDA="$ALIGN_RIDGE_LAMBDA" \
      ++ALIGN_TRAIN_FRACTION="$ALIGN_TRAIN_FRACTION" \
      ++ALIGN_SAVE_DIR="$ALIGN_SAVE_DIR" \
      "${extra_args[@]}"
  done
done
