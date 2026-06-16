cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

CONFIG_PATH="config/oc_extended/phase2"
LAYOUTS=(
  cramped_room2
  # counter_circuit2
  coord_ring2
)

METHODS=(
  # sp:checkpoints/sp/
  e3t:checkpoints/e3t/
  # ph2v4:checkpoints/ph2v4/
  # ph2v4_ablate:checkpoints/ph2v4_ablate/
  # fcp:checkpoints/fcp/
)

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
    echo "Evaluating XP for method=${method}, layout=${layout}, prefix=${prefix}, result_name=${result_name}, perspective_transform=${perspective_transform}"
    extra_args=()
    if [[ -n "${XP_CHECKPOINT_STEP:-}" ]]; then
      extra_args+=(++XP_CHECKPOINT_STEP="$XP_CHECKPOINT_STEP")
    fi

    python -m baselines.overcookedv2.eval_xp \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$method" \
      ++CHECKPOINTS_PREFIX="$prefix" \
      ++PERSPECTIVE_TRANSFORM=$perspective_transform \
      ++XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++XP_SAVE_DIR="xp_results" \
      ++XP_SAVE_VIDEOS=true \
      "${extra_args[@]}"
  done
done
