cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

CONFIG_PATH="config/oc_single/train"
LAYOUTS=(
  cramped_room2
  counter_circuit2
  coord_ring2
)

CHECKPOINTS_PREFIX="checkpoints/single_frontobs/"
RESULT_NAME="$(basename "${CHECKPOINTS_PREFIX%/}")"

for layout in "${LAYOUTS[@]}"; do
  echo "Evaluating single-agent policy for layout=${layout}, prefix=${CHECKPOINTS_PREFIX}, result_name=${RESULT_NAME}"
  python -m baselines.overcookedv2.eval_single \
    --config-path="$CONFIG_PATH" \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++ENV_KWARGS.random_reset=false \
    ++CHECKPOINTS_PREFIX="$CHECKPOINTS_PREFIX" \
    ++SINGLE_TRAINING_METHOD="single" \
    ++SINGLE_LATEST_PER_SEED=true \
    ++SINGLE_RESULT_NAME="$RESULT_NAME" \
    ++SINGLE_SAVE_DIR="single_results"
done
