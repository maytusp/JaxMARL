cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

CONFIG_PATH="config/oc_extended/bf_sp"
LAYOUTS=(
  counter_circuit2
  coord_ring2
  cramped_room2  
)

METHODS=(
  # ph2_v2:checkpoints/ph2v2/
  # ph2_v2_ablate:checkpoints/ph2v2_ablate/
  # sp:checkpoints/sp/
  # bf_sp:checkpoints/bf_sp/actor_n2_tau1k_200k
  # bf_sp:checkpoints/bf_sp/actor_n3_tau1k_200k
  bf_sp:checkpoints/bf_sp/all_n2_tau1k_200k
  bf_sp:checkpoints/bf_sp/all_n3_tau1k_200k
)

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
      ++XP_SAVE_DIR="xp_results"
  done
done
