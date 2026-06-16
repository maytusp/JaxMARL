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
FCP_CHECKPOINTS_PREFIX="checkpoints/fcp/"
SP_PARTNER_CHECKPOINTS_PREFIX="checkpoints/sp/"

LAYOUTS=(
  cramped_room2
  # counter_circuit2
  # coord_ring2
)

for layout in "${LAYOUTS[@]}"; do
  echo "Evaluating FCP egos against SP partners for layout=${layout}"
  python -m baselines.overcookedv2.eval_fcp \
    --config-path="$CONFIG_PATH" \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++ENV_KWARGS.random_reset=true \
    ++FCP_CHECKPOINTS_PREFIX="$FCP_CHECKPOINTS_PREFIX" \
    ++SP_PARTNER_CHECKPOINTS_PREFIX="$SP_PARTNER_CHECKPOINTS_PREFIX" \
    ++FCP_LATEST_PER_SEED=true \
    ++SP_PARTNER_LATEST_PER_SEED=false \
    ++FCP_EVAL_SAVE_DIR="fcp_eval_results"
done
