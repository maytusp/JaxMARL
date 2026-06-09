#!/bin/bash --login
#SBATCH -p gpuL               # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                 # CPU cores available to the host code

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

# MEP has two stages:
# 1. population: train a live entropy-diverse population from scratch.
# 2. br: train the final best-response agent against the frozen MEP pool.
#
# Set RUN_MEP_BR=true to run the final BR stage after population training.

layouts=(coord_ring2 counter_circuit2 cramped_room2)
# layouts=(cramped_room2)

MEP_POOL_PREFIX="${MEP_POOL_PREFIX:-checkpoints/mep_pool/}"
MEP_BR_PREFIX="${MEP_BR_PREFIX:-checkpoints/mep_br/}"
RUN_MEP_POPULATION="${RUN_MEP_POPULATION:-false}"
RUN_MEP_BR="${RUN_MEP_BR:-true}"

for layout in "${layouts[@]}"; do
  if [ "$RUN_MEP_POPULATION" = true ]; then
    python -m baselines.overcookedv2.train_mep \
      --config-path=config/oc_extended/mep_pool/ \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++CHECKPOINTS_PREFIX="$MEP_POOL_PREFIX"
  fi

  if [ "$RUN_MEP_BR" = true ]; then
    python -m baselines.overcookedv2.train_mep \
      --config-path=config/oc_extended/mep_br/ \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PARTNER_CHECKPOINTS_PREFIX="$MEP_POOL_PREFIX" \
      ++CHECKPOINTS_PREFIX="$MEP_BR_PREFIX"
  fi
done
