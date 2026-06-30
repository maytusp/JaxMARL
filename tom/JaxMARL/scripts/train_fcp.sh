#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 2-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

# FCP trains an ego agent against frozen SP partner checkpoints.
PARTNER_CHECKPOINTS_PREFIX="checkpoints/sp/"
CHECKPOINTS_PREFIX="checkpoints/fcp/"
FCP_PARTNER_STAGE_FRACTIONS="[0.1,0.5,1.0]"
FCP_MAX_PARTNERS=200
TOTAL_TIMESTEPS=1.5e8
REW_SHAPING_HORIZON=7.5e7
ENT_COEF=0.05


layouts=(coord_ring2 counter_circuit2 cramped_room2)

for layout in "${layouts[@]}"; do
  echo "Training FCP on ${layout}"
  python -m baselines.overcookedv2.train_fcp \
    --config-path=config/oc_extended/phase2/ \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++ENV_KWARGS.random_reset=false \
    ++PARTNER_CHECKPOINTS_PREFIX="$PARTNER_CHECKPOINTS_PREFIX" \
    ++CHECKPOINTS_PREFIX="$CHECKPOINTS_PREFIX" \
    ++FCP_PARTNER_STAGE_FRACTIONS="$FCP_PARTNER_STAGE_FRACTIONS" \
    ++FCP_MAX_PARTNERS="$FCP_MAX_PARTNERS" \
    ++TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    ++REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    ++ENT_COEF="$ENT_COEF"
done
