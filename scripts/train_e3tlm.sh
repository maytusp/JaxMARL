#!/bin/bash --login
#SBATCH -p gpuA              # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                  # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

TOTAL_TIMESTEPS=1.5e8
REW_SHAPING_HORIZON=7.5e7
ENT_COEF=0.03

layouts=(coord_ring2 counter_circuit2 cramped_room2)

for layout in "${layouts[@]}"; do
  python -m baselines.overcookedv2.train_e3tlm \
    --config-path=config/oc_extended/e3t/ \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++PRETRAINED_CHECKPOINTS_PREFIX=checkpoints/single/ \
    ++CHECKPOINTS_PREFIX=checkpoints/e3tlm/ \
    ++MOA_COEF=1.0 \
    ++TRAIN_KWARGS.ckpt_id=0 \
    ++TRAIN_KWARGS.e3t_beta=0.2 \
    ++TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    ++REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    ++ENT_COEF="$ENT_COEF"
done
