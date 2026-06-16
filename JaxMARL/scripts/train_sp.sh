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

# layouts=(coord_ring2 counter_circuit2 cramped_room2)
# layouts=(cramped_room2)
layouts=(coord_ring counter_circuit cramped_room5x5)

for layout in "${layouts[@]}"; do
  python -m baselines.overcookedv2.train_sp \
    --config-path=config/oc_extended/sp_pool/ \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++CHECKPOINTS_PREFIX=checkpoints/sp/
done
