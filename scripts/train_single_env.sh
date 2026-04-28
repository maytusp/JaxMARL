#!/bin/bash --login
#SBATCH -p gpuA            # A100 GPUs
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

python -m baselines.overcookedv2.train_single --config-path=config/oc_single/train --config-name=counter_circuit2 +ENV_KWARGS.front_obs=true CHECKPOINTS_PREFIX=checkpoints/single_frontobs/
python -m baselines.overcookedv2.train_single --config-path=config/oc_single/train --config-name=coord_ring2 +ENV_KWARGS.front_obs=true CHECKPOINTS_PREFIX=checkpoints/single_frontobs/
python -m baselines.overcookedv2.train_single --config-path=config/oc_single/train --config-name=cramped_room2 +ENV_KWARGS.front_obs=true CHECKPOINTS_PREFIX=checkpoints/single_frontobs/