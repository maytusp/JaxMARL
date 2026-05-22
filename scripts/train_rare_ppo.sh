#!/bin/bash --login
#SBATCH -p gpuL              # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                 # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

python -m baselines.overcooked_rare.train_ppo \
  --config-path=config/ppo \
  --config-name=single_rare_room \
  ++ENV_KWARGS.rare_recipe_prob=0.5 \
  ++CHECKPOINTS_PREFIX=checkpoints/overcooked_rare/rare_prob_05/ppo/
