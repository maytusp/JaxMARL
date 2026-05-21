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

python -m baselines.overcooked_rare.train_stc \
  --config-path=config/stc \
  --config-name=single_rare_room \
  +ENV_KWARGS.front_obs=true \
  ++ENV_KWARGS.rare_recipe_prob=0.01 \
  ++bf.apply_to=actor_only \
  ++bf.num_states=2 \
  ++bf.tau_min=1000.0 \
  ++bf.tau_max=200000.0 \
  ++bf.stc.enabled=true \
  ++bf.stc.capture_mode=advantage \
  ++CHECKPOINTS_PREFIX=checkpoints/overcooked_rare/rare_prob_001/stc_actor_only_n3_tau1k_200k/
