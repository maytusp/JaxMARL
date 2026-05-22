#!/bin/bash --login
#SBATCH -p gpuA             # A100 GPUs
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
  ++ENV_KWARGS.rare_recipe_prob=0.1 \
  ++enable_stc=true \
  ++theta_tag=0.001 \
  ++tag_mode=soft \
  ++tag_temperature=0.001 \
  ++eta_slow=0.00001 \
  ++top_k=null \
  ++top_k_fraction=0.1 \
  ++capture_norm=none \
  ++capture_clip_max=null \
  ++stc_apply_to=actor_only \
  ++stc_exclude_norm=true \
  ++latent_dim=64 \
  ++latent_lr=0.001 \
  ++CHECKPOINTS_PREFIX=checkpoints/overcooked_rare/rare_prob_01/stc_postppo_actor_only/
