#!/bin/bash --login
#SBATCH -p gpuA              # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                 # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

layouts=(counter_circuit coord_ring cramped_room5x5)
self_pred_coefs=(0.05)
self_pred_gammas="[0.0,0.5,0.9]"

INIT_CHECKPOINTS_PREFIX="${INIT_CHECKPOINTS_PREFIX:-checkpoints/ph2v4_ablate/}"
PARTNER_CHECKPOINTS_PREFIX="${PARTNER_CHECKPOINTS_PREFIX:-checkpoints/ph2v4_ablate/}"
CHECKPOINTS_PREFIX="${CHECKPOINTS_PREFIX:-checkpoints/lmpred_pop_ablate/}"

for layout in "${layouts[@]}"; do
  for self_pred_coef in "${self_pred_coefs[@]}"; do
    python -m baselines.overcookedv2.train_lmpred_pop \
      --config-path=config/oc_extended/lmpred_pop \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=false \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++INIT_CHECKPOINTS_PREFIX="$INIT_CHECKPOINTS_PREFIX" \
      ++PARTNER_CHECKPOINTS_PREFIX="$PARTNER_CHECKPOINTS_PREFIX" \
      ++CHECKPOINTS_PREFIX="$CHECKPOINTS_PREFIX"
  done
done
