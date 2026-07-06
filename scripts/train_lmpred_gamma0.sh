#!/bin/bash --login
#SBATCH -p gpuL             # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 3-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
#SBATCH -n 1                  # One Slurm task
#SBATCH -c 12                 # 8 CPU cores available to the host code.
                              # Can use up to 12 CPUs with an A100 GPU.
                              # Can use up to 12 CPUs with an L40s GPU.

# Latest version of CUDA

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

layouts=(counter_circuit coord_ring cramped_room5x5 asymm_advantages forced_coord)
self_pred_coefs=(0.05)
self_pred_gammas="[0.0]"

for layout in "${layouts[@]}"; do
  for self_pred_coef in "${self_pred_coefs[@]}"; do
    python -m baselines.overcookedv2.train_lmpred \
      --config-path=config/oc_extended/sp \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=true \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++CHECKPOINTS_PREFIX="checkpoints/lmpred_gamma0/"
  done
done

#ABLATE
for layout in "${layouts[@]}"; do
  for self_pred_coef in "${self_pred_coefs[@]}"; do
    python -m baselines.overcookedv2.train_lmpred \
      --config-path=config/oc_extended/sp \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=false \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++CHECKPOINTS_PREFIX="checkpoints/lmpred_gamma0_ablate/"
  done
done
