#!/bin/bash --login
#SBATCH -p gpuL             # A100 GPUs
#SBATCH -G 1                  # 1 GPU
#SBATCH -t 2-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
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
self_pred_coefs=(0.05 0.2 0.4)
self_pred_names=(lmpred005 lmpred02 lmpred04)
self_pred_gammas="[0.0,0.5,0.9]"
no_self_pred_coef=0.0


#ABLATE
for layout in "${layouts[@]}"; do
  for i in "${!self_pred_coefs[@]}"; do
    self_pred_coef="${self_pred_coefs[$i]}"
    self_pred_name="${self_pred_names[$i]}"

    python -m baselines.overcookedv2.train_lmpredV2 \
      --config-path=config/oc_extended/sp \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=false \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++CHECKPOINTS_PREFIX="checkpoints/${self_pred_name}_ablate/"
  done
done

# ABLATE, NO SELF-PREDICTION
for layout in "${layouts[@]}"; do
  python -m baselines.overcookedv2.train_lmpredV2 \
    --config-path=config/oc_extended/sp \
    --config-name="$layout" \
    +ENV_KWARGS.front_obs=true \
    ++PERSPECTIVE_TRANSFORM=false \
    ++SELF_PRED_COEF="$no_self_pred_coef" \
    ++SELF_PRED_GAMMAS="$self_pred_gammas" \
    ++WANDB_RUN_SUFFIX="no_self_pred" \
    ++CHECKPOINTS_PREFIX="checkpoints/lmpredV2_ablate_no_self_pred/"
done
