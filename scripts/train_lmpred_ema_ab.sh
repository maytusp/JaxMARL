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

layouts=(counter_circuit coord_ring cramped_room5x5)
ablation_names=(no_self_pred gamma0 gamma09)
ablation_coefs=(0.0 0.05 0.05)
ablation_gammas=("[0.0]" "[0.0]" "[0.9]")

for layout in "${layouts[@]}"; do
  for i in "${!ablation_names[@]}"; do
    ablation_name="${ablation_names[$i]}"
    self_pred_coef="${ablation_coefs[$i]}"
    self_pred_gammas="${ablation_gammas[$i]}"

    python -m baselines.overcookedv2.train_lmpred_ema \
      --config-path=config/oc_extended/phase2 \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=true \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++USE_OTHER_STREAM_EMA=true \
      ++OTHER_STREAM_EMA_DECAY=0.996 \
      ++OTHER_STREAM_EMA_DECAY_END=1.0 \
      ++WANDB_RUN_SUFFIX="$ablation_name" \
      ++CHECKPOINTS_PREFIX="checkpoints/lmpred_ema_${ablation_name}/"
  done
done
