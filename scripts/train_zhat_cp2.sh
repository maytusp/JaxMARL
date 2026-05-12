#!/bin/bash --login
#SBATCH -p gpuL             # A100 GPUs
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

layouts=(coord_ring2 counter_circuit2 cramped_room2)
cp_values=(0.5)

for cp in "${cp_values[@]}"; do
  cp_label="${cp//./}"
  echo "Running predzhat_mask_ema with PREDZ_COEF=$cp"

  for layout in "${layouts[@]}"; do
    python -m baselines.overcookedv2.train_predzhat_mask_ema \
      --config-path=config/oc_extended/phase2/ \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PREDZ_COEF="$cp" \
      ++CHECKPOINTS_PREFIX="checkpoints/predzhat_mask_ema_cp${cp_label}/"
  done
done
