# # !/bin/bash --login
# # SBATCH -p gpuA              # A100 GPUs
# # SBATCH -G 1                  # 1 GPU
# # SBATCH -t 1-0                # Wallclock limit (1-0 is 1 day, 4-0 is the max permitted)
# # SBATCH -n 1                  # One Slurm task
# # SBATCH -c 12                 # 8 CPU cores available to the host code.
# #                               Can use up to 12 CPUs with an A100 GPU.
# #                               Can use up to 12 CPUs with an L40s GPU.

# # Latest version of CUDA
cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

# layouts=(counter_circuit2 coord_ring2 cramped_room2)
layouts=(coord_ring2)
self_pred_coefs=(0.05)
self_pred_gammas="[0.0,0.5,0.9]"

for layout in "${layouts[@]}"; do
  for self_pred_coef in "${self_pred_coefs[@]}"; do
    python -m baselines.overcookedv2.train_ph2v4 \
      --config-path=config/oc_extended/phase2 \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=false \
      ++SELF_PRED_COEF="$self_pred_coef" \
      ++SELF_PRED_GAMMAS="$self_pred_gammas" \
      ++PRETRAINED_CHECKPOINTS_PREFIX="" \
      ++CHECKPOINTS_PREFIX="checkpoints/sp_dual/" \
      ++FINETUNE_OTHER_STREAM=true
  done
done
