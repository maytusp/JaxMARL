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

# layouts=(counter_circuit2 coord_ring2 cramped_room2)
layouts=(cramped_room2)
sf_coefs=(0.05)

for layout in "${layouts[@]}"; do
  for sf_coef in "${sf_coefs[@]}"; do
    python -m baselines.overcookedv2.train_ph2_sf \
      --config-path=config/oc_extended/phase2 \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++PERSPECTIVE_TRANSFORM=false \
      ++PRETRAINED_CHECKPOINTS_PREFIX="checkpoints/single_sf_frontobs_sf${sf_coef}/" \
      ++CHECKPOINTS_PREFIX="checkpoints/ph2sf_ablate/"
  done
done
