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

python -m baselines.IPPO.lmtom_fcp_overcooked_v2 --config-path=config/oc_extended/cpt_inp_agg --config-name=cramped_room2
python -m baselines.IPPO.lmtom_fcp_overcooked_v2 --config-path=config/oc_extended/cpt_inp_agg --config-name=coord_ring2
python -m baselines.IPPO.lmtom_fcp_overcooked_v2 --config-path=config/oc_extended/cpt_inp_agg --config-name=counter_circuit2