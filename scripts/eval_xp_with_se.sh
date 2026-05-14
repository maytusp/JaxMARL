
cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax


LAYOUTS=(
  cramped_room2
  counter_circuit2
  coord_ring2
)

# Format: TRAINING_METHOD:CHECKPOINTS_PREFIX
# CHECKPOINTS_PREFIX is only used to derive result_name, matching eval_xp.sh.
METHODS=(
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp001/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp005/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp01/
  # predzhat_mask_ema:checkpoints/predzhat_mask_ema_cp05/

  ph2_v2:checkpoints/ph2v2/
  ph2_v2_ablate:checkpoints/ph2v2_ablate/
  dual:checkpoints/dual/
  dual_ablation:checkpoints/dual_ablation/

  # sp:checkpoints/sp/
  # predzhat_mask_ema_ablation:checkpoints/predzhat_mask_ema_ablation/
  predzhat_mask:checkpoints/predzhat_mask/
  predzhat_mask_ablation:checkpoints/predzhat_mask_ablation/
  ph2_sp:checkpoints/sp/
  # ph2_v1:checkpoints/ph2v1/
)

for method_spec in "${METHODS[@]}"; do
  method="${method_spec%%:*}"
  prefix="${method_spec#*:}"
  result_name="$(basename "${prefix%/}")"

  for layout in "${LAYOUTS[@]}"; do
    xp_matrix_csv="xp_results/${result_name}/${layout}/xp_matrix.csv"
    echo "Computing XP/SP SE from CSV for method=${method}, layout=${layout}, result_name=${result_name}"
    python -m baselines.overcookedv2.eval_xp_from_csv \
      "$xp_matrix_csv" \
      --overwrite-summary
  done
done
