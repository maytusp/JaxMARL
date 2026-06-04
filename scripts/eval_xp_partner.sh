SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"
echo "Script directory: $SCRIPT_DIR"
echo "Repo directory: $REPO_DIR"

source activate jax

CONFIG_PATH="config/oc_extended/sp_pool"
LAYOUTS=(
  cramped_room2
  # counter_circuit2
  # coord_ring2
)

# Format:
# EGO_METHOD:EGO_CHECKPOINTS_PREFIX:PARTNER_METHOD:PARTNER_CHECKPOINTS_PREFIX:RESULT_NAME
#
# The default entry evaluates self-play ego agents against a fixed self-play
# partner pool. Add PARTNER_XP_SEEDS or PARTNER_XP_CHECKPOINTS below to make
# the partner set smaller/fixed.
PAIRS=(
  # sp:checkpoints/sp/:sp:checkpoints/sp/:sp_ego_vs_sp_partner
  # ph2_v2:checkpoints/ph2v2/:sp:checkpoints/sp/:ph2v2_ego_vs_sp_partner
  # fcp:checkpoints/fcp/:sp:checkpoints/sp/:fcp_ego_vs_sp_partner
  fcp:checkpoints/fcp/:sp:checkpoints/fcp/:fcp_ego_vs_fcp_partner
)

# Optional fixed partner subset.
# Examples:
# PARTNER_XP_SEEDS="[0,1,2]"
# Single checkpoint:
# PARTNER_XP_CHECKPOINTS="baseline_seed_0_step_30000000.msgpack"
#
# Multiple checkpoints use Hydra list syntax:
# PARTNER_XP_CHECKPOINTS="[baseline_seed_0_step_30000000.msgpack,baseline_seed_1_step_30000000.msgpack]"
PARTNER_XP_SEEDS=""
PARTNER_XP_CHECKPOINTS=""

for pair_spec in "${PAIRS[@]}"; do
  IFS=":" read -r ego_method ego_prefix partner_method partner_prefix result_name <<< "$pair_spec"

  ego_perspective_transform=true
  if [[ "$ego_method" == *ablat* ]]; then
    ego_perspective_transform=false
  fi

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating XP partner matrix for ego_method=${ego_method}, partner_method=${partner_method}, layout=${layout}"
    echo "ego_prefix=${ego_prefix}, partner_prefix=${partner_prefix}, result_name=${result_name}, ego_perspective_transform=${ego_perspective_transform}"

    extra_args=()
    if [[ -n "$PARTNER_XP_SEEDS" ]]; then
      extra_args+=(++PARTNER_XP_SEEDS="$PARTNER_XP_SEEDS")
    fi
    if [[ -n "$PARTNER_XP_CHECKPOINTS" ]]; then
      extra_args+=(++PARTNER_XP_CHECKPOINTS="$PARTNER_XP_CHECKPOINTS")
    fi

    python -m baselines.overcookedv2.eval_xp_partner \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$ego_method" \
      ++EGO_TRAINING_METHOD="$ego_method" \
      ++EGO_CHECKPOINTS_PREFIX="$ego_prefix" \
      ++PARTNER_TRAINING_METHOD="$partner_method" \
      ++PARTNER_CHECKPOINTS_PREFIX="$partner_prefix" \
      ++PERSPECTIVE_TRANSFORM="$ego_perspective_transform" \
      ++EGO_XP_LATEST_PER_SEED=true \
      ++PARTNER_XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++XP_SAVE_DIR="xp_partner_results" \
      "${extra_args[@]}"
  done
done
