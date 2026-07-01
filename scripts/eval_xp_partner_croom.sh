#!/bin/bash --login
#SBATCH -p gpuA
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

source activate jax

CONFIG_PATH="config/oc_extended/mep_br"
LAYOUTS=(
  # cramped_room2
  # counter_circuit2
  # coord_ring2
  # counter_circuit
  # coord_ring
  cramped_room5x5
)

# Format:
# EGO_METHOD:EGO_CHECKPOINTS_PREFIX:RESULT_NAME
#
# Evaluates ego seeds against a diverse partner pool. Partner methods and
# checkpoint prefixes are configured by PARTNER_POOL_SPECS below.
PAIRS=(
  ph2v5:checkpoints/ph2v5/:ph2v5_ego_vs_eval_pool_partner
  ph2v5_ablate:checkpoints/ph2v5_ablate/:ph2v5_ablate_ego_vs_eval_pool_partner
  ph2v4:checkpoints/ph2v4/:ph2v4_ego_vs_eval_pool_partner
  ph2v4_ablate:checkpoints/ph2v4_ablate/:ph2v4_ablate_ego_vs_eval_pool_partner
  sp:checkpoints/sp/:sp_ego_vs_eval_pool_partner
  e3t:checkpoints/e3t/:e3t_ego_vs_eval_pool_partner
  mep_br:checkpoints/mep_br/:mep_br_ego_vs_eval_pool_partner
  pbt:checkpoints/pbt/:pbt_ego_vs_eval_pool_partner
  # fcp:checkpoints/fcp/:fcp_ego_vs_eval_pool_partner
)

PARTNER_POOL_SPECS="${PARTNER_POOL_SPECS:-e3t:checkpoints/eval_pools/e3t;mep_br:checkpoints/eval_pools/mep_br;pbt:checkpoints/eval_pools/pbt;sp:checkpoints/eval_pools/sp}"

# Optional fixed partner subset.
# Examples:
# PARTNER_XP_SEEDS="[0,1,2]"
# Single checkpoint:
# PARTNER_XP_CHECKPOINTS="baseline_seed_0_step_30000000.msgpack"
#
# Multiple checkpoints use Hydra list syntax:
# PARTNER_XP_CHECKPOINTS="[baseline_seed_0_step_30000000.msgpack,baseline_seed_1_step_30000000.msgpack]"
PARTNER_XP_SEEDS="${PARTNER_XP_SEEDS:-}"
PARTNER_XP_CHECKPOINTS="${PARTNER_XP_CHECKPOINTS:-}"
EGO_XP_SEEDS="${EGO_XP_SEEDS:-[0,1,2,3,4]}"
EVAL_NUM_EPISODES="${EVAL_NUM_EPISODES:-128}"
EVAL_NUM_STEPS="${EVAL_NUM_STEPS:-}"
XP_EVAL_ALIGNMENT="${XP_EVAL_ALIGNMENT:-true}"
XP_PAIR_BATCH_SIZE="${XP_PAIR_BATCH_SIZE:-32}"
ALIGN_EVAL_NUM_EPISODES="${ALIGN_EVAL_NUM_EPISODES:-$EVAL_NUM_EPISODES}"
ALIGN_EVAL_NUM_STEPS="${ALIGN_EVAL_NUM_STEPS:-}"
ALIGN_PAIR_BATCH_SIZE="${ALIGN_PAIR_BATCH_SIZE:-2}"
ALIGN_MAX_PAIR_BATCH_SIZE="${ALIGN_MAX_PAIR_BATCH_SIZE:-4}"
ALIGN_RIDGE_LAMBDA="${ALIGN_RIDGE_LAMBDA:-0.001}"
ALIGN_TRAIN_FRACTION="${ALIGN_TRAIN_FRACTION:-0.7}"

if (( ALIGN_PAIR_BATCH_SIZE > ALIGN_MAX_PAIR_BATCH_SIZE )); then
  echo "Capping ALIGN_PAIR_BATCH_SIZE from ${ALIGN_PAIR_BATCH_SIZE} to ${ALIGN_MAX_PAIR_BATCH_SIZE}; alignment stores hidden trajectories and can OOM at large pair batches."
  ALIGN_PAIR_BATCH_SIZE="$ALIGN_MAX_PAIR_BATCH_SIZE"
fi

for pair_spec in "${PAIRS[@]}"; do
  IFS=":" read -r ego_method ego_prefix result_name <<< "$pair_spec"

  ego_perspective_transform=true
  if [[ "$ego_method" == *ablat* ]]; then
    ego_perspective_transform=false
  fi

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating XP partner matrix for ego_method=${ego_method}, layout=${layout}"
    echo "ego_prefix=${ego_prefix}, partner_pool_specs=${PARTNER_POOL_SPECS}, result_name=${result_name}, ego_perspective_transform=${ego_perspective_transform}"

    extra_args=()
    if [[ -n "$PARTNER_XP_SEEDS" ]]; then
      extra_args+=(++PARTNER_XP_SEEDS="$PARTNER_XP_SEEDS")
    fi
    if [[ -n "$PARTNER_XP_CHECKPOINTS" ]]; then
      extra_args+=(++PARTNER_XP_CHECKPOINTS="$PARTNER_XP_CHECKPOINTS")
    fi
    if [[ -n "$EVAL_NUM_STEPS" ]]; then
      extra_args+=(++EVAL_NUM_STEPS="$EVAL_NUM_STEPS")
    fi
    if [[ -n "$ALIGN_EVAL_NUM_STEPS" ]]; then
      extra_args+=(++ALIGN_EVAL_NUM_STEPS="$ALIGN_EVAL_NUM_STEPS")
    fi

    python -m baselines.overcookedv2.eval_xp_partner \
      --config-path="$CONFIG_PATH" \
      --config-name="$layout" \
      +ENV_KWARGS.front_obs=true \
      ++ENV_KWARGS.random_reset=false \
      ++TRAINING_METHOD="$ego_method" \
      ++EGO_TRAINING_METHOD="$ego_method" \
      ++EGO_CHECKPOINTS_PREFIX="$ego_prefix" \
      ++PARTNER_POOL_SPECS="'$PARTNER_POOL_SPECS'" \
      ++PERSPECTIVE_TRANSFORM="$ego_perspective_transform" \
      ++EGO_XP_LATEST_PER_SEED=true \
      ++EGO_XP_SEEDS="$EGO_XP_SEEDS" \
      ++PARTNER_XP_LATEST_PER_SEED=true \
      ++XP_RESULT_NAME="$result_name" \
      ++XP_SAVE_DIR="xp_partner_results" \
      ++XP_EVAL_ALIGNMENT="$XP_EVAL_ALIGNMENT" \
      ++EVAL_NUM_EPISODES="$EVAL_NUM_EPISODES" \
      ++XP_PAIR_BATCH_SIZE="$XP_PAIR_BATCH_SIZE" \
      ++ALIGN_EVAL_NUM_EPISODES="$ALIGN_EVAL_NUM_EPISODES" \
      ++ALIGN_PAIR_BATCH_SIZE="$ALIGN_PAIR_BATCH_SIZE" \
      ++ALIGN_MAX_PAIR_BATCH_SIZE="$ALIGN_MAX_PAIR_BATCH_SIZE" \
      ++ALIGN_RIDGE_LAMBDA="$ALIGN_RIDGE_LAMBDA" \
      ++ALIGN_TRAIN_FRACTION="$ALIGN_TRAIN_FRACTION" \
      "${extra_args[@]}"
  done
done
