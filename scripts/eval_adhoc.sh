#!/bin/bash --login
#SBATCH -p gpuL
#SBATCH -G 1
#SBATCH -t 1-0
#SBATCH -n 1
#SBATCH -c 12

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"
echo "Running ad-hoc teamplay evaluation: ego agents paired with a cross-method partner pool."

source activate jax

CONFIG_PATH="config/oc_extended/mep_br"
LAYOUTS=(
  cramped_room5x5
  counter_circuit
  coord_ring
  # asymm_advantages
  # forced_coord
)

# Format:
# EGO_METHOD:EGO_CHECKPOINTS_PREFIX:RESULT_NAME
#
# Ad-hoc teamplay evaluation measures generalisation across training methods.
# Partner policies are intentionally a diverse cross-method pool; they do not
# need to be strong task specialists.
PAIRS=(
  ph2v5:checkpoints/ph2v5/:ph2v5_ad_hoc_teamplay
  ph2v5_ablate:checkpoints/ph2v5_ablate/:ph2v5_ablate_ad_hoc_teamplay
  ph2v4:checkpoints/ph2v4/:ph2v4_ad_hoc_teamplay
  ph2v4_ablate:checkpoints/ph2v4_ablate/:ph2v4_ablate_ad_hoc_teamplay
  sp:checkpoints/sp/:sp_ad_hoc_teamplay
  e3t:checkpoints/e3t/:e3t_ad_hoc_teamplay
  mep_br:checkpoints/mep_br/:mep_br_ad_hoc_teamplay
  pbt:checkpoints/pbt/:pbt_ad_hoc_teamplay
  fcp:checkpoints/fcp/:fcp_ad_hoc_teamplay
  lmpred_ema:checkpoints/lmpred_ema/:lmpred_ema_ad_hoc_teamplay
  lmpred_ema_ablate:checkpoints/lmpred_ema_ablate/:lmpred_ema_ablate_ad_hoc_teamplay
  lmpred_ema_gamma0:checkpoints/lmpred_ema_gamma0/:lmpred_ema_gamma0_ad_hoc_teamplay
  lmpred_ema_gamma09:checkpoints/lmpred_ema_gamma09/:lmpred_ema_gamma09_ad_hoc_teamplay
  lmpred_ema_no_self_pred:checkpoints/lmpred_ema_no_self_pred/:lmpred_ema_no_self_pred_ad_hoc_teamplay
  lmpred:checkpoints/lmpred/:lmpred_ad_hoc_teamplay
  lmpred_ablate:checkpoints/lmpred_ablate/:lmpred_ablate_ad_hoc_teamplay
  # lmpred_gamma0:checkpoints/lmpred_gamma0:lmpred_gamma0_ad_hoc_teamplay
  # lmpred_gamma0_ablate:checkpoints/lmpred_gamma0_ablate:lmpred_gamma0_ablate_ad_hoc_teamplay
  # lmpred_gamma09:checkpoints/lmpred_gamma09:lmpred_gamma09_ad_hoc_teamplay
  # lmpred_gamma09_ablate:checkpoints/lmpred_gamma09_ablate:lmpred_gamma09_ablate_ad_hoc_teamplay
  # lmpred_no_self_pred:checkpoints/lmpred_no_self_pred:lmpred_no_self_pred_ad_hoc_teamplay
  # lmpred_ablate_no_self_pred:checkpoints/lmpred_ablate_no_self_pred:lmpred_ablate_no_self_pred_ad_hoc_teamplay
#   lmpredlow:checkpoints/lmpredlow/:lmpredlow_ad_hoc_teamplay
#   lmpredlow_ablate:checkpoints/lmpredlow_ablate/:lmpredlow_ablate_ad_hoc_teamplay
#   lmpredlow_ema:checkpoints/lmpredlow_ema/:lmpredlow_ema_ad_hoc_teamplay
#   lmpredlow_ema_ablate:checkpoints/lmpredlow_ema_ablate/:lmpredlow_ema_ablate_ad_hoc_teamplay
  # lmpredV2_ablate_no_self_pred:checkpoints/lmpredV2_ablate_no_self_pred/:lmpredV2_ablate_no_self_pred_ad_hoc_teamplay
  # lmpredV2_no_self_pred:checkpoints/lmpredV2_no_self_pred/:lmpredV2_no_self_pred_ad_hoc_teamplay
  # lmpredV2005:checkpoints/lmpredV2005/:lmpredV2005_ad_hoc_teamplay
  # lmpredV2005_ablate:checkpoints/lmpredV2005_ablate/:lmpredV2005_ablate_ad_hoc_teamplay  
  # lmpredV202:checkpoints/lmpredV202/:lmpredV202_ad_hoc_teamplay
  # lmpredV202_ablate:checkpoints/lmpredV202_ablate/:lmpredV202_ablate_ad_hoc_teamplay
  # lmpredV204:checkpoints/lmpredV204/:lmpredV204_ad_hoc_teamplay
  # lmpredV204_ablate:checkpoints/lmpredV204_ablate/:lmpredV204_ablate_ad_hoc_teamplay
  # lmpredV2005_gamma0:checkpoints/lmpredV2005_gamma0/:lmpredV2005_gamma0_ad_hoc_teamplay
  # lmpredV2005_gamma0_ablate:checkpoints/lmpredV2005_gamma0_ablate/:lmpredV2005_gamma0_ablate_ad_hoc_teamplay
  # lmpredV202_gamma0:checkpoints/lmpredV202_gamma0/:lmpredV202_gamma0_ad_hoc_teamplay
  # lmpredV204_gamma0:checkpoints/lmpredV204_gamma0/:lmpredV204_gamma0_ad_hoc_teamplay
  # lmpredV202_gamma0_ablate:checkpoints/lmpredV202_gamma0_ablate/:lmpredV202_gamma0_ablate_ad_hoc_teamplay
  # lmpredV204_gamma0_ablate:checkpoints/lmpredV204_gamma0_ablate/:lmpredV204_gamma0_ablate_ad_hoc_teamplay

)

PARTNER_POOL_SPECS="${PARTNER_POOL_SPECS:-e3t:checkpoints/eval_pools/e3t;mep_br:checkpoints/eval_pools/mep_br;pbt:checkpoints/eval_pools/pbt;sp:checkpoints/eval_pools/sp}"
AD_HOC_SAVE_DIR="${AD_HOC_SAVE_DIR:-ad_hoc_teamplay_results}"

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
  echo "Capping ALIGN_PAIR_BATCH_SIZE from ${ALIGN_PAIR_BATCH_SIZE} to ${ALIGN_MAX_PAIR_BATCH_SIZE}; alignment stores full, blind, and partner hidden trajectories and can OOM at large pair batches."
  ALIGN_PAIR_BATCH_SIZE="$ALIGN_MAX_PAIR_BATCH_SIZE"
fi

for pair_spec in "${PAIRS[@]}"; do
  IFS=":" read -r ego_method ego_prefix result_name <<< "$pair_spec"

  ego_perspective_transform=true
  if [[ "$ego_method" == *ablat* ]]; then
    ego_perspective_transform=false
  fi

  for layout in "${LAYOUTS[@]}"; do
    echo "Evaluating ad-hoc teamplay for ego_method=${ego_method}, layout=${layout}"
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

    python -m baselines.overcookedv2.eval_adhoc \
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
      ++XP_SAVE_DIR="$AD_HOC_SAVE_DIR" \
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
