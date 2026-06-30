#!/bin/bash --login

set -euo pipefail

cd ..
SCRIPT_DIR="$(pwd)"
echo "Script directory: $SCRIPT_DIR"

set +u
source activate jax
set -u

python -m baselines.overcookedv2.xp_summary_table \
  --results-root "xp_results" \
  --output-csv "xp_results/xp_summary_table.csv" \
  --output-md "xp_results/xp_summary_table.md"
