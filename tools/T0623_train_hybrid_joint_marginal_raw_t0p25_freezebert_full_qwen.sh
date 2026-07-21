#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    cat <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash ./tools/T0623_train_hybrid_joint_marginal_raw_t0p25_freezebert_full_qwen.sh

Controlled rerun of T0616 with the same cache, tasks, freeze-BERT setting,
raw correct_conf target at temperature 0.25, and weighted_sum evaluations.

Training loss:
  correct_conf_ce + MSE_WEIGHT * weighted_sum_marginal_mse

The original T0616 used weighted_sum_marginal_mse only. Outputs are written
to a fresh T0623_wsum_hybrid_joint_marginal_raw_t0p25_freezebert_full_* root.
EOF
    exit 0
fi

export RUN_STAMP
export RUN_LABEL="${RUN_LABEL:-T0623_hybrid}"
export OUT_ROOT="${OUT_ROOT:-./T0623_wsum_hybrid_joint_marginal_raw_t0p25_freezebert_full_${RUN_STAMP}}"
export WEIGHTED_SUM_AUX_ONLY=0
export MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
export BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"

exec bash ./tools/T0616_train_wsum_raw_t0p25_freezebert_full_qwen.sh "$@"
