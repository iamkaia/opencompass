#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    cat <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash ./tools/T0623_train_single_all_layers_hybrid_qwen.sh

Reuses the completed T0622 single-all-layers cache. It trains all mrs_only,
mrs_plus, and new_only routers with:

  expert_ce + weighted_sum_mse

Then it runs official weighted_sum evaluation at sharpness 1/2/3. No cache
is rebuilt. All outputs go to a fresh T0623_single_all_layers_hybrid_* root.
EOF
    exit 0
fi

export RUN_STAMP
export OUT_ROOT="${OUT_ROOT:-./T0623_single_all_layers_hybrid_${RUN_STAMP}}"
export CACHE_ROOT="${CACHE_ROOT:-./T0622_single_all_layers_wsum_run1/cache/qwen3_9task_3expert_single_all_layers_800_200}"
export EXPERT_CE_WEIGHT="${EXPERT_CE_WEIGHT:-1.0}"
export MSE_WEIGHT="${MSE_WEIGHT:-1.0}"

bash ./tools/T0622_single_all_layers_weighted_sum_qwen.sh train
bash ./tools/T0622_single_all_layers_weighted_sum_qwen.sh eval
