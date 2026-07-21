#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
export EXP_ROOT="${EXP_ROOT:-./T0615_marginal_only_t0p25_20260615}"
export TARGET_TEMPERATURE=0.25
export AUX_ONLY=1
export JOINT_LOSS=cache_oracle_matrix_kl
export MSE_WEIGHT=1.0
export WEIGHT_TAG=marginal_only
export LOG_PREFIX=T0615
export BEST_METRIC=weighted_sum_marginal_mse

mode="${1:-tasks}"
tasks="${2:-boolq,rte,siqa,piqa,openbookqa,arc_c}"

case "$mode" in
    mrs_only)
        exec bash tools/T0615_train_marginal_only_qwen.sh mrs_only
        ;;
    tasks|mrs_plus)
        exec bash tools/T0615_train_marginal_only_qwen.sh tasks "$tasks"
        ;;
    all)
        # The lower-level launcher refuses to overwrite an existing mrs_only
        # checkpoint, so use this only with a fresh EXP_ROOT.
        exec bash tools/T0615_train_marginal_only_qwen.sh all "$tasks"
        ;;
    -h|--help|help)
        cat <<'EOF'
usage:
  bash tools/T0615_train_marginal_only_t0p25_qwen.sh mrs_only
  bash tools/T0615_train_marginal_only_t0p25_qwen.sh tasks [task_csv]
  bash tools/T0615_train_marginal_only_t0p25_qwen.sh all [task_csv]

Defaults to training the six mrs_plus routers under:
  ./T0615_marginal_only_t0p25_20260615
EOF
        ;;
    *)
        echo "unsupported mode: $mode" >&2
        exit 2
        ;;
esac
