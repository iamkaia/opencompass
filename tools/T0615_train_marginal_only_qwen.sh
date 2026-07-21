#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
export CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
export EXP_ROOT="${EXP_ROOT:-./T0615_marginal_only_20260615}"
export AUX_ONLY=1
# This reproduces correct_conf_or_loss_t1 target construction, including the
# loss-softmax fallback for samples without any correct pair.
export JOINT_LOSS="${JOINT_LOSS:-cache_oracle_matrix_kl}"
export TARGET_TEMPERATURE="${TARGET_TEMPERATURE:-1.0}"
export MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
export WEIGHT_TAG="${WEIGHT_TAG:-marginal_only}"
export LOG_PREFIX="${LOG_PREFIX:-T0615}"
export BEST_METRIC="${BEST_METRIC:-weighted_sum_marginal_mse}"

exec bash tools/T0603_train_weighted_sum_marginal_mse_existing_cache.sh "$@"
