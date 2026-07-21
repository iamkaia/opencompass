#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

OLD_ROOT="${OLD_ROOT:-./T0704_qwen35_two_layer_wsum_t0p25_20260704_132324}"
RUN_STAMP="${RUN_STAMP:-20260706_181134_maxout256}"
OUT_ROOT="${OUT_ROOT:-./T0706_qwen35_two_layer_wsum_t0p25_maxout256_20260706_181134}"

export RUN_STAMP
export OUT_ROOT
export MAX_OUT_LEN="${MAX_OUT_LEN:-256}"
export CACHE_ROOT="${CACHE_ROOT:-$OLD_ROOT/cache/qwen35_9task_3expert_two_layer_800_200}"
export ROUTER_DIR="${ROUTER_DIR:-$OLD_ROOT/routers}"
export SHARPNESS_LIST="${SHARPNESS_LIST:-3.0}"
export INCLUDE_MRS_PLUS="${INCLUDE_MRS_PLUS:-1}"
export INCLUDE_NEW_ONLY="${INCLUDE_NEW_ONLY:-0}"
export EVAL_BASELINES="${EVAL_BASELINES:-1}"
export RESUME="${RESUME:-1}"

bash tools/T0704_train_qwen35_two_layer_wsum.sh eval
bash tools/T0704_train_qwen35_two_layer_wsum.sh baseline
