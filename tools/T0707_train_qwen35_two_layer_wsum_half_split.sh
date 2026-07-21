#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"

export MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
export RUN_STAMP
export OUT_ROOT="${OUT_ROOT:-./T0707_qwen35_two_layer_wsum_t0p25_half16_${RUN_STAMP}}"
export RUN_LABEL="${RUN_LABEL:-T0707_qwen35_half16_two}"
export CACHE_ROOT="${CACHE_ROOT:-$OUT_ROOT/cache/qwen35_9task_3expert_two_layer_half16_800_200}"

exec bash tools/T0704_train_qwen35_two_layer_wsum.sh "$@"
