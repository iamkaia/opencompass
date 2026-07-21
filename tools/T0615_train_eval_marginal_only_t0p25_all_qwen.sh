#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-./T0615_marginal_only_t0p25_all_20260615_${RUN_STAMP}}"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,2.0,3.0}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
ROOT_LOG="$RUN_ROOT/pipeline.log"

log() {
    mkdir -p "$RUN_ROOT"
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$ROOT_LOG"
}

log "[START] train mrs_only and mrs_plus tasks=$TASKS target_temperature=0.25"
EXP_ROOT="$RUN_ROOT/train" \
TARGET_TEMPERATURE=0.25 \
    bash tools/T0615_train_marginal_only_t0p25_qwen.sh all "$TASKS"
log "[DONE] training"

IFS=',' read -r -a sharpness_values <<< "$SHARPNESS_LIST"
for sharpness in "${sharpness_values[@]}"; do
    sharp_tag="$(printf '%s' "$sharpness" | sed 's/-/m/g; s/\./p/g')"
    log "[START] official eval mrs_only and mrs_plus sharpness=$sharpness"
    EXP_ROOT="$RUN_ROOT/train" \
    EVAL_ROOT="$RUN_ROOT/eval_sharp${sharp_tag}" \
    RUN_STAMP="$RUN_STAMP" \
    WEIGHT_TAG=marginal_only \
    SHARPNESS="$sharpness" \
        bash tools/T0603_eval_wsmse1p0_weighted_sum.sh all "$TASKS"
    log "[DONE] official eval sharpness=$sharpness"
done

log "[DONE] pipeline output=$RUN_ROOT"
