#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0615_marginal_only_eval_20260615_${RUN_STAMP}}"
SHARPNESS="${SHARPNESS:-1.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"

T1_ROUTER="${T1_ROUTER:-./T0615_marginal_only_20260615/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmsemarginal_only_3expert_sst2words}"
T0P25_ROUTER="${T0P25_ROUTER:-./T0615_marginal_only_t0p25_20260615/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmsemarginal_only_3expert_sst2words}"

ROOT_LOG="$OUT_ROOT/run.log"

log() {
    mkdir -p "$OUT_ROOT"
    printf '[%s] %s\n' "$(date '+%F %T %Z')" "$*" | tee -a "$ROOT_LOG"
}

require_router() {
    local path="$1"
    [[ -f "$path/router_heads.pt" && -f "$path/router_config.json" ]] || {
        echo "missing router checkpoint: $path" >&2
        exit 1
    }
}

run_variant() {
    local label="$1" router="$2"
    require_router "$router"

    log "[START] $label train-split weighted_sum sharpness=$SHARPNESS"
    PY="$PY" \
    ROUTER_CKPT="$router" \
    ROOT="$OUT_ROOT/$label/train_split" \
    ROOT_LOG="$OUT_ROOT/$label/train_split.root.log" \
    RUN_STAMP="$RUN_STAMP" \
    SHARPNESS="$SHARPNESS" \
    ROUTING_TOPK="$ROUTING_TOPK" \
        bash tools/T0610_eval_router_train_split_qwen.sh weighted_sum

    log "[START] $label official 9-dataset weighted_sum sharpness=$SHARPNESS"
    PY="$PY" \
    EXP_ROOT="$OUT_ROOT/unused_exp_root" \
    OUT_ROOT="$OUT_ROOT/$label/official" \
    RUN_STAMP="$RUN_STAMP" \
    SHARPNESS="$SHARPNESS" \
    ROUTING_TOPK="$ROUTING_TOPK" \
    ROUTING_MODE=weighted_sum \
    MRS_ONLY_ROUTER_CKPT="$router" \
        bash tools/T0615_eval_marginal_only_official_qwen.sh

    log "[DONE] $label"
}

run_variant t1 "$T1_ROUTER"
run_variant t0p25 "$T0P25_ROUTER"
log "[DONE] all evaluations output=$OUT_ROOT"
