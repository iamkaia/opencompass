#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
BASE_RUN_ROOT="${BASE_RUN_ROOT:-./T0615_marginal_only_t0p25_all_20260615_20260615_072440}"
BASE_ROUTER="${BASE_ROUTER:-$BASE_RUN_ROOT/train/routers/router_T0603_qwen3_fp16_mrs_only_taskcls_trainbert_qwenfix_correct_conf_ce_t1_wsmsemarginal_only_3expert_sst2words}"
OUT_ROOT="${OUT_ROOT:-./T0615_new_only_marginal_t0p25_freezebert_20260615}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"

BERT="${BERT:-./task_classifier_ckpt}"
EXPERTS="medmcqa,race,sst2"
LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="$OUT_ROOT/routers"

timestamp() { date '+%F %T %Z'; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

validate_task() {
    case "$1" in
        boolq|rte|siqa|piqa|openbookqa|arc_c) ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

[[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
    echo "missing cache: $CACHE_ROOT" >&2
    exit 1
}
[[ -f "$BASE_ROUTER/router_heads.pt" && -f "$BASE_ROUTER/router_config.json" ]] || {
    echo "missing base mrs_only router: $BASE_ROUTER" >&2
    exit 1
}
mkdir -p "$LOG_DIR" "$ROUTER_DIR"

IFS=',' read -r -a task_list <<< "$TASKS"
for task in "${task_list[@]}"; do
    validate_task "$task"
    output_dir="$ROUTER_DIR/router_T0615_qwen3_fp16_new_only_${task}_from_mrs_marginal_t0p25_freezebert_3expert_sst2words"
    log_file="$LOG_DIR/T0615_train_qwen3_fp16_new_only_${task}_from_mrs_marginal_t0p25_freezebert_${RUN_STAMP}.log"
    [[ ! -e "$output_dir" ]] || { echo "router output exists: $output_dir" >&2; exit 1; }
    [[ ! -e "$log_file" ]] || { echo "log output exists: $log_file" >&2; exit 1; }

    log "[START] task=$task base=$BASE_ROUTER out=$output_dir"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        --load_from "$BASE_ROUTER" \
        --out_dir "$output_dir" \
        --sample_task_names "$task" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size 32 \
        --epochs 10 \
        --lr 2e-4 \
        --joint_loss cache_oracle_matrix_kl \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 0.25 \
        --pair_loss_normalization sample_minmax \
        --weighted_sum_marginal_mse_weight 1.0 \
        --weighted_sum_aux_only \
        --best_metric weighted_sum_marginal_mse \
        --early_stop_patience 2 \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] task=$task out=$output_dir log=$log_file"
done

log "[DONE] all new_only routers output=$OUT_ROOT"
