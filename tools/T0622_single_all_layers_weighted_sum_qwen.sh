#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0622_single_all_layers_wsum_${RUN_STAMP}}"
DATA_ROOT="${DATA_ROOT:-./0602_router_train_dataset}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="medmcqa,race,sst2"
BERT="${BERT:-./task_classifier_ckpt}"
BASE_MODEL="Qwen/Qwen3-4B-Instruct-2507"
LORA_ROOT="./saves/Qwen/Qwen3-4B-Instruct-2507/lora"
MODEL_CONFIG="T0531_mrs_ablation_sst2words_hard_routing.py"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,2.0,3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-800}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-200}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"

CACHE_ROOT="${CACHE_ROOT:-$OUT_ROOT/cache/qwen3_9task_3expert_single_all_layers_800_200}"
EXPERT_CE_WEIGHT="${EXPERT_CE_WEIGHT:-0.0}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
ROUTER_ROOT="$OUT_ROOT/routers"
LOG_ROOT="$OUT_ROOT/logs"
OC_ROOT="$OUT_ROOT/opencompass_official"
RECORD_ROOT="$OUT_ROOT/router_records"
ROOT_LOG="$OUT_ROOT/pipeline.log"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }
temp_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'; }
validate_task() { case "$1" in boolq|rte|siqa|piqa|openbookqa|arc_c) ;; *) echo "bad task=$1" >&2; exit 2;; esac; }
task_dataset() { case "$1" in boolq) echo SuperGLUE_BoolQ_gen;; rte) echo SuperGLUE_RTE_gen;; siqa) echo siqa_gen;; piqa) echo piqa_gen;; openbookqa) echo obqa_main_gen;; arc_c) echo ARC_c_gen;; esac; }

mrs_router() { echo "$ROUTER_ROOT/mrs_only_single_all_layers"; }
plus_router() { echo "$ROUTER_ROOT/mrs_plus_${1}_single_all_layers"; }
new_router() { echo "$ROUTER_ROOT/new_only_${1}_from_mrs_single_all_layers"; }
require_cache() { [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || { echo "missing cache: $CACHE_ROOT" >&2; exit 1; }; }
require_router() { [[ -f "$1/router_heads.pt" && -f "$1/router_config.json" ]] || { echo "missing router: $1" >&2; exit 1; }; }

setup() {
    mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$ROUTER_ROOT" "$OC_ROOT" "$RECORD_ROOT"
    exec > >(tee -a "$ROOT_LOG") 2>&1
    log "out_root=$OUT_ROOT"
}

build_cache() {
    [[ ! -e "$CACHE_ROOT" ]] || { echo "cache exists: $CACHE_ROOT" >&2; exit 1; }
    log "[START] build 3-way single-all-layers cache"
    "$PY" -u build_cached_router_pair_dataset.py \
        --route_space single_all_layers \
        --data_root "$DATA_ROOT" \
        --feature_root "$CACHE_ROOT" \
        --task_names boolq,medmcqa,openbookqa,arc_c,piqa,race,rte,siqa,sst2 \
        --expert_names "$EXPERTS" \
        --base_model_path "$BASE_MODEL" \
        --router_bert_init "$BERT" \
        --batch_size "$CACHE_BATCH_SIZE" --max_train_samples "$MAX_TRAIN_SAMPLES" --max_val_samples "$MAX_VAL_SAMPLES" \
        --first_layer_idx 0 --middle_layer_idx 18 --router_dim 512 \
        --dtype float16 --score_mode official_eval_aligned_generation \
        --cache_prompt_template chat_template --chunk_size 2048 --seed 42 \
        --lora_medmcqa "$LORA_ROOT/sft_medmcqa" \
        --lora_race "$LORA_ROOT/sft_race" \
        --lora_sst2 "$LORA_ROOT/sft_sst2" \
        > "$LOG_ROOT/cache_${RUN_STAMP}.log" 2>&1
    log "[DONE] cache=$CACHE_ROOT"
}

train_one() {
    local label="$1" sample_tasks="$2" out_dir="$3" load_from="${4:-}"
    local -a load_args=()
    require_cache
    [[ ! -e "$out_dir" ]] || { echo "router exists: $out_dir" >&2; exit 1; }
    if [[ -n "$load_from" ]]; then require_router "$load_from"; load_args+=(--load_from "$load_from"); fi
    log "[START] train $label tasks=$sample_tasks"
    "$PY" -u train_single_all_layers_weighted_sum_router.py \
        --feature_roots "$CACHE_ROOT" --bert_init "$BERT" --out_dir "$out_dir" \
        --sample_task_names "$sample_tasks" --expert_names "$EXPERTS" \
        "${load_args[@]}" --batch_size "$TRAIN_BATCH_SIZE" --epochs "$EPOCHS" --lr 2e-4 \
        --router_dim 512 --target_temperature 0.25 --loss_normalization none \
        --target_empty_fallback uniform --early_stop_patience 2 \
        --expert_ce_weight "$EXPERT_CE_WEIGHT" --weighted_sum_mse_weight "$MSE_WEIGHT" \
        > "$LOG_ROOT/train_${label}_${RUN_STAMP}.log" 2>&1
    log "[DONE] router=$out_dir"
}

train_all() {
    local task
    train_one mrs_only "$EXPERTS" "$(mrs_router)"
    for task in "${TASK_LIST[@]}"; do train_one "mrs_plus_$task" "$EXPERTS,$task" "$(plus_router "$task")"; done
    for task in "${TASK_LIST[@]}"; do train_one "new_only_$task" "$task" "$(new_router "$task")" "$(mrs_router)"; done
}

eval_one() {
    local label="$1" router="$2" sharpness="$3"; shift 3
    local tag work_dir record log_file
    require_router "$router"
    tag="$(temp_tag "$sharpness")"
    work_dir="$OC_ROOT/${label}_sharp${tag}_${RUN_STAMP}"
    record="$RECORD_ROOT/${label}_sharp${tag}_${RUN_STAMP}.jsonl"
    log_file="$LOG_ROOT/eval_${label}_sharp${tag}_${RUN_STAMP}.log"
    log "[START] eval $label sharpness=$sharpness datasets=$*"
    T0531_MRS_ROUTER_CKPT="$router" T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" T0531_SHARE_FIRST_WEIGHTS_ALL_LAYERS=1 \
    T0531_ROUTER_RECORD_TAG="T0622_single_${label}_sharp${tag}" T0601_ROUTER_RECORD_PATH="$record" \
        "$PY" -u run.py --models "$MODEL_CONFIG" --datasets "$@" \
        --work-dir "$work_dir" --debug > "$log_file" 2>&1
    log "[DONE] work_dir=$work_dir"
}

eval_all() {
    local sharpness task dataset
    local -a mrs=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
    split_csv "$SHARPNESS_LIST"; local -a sharps=("${SPLIT_RESULT[@]}")
    for sharpness in "${sharps[@]}"; do
        eval_one mrs_only "$(mrs_router)" "$sharpness" SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
        for task in "${TASK_LIST[@]}"; do dataset="$(task_dataset "$task")"; eval_one "mrs_plus_$task" "$(plus_router "$task")" "$sharpness" "$dataset" "${mrs[@]}"; done
        for task in "${TASK_LIST[@]}"; do dataset="$(task_dataset "$task")"; eval_one "new_only_$task" "$(new_router "$task")" "$sharpness" "$dataset" "${mrs[@]}"; done
    done
}

main() {
    local mode="${1:-all}" task
    case "$mode" in all|cache|train|eval) ;; *) echo "usage: bash $0 [all|cache|train|eval]" >&2; exit 2;; esac
    split_csv "$TASKS"; TASK_LIST=("${SPLIT_RESULT[@]}")
    for task in "${TASK_LIST[@]}"; do validate_task "$task"; done
    setup
    log "training_loss=${EXPERT_CE_WEIGHT}*expert_ce + ${MSE_WEIGHT}*weighted_sum_mse cache_root=$CACHE_ROOT"
    [[ "$mode" == all || "$mode" == cache ]] && build_cache
    [[ "$mode" == all || "$mode" == train ]] && train_all
    [[ "$mode" == all || "$mode" == eval ]] && eval_all
    log "[DONE] output=$OUT_ROOT"
}
main "$@"
