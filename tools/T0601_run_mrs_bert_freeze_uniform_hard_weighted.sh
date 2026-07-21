#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
EXP_ROOT="${EXP_ROOT:-./runs/T0601_mrs_router_ablation_${RUN_STAMP}}"
LOG_DIR="$EXP_ROOT/logs"
ROUTER_DIR="$EXP_ROOT/routers"
OC_DIR="$EXP_ROOT/opencompass"
RECORD_DIR="$EXP_ROOT/router_records"

EXPERTS="medmcqa,race,sst2"
BERT_TASKCLS="./task_classifier_ckpt"
BERT_RAW="prajjwal1/bert-tiny"
EVAL_DATASETS=(
    medmcqa_gen_sft_prompt
    race_gen_sft_prompt
    sst2_gen
    SuperGLUE_BoolQ_gen
    siqa_gen
    piqa_gen
    SuperGLUE_RTE_gen
    ARC_c_gen
    obqa_main_gen
)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0601_run_mrs_bert_freeze_uniform_hard_weighted.sh [llama|qwen|both]

Runs, for each selected family:
  1. uniform_taskcls: OpenCompass uniform eval with router_bert_init=./task_classifier_ckpt.
  2. uniform_rawbert: OpenCompass uniform eval with router_bert_init=prajjwal1/bert-tiny.
  3. Train MRS routers with bert_init in {./task_classifier_ckpt, prajjwal1/bert-tiny}
     and bert mode in {freeze_bert, train_bert}.
  4. For each trained router, run OpenCompass hard and weighted_sum eval.

All outputs are written under EXP_ROOT, default:
  ./runs/T0601_mrs_router_ablation_<current timestamp>
EOF
}

require_family() {
    case "$1" in
        llama|qwen) ;;
        *) usage; exit 2 ;;
    esac
}

family_label() {
    case "$1" in
        llama) echo "llama" ;;
        qwen) echo "qwen3_fp16" ;;
    esac
}

family_suffix() {
    case "$1" in
        llama) echo "3expert" ;;
        qwen) echo "3expert_sst2words" ;;
    esac
}

mrs_cache_for() {
    case "$1" in
        llama) echo "./0527_llama_cache_mrs_3expert_official_eval_aligned" ;;
        qwen) echo "./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words" ;;
    esac
}

metadata_router_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

model_config_for() {
    case "$1" in
        llama) echo "T0531_mrs_ablation_hard_routing.py" ;;
        qwen) echo "T0531_mrs_ablation_sst2words_hard_routing.py" ;;
    esac
}

bert_init_for() {
    case "$1" in
        taskcls|uniform_taskcls) echo "$BERT_TASKCLS" ;;
        rawbert|uniform_rawbert) echo "$BERT_RAW" ;;
        *) echo "unknown bert variant: $1" >&2; exit 2 ;;
    esac
}

require_cache() {
    local cache_root="$1"
    if [[ ! -f "$cache_root/train/manifest.json" || ! -f "$cache_root/validation/manifest.json" ]]; then
        echo "missing completed MRS cache: $cache_root" >&2
        echo "build it first with: CUDA_VISIBLE_DEVICES=0 bash tools/T0527_build_cached_mrs_4other_3expert.sh {llama|qwen} mrs" >&2
        exit 1
    fi
}

require_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_heads.pt" || ! -f "$router_root/router_config.json" ]]; then
        echo "missing router checkpoint: $router_root" >&2
        exit 1
    fi
}

prepare_dirs() {
    if [[ -e "$EXP_ROOT" ]]; then
        echo "EXP_ROOT already exists, refusing to reuse it: $EXP_ROOT" >&2
        echo "Set a fresh RUN_STAMP or EXP_ROOT." >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_DIR" "$RECORD_DIR"
}

common_training_args() {
    printf '%s\n' \
        --router_dim 512 \
        --batch_size 32 \
        --epochs 10 \
        --lr 2e-4 \
        --joint_loss correct_conf_ce \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature 1.0 \
        --pair_loss_normalization sample_minmax \
        --best_metric route_correct_acc \
        --early_stop_patience 2 \
        --save_route_records \
        --eval_train_each_epoch
}

train_mrs() {
    local family="$1"
    local bert_variant="$2"
    local bert_mode="$3"
    local cache="$4"
    local bert_init output_dir log_file variant_name
    local -a args bert_mode_arg

    bert_init="$(bert_init_for "$bert_variant")"
    variant_name="${bert_variant}_${bert_mode}"
    output_dir="$ROUTER_DIR/router_T0601_$(family_label "$family")_${variant_name}_mrs_correct_conf_ce_t1_$(family_suffix "$family")"
    log_file="$LOG_DIR/train_$(family_label "$family")_${variant_name}.log"

    if [[ -e "$output_dir" ]]; then
        echo "router output already exists: $output_dir" >&2
        exit 1
    fi
    case "$bert_mode" in
        freezebert) bert_mode_arg=(--freeze_bert) ;;
        trainbert) bert_mode_arg=(--train_bert) ;;
        *) echo "unknown bert mode: $bert_mode" >&2; exit 2 ;;
    esac

    mapfile -t args < <(common_training_args)
    echo "[START] train family=$family variant=$variant_name bert_init=$bert_init out=$output_dir log=$log_file" >&2
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$bert_init" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${bert_mode_arg[@]}" \
        "${args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] train family=$family variant=$variant_name out=$output_dir" >&2
    echo "$output_dir"
}

eval_router() {
    local family="$1"
    local variant="$2"
    local router_ckpt="$3"
    local bert_init="$4"
    local routing_mode="$5"
    local model_config log_file work_dir record_path

    require_router "$router_ckpt"
    model_config="$(model_config_for "$family")"
    log_file="$LOG_DIR/opencompass_$(family_label "$family")_${variant}_${routing_mode}.log"
    work_dir="$OC_DIR/$(family_label "$family")_${variant}_${routing_mode}"
    record_path="$RECORD_DIR/$(family_label "$family")_${variant}_${routing_mode}.jsonl"

    if [[ -e "$work_dir" || -e "$record_path" || -e "$log_file" ]]; then
        echo "eval output already exists for family=$family variant=$variant mode=$routing_mode" >&2
        exit 1
    fi

    echo "[START] eval family=$family variant=$variant mode=$routing_mode ckpt=$router_ckpt log=$log_file" >&2
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$bert_init" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTER_RECORD_TAG="$variant" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${EVAL_DATASETS[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] eval family=$family variant=$variant mode=$routing_mode log=$log_file" >&2
}

run_family() {
    local family="$1"
    local cache metadata_router router_ckpt bert_variant bert_mode bert_init variant

    require_family "$family"
    cache="$(mrs_cache_for "$family")"
    metadata_router="$(metadata_router_for "$family")"
    require_cache "$cache"
    require_router "$metadata_router"

    for variant in uniform_taskcls uniform_rawbert; do
        bert_init="$(bert_init_for "$variant")"
        eval_router "$family" "$variant" "$metadata_router" "$bert_init" uniform
    done

    for bert_variant in taskcls rawbert; do
        for bert_mode in freezebert trainbert; do
            variant="${bert_variant}_${bert_mode}"
            bert_init="$(bert_init_for "$bert_variant")"
            router_ckpt="$(train_mrs "$family" "$bert_variant" "$bert_mode" "$cache")"
            eval_router "$family" "$variant" "$router_ckpt" "$bert_init" hard
            eval_router "$family" "$variant" "$router_ckpt" "$bert_init" weighted_sum
        done
    done
}

main() {
    local family="${1:-both}"
    prepare_dirs
    echo "[INFO] EXP_ROOT=$EXP_ROOT"
    case "$family" in
        llama|qwen) run_family "$family" ;;
        both)
            run_family llama
            run_family qwen
            ;;
        *)
            usage
            exit 2
            ;;
    esac
    echo "[DONE] all outputs are under $EXP_ROOT"
}

main "${1:-both}"
