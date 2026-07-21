#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

EXPERTS="medmcqa,race,sst2"
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
BERT_TASKCLS="./task_classifier_ckpt"
BERT_RAW="prajjwal1/bert-tiny"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0531_run_mrs_trainbert_ablation_hard_routing.sh [llama|qwen|both] [taskcls|rawbert|direct|all]

Variants:
  taskcls  Train only the MRS router from ./task_classifier_ckpt with --train_bert, then run OpenCompass eval.
  rawbert  Train only the MRS router from prajjwal1/bert-tiny with --train_bert, then run OpenCompass eval.
  direct   Do not train a router. Run OpenCompass by averaging the 3 MRS experts with routing_mode=uniform.
  all      Run taskcls, rawbert, and direct.

No replay and no boolq/rte/siqa/piqa router training. Eval datasets are:
  medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen
  SuperGLUE_BoolQ_gen siqa_gen piqa_gen SuperGLUE_RTE_gen ARC_c_gen obqa_main_gen
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

mrs_existing_router_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

bert_for_variant() {
    case "$1" in
        taskcls|direct) echo "$BERT_TASKCLS" ;;
        rawbert) echo "$BERT_RAW" ;;
    esac
}

mrs_output_for() {
    local family="$1"
    local variant="$2"
    printf './router_T0531_%s_%s_mrs_trainbert_correct_conf_ce_t1_%s' \
        "$(family_label "$family")" "$variant" "$(family_suffix "$family")"
}

model_config_for() {
    case "$1" in
        llama) echo "T0531_mrs_ablation_hard_routing.py" ;;
        qwen) echo "T0531_mrs_ablation_sst2words_hard_routing.py" ;;
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

training_args() {
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
        --train_bert \
        --save_route_records \
        --eval_train_each_epoch
}

train_mrs() {
    local family="$1"
    local variant="$2"
    local cache="$3"
    local bert_init="$4"
    local output_dir log_file
    local -a common_args

    output_dir="$(mrs_output_for "$family" "$variant")"
    log_file="T0531_$(family_label "$family")_${variant}_mrs_trainbert.log"

    if [[ -e "$output_dir" && ! -f "$output_dir/router_heads.pt" ]]; then
        echo "router output exists but is incomplete: $output_dir" >&2
        exit 1
    fi
    if [[ -f "$output_dir/router_heads.pt" ]]; then
        echo "[SKIP] train_mrs family=$family variant=$variant output=$output_dir" >&2
        echo "$output_dir"
        return
    fi

    mapfile -t common_args < <(training_args)
    echo "[START] train_mrs family=$family variant=$variant bert_init=$bert_init log=$log_file output=$output_dir" >&2
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$cache" \
        --bert_init "$bert_init" \
        --out_dir "$output_dir" \
        --sample_task_names "$EXPERTS" \
        --expert_names "$EXPERTS" \
        "${common_args[@]}" \
        > "$log_file" 2>&1
    echo "[DONE] train_mrs family=$family variant=$variant output=$output_dir" >&2
    echo "$output_dir"
}

eval_mrs() {
    local family="$1"
    local variant="$2"
    local router_ckpt="$3"
    local bert_init="$4"
    local routing_mode="$5"
    local model_config log_file

    model_config="$(model_config_for "$family")"
    log_file="T0531_opencompass_$(family_label "$family")_${variant}_${routing_mode}_mrs_${RUN_ID}.log"

    echo "[START] eval_mrs family=$family variant=$variant mode=$routing_mode ckpt=$router_ckpt log=$log_file"
    T0531_MRS_ROUTER_CKPT="$router_ckpt" \
    T0531_ROUTER_BERT_INIT="$bert_init" \
    T0531_ROUTING_MODE="$routing_mode" \
    T0531_ROUTER_RECORD_TAG="${variant}_${routing_mode}" \
        "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${EVAL_DATASETS[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] eval_mrs family=$family variant=$variant mode=$routing_mode ckpt=$router_ckpt log=$log_file"
}

run_family_variant() {
    local family="$1"
    local variant="$2"
    local cache bert_init router_ckpt

    require_family "$family"
    cache="$(mrs_cache_for "$family")"
    require_cache "$cache"

    case "$variant" in
        taskcls|rawbert)
            bert_init="$(bert_for_variant "$variant")"
            router_ckpt="$(train_mrs "$family" "$variant" "$cache" "$bert_init")"
            eval_mrs "$family" "$variant" "$router_ckpt" "$bert_init" hard
            ;;
        direct)
            bert_init="$(bert_for_variant direct)"
            router_ckpt="$(mrs_existing_router_for "$family")"
            require_router "$router_ckpt"
            eval_mrs "$family" direct "$router_ckpt" "$bert_init" uniform
            ;;
        *)
            usage
            exit 2
            ;;
    esac
}

main() {
    local family="${1:-both}"
    local variant="${2:-all}"
    local -a families variants
    local f v

    case "$family" in
        llama|qwen) families=("$family") ;;
        both) families=(llama qwen) ;;
        *) usage; exit 2 ;;
    esac

    case "$variant" in
        taskcls|rawbert|direct) variants=("$variant") ;;
        all) variants=(taskcls rawbert direct) ;;
        *) usage; exit 2 ;;
    esac

    for f in "${families[@]}"; do
        for v in "${variants[@]}"; do
            run_family_variant "$f" "$v"
        done
    done
}

main "${1:-both}" "${2:-all}"
