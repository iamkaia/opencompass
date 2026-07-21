#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

FAMILY="${1:-both}"
REQUESTED_TASK="${2:-all}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

TASKS=(mrs boolq rte siqa piqa)

usage() {
    echo "Usage: bash tools/T0527_run_new_only_opencompass_hard_routing.sh {llama|qwen|both} [all|mrs|boolq|rte|siqa|piqa]"
}

validate_request() {
    case "${FAMILY}" in
        llama|qwen|both) ;;
        *) usage; exit 2 ;;
    esac
    case "${REQUESTED_TASK}" in
        all|mrs|boolq|rte|siqa|piqa) ;;
        *) usage; exit 2 ;;
    esac
}

datasets_for_task() {
    case "$1" in
        mrs)
            DATASETS=(medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt)
            ;;
        boolq)
            DATASETS=(SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
            ;;
        rte)
            DATASETS=(SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen)
            ;;
        siqa)
            DATASETS=(siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen)
            ;;
        piqa)
            DATASETS=(piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen)
            ;;
    esac
}

router_ckpt_for() {
    local family="$1"
    local task="$2"
    if [[ "${family}" == "llama" ]]; then
        if [[ "${task}" == "mrs" ]]; then
            echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert"
        else
            echo "./router_T0527_llama_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert"
        fi
    else
        if [[ "${task}" == "mrs" ]]; then
            echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words"
        else
            echo "./router_T0527_qwen3_fp16_new_only_${task}_from_mrs_correct_conf_ce_t1_3expert_sst2words"
        fi
    fi
}

config_for() {
    if [[ "$1" == "llama" ]]; then
        echo "T0527_new_only_hard_routing.py"
    else
        echo "T0527_new_only_sst2words_hard_routing.py"
    fi
}

run_one() {
    local family="$1"
    local task="$2"
    local router_ckpt
    local model_config
    local log_file
    router_ckpt="$(router_ckpt_for "${family}" "${task}")"
    model_config="$(config_for "${family}")"

    if [[ ! -f "${router_ckpt}/router_heads.pt" || ! -f "${router_ckpt}/router_config.json" ]]; then
        echo "Missing completed router checkpoint: ${router_ckpt}" >&2
        exit 1
    fi

    datasets_for_task "${task}"
    if [[ "${family}" == "llama" ]]; then
        log_file="T0527_opencompass_llama_new_only_${task}_hard_routing_${RUN_ID}.log"
    else
        log_file="T0527_opencompass_qwen3_new_only_${task}_sst2words_hard_routing_${RUN_ID}.log"
    fi

    echo "[$(date '+%F %T')] start family=${family} task=${task} config=${model_config} log=${log_file}"
    T0527_NEW_ONLY_TASK="${task}" "${PY}" -u run.py \
        --models "${model_config}" \
        --datasets "${DATASETS[@]}" \
        --debug > "${log_file}" 2>&1
    echo "[$(date '+%F %T')] done family=${family} task=${task} log=${log_file}"
}

run_family() {
    local family="$1"
    local task
    if [[ "${REQUESTED_TASK}" == "all" ]]; then
        for task in "${TASKS[@]}"; do
            run_one "${family}" "${task}"
        done
    else
        run_one "${family}" "${REQUESTED_TASK}"
    fi
}

validate_request
case "${FAMILY}" in
    llama) run_family llama ;;
    qwen) run_family qwen ;;
    both)
        run_family llama
        run_family qwen
        ;;
esac
