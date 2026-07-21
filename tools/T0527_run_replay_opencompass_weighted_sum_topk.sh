#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
RUN_ID="$(date +"%Y%m%d_%H%M%S")"

TASKS=(boolq rte siqa piqa)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0527_run_replay_opencompass_weighted_sum_topk.sh {llama|qwen|both} [all|boolq|rte|siqa|piqa]

This launcher runs replay routers synchronously. With "all" or omitted task,
one router finishes before the next router starts, and each router has its own log.
EOF
}

require_family() {
    case "$1" in
        llama|qwen|both) ;;
        *) usage; exit 2 ;;
    esac
}

require_task() {
    case "$1" in
        all|boolq|rte|siqa|piqa) ;;
        *) usage; exit 2 ;;
    esac
}

router_dir_for() {
    local family="$1"
    local task="$2"
    case "$family" in
        llama) echo "./router_T0527_llama_replay_mrs_${task}_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_replay_mrs_${task}_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

model_config_for() {
    case "$1" in
        llama) echo "T0527_replay_weighted_sum_topk.py" ;;
        qwen) echo "T0527_replay_sst2words_weighted_sum_topk.py" ;;
    esac
}

datasets_for() {
    case "$1" in
        boolq) echo "SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt" ;;
        rte) echo "SuperGLUE_RTE_gen siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen" ;;
        siqa) echo "siqa_gen piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen" ;;
        piqa) echo "piqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt SuperGLUE_BoolQ_gen SuperGLUE_RTE_gen siqa_gen" ;;
    esac
}

run_one() {
    local family="$1"
    local task="$2"
    local model_config router_dir log_file
    local -a datasets
    model_config="$(model_config_for "$family")"
    router_dir="$(router_dir_for "$family" "$task")"
    if [[ ! -f "$router_dir/router_heads.pt" || ! -f "$router_dir/router_config.json" ]]; then
        echo "missing completed router checkpoint: $router_dir" >&2
        exit 1
    fi
    read -r -a datasets <<< "$(datasets_for "$task")"
    if [[ "$family" == "llama" ]]; then
        log_file="T0527_opencompass_llama_replay_mrs_${task}_weighted_sum_top3_${RUN_ID}.log"
    else
        log_file="T0527_opencompass_qwen3_replay_mrs_${task}_sst2words_weighted_sum_top3_${RUN_ID}.log"
    fi
    export T0527_REPLAY_TASK="$task"
    echo "[START] family=$family replay_router=$task log=$log_file"
    "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] family=$family replay_router=$task log=$log_file"
}

run_family() {
    local family="$1"
    local requested_task="$2"
    local task
    if [[ "$requested_task" == "all" ]]; then
        for task in "${TASKS[@]}"; do
            run_one "$family" "$task"
        done
    else
        run_one "$family" "$requested_task"
    fi
}

[[ $# -ge 1 && $# -le 2 ]] || { usage; exit 2; }
FAMILY="$1"
TASK="${2:-all}"
require_family "$FAMILY"
require_task "$TASK"

case "$FAMILY" in
    llama|qwen)
        run_family "$FAMILY" "$TASK"
        ;;
    both)
        run_family llama "$TASK"
        run_family qwen "$TASK"
        ;;
esac
