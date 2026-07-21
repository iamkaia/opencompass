#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

declare -A MODEL_CONFIGS=(
    [llama]="T0527_new_only_weighted_sum_topk.py"
    [qwen]="T0527_new_only_sst2words_weighted_sum_topk.py"
    [llama_base]="hf_llama2_7b_chat_64.py"
    [qwen_base]="hf_qwen3_4b_instruct_2507_64.py"
)
RUN_ID="$(date +"%Y%m%d_%H%M%S")"
RUN_ORDER=(llama qwen llama_base qwen_base)

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0528_run_mrs_arc_obqa_opencompass_weighted_sum.sh {all|llama|qwen|llama_base|qwen_base}

Runs MRS weighted-sum routers or their 64-token base-model baselines on:
ARC-c, ARC-e, openbookqa, and openbookqa_fact.

With "all", models run synchronously in this order:
llama, qwen, llama_base, qwen_base. Each run writes its own log.
EOF
}

run_eval() {
    local family="$1"
    local model_config="${MODEL_CONFIGS[$family]:-}"
    local log_file
    local -a datasets

    if [[ -z "$model_config" ]]; then
        usage
        exit 2
    fi

    case "$family" in
      llama)
        export T0527_NEW_ONLY_TASK="mrs"
        datasets=(ARC_c_gen ARC_e_gen obqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
        log_file="T0528_opencompass_llama_mrs_arc_obqa_weighted_sum_${RUN_ID}.log"
        ;;
      qwen)
        export T0527_NEW_ONLY_TASK="mrs"
        datasets=(obqa_gen ARC_e_gen ARC_c_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
        log_file="T0528_opencompass_qwen3_mrs_arc_obqa_sst2words_weighted_sum_${RUN_ID}.log"
        ;;
      llama_base)
        unset T0527_NEW_ONLY_TASK || true
        datasets=(ARC_c_gen ARC_e_gen obqa_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
        log_file="T0528_opencompass_llama2_7b_chat_base_arc_obqa_64token_${RUN_ID}.log"
        ;;
      qwen_base)
        unset T0527_NEW_ONLY_TASK || true
        datasets=(obqa_gen ARC_e_gen ARC_c_gen sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt)
        log_file="T0528_opencompass_qwen3_4b_instruct_2507_base_arc_obqa_64token_${RUN_ID}.log"
        ;;
    esac

    echo "[START] family=$family log=$log_file datasets=ARC-c ARC-e openbookqa openbookqa_fact sst2_gen race_gen_sft_prompt medmcqa_gen_sft_prompt"
    "$PY" -u run.py \
        --models "$model_config" \
        --datasets "${datasets[@]}" \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] family=$family log=$log_file"
}

[[ $# -eq 1 ]] || { usage; exit 2; }
case "$1" in
    all)
        for family in "${RUN_ORDER[@]}"; do
            run_eval "$family"
        done
        ;;
    llama|qwen|llama_base|qwen_base)
        run_eval "$1"
        ;;
    *)
        usage
        exit 2
        ;;
esac
