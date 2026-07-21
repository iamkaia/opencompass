#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
MODEL_KEY="${1:-}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES

COMMON_ARGS=(
    --data_root router_train_datasets_config_zeroshot_0521
    --samples_per_task 10
    --expert_names medmcqa,race,sst2
    --router_bert_init ./task_classifier_ckpt
    --dtype float16
    --r 8
    --alpha 32
    --router_dim 512
    --max_llm_len 768
    --batch_size 4
)

case "$MODEL_KEY" in
    qwen3)
        OUT_DIR="outputs/router_pair_max_tokens_compare_0521_qwen3"
        LOG_FILE="T0527_compare_router_pair_generation_lengths_0521_qwen3.log"
        MODEL_ARGS=(
            --base_model_path Qwen/Qwen3-4B-Instruct-2507
            --first_layer_idx 0
            --middle_layer_idx 18
            --lora_medmcqa ./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa
            --lora_race ./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race
            --lora_sst2 ./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2
        )
        ;;
    llama)
        OUT_DIR="outputs/router_pair_max_tokens_compare_0521_llama"
        LOG_FILE="T0527_compare_router_pair_generation_lengths_0521_llama.log"
        MODEL_ARGS=(
            --base_model_path meta-llama/Llama-2-7b-chat-hf
            --first_layer_idx 0
            --middle_layer_idx 15
            --lora_medmcqa ./saves/llama2-7b-chat-hf/lora/sft_medmcqa
            --lora_race ./saves/llama2-7b-chat-hf/lora/sft_race
            --lora_sst2 ./saves/llama2-7b-chat-hf/lora/sft_sst2
        )
        ;;
    *)
        echo "usage: CUDA_VISIBLE_DEVICES=0 bash tools/run_compare_router_pair_generation_lengths_0521.sh {qwen3|llama}" >&2
        exit 2
        ;;
esac

nohup "$PY" -u tools/compare_router_pair_generation_lengths.py \
    "${COMMON_ARGS[@]}" \
    "${MODEL_ARGS[@]}" \
    --out_dir "$OUT_DIR" \
    > "$LOG_FILE" 2>&1 &

echo "started: model=$MODEL_KEY pid=$! log=$LOG_FILE out_dir=$OUT_DIR"
