#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

DATA_ROOT="./0527_router_train_dataset"
BERT="./task_classifier_ckpt"
EXPERTS="medmcqa,race,sst2"

usage() {
    echo "usage: CUDA_VISIBLE_DEVICES=0 bash tools/T0527_build_cached_mrs_4other_3expert.sh {llama|qwen} {mrs|4other}" >&2
}

[[ $# -eq 2 ]] || { usage; exit 2; }
FAMILY="$1"
GROUP="$2"

require_dataset() {
    [[ -f "$DATA_ROOT/alignment_meta.json" ]] || {
        echo "missing dataset metadata: $DATA_ROOT/alignment_meta.json" >&2
        exit 1
    }
    "$PY" -u tools/verify_router_train_dataset_split_provenance.py \
        --data_root "$DATA_ROOT"
}

require_new_cache_root() {
    local cache_root="$1"
    if [[ -e "$cache_root" ]]; then
        echo "cache output already exists: $cache_root" >&2
        echo "use a new output name or remove the incomplete/obsolete cache deliberately" >&2
        exit 1
    fi
}

case "$GROUP" in
    mrs)
        TASKS="medmcqa,race,sst2"
        ;;
    4other)
        TASKS="boolq,rte,siqa,piqa"
        ;;
    *)
        usage
        exit 2
        ;;
esac

case "$FAMILY" in
    llama)
        MODEL="meta-llama/Llama-2-7b-chat-hf"
        CACHE_ROOT="./0527_llama_cache_${GROUP}_3expert_official_eval_aligned"
        LOG_FILE="T0527_llama_build_cache_${GROUP}_3expert_official_eval_aligned.log"
        MIDDLE_LAYER_IDX=15
        LORA_MEDMCQA="./saves/llama2-7b-chat-hf/lora/sft_medmcqa"
        LORA_RACE="./saves/llama2-7b-chat-hf/lora/sft_race"
        LORA_SST2="./saves/llama2-7b-chat-hf/lora/sft_sst2"
        ;;
    qwen)
        MODEL="Qwen/Qwen3-4B-Instruct-2507"
        CACHE_ROOT="./0527_qwen3_fp16_cache_${GROUP}_3expert_official_eval_aligned_sst2words"
        LOG_FILE="T0527_qwen3_fp16_build_cache_${GROUP}_3expert_official_eval_aligned_sst2words.log"
        MIDDLE_LAYER_IDX=18
        LORA_MEDMCQA="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa"
        LORA_RACE="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race"
        LORA_SST2="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2"
        ;;
    *)
        usage
        exit 2
        ;;
esac

require_dataset
require_new_cache_root "$CACHE_ROOT"

nohup "$PY" -u build_cached_router_pair_dataset.py \
    --data_root "$DATA_ROOT" \
    --feature_root "$CACHE_ROOT" \
    --task_names "$TASKS" \
    --expert_names "$EXPERTS" \
    --base_model_path "$MODEL" \
    --router_bert_init "$BERT" \
    --batch_size 8 \
    --max_train_samples 200 \
    --max_val_samples 50 \
    --first_layer_idx 0 \
    --middle_layer_idx "$MIDDLE_LAYER_IDX" \
    --router_dim 512 \
    --dtype float16 \
    --score_mode official_eval_aligned_generation \
    --chunk_size 2048 \
    --seed 42 \
    --lora_medmcqa "$LORA_MEDMCQA" \
    --lora_race "$LORA_RACE" \
    --lora_sst2 "$LORA_SST2" \
    > "$LOG_FILE" 2>&1 &

echo "started $FAMILY/$GROUP cache: pid=$! log=$LOG_FILE output=$CACHE_ROOT tasks=$TASKS experts=$EXPERTS"
