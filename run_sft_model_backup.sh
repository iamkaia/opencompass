#!/bin/bash
set -euo pipefail

# ===== 1. 定義 models & datasets（照順序一一對應） =====
MODELS=(
  "hf_llama_recall_sst2"
  "hf_llama_recall_squad2"
  "hf_llama_recall_race"
  "hf_llama_recall_medmcqa"
  "hf_llama_recall_iwslt"
)

default_DATASETS=(
  "sst2_gen"
  "squad20_gen"
  "race_gen"
  "medmcqa_gen"
  "iwslt2017_gen"
)

sft_DATASETS=(
  "sst2_gen"
  "squad20_gen_sft_prompt"
  "race_gen_sft_prompt"
  "medmcqa_gen_sft_prompt"
  "iwslt2017_gen_sft_prompt"
)



'''
# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_default_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"


# ===== 2. 逐一啟動每組 model + dataset =====
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    ds="${default_DATASETS[$i]}"

    log_file="${LOG_DIR}/${model}_on_default_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=${model}  dataset=${ds} ==="
    echo "    → log: ${log_file}"

    nohup python run.py \
        --models "${model}" \
        --datasets "${ds}" \
        --debug \
        > "${log_file}" 2>&1
    wait
done

echo "✅ 所有 5 個 jobs 都已經在背景跑了，用 ps / nvidia-smi / tail 看進度。"

# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_sft_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

# ===== 2. 逐一啟動每組 model + dataset =====
for i in "${!MODELS[@]}"; do
    model="${MODELS[$i]}"
    ds="${sft_DATASETS[$i]}"

    log_file="${LOG_DIR}/${model}_on_sft_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=${model}  dataset=${ds} ==="
    echo "    → log: ${log_file}"

    nohup python run.py \
        --models "${model}" \
        --datasets "${ds}" \
        --debug \
        > "${log_file}" 2>&1
    wait
done

echo "✅ 所有 5 個 jobs 都已經在背景跑了，用 ps / nvidia-smi / tail 看進度。"
'''