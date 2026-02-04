#!/bin/bash
set -euo pipefail

DATASETS=(
  "sst2_gen"
  "squad20_gen"
  "iwslt2017_gen"
  "race_gen"
  "medmcqa_gen"
)

# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_sft_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

# ===== 2. 逐一啟動每組 model + dataset =====
for i in "${!DATASETS[@]}"; do
    ds="${DATASETS[$i]}"

    log_file="${LOG_DIR}/base_model_on_${ds}_in_default_prompt_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: default prompt dataset=${ds} ==="
    echo "    → log: ${log_file}"

    nohup python run.py \
        --models hf_llama2_7b_chat \
        --datasets "${ds}" \
        --debug \
        > "${log_file}" 2>&1 
    wait
done

echo "✅ 所有 5 個 jobs 都已經在背景跑了，用 ps / nvidia-smi / tail 看進度。"