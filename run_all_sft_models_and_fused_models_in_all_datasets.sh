#!/bin/bash
set -euo pipefail

# ===== 1. 定義 models & datasets（照順序一一對應） =====
SFT_MODELS=(
  #"hf_llama_recall_sst2"
  #"hf_llama_recall_squad20"
  "hf_llama_recall_iwslt"
  #"hf_llama_recall_race"
  #"hf_llama_recall_medmcqa"
)

FUSED_MODELS=(
  #"hf_fused_sst2"
  #"hf_fused_squad20"
  "hf_fused_iwslt"
  #"hf_fused_race"
  #"hf_fused_medmcqa"
)

default_DATASETS=(
  "sst2_gen"
)

sft_DATASETS=(
  "sst2_gen"
)

# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_sft_and_fused_models_dataset_test_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

for model in "${SFT_MODELS[@]}"; do
  for ds in "${sft_DATASETS[@]}"; do
	  log_file="${LOG_DIR}/SFT_${model}_on_sft(paper)_prompt_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=SFT_${model}  dataset=${ds} ==="
    echo "    → log: ${log_file}"

    nohup python run.py \
      --models "${model}" \
      --datasets "${ds}" \
      --debug \
      > "${log_file}" 2>&1 &

    pid=$!
    echo "    ↳ PID = $pid，等它跑完再跑下一個"

    # 等這個背景 job 結束，再進到下一個 pair
    wait "$pid"
  done
done

for model in "${FUSED_MODELS[@]}"; do
  for ds in "${sft_DATASETS[@]}"; do
          log_file="${LOG_DIR}/FUSED_${model}_on_sft(paper)_prompt_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=FUSED_${model}  dataset=${ds} ==="
    echo "    → log: ${log_file}"

    nohup python run.py \
      --models "${model}" \
      --datasets "${ds}" \
      --debug \
      > "${log_file}" 2>&1 &

    pid=$!
    echo "    ↳ PID = $pid，等它跑完再跑下一個"

    # 等這個背景 job 結束，再進到下一個 pair
    wait "$pid"
  done
done
