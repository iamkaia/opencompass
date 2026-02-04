#!/bin/bash
set -euo pipefail

# ===== 1. 定義 models & datasets（照順序一一對應） =====
MODELS=(
  "hf_llama_recall_medmcqa"
)

default_DATASETS=(
  "sst2_gen"
  "squad20_gen"
  "race_gen"
  "medmcqa_gen"
  "iwslt2017_gen"
)

sft_DATASETS=(
  "iwslt2017_gen_sft_prompt"
)

# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_sft_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  for ds in "${sft_DATASETS[@]}"; do
    log_file="${LOG_DIR}/${model}_on_sft_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=${model}  dataset=${ds} ==="
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



##### run hf -llama
#!/bin/bash
set -euo pipefail

# ===== 1. 定義 models & datasets（照順序一一對應） =====
MODELS=(
  "hf_fused_sst2"
  "hf_fused_squad20"
  "hf_fused_iwslt"
  "hf_fused_race"
  "hf_fused_medmcqa"
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
  "iwslt2017_gen_sft_prompt"
  "race_gen_sft_prompt"
  "medmcqa_gen_sft_prompt"
)

# 可選：log 統一放一個資料夾裡
LOG_DIR="logs_fused_models_$(date +'%Y%m%d_%H%M%S')"
mkdir -p "$LOG_DIR"

for model in "${MODELS[@]}"; do
  for ds in "${sft_DATASETS[@]}"; do
    log_file="${LOG_DIR}/fused_${model}_on_sft_${ds}_$(date +'%Y%m%d_%H%M%S').log"

    echo "=== Running: model=fused_${model}  dataset=${ds} ==="
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
~                           
