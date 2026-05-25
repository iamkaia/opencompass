#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/u9472191/.conda/envs/opencompass/bin/python}"
DATA_ROOT="${DATA_ROOT:-router_train_datasets_opencompass_eval_0520}"
PROMPT_TIMEOUT="${PROMPT_TIMEOUT:-180}"

items=(
  "boolq|opencompass/configs/datasets/SuperGLUE_BoolQ/SuperGLUE_BoolQ_gen.py"
  "iwslt2017|opencompass/configs/datasets/iwslt2017/iwslt2017_gen_sft_prompt.py"
  "medmcqa|opencompass/configs/datasets/medmcqa/medmcqa_gen_sft_prompt.py"
  "piqa|opencompass/configs/datasets/piqa/piqa_gen.py"
  "race|opencompass/configs/datasets/race/race_gen_sft_prompt.py"
  "rte|opencompass/configs/datasets/SuperGLUE_RTE/SuperGLUE_RTE_gen.py"
  "siqa|opencompass/configs/datasets/siqa/siqa_gen.py"
  "squad2|opencompass/configs/datasets/squad20/squad20_gen_sft_prompt.py"
  "sst2|opencompass/configs/datasets/glue/sst2_gen.py"
)

for item in "${items[@]}"; do
  task="${item%%|*}"
  cfg="${item#*|}"

  echo
  echo "################################################################################"
  echo "[OpenCompass prompt_viewer] ${task}"
  echo "config=${cfg}"
  echo "################################################################################"
  timeout "${PROMPT_TIMEOUT}" "${PYTHON_BIN}" tools/prompt_viewer.py "${cfg}" --all --count 1 \
    || echo "[prompt_viewer failed or timed out] ${task}"

  echo
  echo "################################################################################"
  echo "[Router/BERT dataset] ${task}"
  echo "path=${DATA_ROOT}/${task}/train.jsonl"
  echo "################################################################################"
  "${PYTHON_BIN}" -c 'import json,sys; from pathlib import Path; task=sys.argv[1]; data_root=Path(sys.argv[2]); path=data_root/task/"train.jsonl"; row=json.loads(next(x for x in path.open(encoding="utf-8") if x.strip())); print("target=" + str(row.get("target"))); print("source_index=" + str(row.get("_source_index"))); print(); print(row["text"][:2000])' "${task}" "${DATA_ROOT}"
done
