#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${ROOT_DIR}/.venv-ministral3/bin/python"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/T0622_ministral3_only_official}"

cd "${ROOT_DIR}"

"${PYTHON}" -u run.py \
  --models T0622_ministral3_only_hf.py \
  --datasets \
    SuperGLUE_BoolQ_gen \
    medmcqa_gen_sft_prompt \
    obqa_main_gen \
    ARC_c_gen \
    piqa_gen \
    race_gen_sft_prompt \
    SuperGLUE_RTE_gen \
    siqa_gen \
    sst2_gen \
  --work-dir "${OUT_ROOT}" \
  --max-num-workers 1
