#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-/home/u9472191/.conda/envs/opencompass/bin/python}"
OUT_ROOT="${OUT_ROOT:-${ROOT_DIR}/T0711_gemma4_base_official}"
MODE="${MODE:-all}"
MAX_NUM_WORKERS="${MAX_NUM_WORKERS:-1}"
REUSE="${REUSE:-}"
DEBUG="${DEBUG:-1}"
FORCE_MAX_OUT_LEN="${FORCE_MAX_OUT_LEN:-1}"
DATASETS_CSV="${DATASETS:-SuperGLUE_BoolQ_gen,medmcqa_gen_sft_prompt,obqa_main_gen,ARC_c_gen,piqa_gen,race_gen_sft_prompt,SuperGLUE_RTE_gen,siqa_gen,sst2_gen}"

cd "${ROOT_DIR}"

if [[ "${FORCE_MAX_OUT_LEN}" == "1" ]]; then
  export T0704_MAX_OUT_LEN="${T0704_MAX_OUT_LEN:-128}"
fi

IFS=',' read -r -a DATASET_ARGS <<< "${DATASETS_CSV}"

args=(
  -u run.py
  --models T0711_gemma4_base_hf.py
  --datasets
    "${DATASET_ARGS[@]}"
  --work-dir "${OUT_ROOT}"
  --max-num-workers "${MAX_NUM_WORKERS}"
  --mode "${MODE}"
)

if [[ -n "${REUSE}" ]]; then
  args+=(-r "${REUSE}")
fi

if [[ "${DEBUG}" == "1" ]]; then
  args+=(--debug)
fi

printf '[INFO] python=%s\n' "${PYTHON}"
printf '[INFO] out_root=%s mode=%s reuse=%s max_num_workers=%s\n' \
  "${OUT_ROOT}" "${MODE}" "${REUSE:-<new>}" "${MAX_NUM_WORKERS}"
printf '[INFO] datasets=%s\n' "${DATASETS_CSV}"
printf '[INFO] FORCE_MAX_OUT_LEN=%s T0704_MAX_OUT_LEN=%s\n' \
  "${FORCE_MAX_OUT_LEN}" "${T0704_MAX_OUT_LEN:-<unset>}"

"${PYTHON}" "${args[@]}"
