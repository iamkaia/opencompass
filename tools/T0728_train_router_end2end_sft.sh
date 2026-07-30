#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="${1:-all}"
DATA_ROOT="${DATA_ROOT:-0602_router_train_dataset}"
PYTHON="${PYTHON:-/home/u9472191/.conda/envs/opencompass/bin/python}"
MRS_TASKS="${MRS_TASKS:-medmcqa,race,sst2}"
PLUS_TASKS="${PLUS_TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-Qwen/Qwen3.5-4B}"
ROUTER_BERT_INIT="${ROUTER_BERT_INIT:-./task_classifier_ckpt}"
OUT_PREFIX="${OUT_PREFIX:-T0728_qwen35_router_end2end_sft_half16}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./${OUT_PREFIX}_${RUN_STAMP}}"
ROUTER_DIR="${ROUTER_DIR:-${OUT_ROOT}/routers}"

LORA_ROOT="${LORA_ROOT:-./saves/Qwen/Qwen3.5-4B/lora}"
LORA_PATHS="${LORA_PATHS:-medmcqa=${LORA_ROOT}/sft_medmcqa,race=${LORA_ROOT}/sft_race,sst2=${LORA_ROOT}/sft_sst2}"

BATCH_SIZE="${BATCH_SIZE:-2}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-6}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"
EARLY_STOP_MIN_DELTA="${EARLY_STOP_MIN_DELTA:-0.0}"
MAX_TRAIN_SAMPLES="${MAX_TRAIN_SAMPLES:-800}"
MAX_VAL_SAMPLES="${MAX_VAL_SAMPLES:-200}"
MAX_LLM_LEN="${MAX_LLM_LEN:-768}"
MAX_BERT_LEN="${MAX_BERT_LEN:-512}"
FIRST_LAYER_IDX="${FIRST_LAYER_IDX:-0}"
MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX:-16}"
ROUTER_DIM="${ROUTER_DIM:-512}"
ROUTER_POOLING="${ROUTER_POOLING:-mean}"
ROUTING_SHARPNESS="${ROUTING_SHARPNESS:-1.0}"
DTYPE="${DTYPE:-bfloat16}"
LORA_R="${LORA_R:-8}"
LORA_ALPHA="${LORA_ALPHA:-32}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-opencompass}"
WANDB_NAME="${WANDB_NAME:-qwen35_half16_end2end_sft_router}"
WANDB_GROUP="${WANDB_GROUP:-router_end2end_sft}"
WANDB_TAGS="${WANDB_TAGS:-qwen35,end2end_sft,half16}"

MODEL_CONFIG="${MODEL_CONFIG:-T0728_qwen35_router_end2end_sft_two_layer_bf16_env.py}"
BASE_MODEL_CONFIG="${BASE_MODEL_CONFIG:-T0728_qwen35_base_model_bf16_env.py}"
OC_DIR="${OC_DIR:-${OUT_ROOT}/opencompass_official}"
RECORD_DIR="${RECORD_DIR:-${OUT_ROOT}/router_records}"
LOG_DIR="${LOG_DIR:-${OUT_ROOT}/logs}"
OC_SHARPNESS_LIST="${OC_SHARPNESS_LIST:-1.0,3.0}"
RESUME="${RESUME:-1}"
EVAL_BASELINES="${EVAL_BASELINES:-1}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/train.log}"
OC_ROUTER_BATCH_SIZE="${OC_ROUTER_BATCH_SIZE:-32}"
OC_BASE_BATCH_SIZE="${OC_BASE_BATCH_SIZE:-64}"

MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
OFFICIAL_ALL_DATASETS=(
  SuperGLUE_BoolQ_gen
  medmcqa_gen_sft_prompt
  obqa_main_gen
  ARC_c_gen
  piqa_gen
  race_gen_sft_prompt
  SuperGLUE_RTE_gen
  siqa_gen
  sst2_gen
)

mkdir -p "${ROUTER_DIR}" "${OC_DIR}" "${RECORD_DIR}" "${LOG_DIR}"

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }
split_csv() { local IFS=,; read -r -a SPLIT_RESULT <<< "$1"; }
temp_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g; s/,/_/g'; }

router_dir_for_label() {
  printf '%s/%s_two_layer' "$ROUTER_DIR" "$1"
}

task_dataset() {
  case "$1" in
    boolq) printf '%s\n' SuperGLUE_BoolQ_gen ;;
    rte) printf '%s\n' SuperGLUE_RTE_gen ;;
    siqa) printf '%s\n' siqa_gen ;;
    piqa) printf '%s\n' piqa_gen ;;
    openbookqa) printf '%s\n' obqa_main_gen ;;
    arc_c) printf '%s\n' ARC_c_gen ;;
    *) echo "unknown plus task: $1" >&2; exit 2 ;;
  esac
}

summary_exists() {
  local work_dir="$1"
  [[ -d "$work_dir" ]] || return 1
  find "$work_dir" -path '*/summary/*.md' -type f -print -quit | grep -q .
}

record_and_summary_exist() {
  local record_path="$1" work_dir="$2"
  [[ -s "$record_path" ]] && summary_exists "$work_dir"
}

move_partial_eval_outputs() {
  local work_dir="$1" record="$2" log_file="$3" suffix
  suffix=".partial.$(date +%Y%m%d_%H%M%S)"
  [[ -e "$work_dir" ]] && mv "$work_dir" "${work_dir}${suffix}"
  [[ -e "$record" ]] && mv "$record" "${record}${suffix}"
  [[ -e "$log_file" ]] && mv "$log_file" "${log_file}${suffix}"
}

usage() {
  cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 nohup bash tools/T0728_train_router_end2end_sft.sh all > T0728_qwen35_router_end2end_sft.nohup.log 2>&1 &

Modes:
  train     train mrs_only and every mrs_plus_* router
  eval      run family-matched OpenCompass eval for trained routers
  baseline  run BF16 base_model and uniform eval on all 9 official datasets
  all       train one family, eval it, then continue; baseline eval last

Defaults:
  BASE_MODEL_PATH=Qwen/Qwen3.5-4B
  LORA_ROOT=./saves/Qwen/Qwen3.5-4B/lora
  MIDDLE_LAYER_IDX=16
  MRS_TASKS=medmcqa,race,sst2
  PLUS_TASKS=boolq,rte,siqa,piqa,openbookqa,arc_c
  BATCH_SIZE=2
  EVAL_BATCH_SIZE=2
  EPOCHS=10
  EARLY_STOP_PATIENCE=2
  ROUTING_SHARPNESS=1.0
  WANDB=1
  WANDB_NAME=qwen35_half16_end2end_sft_router
  OC_SHARPNESS_LIST=1.0,3.0

To use more GPU during router training, try:
  BATCH_SIZE=2 EVAL_BATCH_SIZE=2 bash tools/T0728_train_router_end2end_sft.sh all
EOF
}

case "${MODE}" in
  -h|--help|help) usage; exit 0 ;;
esac

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "[RUN] OUT_ROOT=${OUT_ROOT}"
echo "[RUN] ROUTER_DIR=${ROUTER_DIR}"
echo "[RUN] BASE_MODEL_PATH=${BASE_MODEL_PATH}"
echo "[RUN] DATA_ROOT=${DATA_ROOT} MRS_TASKS=${MRS_TASKS} PLUS_TASKS=${PLUS_TASKS} EXPERTS=${EXPERTS}"
echo "[RUN] LORA_PATHS=${LORA_PATHS}"

run_train_family() {
  local label="$1" train_tasks="$2" output_dir="$3"
  local -a wandb_args=()

  if [[ "${RESUME}" == "1" && -f "${output_dir}/router_heads.pt" && -f "${output_dir}/router_config.json" ]]; then
    log "[SKIP] train label=${label} output_dir=${output_dir}"
    return 0
  fi

  mkdir -p "$output_dir"
  if [[ "${WANDB}" == "1" ]]; then
    wandb_args=(
      --wandb
      --wandb_project "${WANDB_PROJECT}"
      --wandb_name "${WANDB_NAME}_${label}"
      --wandb_group "${WANDB_GROUP}"
      --wandb_tags "${WANDB_TAGS},${label}"
    )
  fi

  log "[START] train label=${label} tasks=${train_tasks} middle_layer_idx=${MIDDLE_LAYER_IDX} wandb=${WANDB}"
  "${PYTHON}" -u train_router_end2end_sft.py \
    --data_root "${DATA_ROOT}" \
    --task_names "${train_tasks}" \
    --expert_names "${EXPERTS}" \
    --lora_paths "${LORA_PATHS}" \
    --base_model_path "${BASE_MODEL_PATH}" \
    --router_bert_init "${ROUTER_BERT_INIT}" \
    --output_dir "${output_dir}" \
    --batch_size "${BATCH_SIZE}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" \
    --epochs "${EPOCHS}" \
    --lr "${LR}" \
    --max_train_samples "${MAX_TRAIN_SAMPLES}" \
    --max_val_samples "${MAX_VAL_SAMPLES}" \
    --max_llm_len "${MAX_LLM_LEN}" \
    --max_bert_len "${MAX_BERT_LEN}" \
    --first_layer_idx "${FIRST_LAYER_IDX}" \
    --middle_layer_idx "${MIDDLE_LAYER_IDX}" \
    --router_dim "${ROUTER_DIM}" \
    --router_pooling "${ROUTER_POOLING}" \
    --routing_sharpness "${ROUTING_SHARPNESS}" \
    --early_stop_patience "${EARLY_STOP_PATIENCE}" \
    --early_stop_min_delta "${EARLY_STOP_MIN_DELTA}" \
    --dtype "${DTYPE}" \
    --r "${LORA_R}" \
    --alpha "${LORA_ALPHA}" \
    --add_eos_to_target \
    "${wandb_args[@]}"
  log "[DONE] train label=${label} router=${output_dir}"
}

run_train_all() {
  local label output_dir plus_task train_tasks

  label="mrs_only"
  output_dir="$(router_dir_for_label "$label")"
  run_train_family "$label" "$MRS_TASKS" "$output_dir"

  split_csv "$PLUS_TASKS"
  for plus_task in "${SPLIT_RESULT[@]}"; do
    [[ -n "$plus_task" ]] || continue
    label="mrs_plus_${plus_task}"
    output_dir="$(router_dir_for_label "$label")"
    train_tasks="${MRS_TASKS},${plus_task}"
    run_train_family "$label" "$train_tasks" "$output_dir"
  done
}

run_train_eval_all() {
  local label output_dir plus_task train_tasks

  label="mrs_only"
  output_dir="$(router_dir_for_label "$label")"
  run_train_family "$label" "$MRS_TASKS" "$output_dir"
  run_family_eval "$label" "$output_dir"

  split_csv "$PLUS_TASKS"
  for plus_task in "${SPLIT_RESULT[@]}"; do
    [[ -n "$plus_task" ]] || continue
    label="mrs_plus_${plus_task}"
    output_dir="$(router_dir_for_label "$label")"
    train_tasks="${MRS_TASKS},${plus_task}"
    run_train_family "$label" "$train_tasks" "$output_dir"
    run_family_eval "$label" "$output_dir" "$plus_task"
  done
}

run_eval_one() {
  local label="$1" router_ckpt="$2" routing_mode="$3" sharpness="$4"
  shift 4
  local -a datasets=("$@")
  local tag work_dir record eval_log
  tag="$(temp_tag "$sharpness")"
  work_dir="${OC_DIR}/${label}_${routing_mode}_sharp${tag}_${RUN_STAMP}"
  record="${RECORD_DIR}/${label}_${routing_mode}_sharp${tag}_${RUN_STAMP}.jsonl"
  eval_log="${LOG_DIR}/eval_${label}_${routing_mode}_sharp${tag}_${RUN_STAMP}.log"

  if [[ "${RESUME}" == "1" ]] && record_and_summary_exist "$record" "$work_dir"; then
    log "[SKIP] eval label=${label} mode=${routing_mode} sharpness=${sharpness} work_dir=${work_dir}"
    return 0
  fi
  if [[ "${RESUME}" == "1" && ( -e "$work_dir" || -e "$record" || -e "$eval_log" ) ]]; then
    move_partial_eval_outputs "$work_dir" "$record" "$eval_log"
  fi
  [[ ! -e "$work_dir" && ! -e "$record" && ! -e "$eval_log" ]] || {
    echo "eval output exists: $work_dir $record $eval_log" >&2
    exit 1
  }

  log "[START] opencompass eval label=${label} mode=${routing_mode} sharpness=${sharpness} datasets=${datasets[*]}"
  BASE_MODEL="${BASE_MODEL_PATH}" \
  LORA_ROOT="${LORA_ROOT}" \
  MIDDLE_LAYER_IDX="${MIDDLE_LAYER_IDX}" \
  T0531_MRS_ROUTER_CKPT="${router_ckpt}" \
  T0531_ROUTER_BERT_INIT="${ROUTER_BERT_INIT}" \
  T0531_ROUTING_MODE="${routing_mode}" \
  T0531_ROUTING_SHARPNESS="${sharpness}" \
  T0531_ROUTER_RECORD_TAG="${OUT_PREFIX}_${label}_${routing_mode}_sharp${tag}" \
  T0601_ROUTER_RECORD_PATH="${record}" \
  T0728_ROUTER_DTYPE="bfloat16" \
  T0728_ROUTER_BATCH_SIZE="${OC_ROUTER_BATCH_SIZE}" \
    "${PYTHON}" -u run.py \
      --models "${MODEL_CONFIG}" \
      --datasets "${datasets[@]}" \
      --work-dir "${work_dir}" \
      --debug \
      > "${eval_log}" 2>&1
  log "[DONE] opencompass eval label=${label} work_dir=${work_dir} log=${eval_log}"
}

run_router_eval_all() {
  local mrs_router plus_task label router_ckpt sharpness plus_dataset
  mrs_router="$(router_dir_for_label mrs_only)"
  [[ -f "${mrs_router}/router_heads.pt" && -f "${mrs_router}/router_config.json" ]] || {
    echo "missing mrs_only router checkpoint under ${mrs_router}" >&2
    exit 1
  }

  split_csv "${OC_SHARPNESS_LIST}"
  local -a sharps=("${SPLIT_RESULT[@]}")
  for sharpness in "${sharps[@]}"; do
    run_eval_one "mrs_only" "$mrs_router" "weighted_sum" "$sharpness" "${MRS_DATASETS[@]}"

    split_csv "$PLUS_TASKS"
    for plus_task in "${SPLIT_RESULT[@]}"; do
      [[ -n "$plus_task" ]] || continue
      label="mrs_plus_${plus_task}"
      router_ckpt="$(router_dir_for_label "$label")"
      [[ -f "${router_ckpt}/router_heads.pt" && -f "${router_ckpt}/router_config.json" ]] || {
        echo "missing ${label} router checkpoint under ${router_ckpt}" >&2
        exit 1
      }
      plus_dataset="$(task_dataset "$plus_task")"
      run_eval_one "$label" "$router_ckpt" "weighted_sum" "$sharpness" "$plus_dataset" "${MRS_DATASETS[@]}"
    done
  done
}

run_family_eval() {
  local label="$1" router_ckpt="$2" plus_task="${3:-}" sharpness plus_dataset
  [[ -f "${router_ckpt}/router_heads.pt" && -f "${router_ckpt}/router_config.json" ]] || {
    echo "missing ${label} router checkpoint under ${router_ckpt}" >&2
    exit 1
  }

  split_csv "${OC_SHARPNESS_LIST}"
  local -a sharps=("${SPLIT_RESULT[@]}")
  for sharpness in "${sharps[@]}"; do
    if [[ -n "$plus_task" ]]; then
      plus_dataset="$(task_dataset "$plus_task")"
      run_eval_one "$label" "$router_ckpt" "weighted_sum" "$sharpness" "$plus_dataset" "${MRS_DATASETS[@]}"
    else
      run_eval_one "$label" "$router_ckpt" "weighted_sum" "$sharpness" "${MRS_DATASETS[@]}"
    fi
  done
}

run_base_model_eval() {
  local label work_dir eval_log
  label="base_model_bf16_all9"
  work_dir="${OC_DIR}/${label}_${RUN_STAMP}"
  eval_log="${LOG_DIR}/eval_${label}_${RUN_STAMP}.log"

  if [[ "${RESUME}" == "1" ]] && summary_exists "$work_dir"; then
    log "[SKIP] eval label=${label} work_dir=${work_dir}"
    return 0
  fi
  if [[ "${RESUME}" == "1" && ( -e "$work_dir" || -e "$eval_log" ) ]]; then
    move_partial_eval_outputs "$work_dir" "/tmp/nonexistent_t0728_base_record" "$eval_log"
  fi

  log "[START] opencompass eval label=${label} datasets=${OFFICIAL_ALL_DATASETS[*]}"
  BASE_MODEL="${BASE_MODEL_PATH}" \
  T0728_BASE_TORCH_DTYPE="torch.bfloat16" \
  T0728_BASE_BATCH_SIZE="${OC_BASE_BATCH_SIZE}" \
    "${PYTHON}" -u run.py \
      --models "${BASE_MODEL_CONFIG}" \
      --datasets "${OFFICIAL_ALL_DATASETS[@]}" \
      --work-dir "${work_dir}" \
      --debug \
      > "${eval_log}" 2>&1
  log "[DONE] opencompass eval label=${label} work_dir=${work_dir} log=${eval_log}"
}

run_uniform_eval() {
  local mrs_router
  mrs_router="$(router_dir_for_label mrs_only)"
  [[ -f "${mrs_router}/router_heads.pt" && -f "${mrs_router}/router_config.json" ]] || {
    echo "missing mrs_only router checkpoint for uniform eval under ${mrs_router}" >&2
    exit 1
  }
  run_eval_one "uniform_bf16_all9" "$mrs_router" "uniform" "1.0" "${OFFICIAL_ALL_DATASETS[@]}"
}

run_eval_baselines() {
  if [[ "${EVAL_BASELINES}" != "1" ]]; then
    log "[SKIP] baseline eval EVAL_BASELINES=${EVAL_BASELINES}"
    return 0
  fi
  run_base_model_eval
  run_uniform_eval
}

case "${MODE}" in
  train) run_train_all ;;
  eval) run_router_eval_all ;;
  baseline) run_eval_baselines ;;
  all) run_train_eval_all; run_eval_baselines ;;
  *) usage; exit 2 ;;
esac

log "[DONE] mode=${MODE} out_root=${OUT_ROOT}"
