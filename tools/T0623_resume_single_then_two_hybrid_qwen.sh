#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# This resume script is intentionally tied to the interrupted 20260623 run by default.
# Override RUN_STAMP / OUT_ROOT if you want to resume another compatible run.
RUN_STAMP="${RUN_STAMP:-20260623_024009}"
OUT_ROOT="${OUT_ROOT:-./T0623_single_all_layers_hybrid_${RUN_STAMP}}"
CACHE_ROOT="${CACHE_ROOT:-./T0622_single_all_layers_wsum_run1/cache/qwen3_9task_3expert_single_all_layers_800_200}"
BERT="${BERT:-./task_classifier_ckpt}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="medmcqa,race,sst2"
MODEL_CONFIG="T0531_mrs_ablation_sst2words_hard_routing.py"
ROUTING_TOPK="${ROUTING_TOPK:-}"
RESUME_TAG="${RESUME_TAG:-resume1}"
MIN_AVAIL_GB="${MIN_AVAIL_GB:-10}"
RUN_TWO_LAYER="${RUN_TWO_LAYER:-1}"

ROUTER_ROOT="$OUT_ROOT/routers"
LOG_ROOT="$OUT_ROOT/logs"
OC_ROOT="$OUT_ROOT/opencompass_official"
RECORD_ROOT="$OUT_ROOT/router_records"
ROOT_LOG="$OUT_ROOT/pipeline.log"
MARK_ROOT="$OUT_ROOT/.resume_done"

log() { printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"; }
temp_tag() { printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'; }

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" || "${1:-}" == "help" ]]; then
    cat <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash ./tools/T0623_resume_single_then_two_hybrid_qwen.sh

Purpose:
  Resume the interrupted T0623 single-all-layers hybrid eval without rerunning
  already completed official evals. Failed/incomplete evals are written with a
  resume suffix so existing partial files are not overwritten.

Defaults:
  RUN_STAMP=20260623_024009
  OUT_ROOT=./T0623_single_all_layers_hybrid_20260623_024009
  RESUME_TAG=resume1
  RUN_TWO_LAYER=1
  MIN_AVAIL_GB=10

Outputs:
  single resume logs:
    ./T0623_single_all_layers_hybrid_20260623_024009/logs/*_resume1.log
  single resume work dirs:
    ./T0623_single_all_layers_hybrid_20260623_024009/opencompass_official/*_resume1/
  single resume router records:
    ./T0623_single_all_layers_hybrid_20260623_024009/router_records/*_resume1.jsonl

After the single run is complete, it starts the two-layer hybrid run:
  ./T0623_wsum_hybrid_joint_marginal_raw_t0p25_freezebert_full_20260623_024009/

Set RUN_TWO_LAYER=0 to resume only the single-all-layers eval.
EOF
    exit 0
fi

check_space() {
    local avail_kb min_kb
    avail_kb="$(df -Pk "$ROOT" | awk 'NR==2 {print $4}')"
    min_kb=$((MIN_AVAIL_GB * 1024 * 1024))
    if (( avail_kb < min_kb )); then
        echo "Not enough free space under $ROOT: available=$((avail_kb / 1024 / 1024))G, need_at_least=${MIN_AVAIL_GB}G" >&2
        echo "Free some space first, or rerun with a lower MIN_AVAIL_GB if you really want to risk it." >&2
        exit 1
    fi
}

require_router() {
    [[ -f "$1/router_heads.pt" && -f "$1/router_config.json" ]] || {
        echo "missing router: $1" >&2
        exit 1
    }
}

task_dataset() {
    case "$1" in
        boolq) echo SuperGLUE_BoolQ_gen ;;
        rte) echo SuperGLUE_RTE_gen ;;
        siqa) echo siqa_gen ;;
        piqa) echo piqa_gen ;;
        openbookqa) echo obqa_main_gen ;;
        arc_c) echo ARC_c_gen ;;
        *) echo "bad task=$1" >&2; exit 2 ;;
    esac
}

mrs_router() { echo "$ROUTER_ROOT/mrs_only_single_all_layers"; }
plus_router() { echo "$ROUTER_ROOT/mrs_plus_${1}_single_all_layers"; }
new_router() { echo "$ROUTER_ROOT/new_only_${1}_from_mrs_single_all_layers"; }

was_completed_in_original_pipeline() {
    local label="$1" sharpness="$2" tag work_dir
    tag="$(temp_tag "$sharpness")"
    work_dir="$OC_ROOT/${label}_sharp${tag}_${RUN_STAMP}"
    [[ -f "$ROOT_LOG" ]] && grep -Fq "[DONE] work_dir=$work_dir" "$ROOT_LOG"
}

was_completed_by_resume() {
    local label="$1" sharpness="$2" tag
    tag="$(temp_tag "$sharpness")"
    [[ -f "$MARK_ROOT/${label}_sharp${tag}.done" ]]
}

eval_one_resume() {
    local label="$1" router="$2" sharpness="$3"; shift 3
    local tag work_dir record log_file marker
    require_router "$router"
    tag="$(temp_tag "$sharpness")"
    marker="$MARK_ROOT/${label}_sharp${tag}.done"

    if was_completed_by_resume "$label" "$sharpness"; then
        log "[SKIP] resume marker exists label=$label sharpness=$sharpness"
        return
    fi
    if was_completed_in_original_pipeline "$label" "$sharpness"; then
        log "[SKIP] already completed label=$label sharpness=$sharpness"
        return
    fi

    work_dir="$OC_ROOT/${label}_sharp${tag}_${RUN_STAMP}_${RESUME_TAG}"
    record="$RECORD_ROOT/${label}_sharp${tag}_${RUN_STAMP}_${RESUME_TAG}.jsonl"
    log_file="$LOG_ROOT/eval_${label}_sharp${tag}_${RUN_STAMP}_${RESUME_TAG}.log"

    if [[ -e "$work_dir" || -e "$record" || -e "$log_file" ]]; then
        echo "resume output exists; choose a new RESUME_TAG or inspect/remove the partial files:" >&2
        echo "  $work_dir" >&2
        echo "  $record" >&2
        echo "  $log_file" >&2
        exit 1
    fi

    check_space
    log "[START] resume eval $label sharpness=$sharpness datasets=$*"
    T0531_MRS_ROUTER_CKPT="$router" T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum T0531_ROUTING_SHARPNESS="$sharpness" \
    T0531_ROUTING_TOPK="$ROUTING_TOPK" T0531_SHARE_FIRST_WEIGHTS_ALL_LAYERS=1 \
    T0531_ROUTER_RECORD_TAG="T0623_single_resume_${label}_sharp${tag}" T0601_ROUTER_RECORD_PATH="$record" \
        "$PY" -u run.py --models "$MODEL_CONFIG" --datasets "$@" \
        --work-dir "$work_dir" --debug > "$log_file" 2>&1
    touch "$marker"
    log "[DONE] resume eval label=$label sharpness=$sharpness work_dir=$work_dir log=$log_file"
}

resume_single_eval() {
    local sharpness task dataset
    local -a mrs=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)
    local -a tasks
    IFS=, read -r -a tasks <<< "$TASKS"

    # The interrupted run finished all sharpness=1.0 and part of sharpness=2.0.
    # Calling the full sharpness=2.0/3.0 list is still safe because completed
    # items are skipped from pipeline DONE markers.
    for sharpness in 2.0 3.0; do
        eval_one_resume mrs_only "$(mrs_router)" "$sharpness" SuperGLUE_BoolQ_gen medmcqa_gen_sft_prompt obqa_main_gen ARC_c_gen piqa_gen race_gen_sft_prompt SuperGLUE_RTE_gen siqa_gen sst2_gen
        for task in "${tasks[@]}"; do
            dataset="$(task_dataset "$task")"
            eval_one_resume "mrs_plus_$task" "$(plus_router "$task")" "$sharpness" "$dataset" "${mrs[@]}"
        done
        for task in "${tasks[@]}"; do
            dataset="$(task_dataset "$task")"
            eval_one_resume "new_only_$task" "$(new_router "$task")" "$sharpness" "$dataset" "${mrs[@]}"
        done
    done
}

mkdir -p "$OUT_ROOT" "$LOG_ROOT" "$OC_ROOT" "$RECORD_ROOT" "$MARK_ROOT"
exec > >(tee -a "$ROOT_LOG") 2>&1

log "[RESUME] out_root=$OUT_ROOT run_stamp=$RUN_STAMP resume_tag=$RESUME_TAG"
log "[RESUME] cache_root=$CACHE_ROOT run_two_layer=$RUN_TWO_LAYER min_avail_gb=$MIN_AVAIL_GB"
resume_single_eval
log "[DONE] resumed single-all-layers hybrid eval output=$OUT_ROOT"

if [[ "$RUN_TWO_LAYER" == "1" ]]; then
    check_space
    log "[START] two-layer hybrid run stamp=$RUN_STAMP"
    RUN_STAMP="$RUN_STAMP" bash ./tools/T0623_train_hybrid_joint_marginal_raw_t0p25_freezebert_full_qwen.sh
    log "[DONE] two-layer hybrid run stamp=$RUN_STAMP"
fi
