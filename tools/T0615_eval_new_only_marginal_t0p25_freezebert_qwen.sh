#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

TRAIN_ROOT="${TRAIN_ROOT:-./T0615_new_only_marginal_t0p25_freezebert_20260615}"
OUT_ROOT="${OUT_ROOT:-./T0615_new_only_marginal_t0p25_freezebert_official_eval_20260615}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
SHARPNESS_LIST="${SHARPNESS_LIST:-1.0,2.0,3.0}"
ROUTING_TOPK="${ROUTING_TOPK:-}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
MRS_DATASETS=(medmcqa_gen_sft_prompt race_gen_sft_prompt sst2_gen)

timestamp() { date '+%F %T %Z'; }
log() { mkdir -p "$OUT_ROOT"; printf '[%s] %s\n' "$(timestamp)" "$*" | tee -a "$OUT_ROOT/run.log"; }

dataset_for() {
    case "$1" in
        boolq) echo SuperGLUE_BoolQ_gen ;;
        rte) echo SuperGLUE_RTE_gen ;;
        siqa) echo siqa_gen ;;
        piqa) echo piqa_gen ;;
        openbookqa) echo obqa_main_gen ;;
        arc_c) echo ARC_c_gen ;;
        *) echo "unsupported task: $1" >&2; exit 2 ;;
    esac
}

router_for() {
    echo "$TRAIN_ROOT/routers/router_T0615_qwen3_fp16_new_only_${1}_from_mrs_marginal_t0p25_freezebert_3expert_sst2words"
}

IFS=',' read -r -a sharpness_values <<< "$SHARPNESS_LIST"
IFS=',' read -r -a task_values <<< "$TASKS"

for sharpness in "${sharpness_values[@]}"; do
    sharp_tag="$(printf '%s' "$sharpness" | sed 's/-/m/g; s/\./p/g')"
    eval_root="$OUT_ROOT/sharp${sharp_tag}"
    mkdir -p "$eval_root/logs" "$eval_root/opencompass" "$eval_root/router_records"

    for task in "${task_values[@]}"; do
        router="$(router_for "$task")"
        [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" ]] || {
            echo "missing completed router: $router" >&2
            exit 1
        }
        dataset="$(dataset_for "$task")"
        label="new_only_${task}_marginal_t0p25_freezebert"
        log_file="$eval_root/logs/T0615_${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}.log"
        work_dir="$eval_root/opencompass/T0615_${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}"
        record_path="$eval_root/router_records/T0615_${label}_weighted_sum_sharp${sharp_tag}_${RUN_STAMP}.jsonl"
        [[ ! -e "$log_file" && ! -e "$work_dir" && ! -e "$record_path" ]] || {
            echo "eval output exists: task=$task sharpness=$sharpness" >&2
            exit 1
        }

        log "[START] task=$task sharpness=$sharpness datasets=$dataset ${MRS_DATASETS[*]}"
        T0531_MRS_ROUTER_CKPT="$router" \
        T0531_ROUTER_BERT_INIT="$BERT" \
        T0531_ROUTING_MODE=weighted_sum \
        T0531_ROUTING_SHARPNESS="$sharpness" \
        T0531_ROUTING_TOPK="$ROUTING_TOPK" \
        T0531_ROUTER_RECORD_TAG="T0615_${label}_sharp${sharp_tag}" \
        T0601_ROUTER_RECORD_PATH="$record_path" \
            "$PY" -u run.py \
            --models "$MODEL_CONFIG" \
            --datasets "$dataset" "${MRS_DATASETS[@]}" \
            --work-dir "$work_dir" \
            --debug \
            > "$log_file" 2>&1
        log "[DONE] task=$task sharpness=$sharpness log=$log_file"
    done
done

log "[DONE] all new_only official evaluations output=$OUT_ROOT"
