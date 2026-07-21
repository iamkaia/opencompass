#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0617_wsum_sample_variation_ablation_${RUN_STAMP}}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
TRAIN_SPLIT_DATASET="${TRAIN_SPLIT_DATASET:-router_train_split_gen}"
VARIANTS="${VARIANTS:-mixed_correct_conf_t0p25,mixed_correct_conf_t0p10,mixed_matrixkl_t0p25}"

BATCH_SIZE="${BATCH_SIZE:-32}"
EPOCHS="${EPOCHS:-10}"
LR="${LR:-2e-4}"
MSE_WEIGHT="${MSE_WEIGHT:-1.0}"
PAIR_LOSS_NORMALIZATION="${PAIR_LOSS_NORMALIZATION:-none}"
TARGET_EMPTY_FALLBACK="${TARGET_EMPTY_FALLBACK:-uniform}"
TARGET_DISTRIBUTION_POLICY="${TARGET_DISTRIBUTION_POLICY:-cache_oracle}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-2}"

LOG_DIR="$OUT_ROOT/logs"
ROUTER_DIR="$OUT_ROOT/routers"
OC_TRAIN_SPLIT_DIR="$OUT_ROOT/opencompass_train_split"
RECORD_DIR="$OUT_ROOT/router_records"
VARIATION_REPORT="$OUT_ROOT/sample_variation_summary.tsv"
ROOT_LOG="$OUT_ROOT/pipeline.log"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0617_wsum_sample_variation_ablation_qwen.sh

Runs freeze-BERT weighted_sum training ablations that try to avoid collapsing to
task-level average ratios:
  mixed_correct_conf_t0p25: loss = correct_conf_ce + MSE(first/mid marginals)
  mixed_correct_conf_t0p10: same, sharper target
  mixed_matrixkl_t0p25:     loss = cache_oracle_matrix_kl + MSE(first/mid marginals)

For each variant:
  1. train mrs_only
  2. train new_only_<task>, loading that variant's mrs_only router
  3. write sample_variation_summary.tsv
  4. run OpenCompass train_split only
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

split_csv() {
    local raw="$1"
    local IFS=,
    read -r -a SPLIT_CSV_RESULT <<< "$raw"
}

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$ROUTER_DIR" "$OC_TRAIN_SPLIT_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

require_inputs() {
    [[ -f "$CACHE_ROOT/train/manifest.json" && -f "$CACHE_ROOT/validation/manifest.json" ]] || {
        echo "missing completed cache: $CACHE_ROOT" >&2
        exit 1
    }
}

require_router() {
    local router="$1"
    [[ -f "$router/router_heads.pt" && -f "$router/router_config.json" ]] || {
        echo "missing router checkpoint: $router" >&2
        exit 1
    }
}

variant_joint_loss() {
    case "$1" in
        mixed_correct_conf_t0p25|mixed_correct_conf_t0p10) echo "correct_conf_ce" ;;
        mixed_matrixkl_t0p25) echo "cache_oracle_matrix_kl" ;;
        *) echo "unsupported variant: $1" >&2; exit 2 ;;
    esac
}

variant_temperature() {
    case "$1" in
        mixed_correct_conf_t0p25|mixed_matrixkl_t0p25) echo "0.25" ;;
        mixed_correct_conf_t0p10) echo "0.10" ;;
        *) echo "unsupported variant: $1" >&2; exit 2 ;;
    esac
}

router_path() {
    local variant="$1" label="$2"
    echo "$ROUTER_DIR/router_T0617_${variant}_${label}_freezebert_mixed_wsum_3expert_sst2words"
}

train_router() {
    local variant="$1" label="$2" sample_tasks="$3" out_dir="$4" load_from="${5:-}"
    local joint_loss temperature log_file
    joint_loss="$(variant_joint_loss "$variant")"
    temperature="$(variant_temperature "$variant")"
    log_file="$LOG_DIR/train_${variant}_${label}_${RUN_STAMP}.log"
    [[ ! -e "$out_dir" && ! -e "$log_file" ]] || {
        echo "train output exists: $out_dir $log_file" >&2
        exit 1
    }

    local -a load_args=()
    if [[ -n "$load_from" ]]; then
        require_router "$load_from"
        load_args+=(--load_from "$load_from")
    fi

    log "[START] train variant=$variant label=$label sample_tasks=$sample_tasks load_from=${load_from:-none}"
    "$PY" -u train_internal_two_router_compact_cached_joint.py \
        --feature_roots "$CACHE_ROOT" \
        --bert_init "$BERT" \
        "${load_args[@]}" \
        --out_dir "$out_dir" \
        --sample_task_names "$sample_tasks" \
        --expert_names "$EXPERTS" \
        --router_dim 512 \
        --batch_size "$BATCH_SIZE" \
        --epochs "$EPOCHS" \
        --lr "$LR" \
        --joint_loss "$joint_loss" \
        --supervision_mode oracle_loss \
        --correct_soft_ce_temperature "$temperature" \
        --pair_loss_normalization "$PAIR_LOSS_NORMALIZATION" \
        --target_empty_fallback "$TARGET_EMPTY_FALLBACK" \
        --target_distribution_policy "$TARGET_DISTRIBUTION_POLICY" \
        --weighted_sum_marginal_mse_weight "$MSE_WEIGHT" \
        --best_metric weighted_sum_marginal_mse \
        --early_stop_patience "$EARLY_STOP_PATIENCE" \
        --freeze_bert \
        --save_route_records \
        --eval_train_each_epoch \
        > "$log_file" 2>&1
    log "[DONE] train variant=$variant label=$label out=$out_dir log=$log_file"
}

eval_train_split() {
    local variant="$1" label="$2" router="$3"
    local log_file work_dir record_path
    require_router "$router"
    log_file="$LOG_DIR/eval_train_split_${variant}_${label}_${RUN_STAMP}.log"
    work_dir="$OC_TRAIN_SPLIT_DIR/${variant}_${label}_weighted_sum_${RUN_STAMP}"
    record_path="$RECORD_DIR/train_split_${variant}_${label}_weighted_sum_${RUN_STAMP}.jsonl"
    [[ ! -e "$work_dir" && ! -e "$record_path" && ! -e "$log_file" ]] || {
        echo "eval output exists: $work_dir $record_path $log_file" >&2
        exit 1
    }
    log "[START] train_split variant=$variant label=$label"
    T0531_MRS_ROUTER_CKPT="$router" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=weighted_sum \
    T0531_ROUTING_SHARPNESS=1.0 \
    T0531_ROUTER_RECORD_TAG="T0617_${variant}_${label}_train_split" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$TRAIN_SPLIT_DATASET" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] train_split variant=$variant label=$label log=$log_file"
}

write_variation_report() {
    log "[START] write sample variation report $VARIATION_REPORT"
    "$PY" - <<'PY'
import glob
import json
import os
import statistics as st

out_root = os.environ["OUT_ROOT"]
report_path = os.environ["VARIATION_REPORT"]

def mean_std(records, key):
    vals = [r.get(key) for r in records if r.get(key) is not None]
    if not vals:
        return 0.0
    cols = list(zip(*vals))
    return sum(st.pstdev(col) for col in cols) / len(cols)

rows = []
for router_dir in sorted(glob.glob(os.path.join(out_root, "routers", "*"))):
    best_path = os.path.join(router_dir, "best_metrics.json")
    if not os.path.exists(best_path):
        continue
    with open(best_path, "r", encoding="utf-8") as f:
        best = json.load(f)
    epoch = best.get("best_epoch")
    records_path = os.path.join(router_dir, f"route_records_val_epoch{epoch}.json")
    if not os.path.exists(records_path):
        continue
    with open(records_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])
    by_task = {}
    for record in records:
        by_task.setdefault(record.get("task", "unknown"), []).append(record)
    label = os.path.basename(router_dir).replace("router_T0617_", "")
    metrics = best.get("metrics", {})
    for task, task_records in sorted(by_task.items()):
        rows.append({
            "router": label,
            "best_epoch": epoch,
            "task": task,
            "count": len(task_records),
            "target_first_std": mean_std(task_records, "target_first_weights"),
            "target_mid_std": mean_std(task_records, "target_mid_weights"),
            "pred_first_std": mean_std(task_records, "pred_first_weights"),
            "pred_mid_std": mean_std(task_records, "pred_mid_weights"),
            "best_wsmse": metrics.get("weighted_sum_marginal_mse"),
            "route_correct_acc": metrics.get("route_correct_acc"),
        })

os.makedirs(os.path.dirname(os.path.abspath(report_path)), exist_ok=True)
with open(report_path, "w", encoding="utf-8") as f:
    headers = [
        "router", "best_epoch", "task", "count",
        "target_first_std", "target_mid_std", "pred_first_std", "pred_mid_std",
        "best_wsmse", "route_correct_acc",
    ]
    f.write("\t".join(headers) + "\n")
    for row in rows:
        f.write("\t".join(
            str(row[h]) if not isinstance(row[h], float) else f"{row[h]:.6f}"
            for h in headers
        ) + "\n")
print(f"wrote {report_path}")
PY
    log "[DONE] sample variation report $VARIATION_REPORT"
}

train_all() {
    local variant task mrs_router
    for variant in "${VARIANT_LIST[@]}"; do
        mrs_router="$(router_path "$variant" "mrs_only")"
        train_router "$variant" "mrs_only" "$EXPERTS" "$mrs_router"
        for task in "${TASK_LIST[@]}"; do
            train_router "$variant" "new_only_${task}" "$task" "$(router_path "$variant" "new_only_${task}")" "$mrs_router"
        done
    done
}

eval_all_train_split() {
    local variant task
    for variant in "${VARIANT_LIST[@]}"; do
        eval_train_split "$variant" "mrs_only" "$(router_path "$variant" "mrs_only")"
        for task in "${TASK_LIST[@]}"; do
            eval_train_split "$variant" "new_only_${task}" "$(router_path "$variant" "new_only_${task}")"
        done
    done
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac
    split_csv "$TASKS"
    TASK_LIST=("${SPLIT_CSV_RESULT[@]}")
    split_csv "$VARIANTS"
    VARIANT_LIST=("${SPLIT_CSV_RESULT[@]}")
    setup_root
    require_inputs
    export OUT_ROOT VARIATION_REPORT
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] variants=${VARIANT_LIST[*]}"
    log "[INFO] tasks=${TASK_LIST[*]} experts=$EXPERTS"
    log "[INFO] training=mixed_loss freeze_bert=1 mse_weight=$MSE_WEIGHT pair_loss_normalization=$PAIR_LOSS_NORMALIZATION target_empty_fallback=$TARGET_EMPTY_FALLBACK"
    train_all
    write_variation_report
    eval_all_train_split
    log "[DONE] sample variation ablation output=$OUT_ROOT"
}

main "$@"
