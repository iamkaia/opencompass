#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-./T0617_static_sample_ratio_baseline_${RUN_STAMP}}"
SOURCE_EXP_ROOT="${SOURCE_EXP_ROOT:-./T0603_qwenfix_trainbert_chattemplate_20260603_063927}"
CACHE_ROOT="${CACHE_ROOT:-$SOURCE_EXP_ROOT/caches/qwen3_fp16_0602_9task_3expert_official_eval_aligned_sst2words_chattemplate_qwenfix_800_200}"
TARGET_SPLIT="${TARGET_SPLIT:-train}"
TASKS="${TASKS:-boolq,rte,siqa,piqa,openbookqa,arc_c,medmcqa,race,sst2}"
EXPERTS="${EXPERTS:-medmcqa,race,sst2}"
BERT="${BERT:-./task_classifier_ckpt}"
MODEL_CONFIG="${MODEL_CONFIG:-T0531_mrs_ablation_sst2words_hard_routing.py}"
ROUTER_CKPT="${ROUTER_CKPT:-./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945/routers/router_T0616_qwen3_fp16_mrs_only_freezebert_wsum_correctconf_raw_t0p25_emptyuniform_3expert_sst2words}"
TRAINED_ROOT="${TRAINED_ROOT:-./T0616_wsum_correct_conf_raw_t0p25_freezebert_full_20260616_061945}"
TRAIN_SPLIT_DATASET="${TRAIN_SPLIT_DATASET:-router_train_split_gen}"

LOG_DIR="$OUT_ROOT/logs"
OC_TRAIN_SPLIT_DIR="$OUT_ROOT/opencompass_train_split"
OC_OFFICIAL_DIR="$OUT_ROOT/opencompass_official"
RECORD_DIR="$OUT_ROOT/router_records"
WEIGHT_PATH="$OUT_ROOT/static_sample_ratio_weights_${TARGET_SPLIT}.json"
TARGET_PRED_REPORT="$OUT_ROOT/target_vs_trained_pred_summary.json"
ROOT_LOG="$OUT_ROOT/pipeline.log"

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

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0617_static_sample_ratio_baseline_qwen.sh

This builds per-task static sample-ratio weights from cached router targets:
  first_weights = average target matrix row sums
  mid_weights   = average target matrix column sums

Then it runs OpenCompass with routing_mode=static_weighted_sum:
  1. train_split all 9 router_train datasets
  2. official eval all MRS + new datasets

It also writes target_vs_trained_pred_summary.json when TRAINED_ROOT exists,
so you can compare each task's cached target average with the trained router's
validation predicted average.

Set BUILD_ONLY=1 to only write the weight/report JSON files without running
OpenCompass.
EOF
}

timestamp() { date "+%Y-%m-%d %H:%M:%S %Z"; }
log() { printf '[%s] %s\n' "$(timestamp)" "$*"; }

setup_root() {
    if [[ -e "$OUT_ROOT" ]]; then
        echo "OUT_ROOT already exists; refusing to overwrite: $OUT_ROOT" >&2
        exit 1
    fi
    mkdir -p "$LOG_DIR" "$OC_TRAIN_SPLIT_DIR" "$OC_OFFICIAL_DIR" "$RECORD_DIR"
    exec > >(tee -a "$ROOT_LOG") 2>&1
}

require_inputs() {
    [[ -f "$CACHE_ROOT/$TARGET_SPLIT/manifest.json" ]] || {
        echo "missing cache split manifest: $CACHE_ROOT/$TARGET_SPLIT/manifest.json" >&2
        exit 1
    }
    [[ -f "$ROUTER_CKPT/router_heads.pt" && -f "$ROUTER_CKPT/router_config.json" ]] || {
        echo "missing router checkpoint used for runtime model shape: $ROUTER_CKPT" >&2
        exit 1
    }
}

build_static_weights() {
    log "[START] build static sample-ratio weights split=$TARGET_SPLIT out=$WEIGHT_PATH"
    "$PY" - <<'PY'
import glob
import json
import math
import os
import torch

from train_internal_two_router_compact_cached_joint import (
    CachedLossMatrixDataset,
    build_router_target_distribution,
)

cache_root = os.environ["CACHE_ROOT"]
target_split = os.environ["TARGET_SPLIT"]
tasks = [x for x in os.environ["TASKS"].split(",") if x]
experts = [x for x in os.environ["EXPERTS"].split(",") if x]
out_path = os.environ["WEIGHT_PATH"]
trained_root = os.environ.get("TRAINED_ROOT", "")
target_pred_report = os.environ["TARGET_PRED_REPORT"]

dataset_aliases = {
    "boolq": ["boolq", "boolq_router_train", "BoolQ", "SuperGLUE_BoolQ_gen"],
    "rte": ["rte", "rte_router_train", "RTE", "SuperGLUE_RTE_gen"],
    "siqa": ["siqa", "siqa_router_train", "siqa_gen"],
    "piqa": ["piqa", "piqa_router_train", "piqa_gen"],
    "openbookqa": ["openbookqa", "openbookqa_router_train", "openbookqa_gen", "obqa_main_gen", "openbookqa", "openbookqa_main"],
    "arc_c": ["arc_c", "arc_c_router_train", "ARC-c", "ARC_c_gen"],
    "medmcqa": ["medmcqa", "medmcqa_router_train", "medmcqa_gen_sft_prompt"],
    "race": ["race", "race_router_train", "race_gen_sft_prompt", "race-middle", "race-high"],
    "sst2": ["sst2", "sst2_router_train", "sst2_gen"],
}

ds = CachedLossMatrixDataset(
    feature_root=cache_root,
    split=target_split,
    selected_sample_task_names=tasks,
    selected_expert_names=experts,
)
stats = {
    task: {
        "count": 0,
        "no_correct_count": 0,
        "first": [0.0] * len(experts),
        "mid": [0.0] * len(experts),
        "entropy01_sum": 0.0,
        "pair_l1_uniform_sum": 0.0,
        "first_l1_uniform_sum": 0.0,
        "mid_l1_uniform_sum": 0.0,
    }
    for task in tasks
}
for item in ds.items:
    task = item["task"]
    loss = item["loss_matrix"].unsqueeze(0).float()
    correct = item["correct_matrix"].unsqueeze(0).bool()
    task_id = torch.tensor([item["task_id"]], dtype=torch.long)
    target = build_router_target_distribution(
        pair_logits=torch.zeros(1, len(experts) * len(experts)),
        loss_matrix=loss,
        correct_matrix=correct,
        task_ids=task_id,
        joint_loss="correct_conf_ce",
        loss_normalization="none",
        correct_soft_ce_temperature=0.25,
        self_preserve_weight=1.0,
        target_distribution_policy="cache_oracle",
        target_empty_fallback="uniform",
    )[0]
    flat = target.reshape(-1)
    first = target.sum(dim=1)
    mid = target.sum(dim=0)
    rec = stats[task]
    rec["count"] += 1
    if not bool(correct.reshape(-1).any().item()):
        rec["no_correct_count"] += 1
    p = flat.clamp_min(1e-12)
    rec["entropy01_sum"] += float((-(p * p.log()).sum() / math.log(len(experts) * len(experts))).item())
    rec["pair_l1_uniform_sum"] += float(torch.abs(flat - 1.0 / flat.numel()).sum().item())
    rec["first_l1_uniform_sum"] += float(torch.abs(first - 1.0 / len(experts)).sum().item())
    rec["mid_l1_uniform_sum"] += float(torch.abs(mid - 1.0 / len(experts)).sum().item())
    for i in range(len(experts)):
        rec["first"][i] += float(first[i].item())
        rec["mid"][i] += float(mid[i].item())

weights_by_task = {}
weights_by_dataset = {}
for task, rec in stats.items():
    n = max(int(rec["count"]), 1)
    first = [x / n for x in rec["first"]]
    mid = [x / n for x in rec["mid"]]
    entry = {
        "task": task,
        "experts": experts,
        "source_split": target_split,
        "count": int(rec["count"]),
        "no_correct_ratio": rec["no_correct_count"] / n,
        "entropy01": rec["entropy01_sum"] / n,
        "pair_l1_uniform": rec["pair_l1_uniform_sum"] / n,
        "first_l1_uniform": rec["first_l1_uniform_sum"] / n,
        "mid_l1_uniform": rec["mid_l1_uniform_sum"] / n,
        "first_weights": first,
        "mid_weights": mid,
    }
    weights_by_task[task] = entry
    for alias in dataset_aliases.get(task, [task]):
        weights_by_dataset[alias] = {
            "task": task,
            "experts": experts,
            "first_weights": first,
            "mid_weights": mid,
        }

obj = {
    "type": "static_sample_ratio_weights",
    "target_builder": {
        "joint_loss": "correct_conf_ce",
        "correct_soft_ce_temperature": 0.25,
        "pair_loss_normalization": "none",
        "target_empty_fallback": "uniform",
        "target_distribution_policy": "cache_oracle",
    },
    "cache_root": cache_root,
    "source_split": target_split,
    "experts": experts,
    "weights_by_task": weights_by_task,
    "weights_by_dataset": weights_by_dataset,
}
os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(obj, f, indent=2, ensure_ascii=False)

report = {"weights_by_task": weights_by_task, "trained_pred_by_task": {}}
if trained_root and os.path.isdir(trained_root):
    for router_dir in sorted(glob.glob(os.path.join(trained_root, "routers", "*"))):
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
        sums = {}
        counts = {}
        for record in records:
            task = record.get("task")
            if task not in weights_by_task:
                continue
            pf = record.get("pred_first_weights")
            pm = record.get("pred_mid_weights")
            tf = record.get("target_first_weights")
            tm = record.get("target_mid_weights")
            if pf is None or pm is None:
                continue
            cur = sums.setdefault(
                task,
                {
                    "pred_first": [0.0] * len(experts),
                    "pred_mid": [0.0] * len(experts),
                    "target_first": [0.0] * len(experts),
                    "target_mid": [0.0] * len(experts),
                },
            )
            counts[task] = counts.get(task, 0) + 1
            for i in range(len(experts)):
                cur["pred_first"][i] += float(pf[i])
                cur["pred_mid"][i] += float(pm[i])
                if tf is not None:
                    cur["target_first"][i] += float(tf[i])
                if tm is not None:
                    cur["target_mid"][i] += float(tm[i])
        label = os.path.basename(router_dir)
        report["trained_pred_by_task"][label] = {}
        for task, cur in sums.items():
            n = counts[task]
            report["trained_pred_by_task"][label][task] = {
                key: [x / n for x in values]
                for key, values in cur.items()
            } | {"count": n, "best_epoch": epoch}

with open(target_pred_report, "w", encoding="utf-8") as f:
    json.dump(report, f, indent=2, ensure_ascii=False)

print(f"wrote {out_path}")
print(f"wrote {target_pred_report}")
for task in tasks:
    rec = weights_by_task[task]
    first = ",".join(f"{x:.3f}" for x in rec["first_weights"])
    mid = ",".join(f"{x:.3f}" for x in rec["mid_weights"])
    print(
        f"{task}: no_correct={rec['no_correct_ratio']:.3f} entropy={rec['entropy01']:.3f} "
        f"first=[{first}] mid=[{mid}]"
    )
PY
    log "[DONE] build static sample-ratio weights out=$WEIGHT_PATH report=$TARGET_PRED_REPORT"
}

eval_train_split() {
    local log_file="$LOG_DIR/T0617_static_sample_ratio_train_split_${RUN_STAMP}.log"
    local work_dir="$OC_TRAIN_SPLIT_DIR/static_sample_ratio_${RUN_STAMP}"
    local record_path="$RECORD_DIR/T0617_static_sample_ratio_train_split_${RUN_STAMP}.jsonl"
    log "[START] static train_split work_dir=$work_dir"
    T0531_MRS_ROUTER_CKPT="$ROUTER_CKPT" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=static_weighted_sum \
    T0616_STATIC_WEIGHT_PATH="$WEIGHT_PATH" \
    T0531_ROUTER_RECORD_TAG="T0617_static_sample_ratio_train_split" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "$TRAIN_SPLIT_DATASET" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] static train_split log=$log_file record=$record_path"
}

eval_official() {
    local log_file="$LOG_DIR/T0617_static_sample_ratio_official_${RUN_STAMP}.log"
    local work_dir="$OC_OFFICIAL_DIR/static_sample_ratio_${RUN_STAMP}"
    local record_path="$RECORD_DIR/T0617_static_sample_ratio_official_${RUN_STAMP}.jsonl"
    log "[START] static official work_dir=$work_dir"
    T0531_MRS_ROUTER_CKPT="$ROUTER_CKPT" \
    T0531_ROUTER_BERT_INIT="$BERT" \
    T0531_ROUTING_MODE=static_weighted_sum \
    T0616_STATIC_WEIGHT_PATH="$WEIGHT_PATH" \
    T0531_ROUTER_RECORD_TAG="T0617_static_sample_ratio_official" \
    T0601_ROUTER_RECORD_PATH="$record_path" \
        "$PY" -u run.py \
        --models "$MODEL_CONFIG" \
        --datasets "${OFFICIAL_ALL_DATASETS[@]}" \
        --work-dir "$work_dir" \
        --debug \
        > "$log_file" 2>&1
    log "[DONE] static official log=$log_file record=$record_path"
}

main() {
    case "${1:-}" in
        -h|--help|help) usage; exit 0 ;;
        "") ;;
        *) usage; exit 2 ;;
    esac
    setup_root
    require_inputs
    export CACHE_ROOT TARGET_SPLIT TASKS EXPERTS WEIGHT_PATH TARGET_PRED_REPORT TRAINED_ROOT
    log "[INFO] out_root=$OUT_ROOT"
    log "[INFO] cache_root=$CACHE_ROOT target_split=$TARGET_SPLIT"
    log "[INFO] router_ckpt_for_runtime_shape=$ROUTER_CKPT"
    build_static_weights
    if [[ "${BUILD_ONLY:-0}" == "1" ]]; then
        log "[DONE] build-only static sample-ratio output=$OUT_ROOT"
        exit 0
    fi
    eval_train_split
    eval_official
    log "[DONE] static sample-ratio baseline output=$OUT_ROOT"
}

main "$@"
