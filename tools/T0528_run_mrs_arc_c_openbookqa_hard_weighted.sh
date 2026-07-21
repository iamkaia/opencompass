#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PY="/home/u9472191/.conda/envs/opencompass/bin/python"
CONDA_LIB="/home/u9472191/.conda/envs/opencompass/lib"
export LD_LIBRARY_PATH="$CONDA_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

RUN_ID="$(date +"%Y%m%d_%H%M%S")"
LOG_DIR="T0528_mrs_arc_c_openbookqa_logs_${RUN_ID}"
mkdir -p "$LOG_DIR"

usage() {
    cat >&2 <<'EOF'
usage:
  CUDA_VISIBLE_DEVICES=0 bash tools/T0528_run_mrs_arc_c_openbookqa_hard_weighted.sh [all|llama|qwen]

Runs existing MRS routers on:
  ARC_c_gen obqa_main_gen

For each requested model family, runs both:
  weighted_sum hard_routing

Runs synchronously. Each run writes one log under T0528_mrs_arc_c_openbookqa_logs_*.
EOF
}

target="${1:-all}"
case "$target" in
    all) FAMILIES=(llama qwen) ;;
    llama) FAMILIES=(llama) ;;
    qwen) FAMILIES=(qwen) ;;
    *) usage; exit 2 ;;
esac

router_dir_for() {
    case "$1" in
        llama) echo "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert" ;;
        qwen) echo "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words" ;;
    esac
}

model_config_for() {
    local family="$1"
    local routing="$2"
    case "$family:$routing" in
        llama:weighted_sum) echo "T0528_mrs_arc_obqa_weighted_sum_topk.py" ;;
        llama:hard_routing) echo "T0528_mrs_arc_obqa_hard_routing.py" ;;
        qwen:weighted_sum) echo "T0528_mrs_arc_obqa_sst2words_weighted_sum_topk.py" ;;
        qwen:hard_routing) echo "T0528_mrs_arc_obqa_sst2words_hard_routing.py" ;;
    esac
}

require_router() {
    local router_root="$1"
    if [[ ! -f "$router_root/router_config.json" || ! -f "$router_root/router_heads.pt" ]]; then
        echo "missing MRS router checkpoint under: $router_root" >&2
        exit 1
    fi
}

run_one() {
    local family="$1"
    local routing="$2"
    local model_config log_file
    require_router "$(router_dir_for "$family")"
    model_config="$(model_config_for "$family" "$routing")"
    log_file="$LOG_DIR/opencompass_${family}_mrs_arc_c_openbookqa_${routing}.log"

    echo "[START] family=$family routing=$routing datasets=ARC_c_gen obqa_main_gen log=$log_file"
    "$PY" -u run.py \
        --models "$model_config" \
        --datasets ARC_c_gen obqa_main_gen \
        --debug \
        > "$log_file" 2>&1
    echo "[DONE] family=$family routing=$routing log=$log_file"
}

for family in "${FAMILIES[@]}"; do
    run_one "$family" weighted_sum
    run_one "$family" hard_routing
done
