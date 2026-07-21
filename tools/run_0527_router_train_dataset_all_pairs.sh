#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-/home/u9472191/.conda/envs/opencompass/bin/python}"
exec "$PY" tools/run_0527_router_train_dataset_all_pairs.py
