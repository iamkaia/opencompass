#!/usr/bin/env python3
"""Summarize two-layer router pair diagnostics from saved route records."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any


ROWS = [
    "mrs_only",
    "mrs_plus_boolq",
    "mrs_plus_piqa",
    "mrs_plus_siqa",
    "mrs_plus_rte",
    "mrs_plus_arc-c",
    "mrs_plus_openbookqa",
]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def row_name_from_router_dir(path: Path) -> str:
    name = path.name
    for suffix in ("_two_layer", "_single_all_layers"):
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name.replace("mrs_plus_arc_c", "mrs_plus_arc-c")


def validation_index(record: dict[str, Any]) -> str:
    item_id = str(record.get("item_id", ""))
    match = re.search(r":validation:(\d+)$", item_id)
    if match:
        return match.group(1)
    return str(record.get("prompt_sha1") or item_id)


def pick_two_records(router_dir: Path, split: str) -> Path | None:
    if split == "val":
        best_metrics = router_dir / "best_metrics.json"
        if best_metrics.exists():
            best_epoch = load_json(best_metrics).get("best_epoch")
            if best_epoch is not None:
                candidate = router_dir / f"route_records_val_epoch{best_epoch}.json"
                if candidate.exists():
                    return candidate
        candidates = sorted(router_dir.glob("route_records_val_epoch*.json"))
    else:
        candidates = sorted(router_dir.glob("route_records_train_eval_epoch*.json"))
    return candidates[-1] if candidates else None


def pick_single_records(router_dir: Path, split: str) -> Path | None:
    if split == "val":
        candidate = router_dir / "route_records_val_best.json"
        if candidate.exists():
            return candidate
        candidates = sorted(router_dir.glob("route_records_val*.json"))
    else:
        candidates = sorted(router_dir.glob("route_records_train*.json"))
    return candidates[-1] if candidates else None


def flatten_matrix(matrix: list[list[float]]) -> list[float]:
    return [float(x) for row in matrix for x in row]


def entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p) for p in probs if p > 0.0)


def summarize_two_records(records: list[dict[str, Any]]) -> dict[str, float]:
    n = len(records)
    if n == 0:
        return {}
    pred_diag = 0
    gold_diag = 0
    target_diag = 0
    route_match = 0
    pred_entropy = 0.0
    pred_entropy_norm = 0.0
    target_margin = 0.0
    pred_prob = 0.0
    gold_prob = 0.0
    target_available = 0

    for rec in records:
        matrix = rec.get("pair_prob_matrix") or []
        size = len(matrix)
        pred_id = int(rec.get("pred_pair_id", -1))
        gold_id = int(rec.get("gold_pair_id", -1))
        pred_first, pred_mid = divmod(pred_id, size) if size else (-1, -2)
        gold_first, gold_mid = divmod(gold_id, size) if size else (-1, -2)
        pred_diag += int(pred_first == pred_mid)
        gold_diag += int(gold_first == gold_mid)
        route_match += int(bool(rec.get("match", False)))
        pred_prob += float(rec.get("pred_pair_prob") or 0.0)
        gold_prob += float(rec.get("gold_pair_prob") or 0.0)

        probs = flatten_matrix(matrix)
        if probs:
            ent = entropy(probs)
            pred_entropy += ent
            pred_entropy_norm += ent / math.log(len(probs))

        target = rec.get("router_target_matrix") or []
        target_vals = flatten_matrix(target)
        if target_vals:
            target_available += 1
            order = sorted(range(len(target_vals)), key=lambda i: target_vals[i], reverse=True)
            best = order[0]
            second = order[1] if len(order) > 1 else order[0]
            target_margin += target_vals[best] - target_vals[second]
            first, mid = divmod(best, len(target))
            target_diag += int(first == mid)

    return {
        "n": float(n),
        "pred_diag_ratio": pred_diag / n,
        "pred_offdiag_ratio": 1.0 - pred_diag / n,
        "gold_diag_ratio": gold_diag / n,
        "target_diag_ratio": target_diag / target_available if target_available else float("nan"),
        "route_match_ratio": route_match / n,
        "entropy": pred_entropy / n,
        "entropy_norm": pred_entropy_norm / n,
        "target_margin": target_margin / target_available if target_available else float("nan"),
        "pred_pair_prob": pred_prob / n,
        "gold_pair_prob": gold_prob / n,
    }


def compare_single_two(
    two_records: list[dict[str, Any]], single_records: list[dict[str, Any]]
) -> dict[str, float]:
    single_by_key = {validation_index(rec): rec for rec in single_records}
    matched = 0
    first_eq = 0
    mid_eq = 0
    both_eq = 0
    any_eq = 0
    for rec in two_records:
        single = single_by_key.get(validation_index(rec))
        if single is None:
            continue
        pred_expert = str(single.get("pred_expert", ""))
        pair = str(rec.get("pred_pair", ""))
        if "->" not in pair:
            continue
        first, mid = pair.split("->", 1)
        matched += 1
        first_eq += int(first == pred_expert)
        mid_eq += int(mid == pred_expert)
        both_eq += int(first == pred_expert and mid == pred_expert)
        any_eq += int(first == pred_expert or mid == pred_expert)
    if matched == 0:
        return {
            "single_match_n": 0.0,
            "first_eq_single": float("nan"),
            "mid_eq_single": float("nan"),
            "both_eq_single": float("nan"),
            "any_eq_single": float("nan"),
        }
    return {
        "single_match_n": float(matched),
        "first_eq_single": first_eq / matched,
        "mid_eq_single": mid_eq / matched,
        "both_eq_single": both_eq / matched,
        "any_eq_single": any_eq / matched,
    }


def fmt_pct(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value * 100:.1f}%"


def fmt_num(value: float) -> str:
    if math.isnan(value):
        return "NA"
    return f"{value:.4f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--two-root", required=True, type=Path)
    parser.add_argument("--single-root", type=Path)
    parser.add_argument("--split", choices=["val", "train_eval"], default="val")
    args = parser.parse_args()

    two_dirs = {
        row_name_from_router_dir(path): path
        for path in sorted((args.two_root / "routers").glob("*_two_layer"))
    }
    single_dirs = {}
    if args.single_root:
        single_dirs = {
            row_name_from_router_dir(path): path
            for path in sorted((args.single_root / "routers").glob("*_single_all_layers"))
        }

    print(f"two_root: {args.two_root}")
    if args.single_root:
        print(f"single_root: {args.single_root}")
    print(f"split: {args.split}")
    print()
    print(
        "| row | n | pred diag | pred offdiag | gold diag | target diag | "
        "entropy(norm) | target margin | route match | both eq single | any eq single |"
    )
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in ROWS:
        two_dir = two_dirs.get(row)
        if two_dir is None:
            continue
        two_path = pick_two_records(two_dir, args.split)
        if two_path is None:
            continue
        two_records = load_json(two_path).get("records", [])
        stats = summarize_two_records(two_records)

        cmp_stats = {}
        single_dir = single_dirs.get(row)
        if single_dir is not None:
            single_path = pick_single_records(single_dir, args.split)
            if single_path is not None:
                single_records = load_json(single_path).get("records", [])
                cmp_stats = compare_single_two(two_records, single_records)

        print(
            f"| {row} | {int(stats['n'])} | {fmt_pct(stats['pred_diag_ratio'])} | "
            f"{fmt_pct(stats['pred_offdiag_ratio'])} | {fmt_pct(stats['gold_diag_ratio'])} | "
            f"{fmt_pct(stats['target_diag_ratio'])} | {fmt_num(stats['entropy_norm'])} | "
            f"{fmt_num(stats['target_margin'])} | {fmt_pct(stats['route_match_ratio'])} | "
            f"{fmt_pct(cmp_stats.get('both_eq_single', float('nan')))} | "
            f"{fmt_pct(cmp_stats.get('any_eq_single', float('nan')))} |"
        )


if __name__ == "__main__":
    main()
