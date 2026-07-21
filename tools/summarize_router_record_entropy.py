#!/usr/bin/env python
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List


def flatten_matrix(matrix: Any) -> List[float]:
    return [float(value) for row in matrix for value in row]


def normalize(values: Iterable[float]) -> List[float]:
    clipped = [max(0.0, float(value)) for value in values]
    total = sum(clipped)
    if total <= 0.0:
        return clipped
    return [value / total for value in clipped]


def entropy(probs: Iterable[float]) -> float:
    normalized = normalize(probs)
    return -sum(value * math.log(value + 1e-12) for value in normalized)


def top_stats(probs: Iterable[float]) -> Dict[str, float]:
    sorted_probs = sorted(normalize(probs), reverse=True)
    if not sorted_probs:
        return {
            "top1_prob": 0.0,
            "top2_prob": 0.0,
            "top1_top2_prob_gap": 0.0,
            "top1_top2_logprob_gap": 0.0,
        }
    top1 = sorted_probs[0]
    top2 = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
    return {
        "top1_prob": top1,
        "top2_prob": top2,
        "top1_top2_prob_gap": top1 - top2,
        "top1_top2_logprob_gap": math.log(top1 + 1e-12) - math.log(top2 + 1e-12),
    }


def sharpen_probs(probs: Iterable[float], sharpness: float) -> List[float]:
    normalized = normalize(probs)
    if not normalized:
        return []
    logits = [math.log(value + 1e-12) * float(sharpness) for value in normalized]
    max_logit = max(logits)
    exp_values = [math.exp(value - max_logit) for value in logits]
    total = sum(exp_values)
    return [value / total for value in exp_values]


def mean(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(len(rows), 1)


def rate(rows: List[Dict[str, Any]], key: str) -> float:
    return sum(bool(row[key]) for row in rows) / max(len(rows), 1)


def print_group(name: str, rows: List[Dict[str, Any]], sharpness_values: List[float]) -> None:
    if not rows:
        return
    print(f"\n[{name}] n={len(rows)}")
    for key in (
        "target_entropy_norm",
        "pred_entropy_norm",
        "target_top1_prob",
        "pred_top1_prob",
        "target_top1_top2_prob_gap",
        "pred_top1_top2_prob_gap",
        "target_top1_top2_logprob_gap",
        "pred_top1_top2_logprob_gap",
    ):
        print(f"  {key}: {mean(rows, key):.6f}")
    for sharpness in sharpness_values:
        tag = str(float(sharpness)).replace(".", "p")
        print(f"  pred_entropy_norm_sharp{tag}: {mean(rows, f'pred_entropy_norm_sharp{tag}'):.6f}")
        print(f"  pred_top1_prob_sharp{tag}: {mean(rows, f'pred_top1_prob_sharp{tag}'):.6f}")
    print(f"  pair_match_rate: {rate(rows, 'match'):.6f}")
    print(f"  pred_correct_rate: {rate(rows, 'pred_correct'):.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True, type=Path)
    parser.add_argument(
        "--sharpness",
        default="2,3,5,7",
        help="Comma-separated sharpness values to simulate on pair_prob_matrix.",
    )
    args = parser.parse_args()

    sharpness_values = [float(part.strip()) for part in args.sharpness.split(",") if part.strip()]
    obj = json.loads(args.records.read_text(encoding="utf-8"))
    records = obj["records"] if isinstance(obj, dict) and "records" in obj else obj

    rows: List[Dict[str, Any]] = []
    by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("router_target_matrix") is None:
            continue
        pred = flatten_matrix(record["pair_prob_matrix"])
        target = flatten_matrix(record["router_target_matrix"])
        max_entropy = math.log(len(pred))
        pred_top = top_stats(pred)
        target_top = top_stats(target)
        row = {
            "task": str(record.get("task", "unknown")),
            "pred_entropy_norm": entropy(pred) / max_entropy,
            "target_entropy_norm": entropy(target) / max_entropy,
            "pred_top1_prob": pred_top["top1_prob"],
            "target_top1_prob": target_top["top1_prob"],
            "pred_top1_top2_prob_gap": pred_top["top1_top2_prob_gap"],
            "target_top1_top2_prob_gap": target_top["top1_top2_prob_gap"],
            "pred_top1_top2_logprob_gap": pred_top["top1_top2_logprob_gap"],
            "target_top1_top2_logprob_gap": target_top["top1_top2_logprob_gap"],
            "match": bool(record.get("match")),
            "pred_correct": bool(record.get("pred_correct")),
        }
        for sharpness in sharpness_values:
            tag = str(float(sharpness)).replace(".", "p")
            sharpened = sharpen_probs(pred, sharpness)
            sharpened_top = top_stats(sharpened)
            row[f"pred_entropy_norm_sharp{tag}"] = entropy(sharpened) / max_entropy
            row[f"pred_top1_prob_sharp{tag}"] = sharpened_top["top1_prob"]
        rows.append(row)
        by_task[row["task"]].append(row)

    print(f"file: {args.records}")
    print_group("ALL", rows, sharpness_values)
    for task in sorted(by_task):
        print_group(task, by_task[task], sharpness_values)


if __name__ == "__main__":
    main()
