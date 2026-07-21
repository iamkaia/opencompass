#!/usr/bin/env python3
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import torch


DEFAULT_CACHE_ROOTS = [
    "./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words",
    "./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words",
    "./0527_llama_cache_mrs_3expert_official_eval_aligned",
    "./0527_llama_cache_4other_3expert_official_eval_aligned",
    "./0528_llama_cache_arc_c_openbookqa_3expert_official_eval_aligned",
]


CHOICE_RE = re.compile(r"^[A-H]$")
BINARY_RE = re.compile(r"^(yes|no|true|false|positive|negative)$", re.IGNORECASE)
NUMERIC_RE = re.compile(r"^-?\d+(?:\.\d+)?$")
WHITESPACE_RE = re.compile(r"\s+")


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def iter_cache_items(cache_root: Path) -> Iterable[Tuple[str, List[str], Dict[str, Any]]]:
    for split in ("train", "validation"):
        manifest_path = cache_root / split / "manifest.json"
        if not manifest_path.exists():
            continue
        manifest = load_json(manifest_path)
        expert_names = [str(name) for name in manifest.get("expert_names", [])]
        for filename in manifest["files"]:
            payload = torch.load(cache_root / split / filename, map_location="cpu")
            for item in payload["items"]:
                yield split, expert_names, item


def flatten_prediction_matrix(matrix: Any) -> Iterable[str]:
    if matrix is None:
        return
    for first_row in matrix:
        for cell in first_row:
            if isinstance(cell, list):
                if len(cell) == 1 and not isinstance(cell[0], list):
                    yield str(cell[0])
                else:
                    yield str(cell)
            else:
                yield str(cell)


def iter_prediction_matrix_cells(
    matrix: Any,
    expert_names: List[str],
) -> Iterable[Tuple[str, str]]:
    if matrix is None:
        return
    for first_idx, first_row in enumerate(matrix):
        first_name = (
            expert_names[first_idx]
            if first_idx < len(expert_names)
            else str(first_idx)
        )
        for mid_idx, cell in enumerate(first_row):
            mid_name = (
                expert_names[mid_idx]
                if mid_idx < len(expert_names)
                else str(mid_idx)
            )
            pair_name = f"{first_name}->{mid_name}"
            if isinstance(cell, list):
                if len(cell) == 1 and not isinstance(cell[0], list):
                    yield pair_name, str(cell[0])
                else:
                    yield pair_name, str(cell)
            else:
                yield pair_name, str(cell)


def classify_prediction(text: str) -> str:
    value = text.strip()
    if not value:
        return "empty"
    if "\n" in value:
        return "multi_line"
    if CHOICE_RE.fullmatch(value):
        return "choice_label"
    if BINARY_RE.fullmatch(value):
        return "binary_label"
    if NUMERIC_RE.fullmatch(value):
        return "numeric"

    tokens = [tok for tok in WHITESPACE_RE.split(value) if tok]
    if len(tokens) == 1:
        if len(value) <= 8:
            return "single_token_short"
        return "single_token_long"
    if len(tokens) <= 3:
        return "short_phrase"
    return "long_generation"


def prediction_length(text: str) -> int:
    return len(text.strip())


def compact_preview(text: str, limit: int = 80) -> str:
    value = text.replace("\n", "\\n")
    if len(value) <= limit:
        return value
    return value[: limit - 3] + "..."


def parse_roots(raw: List[str]) -> List[Path]:
    return [Path(root) for root in raw]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Summarize cached router prediction_matrix outputs and classify them."
    )
    parser.add_argument(
        "--cache_roots",
        nargs="+",
        default=DEFAULT_CACHE_ROOTS,
        help="Cache roots to inspect. Defaults to the known 0527/0528 router caches.",
    )
    parser.add_argument("--top_k", type=int, default=30, help="How many exact predictions to print.")
    parser.add_argument(
        "--pair_top_k",
        type=int,
        default=20,
        help="How many exact predictions to keep and print for each task/pair.",
    )
    parser.add_argument(
        "--examples_per_category",
        type=int,
        default=5,
        help="How many example strings to show per category.",
    )
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path to write the full summary as JSON.",
    )
    args = parser.parse_args()

    cache_roots = parse_roots(args.cache_roots)
    exact_counter: Counter[str] = Counter()
    category_counter: Counter[str] = Counter()
    category_examples: Dict[str, List[str]] = defaultdict(list)
    task_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    task_exact_counter: Dict[str, Counter[str]] = defaultdict(Counter)
    task_length_sum: Dict[str, int] = defaultdict(int)
    task_length_count: Dict[str, int] = defaultdict(int)
    task_max_length: Dict[str, int] = defaultdict(int)
    task_pair_exact_counter: Dict[str, Dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    task_pair_category_counter: Dict[str, Dict[str, Counter[str]]] = defaultdict(
        lambda: defaultdict(Counter)
    )
    root_task_pair_exact_counter: Dict[str, Dict[str, Dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    root_task_pair_category_counter: Dict[str, Dict[str, Dict[str, Counter[str]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(Counter))
    )
    split_counter: Counter[str] = Counter()
    total_predictions = 0

    for cache_root in cache_roots:
        if not cache_root.exists():
            print(f"[SKIP] missing cache root: {cache_root}")
            continue
        print(f"[SCAN] {cache_root}")
        root_key = str(cache_root)
        for split, expert_names, item in iter_cache_items(cache_root):
            split_counter[split] += 1
            task = str(item.get("task", "unknown"))
            for pair_name, pred in iter_prediction_matrix_cells(
                item.get("prediction_matrix"), expert_names
            ):
                total_predictions += 1
                exact_counter[pred] += 1
                category = classify_prediction(pred)
                category_counter[category] += 1
                task_counter[task][category] += 1
                task_exact_counter[task][pred] += 1
                task_pair_exact_counter[task][pair_name][pred] += 1
                task_pair_category_counter[task][pair_name][category] += 1
                root_task_pair_exact_counter[root_key][task][pair_name][pred] += 1
                root_task_pair_category_counter[root_key][task][pair_name][category] += 1
                pred_len = prediction_length(pred)
                task_length_sum[task] += pred_len
                task_length_count[task] += 1
                if pred_len > task_max_length[task]:
                    task_max_length[task] = pred_len
                if len(category_examples[category]) < args.examples_per_category:
                    category_examples[category].append(pred)

    print("\n== Summary ==")
    print(f"total_predictions: {total_predictions}")
    print(f"splits_seen: {dict(split_counter)}")

    print("\n== Categories ==")
    for category, count in category_counter.most_common():
        examples = ", ".join(f"'{compact_preview(x)}'" for x in category_examples[category])
        print(f"{category}: {count}  examples=[{examples}]")

    print("\n== Top exact predictions ==")
    for value, count in exact_counter.most_common(args.top_k):
        print(f"{count:8d}  '{compact_preview(value)}'")

    print("\n== Per task ==")
    task_summary: Dict[str, Dict[str, Any]] = {}
    for task, counter in sorted(task_counter.items()):
        total_task = sum(counter.values())
        parts = ", ".join(f"{k}={v}" for k, v in counter.most_common())
        avg_len = (
            task_length_sum[task] / task_length_count[task]
            if task_length_count[task]
            else 0.0
        )
        top_preds = ", ".join(
            f"'{compact_preview(pred)}'({count})"
            for pred, count in task_exact_counter[task].most_common(5)
        )
        task_summary[task] = {
            "total": total_task,
            "avg_len": avg_len,
            "max_len": task_max_length[task],
            "categories": dict(counter),
            "top_predictions": [
                {"prediction": pred, "count": count}
                for pred, count in task_exact_counter[task].most_common(20)
            ],
        }
        print(
            f"{task}: total={total_task}, avg_len={avg_len:.2f}, max_len={task_max_length[task]}, "
            f"categories=[{parts}], top=[{top_preds}]"
        )

    print("\n== Per task / pair ==")
    task_pair_summary: Dict[str, Dict[str, Any]] = {}
    for task in sorted(task_pair_exact_counter):
        task_pair_summary[task] = {}
        for pair_name in sorted(task_pair_exact_counter[task]):
            pred_counter = task_pair_exact_counter[task][pair_name]
            category_counter_for_pair = task_pair_category_counter[task][pair_name]
            total_pair = sum(pred_counter.values())
            top_preds = ", ".join(
                f"'{compact_preview(pred)}'({count})"
                for pred, count in pred_counter.most_common(args.pair_top_k)
            )
            task_pair_summary[task][pair_name] = {
                "total": total_pair,
                "categories": dict(category_counter_for_pair),
                "top_predictions": [
                    {"prediction": pred, "count": count}
                    for pred, count in pred_counter.most_common(args.pair_top_k)
                ],
            }
            print(f"{task} {pair_name}: total={total_pair}, top=[{top_preds}]")

    cache_root_task_pair_summary: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for root_key in sorted(root_task_pair_exact_counter):
        cache_root_task_pair_summary[root_key] = {}
        for task in sorted(root_task_pair_exact_counter[root_key]):
            cache_root_task_pair_summary[root_key][task] = {}
            for pair_name in sorted(root_task_pair_exact_counter[root_key][task]):
                pred_counter = root_task_pair_exact_counter[root_key][task][pair_name]
                category_counter_for_pair = root_task_pair_category_counter[root_key][task][pair_name]
                cache_root_task_pair_summary[root_key][task][pair_name] = {
                    "total": sum(pred_counter.values()),
                    "categories": dict(category_counter_for_pair),
                    "top_predictions": [
                        {"prediction": pred, "count": count}
                        for pred, count in pred_counter.most_common(args.pair_top_k)
                    ],
                }

    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cache_roots": [str(path) for path in cache_roots],
            "total_predictions": total_predictions,
            "splits_seen": dict(split_counter),
            "categories": {
                category: {
                    "count": count,
                    "examples": category_examples[category][: args.examples_per_category],
                }
                for category, count in category_counter.most_common()
            },
            "top_exact_predictions": [
                {"prediction": value, "count": count}
                for value, count in exact_counter.most_common(args.top_k)
            ],
            "tasks": task_summary,
            "task_pair_predictions": task_pair_summary,
            "cache_root_task_pair_predictions": cache_root_task_pair_summary,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
        print(f"\n[SAVE] {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
