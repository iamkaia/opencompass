import argparse
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import torch


def parse_task_names(raw: Optional[str], fallback: Optional[Sequence[str]] = None) -> List[str]:
    if raw:
        tasks = [part.strip() for part in raw.split(",") if part.strip()]
        if tasks:
            return tasks
    if fallback:
        return list(fallback)
    raise ValueError("Failed to resolve task names")


def load_manifest(feature_root: str, split: str) -> Dict:
    path = os.path.join(feature_root, split, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def mean(values: List[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    sorted_values = sorted(float(v) for v in values)
    idx = int(round((len(sorted_values) - 1) * q))
    idx = max(0, min(idx, len(sorted_values) - 1))
    return float(sorted_values[idx])


def counter_rows(counter: Counter, denom: int, limit: int = 10) -> List[Dict]:
    return [
        {"name": name, "count": int(count), "rate": float(count / max(denom, 1))}
        for name, count in counter.most_common(limit)
    ]


def analyze_split(feature_root: str, split: str, task_names: Sequence[str]) -> Dict:
    manifest = load_manifest(feature_root, split)
    source_task_names = list(manifest.get("task_names") or manifest.get("expert_names") or [])
    if not source_task_names:
        raise ValueError(f"Missing task_names/expert_names in {feature_root}/{split}/manifest.json")

    missing = [name for name in task_names if name not in source_task_names]
    if missing:
        raise ValueError(f"Requested task_names={missing} not found in source_task_names={source_task_names}")

    selected_indices = [source_task_names.index(name) for name in task_names]
    selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)
    num_tasks = len(task_names)

    global_gold_pair_counter = Counter()
    global_gap_values: List[float] = []
    global_self_minus_oracle_values: List[float] = []
    task_counts = Counter()
    per_task_gold_pair_counter = defaultdict(Counter)
    per_task_gap_values = defaultdict(list)
    per_task_self_minus_oracle_values = defaultdict(list)
    per_task_pair_costs = {
        task_name: defaultdict(list) for task_name in task_names
    }

    num_samples = 0
    for fn in manifest["files"]:
        payload = torch.load(os.path.join(feature_root, split, fn), map_location="cpu")
        for item in payload["items"]:
            task_name = str(item["task"])
            if task_name not in task_names:
                continue

            loss_matrix = item["loss_matrix"]
            if list(loss_matrix.shape)[:2] != [len(source_task_names), len(source_task_names)]:
                raise ValueError(f"Unexpected loss_matrix shape={tuple(loss_matrix.shape)} for file={fn}")
            loss_matrix = loss_matrix.index_select(0, selected_index_tensor).index_select(1, selected_index_tensor).to(torch.float32)

            task_id = task_names.index(task_name)
            flat_loss = loss_matrix.view(-1)
            sorted_loss, sorted_idx = flat_loss.sort()
            best_pair = int(sorted_idx[0].item())
            best_cost = float(sorted_loss[0].item())
            second_cost = float(sorted_loss[1].item()) if flat_loss.numel() > 1 else best_cost
            gap = second_cost - best_cost
            self_cost = float(loss_matrix[task_id, task_id].item())
            self_minus_oracle = self_cost - best_cost

            first_idx = best_pair // num_tasks
            mid_idx = best_pair % num_tasks
            best_pair_name = f"{task_names[first_idx]}->{task_names[mid_idx]}"

            num_samples += 1
            task_counts[task_name] += 1
            global_gold_pair_counter[best_pair_name] += 1
            global_gap_values.append(gap)
            global_self_minus_oracle_values.append(self_minus_oracle)
            per_task_gold_pair_counter[task_name][best_pair_name] += 1
            per_task_gap_values[task_name].append(gap)
            per_task_self_minus_oracle_values[task_name].append(self_minus_oracle)

            for first_name_idx, first_name in enumerate(task_names):
                for mid_name_idx, mid_name in enumerate(task_names):
                    pair_name = f"{first_name}->{mid_name}"
                    per_task_pair_costs[task_name][pair_name].append(float(loss_matrix[first_name_idx, mid_name_idx].item()))

    per_task = []
    for task_name in task_names:
        count = int(task_counts[task_name])
        if count <= 0:
            continue
        avg_pair_cost_rows = []
        for pair_name, values in per_task_pair_costs[task_name].items():
            avg_pair_cost_rows.append(
                {
                    "name": pair_name,
                    "avg_cost": mean(values),
                }
            )
        avg_pair_cost_rows.sort(key=lambda row: row["avg_cost"])
        per_task.append(
            {
                "task": task_name,
                "count": count,
                "avg_gap": mean(per_task_gap_values[task_name]),
                "p50_gap": quantile(per_task_gap_values[task_name], 0.5),
                "p90_gap": quantile(per_task_gap_values[task_name], 0.9),
                "avg_self_minus_oracle": mean(per_task_self_minus_oracle_values[task_name]),
                "top_gold_pairs": counter_rows(per_task_gold_pair_counter[task_name], count, limit=10),
                "lowest_avg_cost_pairs": avg_pair_cost_rows[:10],
            }
        )

    return {
        "split": split,
        "feature_root": feature_root,
        "loss_normalization": str(manifest.get("loss_normalization") or "none"),
        "loss_normalization_scope": str(manifest.get("loss_normalization_scope") or "raw"),
        "task_names": list(task_names),
        "num_samples": num_samples,
        "avg_gap": mean(global_gap_values),
        "p50_gap": quantile(global_gap_values, 0.5),
        "p90_gap": quantile(global_gap_values, 0.9),
        "avg_self_minus_oracle": mean(global_self_minus_oracle_values),
        "top_gold_pairs": counter_rows(global_gold_pair_counter, num_samples, limit=15),
        "per_task": per_task,
    }


def print_summary(summary: Dict):
    print(
        f"[ANALYZE][{summary['split']}] normalization={summary.get('loss_normalization', 'none')} "
        f"scope={summary.get('loss_normalization_scope', 'raw')} "
        f"num_samples={summary['num_samples']} "
        f"avg_gap={summary['avg_gap']:.4f} "
        f"p50_gap={summary['p50_gap']:.4f} "
        f"p90_gap={summary['p90_gap']:.4f} "
        f"avg_self_minus_oracle={summary['avg_self_minus_oracle']:.4f}"
    )
    top_gold = ", ".join(
        f"{row['name']}:{row['rate']:.2%}" for row in summary.get("top_gold_pairs", [])[:10]
    )
    print(f"[ANALYZE][{summary['split']}] top_gold_pairs={top_gold}")
    for row in summary.get("per_task", []):
        top_gold_name = row["top_gold_pairs"][0]["name"] if row.get("top_gold_pairs") else "-"
        low_cost = ", ".join(
            f"{pair['name']}:{pair['avg_cost']:.4f}" for pair in row.get("lowest_avg_cost_pairs", [])[:5]
        )
        print(
            f"[ANALYZE][{summary['split']}][{row['task']}] n={row['count']} "
            f"avg_gap={row['avg_gap']:.4f} "
            f"p50_gap={row['p50_gap']:.4f} "
            f"p90_gap={row['p90_gap']:.4f} "
            f"avg_self_minus_oracle={row['avg_self_minus_oracle']:.4f} "
            f"top_gold_pair={top_gold_name}"
        )
        print(f"[ANALYZE][{summary['split']}][{row['task']}] lowest_avg_cost_pairs={low_cost}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="comma-separated splits to analyze",
    )
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    requested_splits = [part.strip() for part in str(args.splits).split(",") if part.strip()]
    train_manifest = load_manifest(args.feature_root, requested_splits[0])
    fallback_tasks = train_manifest.get("task_names") or train_manifest.get("expert_names")
    task_names = parse_task_names(args.task_names, fallback=fallback_tasks)

    all_summaries = []
    for split in requested_splits:
        summary = analyze_split(args.feature_root, split, task_names)
        print_summary(summary)
        all_summaries.append(summary)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
