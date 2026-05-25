import argparse
from collections import Counter, defaultdict
import json
import os

import torch


def _normalize_items(obj):
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "items" in obj and isinstance(obj["items"], list):
            return obj["items"]
        keys = list(obj.keys())
        if not keys:
            return []
        first_value = obj[keys[0]]
        if isinstance(first_value, list):
            num_items = len(first_value)
            normalized = []
            for idx in range(num_items):
                item = {}
                for key, value in obj.items():
                    item[key] = value[idx]
                normalized.append(item)
            return normalized
    raise TypeError(f"Unsupported chunk object type: {type(obj)!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--chunk", type=str, default="chunk_00000.pt")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--show_matrices", action="store_true")
    parser.add_argument("--show_pair_ranking", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    manifest_path = os.path.join(args.cache_root, args.split, "manifest.json")
    chunk_path = os.path.join(args.cache_root, args.split, args.chunk)

    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    raw_items = torch.load(chunk_path, map_location="cpu")
    items = _normalize_items(raw_items)

    expert_names = list(manifest["expert_names"])
    print(f"manifest: {manifest_path}")
    print(f"chunk: {chunk_path}")
    print(f"score_mode: {manifest.get('score_mode')}")
    print(f"expert_names: {expert_names}")
    print(f"loaded_items: {len(items)}")

    summary = defaultdict(
        lambda: {
            "total": 0,
            "self_pair": 0,
            "self_first": 0,
            "self_mid": 0,
            "self_top3": 0,
            "self_top5": 0,
            "self_correct": 0,
            "oracle_correct": 0,
            "gap_sum": 0.0,
            "gap_abs_sum": 0.0,
            "oracle_pair_counter": Counter(),
        }
    )
    overall = {
        "total": 0,
        "self_pair": 0,
        "self_first": 0,
        "self_mid": 0,
        "self_top3": 0,
        "self_top5": 0,
        "self_correct": 0,
        "oracle_correct": 0,
        "gap_sum": 0.0,
        "gap_abs_sum": 0.0,
    }

    for item in items:
        task = str(item["task"])
        if task not in expert_names:
            continue
        self_idx = expert_names.index(task)
        first = int(item["first_label"])
        mid = int(item["mid_label"])
        flat_loss = item["loss_matrix"].view(-1)
        sorted_pair_idx = torch.argsort(flat_loss, dim=0)
        self_pair_idx = self_idx * len(expert_names) + self_idx
        oracle_loss = float(item["loss_matrix"][first][mid])
        self_loss = float(item["loss_matrix"][self_idx][self_idx])
        gap = self_loss - oracle_loss
        summary[task]["total"] += 1
        summary[task]["self_pair"] += int(first == self_idx and mid == self_idx)
        summary[task]["self_first"] += int(first == self_idx)
        summary[task]["self_mid"] += int(mid == self_idx)
        summary[task]["self_top3"] += int(bool((sorted_pair_idx[:3] == self_pair_idx).any().item()))
        summary[task]["self_top5"] += int(bool((sorted_pair_idx[:5] == self_pair_idx).any().item()))
        summary[task]["self_correct"] += int(bool(item["correct_matrix"][self_idx][self_idx]))
        summary[task]["oracle_correct"] += int(bool(item["correct_matrix"][first][mid]))
        summary[task]["gap_sum"] += gap
        summary[task]["gap_abs_sum"] += abs(gap)
        summary[task]["oracle_pair_counter"][(expert_names[first], expert_names[mid])] += 1
        overall["total"] += 1
        overall["self_pair"] += int(first == self_idx and mid == self_idx)
        overall["self_first"] += int(first == self_idx)
        overall["self_mid"] += int(mid == self_idx)
        overall["self_top3"] += int(bool((sorted_pair_idx[:3] == self_pair_idx).any().item()))
        overall["self_top5"] += int(bool((sorted_pair_idx[:5] == self_pair_idx).any().item()))
        overall["self_correct"] += int(bool(item["correct_matrix"][self_idx][self_idx]))
        overall["oracle_correct"] += int(bool(item["correct_matrix"][first][mid]))
        overall["gap_sum"] += gap
        overall["gap_abs_sum"] += abs(gap)

    print("oracle=self summary:")
    for task_name in expert_names:
        stats = summary.get(task_name)
        if not stats or stats["total"] == 0:
            continue
        total = stats["total"]
        print(
            f"  task={task_name} total={total} "
            f"self_pair_acc={stats['self_pair'] / total:.4f} "
            f"self_first_acc={stats['self_first'] / total:.4f} "
            f"self_mid_acc={stats['self_mid'] / total:.4f} "
            f"self_top3_acc={stats['self_top3'] / total:.4f} "
            f"self_top5_acc={stats['self_top5'] / total:.4f} "
            f"self_answer_acc={stats['self_correct'] / total:.4f} "
            f"oracle_answer_acc={stats['oracle_correct'] / total:.4f} "
            f"avg_gap={stats['gap_sum'] / total:.6f} "
            f"avg_abs_gap={stats['gap_abs_sum'] / total:.6f}"
        )
        top_pairs = stats["oracle_pair_counter"].most_common(5)
        if top_pairs:
            formatted = ", ".join(
                f"{first}->{mid}:{count}/{total}"
                for (first, mid), count in top_pairs
            )
            print(f"    top_oracle_pairs: {formatted}")
    if overall["total"] > 0:
        total = overall["total"]
        print(
            f"  overall total={total} "
            f"self_pair_acc={overall['self_pair'] / total:.4f} "
            f"self_first_acc={overall['self_first'] / total:.4f} "
            f"self_mid_acc={overall['self_mid'] / total:.4f} "
            f"self_top3_acc={overall['self_top3'] / total:.4f} "
            f"self_top5_acc={overall['self_top5'] / total:.4f} "
            f"self_answer_acc={overall['self_correct'] / total:.4f} "
            f"oracle_answer_acc={overall['oracle_correct'] / total:.4f} "
            f"avg_gap={overall['gap_sum'] / total:.6f} "
            f"avg_abs_gap={overall['gap_abs_sum'] / total:.6f}"
        )

    if args.summary_only:
        return

    filtered_items = items
    if args.task is not None:
        filtered_items = [item for item in items if str(item["task"]) == str(args.task)]

    for idx, item in enumerate(filtered_items[: max(int(args.limit), 0)]):
        first = int(item["first_label"])
        mid = int(item["mid_label"])
        task = str(item["task"])
        self_idx = expert_names.index(task) if task in expert_names else None
        oracle_loss = float(item["loss_matrix"][first][mid])
        self_loss = float(item["loss_matrix"][self_idx][self_idx]) if self_idx is not None else None
        print("=" * 80)
        print(f"sample_idx: {idx}")
        print(f"task: {task}")
        print(f"target: {item['target']}")
        print(f"oracle_pair: {expert_names[first]} -> {expert_names[mid]}")
        print(f"pair_label: {item['pair_label']}")
        print(f"first_label: {first}")
        print(f"mid_label: {mid}")
        print(f"oracle_prediction: {item['prediction_matrix'][first][mid]}")
        print(f"oracle_correct: {bool(item['correct_matrix'][first][mid])}")
        print(f"oracle_loss: {oracle_loss:.6f}")
        if self_idx is not None:
            print(f"self_pair: {expert_names[self_idx]} -> {expert_names[self_idx]}")
            print(f"self_prediction: {item['prediction_matrix'][self_idx][self_idx]}")
            print(f"self_correct: {bool(item['correct_matrix'][self_idx][self_idx])}")
            print(f"self_loss: {self_loss:.6f}")
            print(f"gap_self_minus_oracle: {self_loss - oracle_loss:.6f}")
        else:
            print("self_pair: N/A (task not in expert_names)")
        if args.show_matrices:
            print(f"loss_matrix shape: {tuple(item['loss_matrix'].shape)}")
            print(item["loss_matrix"])
            print(f"correct_matrix shape: {tuple(item['correct_matrix'].shape)}")
            print(item["correct_matrix"])
            print("prediction_matrix:")
            for row in item["prediction_matrix"]:
                print(row)
        if args.show_pair_ranking:
            print("pair_ranking_by_loss:")
            ranked = []
            for row_idx, first_name in enumerate(expert_names):
                for col_idx, mid_name in enumerate(expert_names):
                    ranked.append(
                        (
                            float(item["loss_matrix"][row_idx][col_idx]),
                            first_name,
                            mid_name,
                            bool(item["correct_matrix"][row_idx][col_idx]),
                            item["prediction_matrix"][row_idx][col_idx],
                        )
                    )
            ranked.sort(key=lambda x: x[0])
            for rank, (pair_loss, first_name, mid_name, is_correct, prediction) in enumerate(ranked, start=1):
                print(
                    f"  rank={rank:02d} pair={first_name}->{mid_name} "
                    f"loss={pair_loss:.6f} correct={is_correct} pred={prediction}"
                )


if __name__ == "__main__":
    main()
