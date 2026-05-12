import argparse
import json
import os
from typing import Dict, List, Optional

import torch


def parse_csv(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


def load_manifest(feature_root: str, split: str) -> Dict:
    path = os.path.join(feature_root, split, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_names(manifest: Dict, requested_tasks: Optional[List[str]], requested_experts: Optional[List[str]]):
    source_tasks = list(manifest.get("task_names") or [])
    source_experts = list(manifest.get("expert_names") or source_tasks)
    tasks = requested_tasks or source_tasks
    experts = requested_experts or source_experts

    missing_tasks = [name for name in tasks if name not in source_tasks]
    if missing_tasks:
        raise ValueError(f"Missing sample tasks in cache: {missing_tasks}. Available={source_tasks}")
    missing_experts = [name for name in experts if name not in source_experts]
    if missing_experts:
        raise ValueError(f"Missing experts in cache: {missing_experts}. Available={source_experts}")
    return source_tasks, source_experts, tasks, experts


def collect_split(
    feature_root: str,
    split: str,
    tasks: List[str],
    source_experts: List[str],
    experts: List[str],
):
    manifest = load_manifest(feature_root, split)
    split_dir = os.path.join(feature_root, split)
    expert_indices = torch.tensor([source_experts.index(name) for name in experts], dtype=torch.long)
    num_experts = len(experts)
    num_pairs = num_experts * num_experts

    stats = {
        task: {
            "n": 0,
            "pair_correct": torch.zeros(num_pairs, dtype=torch.long),
            "any_correct": 0,
            "self_correct": 0,
        }
        for task in tasks
    }

    for filename in manifest["files"]:
        payload = torch.load(os.path.join(split_dir, filename), map_location="cpu")
        for item in payload["items"]:
            task = str(item["task"])
            if task not in stats:
                continue
            correct_matrix = item.get("correct_matrix")
            if correct_matrix is None:
                raise ValueError(f"Item in {split}/{filename} has no correct_matrix")
            correct_matrix = correct_matrix.index_select(0, expert_indices).index_select(1, expert_indices).bool()
            flat_correct = correct_matrix.view(-1)

            task_stats = stats[task]
            task_stats["n"] += 1
            task_stats["pair_correct"] += flat_correct.to(torch.long)
            task_stats["any_correct"] += int(flat_correct.any().item())
            if task in experts:
                task_id = experts.index(task)
                task_stats["self_correct"] += int(correct_matrix[task_id, task_id].item())

    return stats


def pair_name(experts: List[str], pair_idx: int) -> str:
    num_experts = len(experts)
    return f"{experts[pair_idx // num_experts]}->{experts[pair_idx % num_experts]}"


def rate(count: int, total: int) -> float:
    return float(count) / max(int(total), 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--sample_task_names", type=str, default=None)
    parser.add_argument("--expert_names", type=str, default=None)
    args = parser.parse_args()

    train_manifest = load_manifest(args.feature_root, "train")
    source_tasks, source_experts, tasks, experts = resolve_names(
        train_manifest,
        parse_csv(args.sample_task_names),
        parse_csv(args.expert_names),
    )
    train_stats = collect_split(args.feature_root, "train", tasks, source_experts, experts)
    val_stats = collect_split(args.feature_root, "validation", tasks, source_experts, experts)

    total_val = 0
    total_self_correct = 0
    total_any_correct = 0
    total_train_selected_correct = 0
    total_val_oracle_correct = 0

    print(f"[INFO] tasks={tasks}")
    print(f"[INFO] experts={experts}")
    print("[TASK_BASELINE] one fixed pair per task")

    for task in tasks:
        train = train_stats[task]
        val = val_stats[task]
        n_train = int(train["n"])
        n_val = int(val["n"])
        if n_train == 0 or n_val == 0:
            print(f"[TASK][{task}] skipped n_train={n_train} n_val={n_val}")
            continue

        train_best_idx = int(train["pair_correct"].argmax().item())
        val_oracle_idx = int(val["pair_correct"].argmax().item())
        train_selected_val_correct = int(val["pair_correct"][train_best_idx].item())
        val_oracle_correct = int(val["pair_correct"][val_oracle_idx].item())

        total_val += n_val
        total_self_correct += int(val["self_correct"])
        total_any_correct += int(val["any_correct"])
        total_train_selected_correct += train_selected_val_correct
        total_val_oracle_correct += val_oracle_correct

        print(
            f"[TASK][{task}] "
            f"train_best_pair={pair_name(experts, train_best_idx)} "
            f"train_best_acc={rate(int(train['pair_correct'][train_best_idx].item()), n_train):.4f} "
            f"val_acc_by_train_best={rate(train_selected_val_correct, n_val):.4f} "
            f"val_oracle_pair={pair_name(experts, val_oracle_idx)} "
            f"val_oracle_acc={rate(val_oracle_correct, n_val):.4f} "
            f"val_self_acc={rate(int(val['self_correct']), n_val):.4f} "
            f"val_any_acc={rate(int(val['any_correct']), n_val):.4f}"
        )

    print(
        "[SUMMARY] "
        f"val_self_acc={rate(total_self_correct, total_val):.4f} "
        f"val_task_pair_train_selected_acc={rate(total_train_selected_correct, total_val):.4f} "
        f"val_task_pair_oracle_acc={rate(total_val_oracle_correct, total_val):.4f} "
        f"val_any_pair_acc={rate(total_any_correct, total_val):.4f} "
        f"n_val={total_val}"
    )


if __name__ == "__main__":
    main()
