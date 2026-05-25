import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List

import torch


def load_items(cache_root: Path, split: str) -> List[Dict]:
    items: List[Dict] = []
    for chunk_path in sorted((cache_root / split).glob("chunk_*.pt")):
        raw = torch.load(chunk_path, map_location="cpu")
        if isinstance(raw, list):
            items.extend(raw)
            continue
        if isinstance(raw, dict) and isinstance(raw.get("items"), list):
            items.extend(raw["items"])
            continue
        if isinstance(raw, dict):
            count = len(next(iter(raw.values())))
            items.extend([{key: value[idx] for key, value in raw.items()} for idx in range(count)])
            continue
        raise TypeError(f"Unsupported chunk layout in {chunk_path}: {type(raw)!r}")
    return items


def sample_key(item: Dict) -> tuple:
    return (
        str(item.get("task", "")),
        str(item.get("target", "")),
        str(item.get("prompt_text", item.get("text", ""))),
    )


def router_target(item: Dict) -> torch.Tensor:
    loss_matrix = torch.as_tensor(item["loss_matrix"], dtype=torch.float32)
    correct = torch.as_tensor(item["correct_matrix"], dtype=torch.bool)
    flat_loss = loss_matrix.flatten()
    min_loss = flat_loss.min()
    scale = (flat_loss.max() - min_loss).clamp_min(1e-8)
    normalized = (flat_loss - min_loss) / scale
    logits = (-normalized).masked_fill(~correct.flatten(), -1e9)
    if not bool(correct.any()):
        return torch.zeros_like(flat_loss)
    return torch.softmax(logits, dim=0)


def compare_split(old_root: Path, new_root: Path, split: str) -> None:
    old_items = load_items(old_root, split)
    new_items = load_items(new_root, split)
    print(f"split={split} old_items={len(old_items)} new_items={len(new_items)}")
    if len(old_items) != len(new_items):
        print("  ERROR: item counts differ; positional comparison stopped")
        return

    stats = defaultdict(lambda: defaultdict(int))
    max_loss_diff = defaultdict(float)
    max_target_diff = defaultdict(float)
    mismatched_keys = 0
    for old_item, new_item in zip(old_items, new_items):
        task = str(old_item["task"])
        if sample_key(old_item) != sample_key(new_item):
            mismatched_keys += 1
            continue
        old_loss = torch.as_tensor(old_item["loss_matrix"], dtype=torch.float32)
        new_loss = torch.as_tensor(new_item["loss_matrix"], dtype=torch.float32)
        old_correct = torch.as_tensor(old_item["correct_matrix"], dtype=torch.bool)
        new_correct = torch.as_tensor(new_item["correct_matrix"], dtype=torch.bool)
        loss_diff = float((old_loss - new_loss).abs().max().item())
        target_diff = float((router_target(old_item) - router_target(new_item)).abs().max().item())
        stats[task]["total"] += 1
        stats[task]["loss_changed"] += int(loss_diff > 1e-6)
        stats[task]["correct_changed"] += int(bool((old_correct != new_correct).any().item()))
        stats[task]["lowest_pair_changed"] += int(int(old_loss.argmin()) != int(new_loss.argmin()))
        stats[task]["router_target_changed"] += int(target_diff > 1e-6)
        max_loss_diff[task] = max(max_loss_diff[task], loss_diff)
        max_target_diff[task] = max(max_target_diff[task], target_diff)

    print(f"  mismatched_sample_keys={mismatched_keys}")
    for task in sorted(stats):
        row = stats[task]
        total = row["total"]
        print(
            f"  task={task} n={total} "
            f"loss_changed={row['loss_changed']}/{total} "
            f"correct_changed={row['correct_changed']}/{total} "
            f"lowest_pair_changed={row['lowest_pair_changed']}/{total} "
            f"router_target_changed={row['router_target_changed']}/{total} "
            f"max_loss_abs_diff={max_loss_diff[task]:.6f} "
            f"max_target_abs_diff={max_target_diff[task]:.6f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_root", required=True, type=Path)
    parser.add_argument("--new_root", required=True, type=Path)
    args = parser.parse_args()
    for split in ("train", "validation"):
        compare_split(args.old_root, args.new_root, split)


if __name__ == "__main__":
    main()
