#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a small eval-only dataset root with sampled validation.jsonl files."
    )
    parser.add_argument(
        "--base_root",
        required=True,
        help="Root containing the original 5 tasks, e.g. ./router_train_datasets_5org",
    )
    parser.add_argument(
        "--extra_roots",
        default="",
        help="Comma-separated extra dataset roots such as ./router_train_datasets_only_piqa,./router_train_datasets_only_copa",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        help="Destination eval-only dataset root to recreate.",
    )
    parser.add_argument(
        "--val_per_task",
        type=int,
        default=200,
        help="Number of validation samples to keep per task.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for sampling.",
    )
    return parser.parse_args()


def read_nonempty_lines(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return [line for line in f if line.strip()]


def write_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.writelines(lines)


def collect_task_dirs(root: Path):
    if not root.is_dir():
        raise FileNotFoundError(f"Missing dataset root: {root}")

    task_map = {}
    skipped = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        val_file = p / "validation.jsonl"
        if val_file.is_file():
            task_map[p.name] = p
        else:
            skipped.append(p.name)
    return task_map, skipped


def main():
    args = parse_args()
    rng = random.Random(args.seed)
    dst_root = Path(args.dst_root)

    source_roots = [Path(args.base_root)]
    if args.extra_roots.strip():
        source_roots.extend(Path(x.strip()) for x in args.extra_roots.split(",") if x.strip())

    merged_task_dirs = {}
    for root in source_roots:
        task_map, skipped = collect_task_dirs(root)
        print(f"[INFO] scanning {root}")
        if skipped:
            print(f"[INFO] skipped non-task directories under {root}: {', '.join(skipped)}")
        for task, task_dir in task_map.items():
            merged_task_dirs[task] = task_dir

    if not merged_task_dirs:
        raise ValueError("No task directories with validation.jsonl found.")

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] building small eval dataset at {dst_root} "
        f"(val_per_task={args.val_per_task}, seed={args.seed})"
    )

    for task in sorted(merged_task_dirs):
        src_file = merged_task_dirs[task] / "validation.jsonl"
        rows = read_nonempty_lines(src_file)
        rng.shuffle(rows)
        kept_rows = rows[: min(args.val_per_task, len(rows))]
        dst_file = dst_root / task / "validation.jsonl"
        write_lines(dst_file, kept_rows)
        print(
            f"[SAVE] task={task} kept={len(kept_rows)} from={len(rows)} -> {dst_file}"
        )

    print(f"[DONE] small eval dataset built at {dst_root}")


if __name__ == "__main__":
    main()
