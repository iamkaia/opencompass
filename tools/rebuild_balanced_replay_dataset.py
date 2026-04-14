#!/usr/bin/env python3
import argparse
import os
import random
import shutil
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild a balanced replay dataset from an existing old5+new dataset root."
    )
    parser.add_argument(
        "--src_root",
        required=True,
        help="Source dataset root containing task subdirectories with train.jsonl/validation.jsonl.",
    )
    parser.add_argument(
        "--dst_root",
        required=True,
        help="Destination balanced replay dataset root to recreate.",
    )
    parser.add_argument(
        "--train_per_task",
        type=int,
        default=1000,
        help="Number of training samples to keep per task.",
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


def main():
    args = parse_args()
    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)
    rng = random.Random(args.seed)

    if not src_root.is_dir():
        raise FileNotFoundError(f"Missing source root: {src_root}")

    tasks = []
    skipped_dirs = []
    for p in sorted(src_root.iterdir()):
        if not p.is_dir():
            continue
        has_train = (p / "train.jsonl").is_file()
        has_val = (p / "validation.jsonl").is_file()
        if has_train and has_val:
            tasks.append(p.name)
        else:
            skipped_dirs.append(p.name)
    if not tasks:
        raise ValueError(f"No valid task directories found under {src_root}")

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True, exist_ok=True)

    print(
        f"[INFO] rebuilding balanced replay dataset from {src_root} -> {dst_root} "
        f"(train_per_task={args.train_per_task}, val_per_task={args.val_per_task}, seed={args.seed})"
    )
    if skipped_dirs:
        print(f"[INFO] skipped non-task directories: {', '.join(skipped_dirs)}")

    for task in tasks:
        for split, keep_n in [("train", args.train_per_task), ("validation", args.val_per_task)]:
            src_file = src_root / task / f"{split}.jsonl"
            if not src_file.is_file():
                raise FileNotFoundError(f"Missing dataset file: {src_file}")

            rows = read_nonempty_lines(src_file)
            rng.shuffle(rows)
            kept_rows = rows[: min(keep_n, len(rows))]

            dst_file = dst_root / task / f"{split}.jsonl"
            write_lines(dst_file, kept_rows)
            print(
                f"[SAVE] task={task} split={split} kept={len(kept_rows)} "
                f"from={len(rows)} -> {dst_file}"
            )

    print(f"[DONE] balanced replay dataset rebuilt at {dst_root}")


if __name__ == "__main__":
    main()
