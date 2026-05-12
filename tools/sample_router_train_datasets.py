import argparse
import json
import os
import random
import shutil
from typing import Dict, List


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: List[Dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_tasks(data_root: str) -> List[str]:
    return [
        name
        for name in sorted(os.listdir(data_root))
        if os.path.isdir(os.path.join(data_root, name))
    ]


def main():
    parser = argparse.ArgumentParser(
        description="Sample a small router dataset root while keeping task/split jsonl layout."
    )
    parser.add_argument("--input_root", type=str, default="router_train_datasets_5org")
    parser.add_argument("--output_root", type=str, default="router_datasets_5org_train25")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--copy_other_splits",
        action="store_true",
        help="Copy non-sampled split files, such as validation/test, into the output root.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.input_root):
        raise FileNotFoundError(f"Missing input_root: {args.input_root}")

    os.makedirs(args.output_root, exist_ok=True)
    rng = random.Random(args.seed)
    summary = {
        "input_root": args.input_root,
        "output_root": args.output_root,
        "split": args.split,
        "num_samples_per_task": args.num_samples,
        "seed": args.seed,
        "tasks": {},
    }

    for task in discover_tasks(args.input_root):
        src_task_dir = os.path.join(args.input_root, task)
        dst_task_dir = os.path.join(args.output_root, task)
        os.makedirs(dst_task_dir, exist_ok=True)

        src_split_path = os.path.join(src_task_dir, f"{args.split}.jsonl")
        if not os.path.exists(src_split_path):
            raise FileNotFoundError(f"Missing split file: {src_split_path}")

        rows = read_jsonl(src_split_path)
        sampled = list(rows)
        rng.shuffle(sampled)
        sampled = sampled[: min(args.num_samples, len(sampled))]
        write_jsonl(os.path.join(dst_task_dir, f"{args.split}.jsonl"), sampled)

        if args.copy_other_splits:
            for filename in sorted(os.listdir(src_task_dir)):
                if not filename.endswith(".jsonl") or filename == f"{args.split}.jsonl":
                    continue
                shutil.copy2(os.path.join(src_task_dir, filename), os.path.join(dst_task_dir, filename))

        summary["tasks"][task] = {
            "source_rows": len(rows),
            "sampled_rows": len(sampled),
            "output_file": os.path.join(dst_task_dir, f"{args.split}.jsonl"),
        }

    with open(os.path.join(args.output_root, "sample_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
