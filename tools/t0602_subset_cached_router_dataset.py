#!/usr/bin/env python3
import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import torch


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_tasks(raw: str):
    return [x.strip() for x in str(raw).split(",") if x.strip()]


def subset_split(src_root: Path, dst_root: Path, split: str, tasks, per_task: int, chunk_size: int):
    src_dir = src_root / split
    dst_dir = dst_root / split
    manifest = load_json(src_dir / "manifest.json")
    selected = []
    counts = Counter()
    task_set = set(tasks)

    for filename in manifest["files"]:
        payload = torch.load(src_dir / filename, map_location="cpu")
        for item in payload["items"]:
            task = str(item.get("task"))
            if task not in task_set:
                continue
            if counts[task] >= per_task:
                continue
            selected.append(item)
            counts[task] += 1
        if all(counts[task] >= per_task for task in tasks):
            break

    missing = [task for task in tasks if counts[task] < per_task]
    if missing:
        raise RuntimeError(f"Not enough samples in {src_dir}: counts={dict(counts)} missing={missing}")

    dst_dir.mkdir(parents=True, exist_ok=True)
    files = []
    for chunk_idx, start in enumerate(range(0, len(selected), chunk_size)):
        filename = f"chunk_{chunk_idx:05d}.pt"
        torch.save({"items": selected[start : start + chunk_size]}, dst_dir / filename)
        files.append(filename)

    out_manifest = dict(manifest)
    out_manifest["num_items"] = len(selected)
    out_manifest["files"] = files
    out_manifest["subset_source_root"] = str(src_root)
    out_manifest["subset_tasks"] = list(tasks)
    out_manifest["subset_per_task"] = int(per_task)
    out_manifest["subset_counts"] = dict(counts)
    save_json(out_manifest, dst_dir / "manifest.json")
    return counts, len(selected), files


def main():
    parser = argparse.ArgumentParser(description="Create a per-task subset of an existing cached router dataset.")
    parser.add_argument("--src_root", type=Path, required=True)
    parser.add_argument("--dst_root", type=Path, required=True)
    parser.add_argument("--tasks", type=str, required=True)
    parser.add_argument("--per_task", type=int, default=10)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"dst_root already exists: {args.dst_root}")
        shutil.rmtree(args.dst_root)
    args.dst_root.mkdir(parents=True, exist_ok=True)

    if (args.src_root / "cache_config.json").exists():
        shutil.copy2(args.src_root / "cache_config.json", args.dst_root / "cache_config.json")

    tasks = parse_tasks(args.tasks)
    summary = {
        "src_root": str(args.src_root),
        "dst_root": str(args.dst_root),
        "tasks": tasks,
        "per_task": int(args.per_task),
        "splits": {},
    }
    for split in ["train", "validation"]:
        counts, total, files = subset_split(args.src_root, args.dst_root, split, tasks, args.per_task, args.chunk_size)
        summary["splits"][split] = {
            "counts": dict(counts),
            "num_items": total,
            "files": files,
        }
        print(f"[SUBSET_CACHE] split={split} total={total} counts={dict(counts)} out={args.dst_root / split}", flush=True)

    save_json(summary, args.dst_root / "subset_config.json")
    print(f"[DONE] subset cache out={args.dst_root}", flush=True)


if __name__ == "__main__":
    main()
