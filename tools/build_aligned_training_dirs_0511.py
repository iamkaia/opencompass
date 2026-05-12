import argparse
import hashlib
import json
import os
import random
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def discover_tasks(input_roots: Sequence[str], requested_tasks: Optional[Sequence[str]]) -> List[str]:
    if requested_tasks:
        tasks = [task.strip() for task in requested_tasks if task.strip()]
    else:
        task_set = set()
        for input_root in input_roots:
            task_set.update(
                name
                for name in os.listdir(input_root)
                if os.path.isdir(os.path.join(input_root, name))
            )
        tasks = sorted(task_set)
    if not tasks:
        raise ValueError("No tasks found.")
    return tasks


def sample_id(task: str, split: str, source_index: int, row: Dict) -> str:
    payload = json.dumps(
        {
            "task": task,
            "split": split,
            "source_index": source_index,
            "text": row.get("text"),
            "target": row.get("target") or row.get("answer") or row.get("output"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{task}:{split}:{source_index}:{digest}"


def canonicalize_row(task: str, split: str, source_index: int, row: Dict) -> Dict:
    text = row.get("source_text") or row.get("text") or row.get("instruction")
    target = row.get("target") or row.get("answer") or row.get("output")
    label = row.get("label") or row.get("task") or task
    if text is None or target is None:
        raise ValueError(
            f"Invalid row task={task} split={split} source_index={source_index}: "
            f"keys={list(row.keys())}"
        )
    return {
        "_sample_id": sample_id(task, split, source_index, row),
        "_source_index": source_index,
        "target": str(target),
        "text": str(text),
        "label": str(label),
    }


def indexed_rows(task: str, split: str, rows: Sequence[Dict]) -> List[Dict]:
    return [canonicalize_row(task, split, idx, row) for idx, row in enumerate(rows)]


def select_rows(
    rows: Sequence[Dict],
    count: Optional[int],
    rng: random.Random,
    excluded_ids: Optional[set] = None,
) -> List[Dict]:
    excluded_ids = excluded_ids or set()
    candidates = [row for row in rows if row["_sample_id"] not in excluded_ids]
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    if count is None:
        return shuffled
    return shuffled[: min(count, len(shuffled))]


def top_up_rows(
    selected: Sequence[Dict],
    candidates: Sequence[Dict],
    count: int,
    rng: random.Random,
    excluded_ids: Optional[set] = None,
) -> List[Dict]:
    output = list(selected)
    if len(output) >= count:
        return output[:count]
    blocked = set(excluded_ids or set())
    blocked.update(row["_sample_id"] for row in output)
    needed = count - len(output)
    output.extend(select_rows(candidates, needed, rng, excluded_ids=blocked))
    return output


def split_paths(task_dir: str) -> Dict[str, str]:
    paths = {}
    for split in ("train", "validation", "val", "test"):
        path = os.path.join(task_dir, f"{split}.jsonl")
        if os.path.exists(path):
            paths[split] = path
    return paths


def classifier_row(row: Dict) -> Dict:
    return {
        "_sample_id": row["_sample_id"],
        "target": row["target"],
        "text": row["text"],
        "label": row["label"],
    }


def router_row(row: Dict) -> Dict:
    return {
        "_sample_id": row["_sample_id"],
        "target": row["target"],
        "text": row["text"],
        "label": row["label"],
        "source_text": row["text"],
    }


def expert_sft_row(row: Dict) -> Dict:
    return {
        "_sample_id": row["_sample_id"],
        "instruction": row["text"],
        "input": "",
        "output": row["target"],
        "label": row["label"],
    }


def write_all_outputs(
    roots: Dict[str, str],
    task: str,
    split: str,
    rows: Sequence[Dict],
) -> None:
    converters = {
        "classifier": classifier_row,
        "router": router_row,
        "expert_sft": expert_sft_row,
    }
    for name, root in roots.items():
        out_path = os.path.join(root, task, f"{split}.jsonl")
        write_jsonl(out_path, (converters[name](row) for row in rows))


def assert_alignment(roots: Dict[str, str], task: str, split: str) -> None:
    ids_by_root = {}
    for name, root in roots.items():
        path = os.path.join(root, task, f"{split}.jsonl")
        ids_by_root[name] = [row["_sample_id"] for row in read_jsonl(path)]
    first_name = next(iter(ids_by_root))
    expected = ids_by_root[first_name]
    for name, ids in ids_by_root.items():
        if ids != expected:
            raise AssertionError(f"Sample mismatch task={task} split={split}: {first_name} vs {name}")


def parse_tasks(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    return [part.strip() for part in raw.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build aligned classifier/router/expert-SFT dataset roots from one canonical jsonl root."
        )
    )
    parser.add_argument("--input_root", type=str, default="datasets_classifier")
    parser.add_argument(
        "--input_roots",
        type=str,
        default=None,
        help="Comma-separated canonical roots. The first root containing a task is used.",
    )
    parser.add_argument("--classifier_root", type=str, default="datasets_classifier_0511")
    parser.add_argument("--router_root", type=str, default="router_train_datasets_0511")
    parser.add_argument("--expert_sft_root", type=str, default="expert_sft_datasets_0511")
    parser.add_argument("--tasks", type=str, default=None)
    parser.add_argument(
        "--val_from_train_tasks",
        type=str,
        default=None,
        help="Comma-separated task names whose validation split should be sampled from train instead.",
    )
    parser.add_argument(
        "--skip_test_tasks",
        type=str,
        default=None,
        help="Comma-separated task names whose test split should not be written.",
    )
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--val_samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=511)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output roots.",
    )
    args = parser.parse_args()

    roots = {
        "classifier": args.classifier_root,
        "router": args.router_root,
        "expert_sft": args.expert_sft_root,
    }
    for root in roots.values():
        if os.path.exists(root) and not args.overwrite:
            raise FileExistsError(f"Output root exists: {root}. Use --overwrite to replace files in it.")
        os.makedirs(root, exist_ok=True)

    input_roots = parse_tasks(args.input_roots) if args.input_roots else [args.input_root]
    tasks = discover_tasks(input_roots, parse_tasks(args.tasks))
    val_from_train_tasks = set(parse_tasks(args.val_from_train_tasks) or [])
    skip_test_tasks = set(parse_tasks(args.skip_test_tasks) or [])
    summary = {
        "input_roots": input_roots,
        "output_roots": roots,
        "tasks": {},
        "train_samples_per_task": args.train_samples,
        "val_samples_per_task": args.val_samples,
        "seed": args.seed,
        "note": "All output roots share identical _sample_id order for each task/split.",
    }

    for task in tasks:
        task_dir = None
        task_source_root = None
        for input_root in input_roots:
            candidate = os.path.join(input_root, task)
            if os.path.isdir(candidate):
                task_dir = candidate
                task_source_root = input_root
                break
        if task_dir is None:
            raise FileNotFoundError(f"Could not find task={task} under input_roots={input_roots}")
        paths = split_paths(task_dir)
        if "train" not in paths:
            raise FileNotFoundError(f"Missing train split: {os.path.join(task_dir, 'train.jsonl')}")

        train_all = indexed_rows(task, "train", read_jsonl(paths["train"]))
        val_source_name = "validation" if "validation" in paths else "val" if "val" in paths else None
        val_all = indexed_rows(task, val_source_name, read_jsonl(paths[val_source_name])) if val_source_name else []
        test_all = indexed_rows(task, "test", read_jsonl(paths["test"])) if "test" in paths else []

        train_rng = random.Random(args.seed + 1009 * (tasks.index(task) + 1))
        val_rng = random.Random(args.seed + 2003 * (tasks.index(task) + 1))
        train_rows = select_rows(train_all, args.train_samples, train_rng)

        if task in val_from_train_tasks:
            train_ids = {row["_sample_id"] for row in train_rows}
            val_rows = select_rows(train_all, args.val_samples, val_rng, excluded_ids=train_ids)
            val_source = "train_fallback_excluding_sampled_train"
        elif val_all:
            val_rows = select_rows(val_all, args.val_samples, val_rng)
            val_source = val_source_name
            if len(val_rows) < args.val_samples:
                train_ids = {row["_sample_id"] for row in train_rows}
                val_rows = top_up_rows(val_rows, train_all, args.val_samples, val_rng, excluded_ids=train_ids)
                val_source = f"{val_source_name}_plus_train_fallback_excluding_sampled_train"
        else:
            train_ids = {row["_sample_id"] for row in train_rows}
            val_rows = select_rows(train_all, args.val_samples, val_rng, excluded_ids=train_ids)
            val_source = "train_fallback_excluding_sampled_train"

        splits: List[Tuple[str, Sequence[Dict]]] = [
            ("train", train_rows),
            ("validation", val_rows),
        ]
        if test_all and task not in skip_test_tasks:
            splits.append(("test", test_all))

        for split, rows in splits:
            write_all_outputs(roots, task, split, rows)
            assert_alignment(roots, task, split)

        summary["tasks"][task] = {
            "source_root": task_source_root,
            "source_train_rows": len(train_all),
            "source_validation_rows": len(val_all),
            "source_validation_split": val_source,
            "source_test_rows": len(test_all),
            "written_train_rows": len(train_rows),
            "written_validation_rows": len(val_rows),
            "written_test_rows": len(test_all) if task not in skip_test_tasks else 0,
            "train_sample_ids": [row["_sample_id"] for row in train_rows[:5]],
            "validation_sample_ids": [row["_sample_id"] for row in val_rows[:5]],
        }

    for root in roots.values():
        with open(os.path.join(root, "alignment_meta.json"), "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
