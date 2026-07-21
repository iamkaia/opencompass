import argparse
import json
from pathlib import Path


def read_ids(path: Path) -> set[str]:
    with path.open("r", encoding="utf-8") as f:
        return {json.loads(line)["_sample_id"] for line in f if line.strip()}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify that a router dataset is sourced from raw train and is disjoint from OpenCompass eval."
    )
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()
    root = Path(args.data_root)
    with (root / "alignment_meta.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    failures = []
    for task, task_info in metadata["tasks"].items():
        train_ids = read_ids(root / task / "train.jsonl")
        val_ids = read_ids(root / task / "validation.jsonl")
        if train_ids & val_ids:
            failures.append(f"{task}: generated train/validation rows overlap")
        if len(train_ids) != metadata["train_samples_per_task"]:
            failures.append(f"{task}: incorrect written train count")
        if len(val_ids) != metadata["val_samples_per_task"]:
            failures.append(f"{task}: incorrect written validation count")
        for abbr, source in task_info["sources_by_abbr"].items():
            router = source["router_source"]
            evaluation = source["opencompass_eval_source"]
            check = source["overlap_check"]
            if router["raw_split"] != "train":
                failures.append(f"{task}/{abbr}: router source is not raw train")
            if not check["raw_split_disjoint"] or not check["resolved_path_disjoint"]:
                failures.append(f"{task}/{abbr}: router/eval provenance overlaps")
            print(
                f"{task}/{abbr}: router={router['raw_split']}:{router['row_count']} "
                f"eval={evaluation['raw_split']}:{evaluation['row_count']} "
                f"overlap={check['status']}"
            )
    if failures:
        raise SystemExit("\n".join(failures))
    print(f"PASS: {len(metadata['tasks'])} tasks use raw train and are disjoint from OpenCompass eval sources.")


if __name__ == "__main__":
    main()
