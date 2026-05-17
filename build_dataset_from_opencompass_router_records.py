import argparse
import json
import os
from collections import defaultdict


def normalize_task_name(dataset_name: str) -> str:
    dataset_name = str(dataset_name)
    if dataset_name.startswith("race"):
        return "race"
    if dataset_name.startswith("sst2"):
        return "sst2"
    if dataset_name.startswith("medmcqa"):
        return "medmcqa"
    return dataset_name


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", required=True)
    parser.add_argument("--out_root", required=True)
    parser.add_argument(
        "--target",
        default="A",
        help="Dummy target used only for routing alignment cache construction.",
    )
    args = parser.parse_args()

    by_task = defaultdict(list)
    with open(args.records, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            prompt = record.get("prompt_text")
            if prompt is None:
                raise ValueError(
                    "OpenCompass records do not contain prompt_text. "
                    "Rerun OpenCompass after updating router_moe_llama_internal_compact_cached_joint.py."
                )
            task = normalize_task_name(record.get("dataset", "unknown"))
            by_task[task].append(
                {
                    "_opencompass_dataset": record.get("dataset"),
                    "_opencompass_sample_index": record.get("sample_index"),
                    "_opencompass_prompt_sha1": record.get("prompt_sha1"),
                    "text": prompt,
                    "source_text": prompt,
                    "target": str(args.target),
                    "label": task,
                }
            )

    os.makedirs(args.out_root, exist_ok=True)
    for task, rows in sorted(by_task.items()):
        task_dir = os.path.join(args.out_root, task)
        os.makedirs(task_dir, exist_ok=True)
        for split in ["train", "validation"]:
            path = os.path.join(task_dir, f"{split}.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[WRITE] {path} n={len(rows)}")


if __name__ == "__main__":
    main()
