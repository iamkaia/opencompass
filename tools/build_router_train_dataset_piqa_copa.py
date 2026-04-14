#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Iterable, List

from datasets import DatasetDict, load_dataset


def write_jsonl(path: str, rows: Iterable[Dict]):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wrap_inst(source_text: str) -> str:
    return f"<s>[INST] {source_text} [/INST]"


def piqa_row_to_router(ex: Dict) -> Dict:
    label = ex["label"]
    if isinstance(label, str):
        label = int(label)
    target = "AB"[label]
    source_text = f'{ex["goal"]}\nA. {ex["sol1"]}\nB. {ex["sol2"]}\nAnswer:'
    return {
        "target": target,
        "text": wrap_inst(source_text),
        "label": "piqa",
        "source_text": source_text,
    }


def copa_row_to_router(ex: Dict) -> Dict:
    label = ex["label"]
    if isinstance(label, str):
        label = int(label)
    target = "AB"[label]
    source_text = (
        f'{ex["premise"]}\n'
        f'Question: Which may be the {ex["question"]}?\n'
        f'A. {ex["choice1"]}\n'
        f'B. {ex["choice2"]}\n'
        f'Answer:'
    )
    return {
        "target": target,
        "text": wrap_inst(source_text),
        "label": "copa",
        "source_text": source_text,
    }


def save_split(ds, out_path: str, fn):
    rows = [fn(ex) for ex in ds]
    write_jsonl(out_path, rows)
    print(f"[SAVE] {out_path} rows={len(rows)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_root",
        type=str,
        default="router_train_datasets",
        help="output root, e.g. router_train_datasets",
    )
    parser.add_argument(
        "--piqa_source",
        type=str,
        default="baber/piqa",
        help="HF dataset id for PIQA. Default uses a mirror with train/validation/test labels.",
    )
    parser.add_argument(
        "--skip_piqa",
        action="store_true",
    )
    parser.add_argument(
        "--skip_copa",
        action="store_true",
    )
    args = parser.parse_args()

    meta = {
        "piqa_source": None,
        "copa_source": None,
    }

    if not args.skip_piqa:
        print(f"[LOAD] PIQA from {args.piqa_source}")
        piqa = load_dataset(args.piqa_source)
        if not isinstance(piqa, DatasetDict):
            raise TypeError(f"Expected DatasetDict for PIQA, got {type(piqa)}")
        for split in ["train", "validation", "test"]:
            if split not in piqa:
                raise KeyError(f"PIQA source {args.piqa_source} missing split={split}")
            save_split(
                piqa[split],
                os.path.join(args.out_root, "piqa", f"{split}.jsonl"),
                piqa_row_to_router,
            )
        meta["piqa_source"] = args.piqa_source

    if not args.skip_copa:
        print("[LOAD] COPA from super_glue/copa")
        copa = load_dataset("super_glue", "copa")
        if not isinstance(copa, DatasetDict):
            raise TypeError(f"Expected DatasetDict for COPA, got {type(copa)}")
        for split in ["train", "validation", "test"]:
            if split not in copa:
                raise KeyError(f"COPA dataset missing split={split}")
            split_ds = copa[split]
            if "label" not in split_ds.column_names:
                raise ValueError(
                    "COPA test split does not contain labels in this source, "
                    "so it cannot be converted into router_train_datasets target format."
                )
            save_split(
                split_ds,
                os.path.join(args.out_root, "copa", f"{split}.jsonl"),
                copa_row_to_router,
            )
        meta["copa_source"] = "super_glue/copa"

    meta_path = os.path.join(args.out_root, "piqa_copa_build_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[DONE] meta saved to {meta_path}")


if __name__ == "__main__":
    main()
