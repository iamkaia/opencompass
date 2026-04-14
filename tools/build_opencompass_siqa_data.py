#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path

from datasets import load_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build OpenCompass SIQA raw files under ./data/siqa from Hugging Face social_i_qa."
    )
    parser.add_argument(
        "--out_dir",
        default="./data/siqa",
        help="Output directory for OpenCompass SIQA files.",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Pass trust_remote_code=True when loading social_i_qa.",
    )
    return parser.parse_args()


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_labels(path: Path, labels):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for label in labels:
            f.write(f"{label}\n")


def convert_split(ds_split):
    rows = []
    labels = []
    for ex in ds_split:
        label = int(ex["label"])
        rows.append(
            {
                "context": ex["context"],
                "question": ex["question"],
                "answerA": ex["answerA"],
                "answerB": ex["answerB"],
                "answerC": ex["answerC"],
            }
        )
        labels.append(label)
    return rows, labels


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    ds = load_dataset("social_i_qa", trust_remote_code=args.trust_remote_code)

    train_rows, train_labels = convert_split(ds["train"])
    val_rows, val_labels = convert_split(ds["validation"])

    write_jsonl(out_dir / "train.jsonl", train_rows)
    write_labels(out_dir / "train-labels.lst", train_labels)
    write_jsonl(out_dir / "dev.jsonl", val_rows)
    write_labels(out_dir / "dev-labels.lst", val_labels)

    print(f"[DONE] wrote SIQA OpenCompass files to {out_dir}")
    print(f"[SAVE] {out_dir / 'train.jsonl'} rows={len(train_rows)}")
    print(f"[SAVE] {out_dir / 'train-labels.lst'} rows={len(train_labels)}")
    print(f"[SAVE] {out_dir / 'dev.jsonl'} rows={len(val_rows)}")
    print(f"[SAVE] {out_dir / 'dev-labels.lst'} rows={len(val_labels)}")


if __name__ == "__main__":
    main()
