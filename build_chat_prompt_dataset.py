import argparse
import json
import os
from pathlib import Path

from transformers import AutoTokenizer


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]
SPLITS = ["train", "validation", "test"]


def read_jsonl(path: str):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def wrap_with_chat_template(tokenizer, text: str) -> str:
    messages = [{"role": "user", "content": text}]
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    # Fallback for chat models without an exposed template.
    return f"<s>[INST] {text.strip()} [/INST]"


def convert_row(tokenizer, row: dict) -> dict:
    text = row.get("text", "")
    if not text:
        raise ValueError(f"Row missing text field: {row}")

    new_row = dict(row)
    new_row["source_text"] = text
    new_row["text"] = wrap_with_chat_template(tokenizer, text)
    return new_row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--model_path", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    src_root = Path(args.src_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    meta = {
        "src_root": str(src_root),
        "model_path": args.model_path,
        "conversion": "wrap existing datasets_classifier text with tokenizer.apply_chat_template(add_generation_prompt=True)",
    }
    with open(out_root / "conversion_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    for task in TASK_NAMES:
        task_src = src_root / task
        if not task_src.exists():
            print(f"[WARN] missing task dir: {task_src}")
            continue

        task_out = out_root / task
        task_out.mkdir(parents=True, exist_ok=True)

        for split in SPLITS:
            src_file = task_src / f"{split}.jsonl"
            if not src_file.exists():
                continue

            rows = read_jsonl(str(src_file))
            new_rows = [convert_row(tokenizer, row) for row in rows]
            out_file = task_out / f"{split}.jsonl"
            write_jsonl(str(out_file), new_rows)
            print(f"[DONE] {task}/{split}: {len(new_rows)} -> {out_file}")


if __name__ == "__main__":
    main()
