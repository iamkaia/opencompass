import argparse
import json
import os
from typing import Dict, Iterable, List

from datasets import load_dataset


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def commonsenseqa_rows(split) -> List[Dict]:
    rows = []
    for ex in split:
        labels = list(ex["choices"]["label"])
        texts = list(ex["choices"]["text"])
        choices = {label: text for label, text in zip(labels, texts)}
        target = str(ex.get("answerKey") or "").strip()
        if not target:
            continue
        prompt = (
            f"{ex['question']}\n"
            f"A. {choices.get('A', '')}\n"
            f"B. {choices.get('B', '')}\n"
            f"C. {choices.get('C', '')}\n"
            f"D. {choices.get('D', '')}\n"
            f"E. {choices.get('E', '')}\n"
            "Answer:"
        )
        rows.append({"text": prompt, "target": target, "label": "commonsenseqa"})
    return rows


def rte_rows(split) -> List[Dict]:
    rows = []
    label_map = {0: "A", 1: "B"}
    for ex in split:
        label = int(ex.get("label", -1))
        if label not in label_map:
            continue
        prompt = (
            f"{ex['premise']}\n"
            f"{ex['hypothesis']}\n"
            "Is the sentence below entailed by the sentence above?\n"
            "A. Yes\n"
            "B. No\n"
            "Answer:"
        )
        rows.append({"text": prompt, "target": label_map[label], "label": "rte"})
    return rows


def piqa_rows(split) -> List[Dict]:
    rows = []
    for ex in split:
        label = int(ex.get("label", -1))
        if label not in (0, 1):
            continue
        prompt = f"{ex['goal']}\nA. {ex['sol1']}\nB. {ex['sol2']}\nAnswer:"
        rows.append({"text": prompt, "target": "A" if label == 0 else "B", "label": "piqa"})
    return rows


def siqa_rows(split) -> List[Dict]:
    rows = []
    label_map = {0: "A", 1: "B", 2: "C", 1.0: "A", 2.0: "B", 3.0: "C"}
    for ex in split:
        raw_label = ex.get("label", ex.get("answer"))
        try:
            label = int(raw_label)
        except Exception:
            continue
        if label in (1, 2, 3):
            target = {1: "A", 2: "B", 3: "C"}[label]
        elif label in (0, 1, 2):
            target = {0: "A", 1: "B", 2: "C"}[label]
        else:
            continue
        prompt = (
            f"Context: {ex['context']}\n"
            f"Question: {ex['question']}\n"
            f"A. {ex['answerA']}\n"
            f"B. {ex['answerB']}\n"
            f"C. {ex['answerC']}\n"
            "Answer:"
        )
        rows.append({"text": prompt, "target": target, "label": "siqa"})
    return rows


def boolq_rows(split) -> List[Dict]:
    rows = []
    for ex in split:
        if "label" not in ex:
            continue
        target = "A" if bool(ex["label"]) else "B"
        prompt = f"Passage: {ex['passage']}\nQuestion: {ex['question']}\nA. Yes\nB. No\nAnswer:"
        rows.append({"text": prompt, "target": target, "label": "boolq"})
    return rows


def copa_rows(split) -> List[Dict]:
    rows = []
    for ex in split:
        label = int(ex.get("label", -1))
        if label not in (0, 1):
            continue
        relation = "cause" if ex["question"] == "cause" else "effect"
        prompt = (
            f"{ex['premise']}\n"
            f"Question: Which may be the {relation}?\n"
            f"A. {ex['choice1']}\n"
            f"B. {ex['choice2']}\n"
            "Answer:"
        )
        rows.append({"text": prompt, "target": "A" if label == 0 else "B", "label": "copa"})
    return rows


def hellaswag_rows(split) -> List[Dict]:
    rows = []
    for ex in split:
        raw_label = ex.get("label", "")
        try:
            label = int(raw_label)
        except Exception:
            continue
        if label not in (0, 1, 2, 3):
            continue
        endings = list(ex["endings"])
        prompt = (
            f"{ex['ctx']}\n"
            f"A. {endings[0]}\n"
            f"B. {endings[1]}\n"
            f"C. {endings[2]}\n"
            f"D. {endings[3]}\n"
            "Answer:"
        )
        rows.append({"text": prompt, "target": "ABCD"[label], "label": "hellaswag"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", type=str, default="hf_external_sources_0511")
    args = parser.parse_args()

    commonsenseqa = load_dataset("commonsense_qa")
    rte = load_dataset("super_glue", "rte")
    piqa = load_dataset("piqa")
    siqa = load_dataset("social_i_qa", trust_remote_code=True)
    boolq = load_dataset("super_glue", "boolq")
    copa = load_dataset("super_glue", "copa")
    hellaswag = load_dataset("hellaswag")

    jobs = {
        "piqa": {
            "train": piqa_rows(piqa["train"]),
            "validation": piqa_rows(piqa["validation"]),
            "test": piqa_rows(piqa["validation"]),
        },
        "siqa": {
            "train": siqa_rows(siqa["train"]),
            "validation": siqa_rows(siqa["validation"]),
            "test": siqa_rows(siqa["validation"]),
        },
        "boolq": {
            "train": boolq_rows(boolq["train"]),
            "validation": boolq_rows(boolq["validation"]),
        },
        "copa": {
            "train": copa_rows(copa["train"]),
            "validation": copa_rows(copa["validation"]),
        },
        "hellaswag": {
            "train": hellaswag_rows(hellaswag["train"]),
            "validation": hellaswag_rows(hellaswag["validation"]),
        },
        "commonsenseqa": {
            "train": commonsenseqa_rows(commonsenseqa["train"]),
            "validation": commonsenseqa_rows(commonsenseqa["validation"]),
            "test": commonsenseqa_rows(commonsenseqa["test"]),
        },
        "rte": {
            "train": rte_rows(rte["train"]),
            "validation": rte_rows(rte["validation"]),
            "test": rte_rows(rte["test"]),
        },
    }

    summary = {}
    for task, splits in jobs.items():
        summary[task] = {}
        for split, rows in splits.items():
            if not rows:
                continue
            write_jsonl(os.path.join(args.output_root, task, f"{split}.jsonl"), rows)
            summary[task][split] = len(rows)

    with open(os.path.join(args.output_root, "source_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
