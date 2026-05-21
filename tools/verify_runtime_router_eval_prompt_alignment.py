import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from transformers import AutoTokenizer


DEFAULT_TASKS = [
    "boolq",
    "commonsenseqa",
    "hellaswag",
    "iwslt2017",
    "medmcqa",
    "piqa",
    "race",
    "rte",
    "siqa",
    "squad2",
    "sst2",
]


def parse_tasks(raw: str) -> List[str]:
    if not raw:
        return list(DEFAULT_TASKS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def read_jsonl(path: Path) -> List[Dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_opencompass_history(prompt_text: str) -> List[Dict]:
    # Mirrors the align-check eval config:
    # prompt_template=dict(round=[dict(role="HUMAN", prompt="{text}")])
    return [dict(role="HUMAN", prompt=str(prompt_text))]


def router_model_to_prompt_str(history: List[Dict], tokenizer) -> str:
    messages = []
    for item in history:
        role = item.get("role", "user")
        if role in {"HUMAN", "USER"}:
            role = "user"
        elif role in {"ASSISTANT", "BOT"}:
            role = "assistant"
        elif role == "SYSTEM":
            role = "system"
        content = item.get("content")
        if content is None:
            content = item.get("prompt", "")
        messages.append({"role": role, "content": str(content)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def opencompass_eval_runtime_prompt(history: List[Dict], tokenizer) -> str:
    messages = []
    for item in history:
        role = {
            "HUMAN": "user",
            "BOT": "assistant",
            "SYSTEM": "system",
        }.get(item.get("role", "HUMAN"), item.get("role", "HUMAN"))
        prompt = item.get("prompt", "")
        if prompt == "":
            continue
        if messages and messages[-1]["role"] == role:
            messages[-1]["content"] += "\n" + str(prompt)
        else:
            messages.append({"role": role, "content": str(prompt)})
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def preview(text: str, limit: int = 180) -> str:
    text = text.replace("\n", "\\n")
    if len(text) <= limit:
        return text
    return text[:limit] + f"... <truncated {len(text) - limit} chars>"


def verify_rows(rows: Iterable[Dict], tokenizer) -> Dict:
    checked = 0
    raw_source_mismatch = 0
    history_text_mismatch = 0
    runtime_prompt_mismatch = 0
    examples = []

    for row in rows:
        checked += 1
        raw_text = str(row.get("text", ""))
        source_text = str(row.get("source_text", raw_text))
        history = build_opencompass_history(raw_text)
        history_prompt = str(history[0]["prompt"])
        router_prompt = router_model_to_prompt_str(history, tokenizer)
        eval_prompt = opencompass_eval_runtime_prompt(history, tokenizer)

        row_has_issue = False
        issue = {
            "sample_id": row.get("_sample_id", ""),
            "raw_text_preview": preview(raw_text),
            "source_text_preview": preview(source_text),
            "history_prompt_preview": preview(history_prompt),
            "router_prompt_preview": preview(router_prompt),
            "eval_prompt_preview": preview(eval_prompt),
        }

        if raw_text != source_text:
            raw_source_mismatch += 1
            row_has_issue = True
        if history_prompt != raw_text:
            history_text_mismatch += 1
            row_has_issue = True
        if router_prompt != eval_prompt:
            runtime_prompt_mismatch += 1
            row_has_issue = True

        if row_has_issue and len(examples) < 3:
            examples.append(issue)

    return {
        "checked": checked,
        "raw_source_mismatch": raw_source_mismatch,
        "history_text_mismatch": history_text_mismatch,
        "runtime_prompt_mismatch": runtime_prompt_mismatch,
        "examples": examples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify that router dataset rows, OpenCompass eval history, and "
            "the final chat-templated runtime prompt are aligned."
        )
    )
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--model_path", default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError(
            f"Tokenizer for model_path={args.model_path!r} does not support apply_chat_template."
        )

    data_root = Path(args.data_root)
    summary = {
        "data_root": str(data_root),
        "model_path": args.model_path,
        "tasks": {},
    }
    total_checked = 0
    total_raw_source_mismatch = 0
    total_history_text_mismatch = 0
    total_runtime_prompt_mismatch = 0

    for task in parse_tasks(args.tasks):
        task_rows = []
        for split in ("train", "validation"):
            path = data_root / task / f"{split}.jsonl"
            if not path.exists():
                continue
            task_rows.extend(read_jsonl(path))
        if not task_rows:
            print(f"{task}: checked=0 skipped=no_rows")
            continue

        result = verify_rows(task_rows, tokenizer)
        summary["tasks"][task] = result
        total_checked += result["checked"]
        total_raw_source_mismatch += result["raw_source_mismatch"]
        total_history_text_mismatch += result["history_text_mismatch"]
        total_runtime_prompt_mismatch += result["runtime_prompt_mismatch"]
        print(
            f"{task}: checked={result['checked']} "
            f"raw_source_mismatch={result['raw_source_mismatch']} "
            f"history_text_mismatch={result['history_text_mismatch']} "
            f"runtime_prompt_mismatch={result['runtime_prompt_mismatch']}"
        )
        for example in result["examples"]:
            print(json.dumps(example, ensure_ascii=False, indent=2))

    summary["totals"] = {
        "checked": total_checked,
        "raw_source_mismatch": total_raw_source_mismatch,
        "history_text_mismatch": total_history_text_mismatch,
        "runtime_prompt_mismatch": total_runtime_prompt_mismatch,
    }
    print(
        "TOTAL "
        f"checked={total_checked} "
        f"raw_source_mismatch={total_raw_source_mismatch} "
        f"history_text_mismatch={total_history_text_mismatch} "
        f"runtime_prompt_mismatch={total_runtime_prompt_mismatch}"
    )


if __name__ == "__main__":
    main()
