import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def stringify_origin_prompt(origin_prompt: Any) -> str:
    if isinstance(origin_prompt, str):
        return origin_prompt
    if isinstance(origin_prompt, list):
        parts: List[str] = []
        for item in origin_prompt:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("prompt", item.get("content", ""))))
            else:
                parts.append(str(item))
        return "".join(parts)
    if isinstance(origin_prompt, dict):
        return str(origin_prompt.get("prompt", origin_prompt.get("content", "")))
    return str(origin_prompt)


def load_cache_item(
    cache_root: Path,
    split: str,
    sample_index: int,
    cache_task: Optional[str] = None,
) -> Dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "This tool needs torch to read cached .pt chunks. Run it with the "
            "same Python environment you use for router training."
        ) from exc

    split_dir = cache_root / split
    manifest = load_json(split_dir / "manifest.json")
    seen = 0
    seen_matching_task = 0
    for filename in manifest["files"]:
        payload = torch.load(split_dir / filename, map_location="cpu")
        items = payload["items"]
        if cache_task is None:
            if sample_index < seen + len(items):
                return items[sample_index - seen]
        else:
            for item in items:
                if str(item.get("task")) != str(cache_task):
                    continue
                if seen_matching_task == sample_index:
                    return item
                seen_matching_task += 1
        seen += len(items)
    if cache_task is None:
        raise IndexError(f"sample_index={sample_index} out of range for {split_dir}; total={seen}")
    raise IndexError(
        f"sample_index={sample_index} out of range for task={cache_task} "
        f"in {split_dir}; matching_total={seen_matching_task}"
    )


def load_prediction_item(prediction_json: Path, sample_index: int) -> Dict[str, Any]:
    payload = load_json(prediction_json)
    key = str(sample_index)
    if isinstance(payload, dict) and key in payload:
        return payload[key]
    if isinstance(payload, list):
        return payload[sample_index]
    raise KeyError(f"Cannot find sample index {sample_index} in {prediction_json}")


def maybe_apply_chat_template(origin_prompt: Any, tokenizer_path: Optional[str]) -> Optional[str]:
    if not tokenizer_path:
        return None
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    if not hasattr(tokenizer, "apply_chat_template"):
        return None

    if isinstance(origin_prompt, list):
        messages = []
        for item in origin_prompt:
            if not isinstance(item, dict):
                messages.append({"role": "user", "content": str(item)})
                continue
            role = str(item.get("role", "user"))
            if role in {"HUMAN", "USER"}:
                role = "user"
            elif role in {"BOT", "ASSISTANT"}:
                role = "assistant"
            elif role == "SYSTEM":
                role = "system"
            content = item.get("content", item.get("prompt", ""))
            messages.append({"role": role, "content": str(content)})
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    return tokenizer.apply_chat_template(
        [{"role": "user", "content": stringify_origin_prompt(origin_prompt)}],
        tokenize=False,
        add_generation_prompt=True,
    )


def preview(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... <truncated {len(text) - limit} chars>"


def main():
    parser = argparse.ArgumentParser(
        description="Compare cached router prompt_text with OpenCompass prediction origin_prompt."
    )
    parser.add_argument("--cache_root", type=Path, required=True)
    parser.add_argument("--split", type=str, default="validation")
    parser.add_argument("--prediction_json", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--cache_task",
        type=str,
        default=None,
        help="When the cache contains multiple tasks, select the Nth item for this task.",
    )
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--preview_chars", type=int, default=800)
    args = parser.parse_args()

    cache_item = load_cache_item(args.cache_root, args.split, args.index, args.cache_task)
    pred_item = load_prediction_item(args.prediction_json, args.index)

    cache_prompt = str(cache_item.get("prompt_text", cache_item.get("text", "")))
    oc_prompt = stringify_origin_prompt(pred_item.get("origin_prompt", ""))

    print(f"cache_root: {args.cache_root}")
    print(f"split: {args.split}")
    print(f"prediction_json: {args.prediction_json}")
    print(f"index: {args.index}")
    print(f"cache_task: {cache_item.get('task')}")
    print(f"cache_target: {cache_item.get('target')!r}")
    print(f"opencompass_gold: {pred_item.get('gold')!r}")
    print()
    print(f"cache_prompt_sha1: {sha1_text(cache_prompt)}")
    print(f"opencompass_prompt_sha1: {sha1_text(oc_prompt)}")
    print(f"exact_match: {cache_prompt == oc_prompt}")
    print(f"cache_len: {len(cache_prompt)}")
    print(f"opencompass_len: {len(oc_prompt)}")
    print()
    print("=== cache prompt ===")
    print(preview(repr(cache_prompt), args.preview_chars))
    print()
    print("=== opencompass origin_prompt ===")
    print(preview(repr(oc_prompt), args.preview_chars))

    chat_prompt = maybe_apply_chat_template(pred_item.get("origin_prompt", ""), args.tokenizer)
    if chat_prompt is not None:
        print()
        print("=== opencompass origin_prompt after tokenizer.apply_chat_template ===")
        print(f"chat_template_sha1: {sha1_text(chat_prompt)}")
        print(f"same_as_cache_prompt: {chat_prompt == cache_prompt}")
        print(preview(repr(chat_prompt), args.preview_chars))


if __name__ == "__main__":
    main()
