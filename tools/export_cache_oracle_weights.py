import argparse
import hashlib
import json
import os
from typing import Iterable, List, Optional

import torch
from transformers import AutoTokenizer


TASKS = ["medmcqa", "race", "sst2"]


def sha1(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def load_manifest_files(cache_root: str, split: str) -> List[str]:
    manifest_path = os.path.join(cache_root, split, "manifest.json")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)
    files = manifest.get("files") or manifest.get("chunks")
    if not files:
        raise ValueError(f"No files/chunks in {manifest_path}")
    return [os.path.join(cache_root, split, name) for name in files]


def normalize_minmax(loss: torch.Tensor) -> torch.Tensor:
    loss = loss.float()
    finite = torch.isfinite(loss)
    if not bool(finite.any().item()):
        return torch.zeros_like(loss)
    valid = loss[finite]
    low = valid.min()
    high = valid.max()
    if float((high - low).abs().item()) < 1e-12:
        return torch.zeros_like(loss)
    return (loss - low) / (high - low)


def normalized_loss_values(loss_matrix: torch.Tensor, method: str) -> torch.Tensor:
    method = str(method)
    if method == "sample_minmax":
        return normalize_minmax(loss_matrix)
    if method == "none":
        return loss_matrix.float()
    raise ValueError(f"Unknown loss_normalization={method!r}")


def empty_target_distribution(
    loss_matrix: torch.Tensor,
    fallback: str,
    temperature: float,
    loss_normalization: str,
) -> torch.Tensor:
    num_tasks = loss_matrix.size(0)
    fallback = str(fallback)
    if fallback == "zero":
        return torch.zeros(num_tasks, num_tasks, dtype=loss_matrix.dtype)
    if fallback == "uniform":
        return torch.full(
            (num_tasks, num_tasks),
            1.0 / max(num_tasks * num_tasks, 1),
            dtype=torch.float32,
        )
    if fallback == "loss_softmax":
        normalized = normalized_loss_values(loss_matrix, loss_normalization).view(-1)
        logits = -normalized / max(float(temperature), 1e-6)
        return torch.softmax(logits, dim=-1).view(num_tasks, num_tasks)
    raise ValueError(f"Unknown empty_target_fallback={fallback!r}")


def distribution_from_loss(
    loss_matrix: torch.Tensor,
    correct_matrix: Optional[torch.Tensor],
    task_name: Optional[str],
    mode: str,
    temperature: float,
    empty_target_fallback: str = "zero",
    loss_normalization: str = "sample_minmax",
) -> torch.Tensor:
    num_tasks = loss_matrix.size(0)
    flat_loss = loss_matrix.float().view(-1)
    if mode == "self_if_available_else_correct_conf_ce" and task_name in TASKS:
        task_idx = TASKS.index(str(task_name))
        target = torch.zeros_like(flat_loss)
        target[task_idx * num_tasks + task_idx] = 1.0
        return target.view(num_tasks, num_tasks)

    if mode == "argmin":
        target = torch.zeros_like(flat_loss)
        target[int(flat_loss.argmin().item())] = 1.0
        return target.view(num_tasks, num_tasks)

    normalized = normalized_loss_values(loss_matrix, loss_normalization).view(-1)
    logits = -normalized / max(float(temperature), 1e-6)
    if mode in {"correct_conf_ce", "self_if_available_else_correct_conf_ce"}:
        if correct_matrix is None:
            return empty_target_distribution(loss_matrix, empty_target_fallback, temperature, loss_normalization)
        flat_correct = correct_matrix.bool().view(-1)
        if not bool(flat_correct.any().item()):
            return empty_target_distribution(loss_matrix, empty_target_fallback, temperature, loss_normalization)
        logits = logits.masked_fill(~flat_correct, float("-inf"))
    elif mode == "correct_conf_or_loss" and correct_matrix is not None:
        flat_correct = correct_matrix.bool().view(-1)
        if bool(flat_correct.any().item()):
            logits = logits.masked_fill(~flat_correct, float("-inf"))
    elif mode != "loss_softmax":
        raise ValueError(f"Unknown mode={mode!r}")
    return torch.softmax(logits, dim=-1).view(num_tasks, num_tasks)


def apply_cache_prompt_template(prompt: str, tokenizer, model_path: str) -> str:
    text = str(prompt)
    if "<|im_start|>" in text or "<|im_end|>" in text:
        return text
    if "qwen" in str(model_path).lower() and not text.endswith("\n"):
        text = text + "\n"
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
    )


def iter_items(cache_root: str, split: str) -> Iterable[dict]:
    for chunk_path in load_manifest_files(cache_root, split):
        obj = torch.load(chunk_path, map_location="cpu")
        for item in obj["items"]:
            yield item


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--split", default="train", choices=["train", "validation"])
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mode",
        default="correct_conf_or_loss",
        choices=[
            "correct_conf_or_loss",
            "correct_conf_ce",
            "self_if_available_else_correct_conf_ce",
            "loss_softmax",
            "argmin",
        ],
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument(
        "--loss_normalization",
        default="sample_minmax",
        choices=["sample_minmax", "none"],
        help="Normalize each sample loss matrix before softmax, or use raw loss values with none.",
    )
    parser.add_argument(
        "--empty_target_fallback",
        default="zero",
        choices=["zero", "uniform", "loss_softmax"],
        help=(
            "Fallback for correct_conf_ce-style samples with no correct pair. "
            "Use zero to preserve the raw training target, or uniform/loss_softmax "
            "when exporting runtime oracle weights that must contain positive mass."
        ),
    )
    parser.add_argument("--base_model_path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--local_files_only", action="store_true")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model_path,
        local_files_only=bool(args.local_files_only),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    count = 0
    with open(args.out, "w", encoding="utf-8") as f:
        for item in iter_items(args.cache_root, args.split):
            prompt_text = item.get("prompt_text") or item.get("text") or item.get("source_text")
            if prompt_text is None:
                raise KeyError("Cache item missing prompt_text/text/source_text")
            runtime_prompt = apply_cache_prompt_template(prompt_text, tokenizer, args.base_model_path)
            target = distribution_from_loss(
                item["loss_matrix"],
                item.get("correct_matrix"),
                item.get("task"),
                mode=args.mode,
                temperature=args.temperature,
                empty_target_fallback=args.empty_target_fallback,
                loss_normalization=args.loss_normalization,
            )
            first_weights = target.sum(dim=1)
            mid_weights = target.sum(dim=0)
            pred_pair = int(target.view(-1).argmax().item())
            first_idx = pred_pair // target.size(1)
            mid_idx = pred_pair % target.size(1)
            rec = {
                "prompt_sha1": sha1(runtime_prompt),
                "raw_prompt_sha1": sha1(prompt_text),
                "item_id": item.get("item_id"),
                "task": item.get("task"),
                "mode": args.mode,
                "temperature": float(args.temperature),
                "loss_normalization": str(args.loss_normalization),
                "empty_target_fallback": str(args.empty_target_fallback),
                "target_mass": float(target.sum().item()),
                "first_task": TASKS[first_idx],
                "mid_task": TASKS[mid_idx],
                "first_eid": first_idx + 1,
                "mid_eid": mid_idx + 1,
                "first_weights": [float(x) for x in first_weights.tolist()],
                "mid_weights": [float(x) for x in mid_weights.tolist()],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    print(f"wrote {count} oracle weights to {args.out}")


if __name__ == "__main__":
    main()
