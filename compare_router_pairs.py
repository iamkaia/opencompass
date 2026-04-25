import argparse
import json
import os
from collections import Counter

import torch
from transformers import AutoTokenizer

from extract_llama_vectors_chunked import LlamaVectorExtractor, build_text, read_jsonl
from opencompass.models.unified_moe_core_internal_router_compact import (
    UnifiedMoECoreInternalRouterCompact,
)


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


def load_texts(data_root: str, task: str, split: str, max_samples: int):
    fp = os.path.join(data_root, task, f"{split}.jsonl")
    rows = read_jsonl(fp)
    texts = [build_text(row) for row in rows[:max_samples]]
    return texts


def load_prompts_from_jsonl(prompt_file: str, max_samples: int | None):
    texts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_samples is not None and i >= max_samples:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            texts.append(obj["prompt"])
    return texts


@torch.no_grad()
def get_offline_pairs(texts, extractor, tokenizer):
    device = next(extractor.model.parameters()).device
    inv_expert_to_task = {v: k for k, v in extractor.task_to_expert_id.items()}
    pairs = []

    for text in texts:
        enc = tokenizer(
            [text],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        first_vec, _ = extractor.extract_vectors(input_ids, attention_mask, task_name=None)
        first_eid = int(extractor.predict_first_expert([text], first_vec)[0].item())

        first_task = inv_expert_to_task[first_eid]
        _, mid_vec = extractor.extract_vectors(input_ids, attention_mask, task_name=first_task)
        mid_eid = int(extractor.predict_mid_expert([text], mid_vec)[0].item())

        pairs.append((first_eid, mid_eid))

    return pairs


@torch.no_grad()
def get_runtime_pairs(texts, core):
    pairs = []

    for text in texts:
        core._reset_runtime_cache()
        core._encode_bert_memory(text)

        inp = core.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=core.max_seq_len,
        ).to(core.model.device)

        _ = core.model(
            **inp,
            use_cache=False,
            return_dict=True,
        )

        pairs.append((int(core.cached_first_eid), int(core.cached_mid_eid)))

    return pairs


def summarize_pairs(name, pairs, task_names):
    c = Counter()
    for first_eid, mid_eid in pairs:
        first_task = task_names[first_eid - 1]
        mid_task = task_names[mid_eid - 1]
        c[f"{first_task}->{mid_task}"] += 1
    print(f"\n[{name}]")
    for pair, count in c.most_common():
        print(f"{pair}: {count}")
    return c


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--task", type=str, default="sst2", choices=TASK_NAMES)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--prompt_file", type=str, default=None)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_ckpt_dir", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, required=True)
    parser.add_argument("--lora_iwslt", type=str, required=True)
    parser.add_argument("--lora_medmcqa", type=str, required=True)
    parser.add_argument("--lora_race", type=str, required=True)
    parser.add_argument("--lora_squad2", type=str, required=True)
    parser.add_argument("--lora_sst2", type=str, required=True)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--router_dim", type=int, default=512)
    args = parser.parse_args()

    if args.prompt_file is not None:
        texts = load_prompts_from_jsonl(args.prompt_file, args.max_samples)
        print(f"[INFO] Loaded {len(texts)} prompts from {args.prompt_file}")
    else:
        if args.data_root is None:
            raise ValueError("Either --prompt_file or --data_root must be provided")
        texts = load_texts(args.data_root, args.task, args.split, args.max_samples)
        print(f"[INFO] Loaded {len(texts)} texts from {args.task}/{args.split}")

    lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = LlamaVectorExtractor(
        base_model_path=args.base_model_path,
        dtype=args.dtype,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        lora_paths=lora_paths,
        r=args.r,
        alpha=args.alpha,
        apply_task_lora_for_mid=True,
        use_router_pred_for_mid=True,
        router_ckpt_dir=args.router_ckpt_dir,
        router_bert_init=args.router_bert_init,
        router_dim=args.router_dim,
    ).to(device)

    runtime_core = UnifiedMoECoreInternalRouterCompact(
        base_model_path=args.base_model_path,
        router_ckpt_dir=args.router_ckpt_dir,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        router_dim=args.router_dim,
        device_map="auto",
        max_seq_len=2048,
        force_first_task=None,
        force_mid_task=None,
    )

    llama_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llama_tokenizer.pad_token_id is None:
        llama_tokenizer.pad_token = llama_tokenizer.eos_token

    offline_pairs = get_offline_pairs(texts, extractor, llama_tokenizer)
    runtime_pairs = get_runtime_pairs(texts, runtime_core)

    offline_counter = summarize_pairs("offline", offline_pairs, TASK_NAMES)
    runtime_counter = summarize_pairs("runtime", runtime_pairs, TASK_NAMES)

    same = sum(1 for x, y in zip(offline_pairs, runtime_pairs) if x == y)
    total = len(offline_pairs)
    print(f"\n[COMPARE]\nexact_pair_match={same}/{total} ({same / max(total, 1):.4f})")

    out = {
        "task": args.task,
        "split": args.split,
        "prompt_file": args.prompt_file,
        "max_samples": args.max_samples,
        "offline_pair_counter": dict(offline_counter),
        "runtime_pair_counter": dict(runtime_counter),
        "exact_pair_match": same,
        "total": total,
        "match_ratio": same / max(total, 1),
    }
    out_path = f"compare_pairs_{args.task}_{args.split}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"[SAVE] {out_path}")


if __name__ == "__main__":
    main()
