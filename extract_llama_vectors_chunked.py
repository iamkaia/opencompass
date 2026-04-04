# extract_llama_vectors_chunked.py
import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from opencompass.models.unified_moe_core_internal_router_compact import (
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
)


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]
TASK2ID = {t: i for i, t in enumerate(TASK_NAMES)}


def read_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def build_text(ex: Dict) -> str:
    if "text" in ex and ex["text"]:
        return str(ex["text"])
    if "prompt" in ex and ex["prompt"]:
        return str(ex["prompt"])

    parts = []
    for k in ["instruction", "input", "question", "context", "src", "sentence", "article"]:
        if k in ex and ex[k] is not None and str(ex[k]).strip():
            parts.append(f"{k}: {ex[k]}")

    if "choices" in ex and ex["choices"]:
        if isinstance(ex["choices"], list):
            parts.append("choices: " + " | ".join(map(str, ex["choices"])))
        else:
            parts.append(f"choices: {ex['choices']}")

    if not parts:
        for k, v in ex.items():
            if k.lower() in ["label", "labels", "answer", "answers", "target", "response", "output"]:
                continue
            if isinstance(v, (str, int, float)) and str(v).strip():
                parts.append(f"{k}: {v}")

    if not parts:
        raise ValueError(f"Cannot build text from example keys: {list(ex.keys())}")

    return "\n".join(parts)


def load_split_from_root(data_root: str, split_name: str) -> List[Dict]:
    samples = []
    for task in TASK_NAMES:
        fp = os.path.join(data_root, task, f"{split_name}.jsonl")
        if not os.path.exists(fp):
            print(f"[WARN] missing file: {fp}")
            continue

        raw = read_jsonl(fp)
        label = TASK2ID[task]

        for ex in raw:
            samples.append(
                {
                    "text": build_text(ex),
                    "label": label,
                    "task": task,
                }
            )

        print(f"[LOAD] {task}/{split_name}.jsonl -> {len(raw)} samples")

    if len(samples) == 0:
        raise ValueError(f"No samples loaded from {data_root} split={split_name}")

    return samples


def make_balanced_subset(
    samples: List[Dict],
    max_samples: int,
    seed: int = 42,
) -> List[Dict]:
    """
    自動 balanced sampling：
    盡量讓每個 task 抽到一樣多的樣本。
    """
    rng = random.Random(seed)

    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)

    for task in TASK_NAMES:
        rng.shuffle(by_task[task])

    if max_samples is None:
        return samples

    max_samples = int(max_samples)
    num_tasks = len(TASK_NAMES)
    per_task = max_samples // num_tasks
    remainder = max_samples % num_tasks

    selected = []

    # 先每個 task 抽 per_task
    leftovers = {}
    for task in TASK_NAMES:
        task_samples = by_task[task]
        take_n = min(per_task, len(task_samples))
        selected.extend(task_samples[:take_n])
        leftovers[task] = task_samples[take_n:]

    # 剩下的名額再依序補
    if remainder > 0:
        for task in TASK_NAMES:
            if remainder == 0:
                break
            if len(leftovers[task]) > 0:
                selected.append(leftovers[task][0])
                leftovers[task] = leftovers[task][1:]
                remainder -= 1

    # 如果某些 task 不夠，總數可能小於 max_samples
    # 再從所有剩餘樣本中補齊
    if len(selected) < max_samples:
        remain_pool = []
        for task in TASK_NAMES:
            remain_pool.extend(leftovers[task])
        rng.shuffle(remain_pool)
        need = max_samples - len(selected)
        selected.extend(remain_pool[:need])

    rng.shuffle(selected)

    # 印出實際分布
    stat = defaultdict(int)
    for s in selected:
        stat[s["task"]] += 1
    print("[BALANCED_SUBSET]")
    for task in TASK_NAMES:
        print(f"  {task}: {stat[task]}")

    return selected


class RouterTextDataset(Dataset):
    def __init__(self, data_root: str, split_name: str, max_samples=None, balanced=False, seed=42):
        self.samples = load_split_from_root(data_root, split_name)

        if max_samples is not None:
            if balanced:
                self.samples = make_balanced_subset(self.samples, max_samples=max_samples, seed=seed)
            else:
                self.samples = self.samples[: int(max_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        return {
            "idx": idx,
            "text": item["text"],
            "label": item["label"],
            "task": item["task"],
        }


class Collator:
    def __call__(self, batch: List[Dict]) -> Dict:
        return {
            "idx": [x["idx"] for x in batch],
            "text": [x["text"] for x in batch],
            "label": [x["label"] for x in batch],
            "task": [x["task"] for x in batch],
        }


class LlamaVectorExtractor(torch.nn.Module):
    """
    抽兩個位置的 hidden sequence，再取最後有效 token 向量：
      - first_layer_idx attention 前
      - middle_layer_idx attention 前
    """
    def __init__(
        self,
        base_model_path: str,
        dtype: str,
        first_layer_idx: int,
        middle_layer_idx: int,
        lora_paths: Dict[str, str] | None = None,
        r: int = 8,
        alpha: int = 32,
        apply_task_lora_for_mid: bool = False,
    ):
        super().__init__()
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.apply_task_lora_for_mid = bool(apply_task_lora_for_mid)
        self.task_to_expert_id = {task: TASK2ID[task] + 1 for task in TASK_NAMES}

        if self.apply_task_lora_for_mid:
            if not lora_paths:
                raise ValueError("apply_task_lora_for_mid=True requires lora_paths")
            self.model = patch_llama_with_hard_routed_lora(
                self.model,
                num_experts=1 + len(TASK_NAMES),
                r=int(r),
                alpha=int(alpha),
            )
            for task in TASK_NAMES:
                if task not in lora_paths:
                    raise KeyError(f"Missing LoRA path for task: {task}")
                load_lora_into_expert(
                    self.model,
                    lora_paths[task],
                    self.task_to_expert_id[task],
                )

        self.cached_first_before_attn = None
        self.cached_mid_before_attn = None
        self._install_hooks()

    def _install_hooks(self):
        def first_pre_hook(module, args):
            self.cached_first_before_attn = args[0].detach()
            return None

        def mid_pre_hook(module, args):
            self.cached_mid_before_attn = args[0].detach()
            return None

        self.model.model.layers[self.first_layer_idx].input_layernorm.register_forward_pre_hook(first_pre_hook)
        self.model.model.layers[self.middle_layer_idx].input_layernorm.register_forward_pre_hook(mid_pre_hook)

    @staticmethod
    def gather_last_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        vec = hidden_states[batch_idx, last_idx, :]
        return vec

    @torch.no_grad()
    def extract_vectors(self, input_ids, attention_mask, task_name: str | None = None):
        self.cached_first_before_attn = None
        self.cached_mid_before_attn = None

        if self.apply_task_lora_for_mid:
            if task_name is None:
                raise ValueError("task_name is required when apply_task_lora_for_mid=True")
            if task_name not in self.task_to_expert_id:
                raise KeyError(f"Unknown task_name={task_name}")

            # Runtime alignment:
            # 1. first router sees base-model states before any routed LoRA is active.
            # 2. mid router should see states after first-half layers have used the task expert.
            set_all_experts(self.model, NULL_EXPERT_ID)
            set_layer_range_expert(
                self.model,
                self.first_layer_idx,
                self.middle_layer_idx - 1,
                self.task_to_expert_id[task_name],
            )

        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )

        if self.cached_first_before_attn is None:
            raise RuntimeError("first before-attention hidden is None")
        if self.cached_mid_before_attn is None:
            raise RuntimeError("mid before-attention hidden is None")

        first_vec = self.gather_last_valid(self.cached_first_before_attn, attention_mask)
        mid_vec = self.gather_last_valid(self.cached_mid_before_attn, attention_mask)
        return first_vec, mid_vec


def flush_chunk(chunk_items, split_out_dir: str, chunk_id: int):
    if len(chunk_items) == 0:
        return None
    out_path = os.path.join(split_out_dir, f"chunk_{chunk_id:06d}.pt")
    torch.save({"items": chunk_items}, out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--split", type=str, choices=["train", "validation"], required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_llama_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--chunk_size", type=int, default=2048)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--balanced", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--apply_task_lora_for_mid", action="store_true")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora_iwslt", type=str, default=None)
    parser.add_argument("--lora_medmcqa", type=str, default=None)
    parser.add_argument("--lora_race", type=str, default=None)
    parser.add_argument("--lora_squad2", type=str, default=None)
    parser.add_argument("--lora_sst2", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    ds = RouterTextDataset(
        args.data_root,
        args.split,
        max_samples=args.max_samples,
        balanced=args.balanced,
        seed=args.seed,
    )
    dl = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_paths = None
    if args.apply_task_lora_for_mid:
        lora_paths = {
            "iwslt2017": args.lora_iwslt,
            "medmcqa": args.lora_medmcqa,
            "race": args.lora_race,
            "squad2": args.lora_squad2,
            "sst2": args.lora_sst2,
        }
        missing = [task for task, path in lora_paths.items() if not path]
        if missing:
            raise ValueError(
                "apply_task_lora_for_mid=True requires all LoRA paths. "
                f"Missing: {missing}"
            )

    extractor = LlamaVectorExtractor(
        base_model_path=args.base_model_path,
        dtype=args.dtype,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        lora_paths=lora_paths,
        r=args.r,
        alpha=args.alpha,
        apply_task_lora_for_mid=args.apply_task_lora_for_mid,
    ).to(device)

    split_out_dir = os.path.join(args.out_root, args.split)
    os.makedirs(split_out_dir, exist_ok=True)

    manifest = {
        "split": args.split,
        "num_samples": len(ds),
        "chunk_size": args.chunk_size,
        "files": [],
    }

    chunk_items = []
    chunk_id = 0
    total = 0

    for step, batch in enumerate(dl):
        texts = batch["text"]
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.max_llama_len,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        bs = len(texts)
        first_vec = torch.empty(bs, extractor.model.config.hidden_size, dtype=torch.float16)
        mid_vec = torch.empty(bs, extractor.model.config.hidden_size, dtype=torch.float16)

        # A routed LoRA setup can only choose one expert per forward pass,
        # so we extract homogeneous task sub-batches and then stitch them back.
        task_to_indices = defaultdict(list)
        for i, task in enumerate(batch["task"]):
            task_to_indices[task].append(i)

        with torch.no_grad():
            for task, indices in task_to_indices.items():
                sub_ids = input_ids[indices]
                sub_mask = attention_mask[indices]
                sub_first, sub_mid = extractor.extract_vectors(
                    sub_ids,
                    sub_mask,
                    task_name=task if args.apply_task_lora_for_mid else None,
                )
                first_vec[indices] = sub_first.to(dtype=torch.float16).cpu()
                mid_vec[indices] = sub_mid.to(dtype=torch.float16).cpu()

        for i in range(bs):
            chunk_items.append(
                {
                    "id": int(batch["idx"][i]),
                    "text": batch["text"][i],
                    "label": int(batch["label"][i]),
                    "task": batch["task"][i],
                    "first_vec": first_vec[i].clone(),
                    "mid_vec": mid_vec[i].clone(),
                }
            )
            total += 1

            if len(chunk_items) >= args.chunk_size:
                out_path = flush_chunk(chunk_items, split_out_dir, chunk_id)
                manifest["files"].append(os.path.basename(out_path))
                print(f"[FLUSH] chunk_id={chunk_id} items={len(chunk_items)} total_saved={total}")
                chunk_items = []
                chunk_id += 1

        if step % 10 == 0:
            print(f"[EXTRACT] step={step} total_collected={total}")

    if len(chunk_items) > 0:
        out_path = flush_chunk(chunk_items, split_out_dir, chunk_id)
        manifest["files"].append(os.path.basename(out_path))
        print(f"[FLUSH] chunk_id={chunk_id} items={len(chunk_items)} total_saved={total}")

    cfg = {
        "task_names": TASK_NAMES,
        "first_layer_idx": args.first_layer_idx,
        "middle_layer_idx": args.middle_layer_idx,
        "max_llama_len": args.max_llama_len,
        "dtype": args.dtype,
        "feature_type": "compact_last_valid_token_vector",
        "balanced": args.balanced,
        "seed": args.seed,
        "apply_task_lora_for_mid": args.apply_task_lora_for_mid,
        "r": args.r,
        "alpha": args.alpha,
    }
    with open(os.path.join(args.out_root, "feature_config.json"), "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)

    with open(os.path.join(split_out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"[DONE] saved {total} samples to {split_out_dir}")


if __name__ == "__main__":
    main()
