import argparse
import json
import math
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from opencompass.models.router_moe_components import BertExternalEncoder, CompactCrossAttentionRouter
from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
)


DEFAULT_TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


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


def discover_tasks(data_root: str, requested_tasks: Optional[Sequence[str]]) -> List[str]:
    if requested_tasks:
        tasks = [str(task).strip() for task in requested_tasks if str(task).strip()]
    else:
        tasks = []
        for name in sorted(os.listdir(data_root)):
            full = os.path.join(data_root, name)
            if os.path.isdir(full):
                tasks.append(name)
        if not tasks:
            tasks = list(DEFAULT_TASK_NAMES)
    if not tasks:
        raise ValueError(f"No tasks found under data_root={data_root}")
    return tasks


def make_balanced_subset(samples: List[Dict], task_names: Sequence[str], max_samples: int, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    by_task = defaultdict(list)
    for s in samples:
        by_task[s["task"]].append(s)

    for task in task_names:
        rng.shuffle(by_task[task])

    max_samples = int(max_samples)
    num_tasks = len(task_names)
    per_task = max_samples // max(num_tasks, 1)
    remainder = max_samples % max(num_tasks, 1)

    selected = []
    leftovers = {}
    for task in task_names:
        task_samples = by_task[task]
        take_n = min(per_task, len(task_samples))
        selected.extend(task_samples[:take_n])
        leftovers[task] = task_samples[take_n:]

    if remainder > 0:
        for task in task_names:
            if remainder == 0:
                break
            if leftovers[task]:
                selected.append(leftovers[task][0])
                leftovers[task] = leftovers[task][1:]
                remainder -= 1

    if len(selected) < max_samples:
        remain_pool = []
        for task in task_names:
            remain_pool.extend(leftovers[task])
        rng.shuffle(remain_pool)
        selected.extend(remain_pool[: max_samples - len(selected)])

    rng.shuffle(selected)
    return selected


class RouterTextDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split_name: str,
        task_names: Sequence[str],
        max_samples: Optional[int] = None,
        balanced: bool = False,
        seed: int = 42,
    ):
        self.task_names = list(task_names)
        self.task2id = {task: idx for idx, task in enumerate(self.task_names)}
        self.samples = self._load_split_from_root(data_root, split_name)
        if max_samples is not None:
            if balanced:
                self.samples = make_balanced_subset(
                    self.samples,
                    task_names=self.task_names,
                    max_samples=max_samples,
                    seed=seed,
                )
            else:
                self.samples = self.samples[: int(max_samples)]

    def _load_split_from_root(self, data_root: str, split_name: str) -> List[Dict]:
        samples = []
        for task in self.task_names:
            fp = os.path.join(data_root, task, f"{split_name}.jsonl")
            if not os.path.exists(fp):
                raise FileNotFoundError(f"Missing dataset file: {fp}")
            raw = read_jsonl(fp)
            label = self.task2id[task]
            for ex in raw:
                samples.append({"text": build_text(ex), "label": label, "task": task})
            print(f"[LOAD] {task}/{split_name}.jsonl -> {len(raw)} samples")

        if not samples:
            raise ValueError(f"No samples loaded from {data_root} split={split_name}")
        return samples

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


@dataclass
class Batch:
    texts: List[str]
    labels: torch.Tensor
    tasks: List[str]


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            labels=torch.tensor([x["label"] for x in batch], dtype=torch.long),
            tasks=[x["task"] for x in batch],
        )


class LlamaVectorExtractor(nn.Module):
    """
    Extract before-attention vectors from two LLaMA layers, then pool to one vector per sample.
    """

    def __init__(
        self,
        base_model_path: str,
        dtype: str,
        first_layer_idx: int,
        middle_layer_idx: int,
        task_names: Sequence[str],
        lora_paths: Optional[Dict[str, str]] = None,
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
        self.cached_first_before_attn = None
        self.cached_mid_before_attn = None
        self.apply_task_lora_for_mid = bool(apply_task_lora_for_mid)
        self.task_to_expert_id = {task: idx + 1 for idx, task in enumerate(task_names)}

        if self.apply_task_lora_for_mid:
            if not lora_paths:
                raise ValueError("apply_task_lora_for_mid=True requires lora_paths")
            self.model = patch_llama_with_hard_routed_lora(
                self.model,
                num_experts=1 + len(task_names),
                r=int(r),
                alpha=int(alpha),
            )
            for task in task_names:
                if task not in lora_paths or not lora_paths[task]:
                    raise KeyError(f"Missing LoRA path for task: {task}")
                load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])

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
        return hidden_states[batch_idx, last_idx, :]

    @torch.no_grad()
    def extract_vectors(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        task_name: Optional[str] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.cached_first_before_attn = None
        self.cached_mid_before_attn = None

        if self.apply_task_lora_for_mid:
            set_all_experts(self.model, NULL_EXPERT_ID)
            if task_name is not None:
                if task_name not in self.task_to_expert_id:
                    raise KeyError(f"Unknown task_name={task_name}")
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

        if self.cached_first_before_attn is None or self.cached_mid_before_attn is None:
            raise RuntimeError("Failed to capture first/mid hidden states")

        first_vec = self.gather_last_valid(self.cached_first_before_attn, attention_mask)
        mid_vec = self.gather_last_valid(self.cached_mid_before_attn, attention_mask)
        return first_vec, mid_vec


class InternalTwoRouterCompactModel(nn.Module):
    def __init__(self, bert_init: str, llama_hidden_size: int, router_dim: int, num_tasks: int):
        super().__init__()
        self.bert = BertExternalEncoder(bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        self.router_first = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=num_tasks,
        )
        self.router_mid = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=num_tasks,
        )

    def encode_bert(self, bert_input_ids, bert_attention_mask, bert_token_type_ids):
        return self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )

    def forward_first(self, bert_prev, bert_last, bert_attention_mask, first_vec):
        logits_first = self.router_first(
            llama_vec=first_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        return logits_first

    def forward_mid(self, bert_prev, bert_last, bert_attention_mask, mid_vec):
        logits_mid = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        return logits_mid

    def forward(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, first_vec, mid_vec):
        bert_prev, bert_last = self.encode_bert(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
        )
        logits_first = self.forward_first(
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
            first_vec=first_vec,
        )
        logits_mid = self.forward_mid(
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
            mid_vec=mid_vec,
        )
        return logits_first, logits_mid


def set_trainable(model, mode: str, freeze_bert: bool = False):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "stage1":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_first.parameters():
            p.requires_grad = True
    elif mode == "stage2":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_mid.parameters():
            p.requires_grad = True
    elif mode == "joint":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_first.parameters():
            p.requires_grad = True
        for p in model.router_mid.parameters():
            p.requires_grad = True
    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_loss(logits_first, logits_mid, labels, mode: str):
    ce = nn.CrossEntropyLoss()
    if mode == "stage1":
        return ce(logits_first, labels)
    if mode == "stage2":
        return ce(logits_mid, labels)
    if mode == "joint":
        return 0.5 * (ce(logits_first, labels) + ce(logits_mid, labels))
    raise ValueError(f"Unknown mode: {mode}")


def extract_batch_vectors(
    extractor: LlamaVectorExtractor,
    llm_tokenizer,
    texts: List[str],
    device: torch.device,
    max_llama_len: int,
    apply_task_lora_for_mid: bool,
    mid_task_names: Optional[List[str]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    enc = llm_tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_llama_len,
    )
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)

    bs = len(texts)
    hidden_size = extractor.model.config.hidden_size
    first_vec = torch.empty(bs, hidden_size, dtype=torch.float32, device=device)
    mid_vec = torch.empty(bs, hidden_size, dtype=torch.float32, device=device)

    if not apply_task_lora_for_mid:
        first_vec, mid_vec = extractor.extract_vectors(input_ids, attention_mask)
        return first_vec.to(torch.float32), mid_vec.to(torch.float32)

    base_first, _ = extractor.extract_vectors(input_ids, attention_mask, task_name=None)
    first_vec.copy_(base_first.to(torch.float32))

    if mid_task_names is None:
        _, base_mid = extractor.extract_vectors(input_ids, attention_mask, task_name=None)
        mid_vec.copy_(base_mid.to(torch.float32))
        return first_vec, mid_vec

    task_to_indices = defaultdict(list)
    for i, task in enumerate(mid_task_names):
        task_to_indices[task].append(i)

    for task, indices in task_to_indices.items():
        sub_ids = input_ids[indices]
        sub_mask = attention_mask[indices]
        _, sub_mid = extractor.extract_vectors(sub_ids, sub_mask, task_name=task)
        mid_vec[indices] = sub_mid.to(torch.float32)

    return first_vec, mid_vec


@torch.no_grad()
def evaluate(
    extractor: LlamaVectorExtractor,
    model: InternalTwoRouterCompactModel,
    loader: DataLoader,
    llm_tokenizer,
    bert_tokenizer,
    device: torch.device,
    max_llama_len: int,
    max_bert_len: int,
    mode: str,
    apply_task_lora_for_mid: bool,
    task_names: Sequence[str],
):
    model.eval()

    total = 0
    total_loss = 0.0
    correct_first = 0
    correct_mid = 0

    progress = tqdm(loader, total=len(loader), desc="eval", dynamic_ncols=True) if tqdm is not None else loader
    for batch in progress:
        labels = batch.labels.to(device)
        bert_enc = bert_tokenizer(
            batch.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_bert_len,
        )
        bert_input_ids = bert_enc["input_ids"].to(device)
        bert_attention_mask = bert_enc["attention_mask"].to(device)
        bert_token_type_ids = bert_enc.get("token_type_ids")
        if bert_token_type_ids is not None:
            bert_token_type_ids = bert_token_type_ids.to(device)

        first_vec, mid_vec = extract_batch_vectors(
            extractor=extractor,
            llm_tokenizer=llm_tokenizer,
            texts=batch.texts,
            device=device,
            max_llama_len=max_llama_len,
            apply_task_lora_for_mid=apply_task_lora_for_mid,
        )
        bert_prev, bert_last = model.encode_bert(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
        )
        logits_first = model.forward_first(
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
            first_vec=first_vec,
        )

        if apply_task_lora_for_mid:
            pred_first = logits_first.argmax(dim=-1)
            pred_task_names = [task_names[idx] for idx in pred_first.cpu().tolist()]
            _, mid_vec = extract_batch_vectors(
                extractor=extractor,
                llm_tokenizer=llm_tokenizer,
                texts=batch.texts,
                device=device,
                max_llama_len=max_llama_len,
                apply_task_lora_for_mid=True,
                mid_task_names=pred_task_names,
            )

        logits_mid = model.forward_mid(
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
            mid_vec=mid_vec,
        )
        loss = compute_loss(logits_first, logits_mid, labels, mode)

        pred_first = logits_first.argmax(dim=-1)
        pred_mid = logits_mid.argmax(dim=-1)
        bs = labels.size(0)
        total += bs
        total_loss += loss.item() * bs
        correct_first += (pred_first == labels).sum().item()
        correct_mid += (pred_mid == labels).sum().item()

        if tqdm is not None:
            progress.set_postfix(
                loss=f"{(total_loss / max(total, 1)):.4f}",
                first=f"{(correct_first / max(total, 1)):.4f}",
                mid=f"{(correct_mid / max(total, 1)):.4f}",
            )

    acc_first = correct_first / max(total, 1)
    acc_mid = correct_mid / max(total, 1)
    return {
        "loss": total_loss / max(total, 1),
        "acc_first": acc_first,
        "acc_mid": acc_mid,
        "score": acc_first if mode == "stage1" else acc_mid if mode == "stage2" else 0.5 * (acc_first + acc_mid),
    }


def save_ckpt(
    model: InternalTwoRouterCompactModel,
    out_dir: str,
    task_names: Sequence[str],
    first_layer_idx: int,
    middle_layer_idx: int,
    max_bert_len: int,
    max_llama_len: int,
    metrics: Dict,
    epoch: int,
    mode: str,
    freeze_bert: bool,
    apply_task_lora_for_mid: bool,
):
    os.makedirs(out_dir, exist_ok=True)
    model.bert.encoder.save_pretrained(os.path.join(out_dir, "encoder"))
    torch.save(
        {
            "bert_encoder": model.bert.state_dict(),
            "router_first": model.router_first.state_dict(),
            "router_mid": model.router_mid.state_dict(),
        },
        os.path.join(out_dir, "router_heads.pt"),
    )
    save_json(
        {
            "task_names": list(task_names),
            "first_layer_idx": first_layer_idx,
            "middle_layer_idx": middle_layer_idx,
            "router_max_len": max_bert_len,
            "max_llama_len": max_llama_len,
            "router_feature_type": "compact_last_valid_token_vector",
            "mode": mode,
            "freeze_bert": freeze_bert,
            "apply_task_lora_for_mid": apply_task_lora_for_mid,
        },
        os.path.join(out_dir, "router_config.json"),
    )
    save_json({"best_epoch": epoch, "metrics": metrics}, os.path.join(out_dir, "best_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task_names", type=str, default=None, help="comma-separated task names")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--bert_init", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["stage1", "stage2", "joint"], default="joint")
    parser.add_argument("--load_from", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--max_llama_len", type=int, default=512)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--balanced_train", action="store_true")
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")

    parser.add_argument("--apply_task_lora_for_mid", action="store_true")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora_iwslt", type=str, default=None)
    parser.add_argument("--lora_medmcqa", type=str, default=None)
    parser.add_argument("--lora_race", type=str, default=None)
    parser.add_argument("--lora_squad2", type=str, default=None)
    parser.add_argument("--lora_sst2", type=str, default=None)

    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="router_compact")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="comma-separated wandb tags")
    args = parser.parse_args()

    global tqdm
    if args.disable_tqdm:
        tqdm = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    requested_tasks = parse_csv_arg(args.task_names)
    task_names = discover_tasks(args.data_root, requested_tasks)
    print(f"[INFO] task_names={task_names}")

    train_ds = RouterTextDataset(
        data_root=args.data_root,
        split_name="train",
        task_names=task_names,
        max_samples=args.max_train_samples,
        balanced=args.balanced_train,
        seed=args.seed,
    )
    val_ds = RouterTextDataset(
        data_root=args.data_root,
        split_name="validation",
        task_names=task_names,
        max_samples=args.max_val_samples,
        balanced=False,
        seed=args.seed,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_init)

    lora_paths = None
    if args.apply_task_lora_for_mid:
        lora_paths = {
            "iwslt2017": args.lora_iwslt,
            "medmcqa": args.lora_medmcqa,
            "race": args.lora_race,
            "squad2": args.lora_squad2,
            "sst2": args.lora_sst2,
        }
        missing = [task for task in task_names if not lora_paths.get(task)]
        if missing:
            raise ValueError(f"apply_task_lora_for_mid=True requires LoRA paths for all tasks. Missing: {missing}")

    extractor = LlamaVectorExtractor(
        base_model_path=args.base_model_path,
        dtype=args.dtype,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        task_names=task_names,
        lora_paths=lora_paths,
        r=args.r,
        alpha=args.alpha,
        apply_task_lora_for_mid=args.apply_task_lora_for_mid,
    ).to(device)

    llama_hidden_size = extractor.model.config.hidden_size
    model = InternalTwoRouterCompactModel(
        bert_init=args.bert_init,
        llama_hidden_size=llama_hidden_size,
        router_dim=args.router_dim,
        num_tasks=len(task_names),
    ).to(device)

    if args.load_from is not None:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[LOAD] loaded from {args.load_from}")

    set_trainable(model, args.mode, freeze_bert=args.freeze_bert)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = max(steps_per_epoch * args.epochs, 1)
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise ImportError("--wandb was set but wandb is not installed") from e
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=parse_csv_arg(args.wandb_tags),
            config=vars(args),
            dir=args.out_dir,
        )
        print(f"[INFO] wandb enabled project={args.wandb_project}")

    print(f"[INFO] train_size={len(train_ds)}")
    print(f"[INFO] val_size={len(val_ds)}")
    print(f"[INFO] mode={args.mode}")
    print(f"[INFO] freeze_bert={args.freeze_bert}")
    print(f"[INFO] apply_task_lora_for_mid={args.apply_task_lora_for_mid}")

    best_score = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_total = 0
        running_first = 0
        running_mid = 0

        progress = tqdm(train_loader, total=len(train_loader), desc=f"train {epoch+1}/{args.epochs}", dynamic_ncols=True) if tqdm is not None else train_loader
        for step, batch in enumerate(progress):
            labels = batch.labels.to(device)
            bert_enc = bert_tokenizer(
                batch.texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_bert_len,
            )
            bert_input_ids = bert_enc["input_ids"].to(device)
            bert_attention_mask = bert_enc["attention_mask"].to(device)
            bert_token_type_ids = bert_enc.get("token_type_ids")
            if bert_token_type_ids is not None:
                bert_token_type_ids = bert_token_type_ids.to(device)

            first_vec, mid_vec = extract_batch_vectors(
                extractor=extractor,
                llm_tokenizer=llm_tokenizer,
                texts=batch.texts,
                device=device,
                max_llama_len=args.max_llama_len,
                apply_task_lora_for_mid=args.apply_task_lora_for_mid,
            )
            bert_prev, bert_last = model.encode_bert(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
            )
            logits_first = model.forward_first(
                bert_prev=bert_prev,
                bert_last=bert_last,
                bert_attention_mask=bert_attention_mask,
                first_vec=first_vec,
            )

            if args.apply_task_lora_for_mid:
                with torch.no_grad():
                    pred_first = logits_first.argmax(dim=-1)
                    pred_task_names = [task_names[idx] for idx in pred_first.detach().cpu().tolist()]
                    _, mid_vec = extract_batch_vectors(
                        extractor=extractor,
                        llm_tokenizer=llm_tokenizer,
                        texts=batch.texts,
                        device=device,
                        max_llama_len=args.max_llama_len,
                        apply_task_lora_for_mid=True,
                        mid_task_names=pred_task_names,
                    )

            logits_mid = model.forward_mid(
                bert_prev=bert_prev,
                bert_last=bert_last,
                bert_attention_mask=bert_attention_mask,
                mid_vec=mid_vec,
            )

            loss = compute_loss(logits_first, logits_mid, labels, args.mode)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_first = logits_first.argmax(dim=-1)
            pred_mid = logits_mid.argmax(dim=-1)
            bs = labels.size(0)
            running_total += bs
            running_loss += loss.item() * bs
            running_first += (pred_first == labels).sum().item()
            running_mid += (pred_mid == labels).sum().item()

            if tqdm is not None:
                progress.set_postfix(
                    loss=f"{(running_loss / max(running_total, 1)):.4f}",
                    first=f"{(running_first / max(running_total, 1)):.4f}",
                    mid=f"{(running_mid / max(running_total, 1)):.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                )

            if step % args.log_every == 0:
                print(
                    f"[TRAIN] epoch={epoch+1} step={step}/{len(train_loader)} "
                    f"loss={loss.item():.6f} lr={scheduler.get_last_lr()[0]:.8f}"
                )

        train_metrics = {
            "loss": running_loss / max(running_total, 1),
            "acc_first": running_first / max(running_total, 1),
            "acc_mid": running_mid / max(running_total, 1),
        }
        train_metrics["score"] = (
            train_metrics["acc_first"]
            if args.mode == "stage1"
            else train_metrics["acc_mid"]
            if args.mode == "stage2"
            else 0.5 * (train_metrics["acc_first"] + train_metrics["acc_mid"])
        )

        val_metrics = evaluate(
            extractor=extractor,
            model=model,
            loader=val_loader,
            llm_tokenizer=llm_tokenizer,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_llama_len=args.max_llama_len,
            max_bert_len=args.max_bert_len,
            mode=args.mode,
            apply_task_lora_for_mid=args.apply_task_lora_for_mid,
            task_names=task_names,
        )

        print(
            f"[EVAL] epoch={epoch+1} "
            f"train_loss={train_metrics['loss']:.6f} "
            f"train_first={train_metrics['acc_first']:.4f} "
            f"train_mid={train_metrics['acc_mid']:.4f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_first={val_metrics['acc_first']:.4f} "
            f"val_mid={val_metrics['acc_mid']:.4f} "
            f"score={val_metrics['score']:.4f}"
        )

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/first_acc": train_metrics["acc_first"],
                    "train/mid_acc": train_metrics["acc_mid"],
                    "train/score": train_metrics["score"],
                    "train/lr": scheduler.get_last_lr()[0],
                    "val/loss": val_metrics["loss"],
                    "val/first_acc": val_metrics["acc_first"],
                    "val/mid_acc": val_metrics["acc_mid"],
                    "val/score": val_metrics["score"],
                    "val/best_score": max(best_score, val_metrics["score"]),
                }
            )

        improved = val_metrics["score"] > (best_score + args.early_stop_min_delta)
        if improved:
            best_score = val_metrics["score"]
            best_epoch = epoch + 1
            no_improve_epochs = 0
            save_ckpt(
                model=model,
                out_dir=args.out_dir,
                task_names=task_names,
                first_layer_idx=args.first_layer_idx,
                middle_layer_idx=args.middle_layer_idx,
                max_bert_len=args.max_bert_len,
                max_llama_len=args.max_llama_len,
                metrics={"epoch": epoch + 1, "train": train_metrics, "val": val_metrics},
                epoch=epoch + 1,
                mode=args.mode,
                freeze_bert=args.freeze_bert,
                apply_task_lora_for_mid=args.apply_task_lora_for_mid,
            )
            print(f"[SAVE] best checkpoint updated at epoch={epoch+1}")
        else:
            no_improve_epochs += 1
            print(
                f"[EARLY_STOP] no improvement for {no_improve_epochs} epoch(s). "
                f"best_score={best_score:.4f} at epoch={best_epoch}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(f"[EARLY_STOP] stop training because patience={args.early_stop_patience} is reached.")
                break

        if args.save_every_epoch:
            epoch_dir = os.path.join(args.out_dir, f"epoch_{epoch+1}")
            save_ckpt(
                model=model,
                out_dir=epoch_dir,
                task_names=task_names,
                first_layer_idx=args.first_layer_idx,
                middle_layer_idx=args.middle_layer_idx,
                max_bert_len=args.max_bert_len,
                max_llama_len=args.max_llama_len,
                metrics={"epoch": epoch + 1, "train": train_metrics, "val": val_metrics},
                epoch=epoch + 1,
                mode=args.mode,
                freeze_bert=args.freeze_bert,
                apply_task_lora_for_mid=args.apply_task_lora_for_mid,
            )

    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_score"] = best_score
        wandb_run.finish()

    print(f"[DONE] best_score={best_score:.4f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
