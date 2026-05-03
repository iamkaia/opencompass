import argparse
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_backbone_specs import get_decoder_layers, infer_backbone_spec
from opencompass.models.router_moe_components import (
    BertExternalEncoder,
    CompactCrossAttentionRouter,
    PromptVectorExtractor,
)
from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
)


DEFAULT_EXPERT_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_tasks(data_root: str, requested_tasks: Optional[Sequence[str]] = None) -> List[str]:
    if requested_tasks:
        tasks = [str(task).strip() for task in requested_tasks if str(task).strip()]
    else:
        tasks = []
        if os.path.isdir(data_root):
            for name in sorted(os.listdir(data_root)):
                full = os.path.join(data_root, name)
                if os.path.isdir(full):
                    tasks.append(name)
    if not tasks:
        raise ValueError(f"No tasks found under data_root={data_root}")
    return tasks


def discover_expert_names(
    requested_experts: Optional[Sequence[str]],
    all_lora_paths: Dict[str, Optional[str]],
) -> tuple[List[str], str]:
    if requested_experts:
        expert_names = [str(name).strip() for name in requested_experts if str(name).strip()]
        source = "manual"
    else:
        expert_names = [name for name, path in all_lora_paths.items() if path]
        if not expert_names:
            expert_names = list(DEFAULT_EXPERT_NAMES)
            source = "default"
        else:
            source = "inferred_from_lora_paths"
    return expert_names, source


class RouterEvalDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tasks: Sequence[str],
        task2id: Dict[str, int],
        max_samples: Optional[int] = None,
    ):
        self.items = []
        for task in tasks:
            path = os.path.join(data_root, task, f"{split}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")
            rows = read_jsonl(path)
            if max_samples is not None:
                rows = rows[: int(max_samples)]
            for row in rows:
                prompt = row.get("text") or row.get("source_text")
                target = row.get("target") or row.get("answer") or row.get("output")
                if not prompt or target is None:
                    raise ValueError(f"Invalid row in {path}: keys={list(row.keys())}")
                self.items.append(
                    {
                        "task": task,
                        "task_id": task2id[task],
                        "text": str(prompt),
                        "source_text": str(row.get("source_text") or prompt),
                        "target": str(target),
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


@dataclass
class Batch:
    texts: List[str]
    source_texts: List[str]
    targets: List[str]
    task_ids: torch.Tensor
    task_names: List[str]


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            source_texts=[x["source_text"] for x in batch],
            targets=[x["target"] for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            task_names=[x["task"] for x in batch],
        )


class RouterEvalModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        router_bert_init: str,
        lora_paths: Dict[str, str],
        first_layer_idx: int,
        middle_layer_idx: int,
        router_dim: int,
        dtype: str,
        r: int,
        alpha: int,
        expert_names: Sequence[str],
    ):
        super().__init__()
        self.expert_names = list(expert_names)
        self.expert2id = {task: idx for idx, task in enumerate(self.expert_names)}
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.backbone_spec = infer_backbone_spec(self.model)

        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.expert_names),
            r=r,
            alpha=alpha,
        )
        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))
        self.task_to_expert_id = {task: self.expert2id[task] + 1 for task in self.expert_names}

        for task in self.expert_names:
            load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])

        self.vector_extractor = PromptVectorExtractor(
            self.model,
            self.first_layer_idx,
            self.middle_layer_idx,
            pooling="last_token",
            pooling_last_k=1,
        )
        self.bert = BertExternalEncoder(router_bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        llama_hidden_size = self.model.config.hidden_size
        self.router_first = CompactCrossAttentionRouter(llama_hidden_size, bert_hidden_size, router_dim, len(self.expert_names))
        self.router_mid = CompactCrossAttentionRouter(llama_hidden_size, bert_hidden_size, router_dim, len(self.expert_names))

    def load_router_weights(self, ckpt_dir: str):
        cfg_path = os.path.join(ckpt_dir, "router_config.json")
        ckpt_expert_names = None
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            ckpt_expert_names = cfg.get("expert_names") or cfg.get("task_names")
        state = torch.load(os.path.join(ckpt_dir, "router_heads.pt"), map_location="cpu")

        def _load(module: nn.Module, saved_state: Dict[str, torch.Tensor]):
            if not ckpt_expert_names or list(ckpt_expert_names) == list(self.expert_names):
                module.load_state_dict(saved_state)
                return
            current = module.state_dict()
            loaded = {}
            for key, value in current.items():
                if key not in saved_state:
                    continue
                src = saved_state[key]
                if key in ("classifier.weight", "classifier.bias"):
                    remapped = value.clone()
                    for new_idx, task in enumerate(self.expert_names):
                        if task in ckpt_expert_names:
                            old_idx = ckpt_expert_names.index(task)
                            remapped[new_idx] = src[old_idx]
                    loaded[key] = remapped
                else:
                    loaded[key] = src
            module.load_state_dict(loaded, strict=False)

        _load(self.router_first, state["router_first"])
        _load(self.router_mid, state["router_mid"])
        if "bert_encoder" in state:
            self.bert.load_state_dict(state["bert_encoder"], strict=False)

    @torch.no_grad()
    def extract_prompt_vectors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        set_all_experts(self.model, NULL_EXPERT_ID)
        return self.vector_extractor.extract(input_ids, attention_mask)

    @torch.no_grad()
    def score_all_route_pairs(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
        batch_size = input_ids.size(0)
        num_experts = len(self.expert_names)
        loss_matrix = torch.empty(batch_size, num_experts, num_experts, dtype=torch.float32, device=input_ids.device)
        for first_idx, first_name in enumerate(self.expert_names):
            first_eid = self.task_to_expert_id[first_name]
            for mid_idx, mid_name in enumerate(self.expert_names):
                mid_eid = self.task_to_expert_id[mid_name]
                set_all_experts(self.model, NULL_EXPERT_ID)
                set_layer_range_expert(self.model, self.first_layer_idx, self.middle_layer_idx - 1, first_eid)
                set_layer_range_expert(self.model, self.middle_layer_idx, self.num_layers - 1, mid_eid)
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
                loss_matrix[:, first_idx, mid_idx] = compute_sequence_nll(logits, labels)
        set_all_experts(self.model, NULL_EXPERT_ID)
        return loss_matrix

    def forward_router(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, first_vec, mid_vec):
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        logits_first = self.router_first(first_vec, bert_prev, bert_last, bert_attention_mask)
        logits_mid = self.router_mid(mid_vec, bert_prev, bert_last, bert_attention_mask)
        return logits_first, logits_mid


def build_lm_batch(tokenizer, prompts: Sequence[str], targets: Sequence[str], max_length: int, add_eos_to_target: bool):
    pad_id = tokenizer.pad_token_id
    prompt_ids_list, target_ids_list = [], []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if add_eos_to_target and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        prompt_ids_list.append(prompt_ids)
        target_ids_list.append(target_ids)

    max_prompt_len = min(max((len(x) for x in prompt_ids_list), default=1), max_length)
    max_full_len = 1
    full_seqs, full_labels = [], []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        full_ids = (prompt_ids + target_ids)[:max_length]
        usable_prompt_len = min(len(prompt_ids), len(full_ids))
        seq_labels = [-100] * usable_prompt_len + full_ids[usable_prompt_len:]
        full_seqs.append(full_ids)
        full_labels.append(seq_labels)
        max_full_len = max(max_full_len, len(full_ids))

    prompt_input_ids, prompt_attention_masks = [], []
    input_ids, attention_masks, labels = [], [], []
    for prompt_ids, full_ids, seq_labels in zip(prompt_ids_list, full_seqs, full_labels):
        prompt_ids = prompt_ids[:max_prompt_len]
        prompt_pad = max_prompt_len - len(prompt_ids)
        prompt_input_ids.append(prompt_ids + [pad_id] * prompt_pad)
        prompt_attention_masks.append([1] * len(prompt_ids) + [0] * prompt_pad)

        pad_len = max_full_len - len(full_ids)
        input_ids.append(full_ids + [pad_id] * pad_len)
        labels.append(seq_labels + [-100] * pad_len)
        attention_masks.append([1] * len(full_ids) + [0] * pad_len)

    return {
        "prompt_input_ids": torch.tensor(prompt_input_ids, dtype=torch.long),
        "prompt_attention_mask": torch.tensor(prompt_attention_masks, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def compute_sequence_nll(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    vocab_size = shift_logits.size(-1)
    per_token_loss = nn.functional.cross_entropy(
        shift_logits.view(-1, vocab_size),
        shift_labels.view(-1),
        reduction="none",
        ignore_index=-100,
    ).view(shift_labels.size())
    valid_mask = (shift_labels != -100).to(per_token_loss.dtype)
    denom = valid_mask.sum(dim=1).clamp_min(1.0)
    return (per_token_loss * valid_mask).sum(dim=1) / denom


def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


@torch.no_grad()
def evaluate_router(
    model: RouterEvalModel,
    loader: DataLoader,
    llm_tokenizer,
    bert_tokenizer,
    device: torch.device,
    max_llm_len: int,
    max_bert_len: int,
    add_eos_to_target: bool,
    eval_metric: str,
    eval_task_names: Sequence[str],
    expert_names: Sequence[str],
):
    task_to_eval_id = {task: idx for idx, task in enumerate(eval_task_names)}
    expert_to_id = {task: idx for idx, task in enumerate(expert_names)}
    if eval_metric == "task_label":
        missing = [task for task in eval_task_names if task not in expert_to_id]
        if missing:
            raise ValueError(
                f"eval_metric=task_label requires eval tasks to exist in expert_names. Missing: {missing}"
            )

    totals = defaultdict(float)
    per_task = defaultdict(lambda: defaultdict(float))

    for batch in loader:
        lm_batch = build_lm_batch(llm_tokenizer, batch.texts, batch.targets, max_llm_len, add_eos_to_target)
        prompt_input_ids = lm_batch["prompt_input_ids"].to(device)
        prompt_attention_mask = lm_batch["prompt_attention_mask"].to(device)
        input_ids = lm_batch["input_ids"].to(device)
        attention_mask = lm_batch["attention_mask"].to(device)
        labels = lm_batch["labels"].to(device)

        bert_enc = bert_tokenizer(
            batch.source_texts,
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

        first_vec, mid_vec = model.extract_prompt_vectors(prompt_input_ids, prompt_attention_mask)
        logits_first, logits_mid = model.forward_router(
            bert_input_ids,
            bert_attention_mask,
            bert_token_type_ids,
            first_vec.to(torch.float32),
            mid_vec.to(torch.float32),
        )
        pred_first = logits_first.argmax(dim=-1)
        pred_mid = logits_mid.argmax(dim=-1)

        if eval_metric == "best_pair":
            loss_matrix = model.score_all_route_pairs(input_ids, attention_mask, labels)
            flat_best = loss_matrix.view(loss_matrix.size(0), -1).argmin(dim=-1)
            gold_first = flat_best // loss_matrix.size(2)
            gold_mid = flat_best % loss_matrix.size(2)
            batch_loss = loss_matrix.view(loss_matrix.size(0), -1).min(dim=-1).values
        else:
            gold_ids = torch.tensor([expert_to_id[name] for name in batch.task_names], device=device, dtype=torch.long)
            gold_first = gold_ids
            gold_mid = gold_ids
            batch_loss = 0.5 * (
                nn.functional.cross_entropy(logits_first, gold_first, reduction="none") +
                nn.functional.cross_entropy(logits_mid, gold_mid, reduction="none")
            )

        first_correct = (pred_first == gold_first).to(torch.float32)
        mid_correct = (pred_mid == gold_mid).to(torch.float32)
        joint_correct = 0.5 * (first_correct + mid_correct)

        for i, task in enumerate(batch.task_names):
            per_task[task]["count"] += 1
            per_task[task]["loss"] += float(batch_loss[i].item())
            per_task[task]["first_acc"] += float(first_correct[i].item())
            per_task[task]["mid_acc"] += float(mid_correct[i].item())
            per_task[task]["joint_acc"] += float(joint_correct[i].item())

        batch_size = len(batch.task_names)
        totals["count"] += batch_size
        totals["loss"] += float(batch_loss.sum().item())
        totals["first_acc"] += float(first_correct.sum().item())
        totals["mid_acc"] += float(mid_correct.sum().item())
        totals["joint_acc"] += float(joint_correct.sum().item())

    overall = {
        "loss": totals["loss"] / max(totals["count"], 1.0),
        "first_acc": totals["first_acc"] / max(totals["count"], 1.0),
        "mid_acc": totals["mid_acc"] / max(totals["count"], 1.0),
        "joint_acc": totals["joint_acc"] / max(totals["count"], 1.0),
        "count": int(totals["count"]),
    }
    per_task_metrics = {}
    for task, stat in per_task.items():
        count = max(stat["count"], 1.0)
        per_task_metrics[task] = {
            "loss": stat["loss"] / count,
            "first_acc": stat["first_acc"] / count,
            "mid_acc": stat["mid_acc"] / count,
            "joint_acc": stat["joint_acc"] / count,
            "count": int(stat["count"]),
        }
    return overall, per_task_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--split", type=str, default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--eval_metric", type=str, default="best_pair", choices=["best_pair", "task_label"])
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, required=True)
    parser.add_argument("--load_router_ckpt_dir", type=str, required=True)
    parser.add_argument("--lora_iwslt", type=str, default=None)
    parser.add_argument("--lora_medmcqa", type=str, default=None)
    parser.add_argument("--lora_race", type=str, default=None)
    parser.add_argument("--lora_squad2", type=str, default=None)
    parser.add_argument("--lora_sst2", type=str, default=None)
    parser.add_argument("--lora_piqa", type=str, default=None)
    parser.add_argument("--lora_copa", type=str, default=None)
    parser.add_argument("--expert_names", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--add_eos_to_target", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")

    requested_tasks = parse_csv_arg(args.task_names)
    eval_task_names = discover_tasks(args.data_root, requested_tasks)
    print(f"[INFO] eval_task_names={eval_task_names}")

    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        "piqa": args.lora_piqa,
        "copa": args.lora_copa,
    }
    expert_names, source = discover_expert_names(parse_csv_arg(args.expert_names), all_lora_paths)
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] expert_name_source={source}")

    lora_paths = {}
    missing_loras = []
    for task in expert_names:
        path = all_lora_paths.get(task)
        if not path:
            missing_loras.append(task)
        else:
            lora_paths[task] = path
    if missing_loras:
        raise ValueError(f"Missing LoRA paths for selected experts: {missing_loras}")

    eval_ds = RouterEvalDataset(
        data_root=args.data_root,
        split=args.split,
        tasks=eval_task_names,
        task2id={task: idx for idx, task in enumerate(eval_task_names)},
        max_samples=args.max_samples,
    )
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    bert_tokenizer = AutoTokenizer.from_pretrained(args.router_bert_init)

    model = RouterEvalModel(
        base_model_path=args.base_model_path,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        router_dim=args.router_dim,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        expert_names=expert_names,
    )
    model.load_router_weights(args.load_router_ckpt_dir)
    print(f"[INFO] loaded router weights from {args.load_router_ckpt_dir}")
    model.to(device)

    overall, per_task_metrics = evaluate_router(
        model=model,
        loader=eval_loader,
        llm_tokenizer=llm_tokenizer,
        bert_tokenizer=bert_tokenizer,
        device=device,
        max_llm_len=args.max_llm_len,
        max_bert_len=args.max_bert_len,
        add_eos_to_target=args.add_eos_to_target,
        eval_metric=args.eval_metric,
        eval_task_names=eval_task_names,
        expert_names=expert_names,
    )

    print(
        f"[EVAL] metric={args.eval_metric} split={args.split} "
        f"loss={overall['loss']:.4f} first_acc={overall['first_acc']:.4f} "
        f"mid_acc={overall['mid_acc']:.4f} joint_acc={overall['joint_acc']:.4f} "
        f"count={overall['count']}"
    )
    print("[PER_TASK]")
    for task in eval_task_names:
        if task not in per_task_metrics:
            continue
        m = per_task_metrics[task]
        print(
            f"  {task}: loss={m['loss']:.4f} first_acc={m['first_acc']:.4f} "
            f"mid_acc={m['mid_acc']:.4f} joint_acc={m['joint_acc']:.4f} count={m['count']}"
        )


if __name__ == "__main__":
    main()
