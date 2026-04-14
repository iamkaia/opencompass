import argparse
import importlib.util
import json
import math
import os
import random
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


def _load_router_core_module():
    module_path = os.path.join(
        os.path.dirname(__file__),
        "opencompass",
        "models",
        "unified_moe_core_internal_router_compact.py",
    )
    spec = importlib.util.spec_from_file_location("router_core_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load router core module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_router_core = _load_router_core_module()
NULL_EXPERT_ID = _router_core.NULL_EXPERT_ID
BertExternalEncoder = _router_core.BertExternalEncoder
CompactCrossAttentionRouter = _router_core.CompactCrossAttentionRouter
load_lora_into_expert = _router_core.load_lora_into_expert
patch_llama_with_hard_routed_lora = _router_core.patch_llama_with_hard_routed_lora
set_all_experts = _router_core.set_all_experts
set_layer_range_expert = _router_core.set_layer_range_expert


DEFAULT_EXPERT_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


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
            tasks = list(DEFAULT_EXPERT_NAMES)
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
    if not expert_names:
        raise ValueError("No expert names selected.")
    return expert_names, source


class RouterTrainDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tasks: Sequence[str],
        task2id: Dict[str, int],
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        items: List[Dict] = []
        for task in tasks:
            path = os.path.join(data_root, task, f"{split}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")
            rows = read_jsonl(path)
            for row in rows:
                prompt = row.get("text") or row.get("source_text")
                target = row.get("target") or row.get("answer") or row.get("output")
                if not prompt or target is None:
                    raise ValueError(f"Invalid row in {path}: keys={list(row.keys())}")
                items.append(
                    {
                        "task": task,
                        "task_id": task2id[task],
                        "text": str(prompt),
                        "source_text": str(row.get("source_text") or prompt),
                        "target": str(target),
                    }
                )

        rng = random.Random(seed)
        rng.shuffle(items)
        if max_samples is not None:
            items = items[: int(max_samples)]
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


def build_dataset(
    data_root: str,
    split: str,
    requested_tasks: Optional[Sequence[str]],
    max_samples: Optional[int],
    seed: int,
) -> tuple[RouterTrainDataset, List[str]]:
    task_names = discover_tasks(data_root, requested_tasks=requested_tasks)
    task2id = {task: idx for idx, task in enumerate(task_names)}
    dataset = RouterTrainDataset(
        data_root=data_root,
        split=split,
        tasks=task_names,
        task2id=task2id,
        max_samples=max_samples,
        seed=seed,
    )
    return dataset, task_names


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


class PromptVectorExtractor(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        first_layer_idx: int,
        middle_layer_idx: int,
    ):
        super().__init__()
        self.model = model
        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.cached_first = None
        self.cached_mid = None
        self._install_hooks()

    def _install_hooks(self):
        def first_pre_hook(module, args):
            self.cached_first = args[0].detach()
            return None

        def mid_pre_hook(module, args):
            self.cached_mid = args[0].detach()
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
    def extract(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.cached_first = None
        self.cached_mid = None
        _ = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_attentions=False,
            return_dict=True,
        )
        if self.cached_first is None or self.cached_mid is None:
            raise RuntimeError("Failed to capture hidden states for router vectors.")
        first_vec = self.gather_last_valid(self.cached_first, attention_mask)
        mid_vec = self.gather_last_valid(self.cached_mid, attention_mask)
        return first_vec, mid_vec


class JointAnswerSupervisionRouterModel(nn.Module):
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

        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.expert_names),
            r=r,
            alpha=alpha,
        )

        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.num_layers = len(self.model.model.layers)
        self.task_to_expert_id = {task: self.expert2id[task] + 1 for task in self.expert_names}

        for task in self.expert_names:
            if task not in lora_paths:
                raise KeyError(f"Missing LoRA path for task: {task}")
            load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])

        self.vector_extractor = PromptVectorExtractor(
            model=self.model,
            first_layer_idx=self.first_layer_idx,
            middle_layer_idx=self.middle_layer_idx,
        )

        self.bert = BertExternalEncoder(router_bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        llama_hidden_size = self.model.config.hidden_size

        self.router_first = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=len(self.expert_names),
        )
        self.router_mid = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=len(self.expert_names),
        )

    def load_router_weights(self, ckpt_dir: str):
        ckpt_expert_names = None
        cfg_path = os.path.join(ckpt_dir, "router_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                ckpt_cfg = json.load(f)
            ckpt_expert_names = ckpt_cfg.get("expert_names") or ckpt_cfg.get("task_names")
        state = torch.load(os.path.join(ckpt_dir, "router_heads.pt"), map_location="cpu")

        def _load_router_with_task_remap(module: nn.Module, saved_state: Dict[str, torch.Tensor], which: str):
            if not ckpt_expert_names or list(ckpt_expert_names) == list(self.expert_names):
                module.load_state_dict(saved_state)
                return

            current_state = module.state_dict()
            loaded = {}
            for key, value in current_state.items():
                if key not in saved_state:
                    continue
                src = saved_state[key]
                if key == "classifier.weight":
                    remapped = value.clone()
                    for new_idx, task in enumerate(self.expert_names):
                        if task in ckpt_expert_names:
                            old_idx = ckpt_expert_names.index(task)
                            remapped[new_idx] = src[old_idx]
                    loaded[key] = remapped
                elif key == "classifier.bias":
                    remapped = value.clone()
                    for new_idx, task in enumerate(self.expert_names):
                        if task in ckpt_expert_names:
                            old_idx = ckpt_expert_names.index(task)
                            remapped[new_idx] = src[old_idx]
                    loaded[key] = remapped
                else:
                    loaded[key] = src
            module.load_state_dict(loaded, strict=False)
            print(
                f"[INFO] remapped {which} checkpoint expert heads from "
                f"{ckpt_expert_names} to {self.expert_names}"
            )

        _load_router_with_task_remap(self.router_first, state["router_first"], which="router_first")
        _load_router_with_task_remap(self.router_mid, state["router_mid"], which="router_mid")
        if "bert_encoder" in state:
            self.bert.load_state_dict(state["bert_encoder"], strict=False)

    def set_trainable(self, freeze_bert: bool):
        for p in self.router_first.parameters():
            p.requires_grad = True
        for p in self.router_mid.parameters():
            p.requires_grad = True
        for p in self.bert.parameters():
            p.requires_grad = not freeze_bert

    @torch.no_grad()
    def extract_prompt_vectors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        set_all_experts(self.model, NULL_EXPERT_ID)
        return self.vector_extractor.extract(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def score_all_route_pairs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = input_ids.size(0)
        num_tasks = len(self.expert_names)
        loss_matrix = torch.empty(batch_size, num_tasks, num_tasks, dtype=torch.float32, device=input_ids.device)

        for first_tid, first_task in enumerate(self.expert_names):
            first_eid = self.task_to_expert_id[first_task]
            for mid_tid, mid_task in enumerate(self.expert_names):
                mid_eid = self.task_to_expert_id[mid_task]
                set_all_experts(self.model, NULL_EXPERT_ID)
                set_layer_range_expert(self.model, self.first_layer_idx, self.middle_layer_idx - 1, first_eid)
                set_layer_range_expert(self.model, self.middle_layer_idx, self.num_layers - 1, mid_eid)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                    return_dict=True,
                ).logits
                combo_loss = compute_sequence_nll(logits=logits, labels=labels)
                loss_matrix[:, first_tid, mid_tid] = combo_loss

        set_all_experts(self.model, NULL_EXPERT_ID)
        return loss_matrix

    def forward_router(
        self,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
        bert_token_type_ids: Optional[torch.Tensor],
        first_vec: torch.Tensor,
        mid_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        logits_first = self.router_first(
            llama_vec=first_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        logits_mid = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        return logits_first, logits_mid


def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    add_eos_to_target: bool,
) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is required.")

    prompt_ids_list = []
    target_ids_list = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if add_eos_to_target and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        prompt_ids_list.append(prompt_ids)
        target_ids_list.append(target_ids)

    input_ids = []
    labels = []
    attention_masks = []
    prompt_input_ids = []
    prompt_attention_masks = []

    max_prompt_len = min(max((len(x) for x in prompt_ids_list), default=1), max_length)
    max_full_len = 1
    full_seqs = []
    full_labels = []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        full_ids = (prompt_ids + target_ids)[:max_length]
        usable_prompt_len = min(len(prompt_ids), len(full_ids))
        seq_labels = [-100] * usable_prompt_len + full_ids[usable_prompt_len:]
        full_seqs.append(full_ids)
        full_labels.append(seq_labels)
        max_full_len = max(max_full_len, len(full_ids))

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


def compute_router_loss(
    logits_first: torch.Tensor,
    logits_mid: torch.Tensor,
    loss_matrix: torch.Tensor,
    pseudo_ce_weight: float,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor]:
    prob_first = torch.softmax(logits_first, dim=-1)
    prob_mid = torch.softmax(logits_mid, dim=-1)
    joint_prob = prob_first.unsqueeze(2) * prob_mid.unsqueeze(1)
    expected_loss = (joint_prob * loss_matrix).sum(dim=(1, 2)).mean()

    flat_best = loss_matrix.view(loss_matrix.size(0), -1).argmin(dim=-1)
    best_first = flat_best // loss_matrix.size(2)
    best_mid = flat_best % loss_matrix.size(2)

    total_loss = expected_loss
    ce_first = torch.tensor(0.0, device=logits_first.device)
    ce_mid = torch.tensor(0.0, device=logits_first.device)
    if pseudo_ce_weight > 0:
        ce_first = nn.functional.cross_entropy(logits_first, best_first)
        ce_mid = nn.functional.cross_entropy(logits_mid, best_mid)
        total_loss = total_loss + pseudo_ce_weight * 0.5 * (ce_first + ce_mid)

    metrics = {
        "expected_loss": float(expected_loss.detach().item()),
        "pseudo_ce_first": float(ce_first.detach().item()),
        "pseudo_ce_mid": float(ce_mid.detach().item()),
        "best_pair_loss": float(loss_matrix.view(loss_matrix.size(0), -1).min(dim=-1).values.mean().item()),
    }
    return total_loss, metrics, best_first, best_mid


def compute_routing_accuracy_stats(
    pred_first: torch.Tensor,
    pred_mid: torch.Tensor,
    best_first: torch.Tensor,
    best_mid: torch.Tensor,
    task_ids: torch.Tensor,
) -> Dict[str, float]:
    pred_first = pred_first.detach()
    pred_mid = pred_mid.detach()
    best_first = best_first.detach()
    best_mid = best_mid.detach()
    task_ids = task_ids.to(device=pred_first.device)

    first_correct = (pred_first == best_first).float().mean().item()
    mid_correct = (pred_mid == best_mid).float().mean().item()
    self_first_acc = (pred_first == task_ids).float().mean().item()
    self_mid_acc = (pred_mid == task_ids).float().mean().item()
    oracle_first_self_acc = (best_first == task_ids).float().mean().item()
    oracle_mid_self_acc = (best_mid == task_ids).float().mean().item()
    pred_self_pair_acc = ((pred_first == task_ids) & (pred_mid == task_ids)).float().mean().item()
    oracle_self_pair_acc = ((best_first == task_ids) & (best_mid == task_ids)).float().mean().item()
    pair_acc = ((pred_first == best_first) & (pred_mid == best_mid)).float().mean().item()

    return {
        "first_acc": first_correct,
        "mid_acc": mid_correct,
        "joint_acc": 0.5 * (first_correct + mid_correct),
        "pair_acc": pair_acc,
        "self_first_acc": self_first_acc,
        "self_mid_acc": self_mid_acc,
        "self_joint_acc": 0.5 * (self_first_acc + self_mid_acc),
        "self_pair_acc": pred_self_pair_acc,
        "oracle_first_self_acc": oracle_first_self_acc,
        "oracle_mid_self_acc": oracle_mid_self_acc,
        "oracle_self_joint_acc": 0.5 * (oracle_first_self_acc + oracle_mid_self_acc),
        "oracle_self_pair_acc": oracle_self_pair_acc,
    }


def save_runtime_compatible_router_bundle(
    model: JointAnswerSupervisionRouterModel,
    bert_tokenizer,
    output_dir: str,
    state: Dict,
    router_config: Dict,
):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, "router_heads.pt"))
    save_json(router_config, os.path.join(output_dir, "router_config.json"))

    encoder_dir = os.path.join(output_dir, "encoder")
    model.bert.encoder.save_pretrained(encoder_dir)
    bert_tokenizer.save_pretrained(encoder_dir)


@torch.no_grad()
def evaluate(
    model: JointAnswerSupervisionRouterModel,
    loader: DataLoader,
    llm_tokenizer,
    bert_tokenizer,
    device: torch.device,
    max_llm_len: int,
    max_bert_len: int,
    add_eos_to_target: bool,
    pseudo_ce_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}

    progress = make_progress(loader, total=len(loader), desc="eval")
    for batch in progress:
        lm_batch = build_lm_batch(
            tokenizer=llm_tokenizer,
            prompts=batch.texts,
            targets=batch.targets,
            max_length=max_llm_len,
            add_eos_to_target=add_eos_to_target,
        )
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

        first_vec, mid_vec = model.extract_prompt_vectors(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
        )
        loss_matrix = model.score_all_route_pairs(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        logits_first, logits_mid = model.forward_router(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=first_vec.to(torch.float32),
            mid_vec=mid_vec.to(torch.float32),
        )
        loss, _, best_first, best_mid = compute_router_loss(
            logits_first=logits_first,
            logits_mid=logits_mid,
            loss_matrix=loss_matrix,
            pseudo_ce_weight=pseudo_ce_weight,
        )

        pred_first = logits_first.argmax(dim=-1)
        pred_mid = logits_mid.argmax(dim=-1)
        batch_size = pred_first.size(0)
        batch_stats = compute_routing_accuracy_stats(
            pred_first=pred_first,
            pred_mid=pred_mid,
            best_first=best_first,
            best_mid=best_mid,
            task_ids=batch.task_ids,
        )

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for key, value in batch_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size

        if tqdm is not None:
            progress.set_postfix(
                loss=f"{(total_loss / max(total_samples, 1)):.4f}",
                first=f"{(metric_totals.get('first_acc', 0.0) / max(total_samples, 1)):.4f}",
                mid=f"{(metric_totals.get('mid_acc', 0.0) / max(total_samples, 1)):.4f}",
                self_acc=f"{(metric_totals.get('self_joint_acc', 0.0) / max(total_samples, 1)):.4f}",
            )

    denom = max(total_samples, 1)
    result = {"loss": total_loss / denom}
    for key, value in metric_totals.items():
        result[key] = value / denom
    return result


def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


def make_progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="router_train_datasets")
    parser.add_argument(
        "--task_names",
        type=str,
        default=None,
        help="comma-separated task names. default: discover all subdirs under data_root",
    )
    parser.add_argument(
        "--eval_data_root",
        type=str,
        default=None,
        help="optional evaluation dataset root. default: use data_root",
    )
    parser.add_argument(
        "--eval_task_names",
        type=str,
        default=None,
        help="comma-separated eval task names. default: use discovered tasks under eval_data_root",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="skip training and only evaluate router checkpoint on eval datasets",
    )
    parser.add_argument(
        "--expert_names",
        type=str,
        default=None,
        help="comma-separated expert names to route among. default: infer from provided LoRA paths",
    )
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--load_router_ckpt_dir", type=str, default=None)
    parser.add_argument("--lora_iwslt", type=str, default=None)
    parser.add_argument("--lora_medmcqa", type=str, default=None)
    parser.add_argument("--lora_race", type=str, default=None)
    parser.add_argument("--lora_squad2", type=str, default=None)
    parser.add_argument("--lora_sst2", type=str, default=None)
    parser.add_argument("--lora_piqa", type=str, default=None)
    parser.add_argument("--lora_copa", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_val_samples", type=int, default=None)
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--pseudo_ce_weight", type=float, default=0.5)
    parser.add_argument("--add_eos_to_target", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="opencompass")
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

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise ImportError("--wandb was set but wandb is not installed") from e
        wandb_tags = parse_csv_arg(args.wandb_tags)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=wandb_tags,
            config=vars(args),
            dir=args.output_dir,
        )
        print(f"[INFO] wandb enabled project={args.wandb_project}")

    requested_tasks = parse_csv_arg(args.task_names)
    requested_eval_tasks = parse_csv_arg(args.eval_task_names)
    eval_data_root = args.eval_data_root or args.data_root

    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        "piqa": args.lora_piqa,
        "copa": args.lora_copa,
    }
    requested_experts = parse_csv_arg(args.expert_names)
    expert_names, expert_name_source = discover_expert_names(requested_experts, all_lora_paths)
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] expert_name_source={expert_name_source}")
    lora_paths = {}
    missing_loras = []
    for task in expert_names:
        path = all_lora_paths.get(task)
        if not path:
            missing_loras.append(task)
        else:
            lora_paths[task] = path
    if missing_loras:
        raise ValueError(
            "Missing LoRA paths for selected tasks. "
            f"Please provide CLI args for: {missing_loras}"
        )

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    bert_tokenizer = AutoTokenizer.from_pretrained(args.router_bert_init)

    train_task_names: List[str] = []
    train_loader = None
    val_loader = None
    if not args.eval_only:
        train_ds, train_task_names = build_dataset(
            data_root=args.data_root,
            split="train",
            requested_tasks=requested_tasks,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        print(f"[INFO] train_task_names={train_task_names}")
        val_ds, val_task_names = build_dataset(
            data_root=args.data_root,
            split="validation",
            requested_tasks=requested_tasks,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
        print(f"[INFO] val_task_names={val_task_names}")
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
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=Collator(),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        print("[INFO] eval_only=True, skipping training dataset construction")

    eval_ds, eval_task_names = build_dataset(
        data_root=eval_data_root,
        split=args.eval_split,
        requested_tasks=requested_eval_tasks,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(f"[INFO] eval_task_names={eval_task_names}")
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = JointAnswerSupervisionRouterModel(
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
    if args.load_router_ckpt_dir:
        model.load_router_weights(args.load_router_ckpt_dir)
        print(f"[INFO] loaded router weights from {args.load_router_ckpt_dir}")

    model.set_trainable(freeze_bert=args.freeze_bert)
    model.to(device)

    config = vars(args).copy()
    config["train_task_names"] = train_task_names
    config["expert_names"] = expert_names
    config["eval_task_names"] = eval_task_names
    save_json(config, os.path.join(args.output_dir, "train_config.json"))

    if args.eval_only:
        if not args.load_router_ckpt_dir:
            raise ValueError("--eval_only requires --load_router_ckpt_dir")
        eval_metrics = evaluate(
            model=model,
            loader=eval_loader,
            llm_tokenizer=llm_tokenizer,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_llm_len=args.max_llm_len,
            max_bert_len=args.max_bert_len,
            add_eos_to_target=args.add_eos_to_target,
            pseudo_ce_weight=args.pseudo_ce_weight,
        )
        print(
            f"[EVAL] split={args.eval_split} loss={eval_metrics['loss']:.4f} "
            f"first_acc={eval_metrics['first_acc']:.4f} "
            f"mid_acc={eval_metrics['mid_acc']:.4f} "
            f"joint_acc={eval_metrics['joint_acc']:.4f} "
            f"pair_acc={eval_metrics['pair_acc']:.4f} "
            f"self_first={eval_metrics['self_first_acc']:.4f} "
            f"self_mid={eval_metrics['self_mid_acc']:.4f} "
            f"self_joint={eval_metrics['self_joint_acc']:.4f} "
            f"self_pair={eval_metrics['self_pair_acc']:.4f} "
            f"oracle_first_self={eval_metrics['oracle_first_self_acc']:.4f} "
            f"oracle_mid_self={eval_metrics['oracle_mid_self_acc']:.4f} "
            f"oracle_self_pair={eval_metrics['oracle_self_pair_acc']:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log({f"eval/{k}": v for k, v in eval_metrics.items()})
            wandb_run.finish()
        return

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_val = float("inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0

        train_progress = make_progress(
            train_loader,
            total=len(train_loader),
            desc=f"train epoch {epoch}/{args.epochs}",
        )
        for step, batch in enumerate(train_progress, start=1):
            lm_batch = build_lm_batch(
                tokenizer=llm_tokenizer,
                prompts=batch.texts,
                targets=batch.targets,
                max_length=args.max_llm_len,
                add_eos_to_target=args.add_eos_to_target,
            )
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
                max_length=args.max_bert_len,
            )
            bert_input_ids = bert_enc["input_ids"].to(device)
            bert_attention_mask = bert_enc["attention_mask"].to(device)
            bert_token_type_ids = bert_enc.get("token_type_ids")
            if bert_token_type_ids is not None:
                bert_token_type_ids = bert_token_type_ids.to(device)

            with torch.no_grad():
                first_vec, mid_vec = model.extract_prompt_vectors(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                )
                loss_matrix = model.score_all_route_pairs(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

            logits_first, logits_mid = model.forward_router(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=first_vec.to(torch.float32),
                mid_vec=mid_vec.to(torch.float32),
            )
            loss, metrics, best_first, best_mid = compute_router_loss(
                logits_first=logits_first,
                logits_mid=logits_mid,
                loss_matrix=loss_matrix,
                pseudo_ce_weight=args.pseudo_ce_weight,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = len(batch.texts)
            running_loss += loss.item() * batch_size
            running_count += batch_size

            if tqdm is not None:
                batch_stats = compute_routing_accuracy_stats(
                    pred_first=logits_first.argmax(dim=-1),
                    pred_mid=logits_mid.argmax(dim=-1),
                    best_first=best_first,
                    best_mid=best_mid,
                    task_ids=batch.task_ids,
                )
                train_progress.set_postfix(
                    loss=f"{(running_loss / max(running_count, 1)):.4f}",
                    exp=f"{metrics['expected_loss']:.4f}",
                    pair=f"{metrics['best_pair_loss']:.4f}",
                    f_acc=f"{batch_stats['first_acc']:.4f}",
                    m_acc=f"{batch_stats['mid_acc']:.4f}",
                    self_acc=f"{batch_stats['self_joint_acc']:.4f}",
                )

            if step % 10 == 0 or step == len(train_loader):
                pred_first = logits_first.argmax(dim=-1)
                pred_mid = logits_mid.argmax(dim=-1)
                batch_stats = compute_routing_accuracy_stats(
                    pred_first=pred_first,
                    pred_mid=pred_mid,
                    best_first=best_first,
                    best_mid=best_mid,
                    task_ids=batch.task_ids,
                )
                avg_loss = running_loss / max(running_count, 1)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": step + (epoch - 1) * len(train_loader),
                            "train/loss": avg_loss,
                            "train/expected_loss": metrics["expected_loss"],
                            "train/best_pair_loss": metrics["best_pair_loss"],
                            "train/first_acc": batch_stats["first_acc"],
                            "train/mid_acc": batch_stats["mid_acc"],
                            "train/joint_acc": batch_stats["joint_acc"],
                            "train/pair_acc": batch_stats["pair_acc"],
                            "train/self_first_acc": batch_stats["self_first_acc"],
                            "train/self_mid_acc": batch_stats["self_mid_acc"],
                            "train/self_joint_acc": batch_stats["self_joint_acc"],
                            "train/self_pair_acc": batch_stats["self_pair_acc"],
                            "train/oracle_first_self_acc": batch_stats["oracle_first_self_acc"],
                            "train/oracle_mid_self_acc": batch_stats["oracle_mid_self_acc"],
                            "train/oracle_self_joint_acc": batch_stats["oracle_self_joint_acc"],
                            "train/oracle_self_pair_acc": batch_stats["oracle_self_pair_acc"],
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                    )
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} expected={metrics['expected_loss']:.4f} "
                    f"best_pair={metrics['best_pair_loss']:.4f} "
                    f"first_acc={batch_stats['first_acc']:.4f} mid_acc={batch_stats['mid_acc']:.4f} "
                    f"self_first={batch_stats['self_first_acc']:.4f} self_mid={batch_stats['self_mid_acc']:.4f} "
                    f"oracle_self_pair={batch_stats['oracle_self_pair_acc']:.4f}"
                )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            llm_tokenizer=llm_tokenizer,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_llm_len=args.max_llm_len,
            max_bert_len=args.max_bert_len,
            add_eos_to_target=args.add_eos_to_target,
            pseudo_ce_weight=args.pseudo_ce_weight,
        )
        print(
            f"[VAL] epoch={epoch} loss={val_metrics['loss']:.4f} "
            f"first_acc={val_metrics['first_acc']:.4f} "
            f"mid_acc={val_metrics['mid_acc']:.4f} "
            f"self_first={val_metrics['self_first_acc']:.4f} "
            f"self_mid={val_metrics['self_mid_acc']:.4f} "
            f"oracle_self_pair={val_metrics['oracle_self_pair_acc']:.4f}"
        )
        if wandb_run is not None:
            payload = {
                "val/epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/first_acc": val_metrics["first_acc"],
                "val/mid_acc": val_metrics["mid_acc"],
                "val/joint_acc": val_metrics["joint_acc"],
                "val/best_val": min(best_val, val_metrics["loss"]),
            }
            wandb_run.log(payload)

        state = {
            "router_first": model.router_first.state_dict(),
            "router_mid": model.router_mid.state_dict(),
            "bert_encoder": model.bert.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }

        if args.save_every_epoch:
            torch.save(state, os.path.join(args.output_dir, f"router_heads_epoch{epoch}.pt"))

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            save_runtime_compatible_router_bundle(
                model=model,
                bert_tokenizer=bert_tokenizer,
                output_dir=args.output_dir,
                state=state,
                router_config={
                    "task_names": expert_names,
                    "expert_names": expert_names,
                    "train_task_names": train_task_names,
                    "first_layer_idx": args.first_layer_idx,
                    "middle_layer_idx": args.middle_layer_idx,
                    "router_max_len": args.max_bert_len,
                    "router_feature_type": "answer_supervision_prompt_last_valid_token",
                    "best_epoch": best_epoch,
                    "best_val_loss": best_val,
                },
            )
            print(f"[SAVE] best checkpoint updated at epoch={epoch}")

    print(f"[DONE] best_val={best_val:.4f} best_epoch={best_epoch}")
    if wandb_run is not None:
        wandb_run.summary["best_val"] = best_val
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()


if __name__ == "__main__":
    main()
