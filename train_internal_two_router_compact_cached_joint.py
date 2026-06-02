import argparse
import hashlib
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from opencompass.models.router_moe_components import BertExternalEncoder, CompactRouterFeatureEncoder
from task_eval_specs import normalize_router_option_label, router_option_labels
from router_pair_common import (
    add_joint_loss_arguments,
    build_oracle_debug_summary,
    build_routing_summary,
    compute_pair_losses,
    compute_route_score_stats,
    compute_routing_accuracy_stats,
    flatten_routing_summary,
    init_oracle_debug_accumulator,
    normalize_pair_loss_matrix,
    print_oracle_debug_summary,
    print_routing_summary,
    update_oracle_debug_accumulator,
)

FEATURE_CONTRACT_KEYS = (
    "first_layer_idx",
    "middle_layer_idx",
    "router_pooling",
    "router_pooling_last_k",
    "llama_hidden_size",
)


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def prompt_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def parse_task_names(raw: Optional[str], fallback: Optional[Sequence[str]] = None) -> List[str]:
    if raw:
        tasks = [part.strip() for part in raw.split(",") if part.strip()]
        if tasks:
            return tasks
    if fallback:
        return list(fallback)
    raise ValueError("Failed to resolve task names")


def parse_optional_task_names(raw: Optional[str]) -> Optional[List[str]]:
    if raw:
        tasks = [part.strip() for part in raw.split(",") if part.strip()]
        if tasks:
            return tasks
    return None


def parse_float_list(raw: Optional[str], default: Sequence[float]) -> List[float]:
    if raw:
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
        if values:
            return values
    return [float(value) for value in default]


def parse_feature_roots(feature_root: Optional[str], feature_roots: Optional[str]) -> List[str]:
    if feature_roots:
        roots = [part.strip() for part in feature_roots.split(",") if part.strip()]
    elif feature_root:
        roots = [str(feature_root)]
    else:
        roots = []
    if not roots:
        raise ValueError("Set --feature_root or --feature_roots")
    return roots


def metric_float_tag(value: float) -> str:
    return str(float(value)).replace("-", "m").replace(".", "p")


def build_router_target_distribution(
    *,
    pair_logits: torch.Tensor,
    loss_matrix: torch.Tensor,
    correct_matrix: torch.Tensor,
    task_ids: torch.Tensor,
    joint_loss: str,
    loss_normalization: str,
    correct_soft_ce_temperature: float,
    self_preserve_weight: float,
) -> Optional[torch.Tensor]:
    num_tasks = loss_matrix.size(2)
    flat_correct = correct_matrix.to(device=pair_logits.device, dtype=torch.float32).view(loss_matrix.size(0), -1)
    correct_counts = flat_correct.sum(dim=-1, keepdim=True)
    correct_available = correct_counts > 0
    zero_target = torch.zeros_like(flat_correct)

    if str(joint_loss) == "correct_soft_ce":
        target = torch.where(
            correct_available,
            flat_correct / correct_counts.clamp_min(1.0),
            zero_target,
        )
        return target.view(loss_matrix.size(0), num_tasks, num_tasks)

    if str(joint_loss) in {"correct_conf_ce", "correct_conf_ce_plus_margin", "self_preserving_correct_conf_ce"}:
        normalized_loss_matrix = normalize_pair_loss_matrix(
            loss_matrix=loss_matrix,
            method=loss_normalization,
        )
        normalized_flat_loss = normalized_loss_matrix.view(normalized_loss_matrix.size(0), -1)
        temperature = max(float(correct_soft_ce_temperature), 1e-6)
        correct_conf_logits = -normalized_flat_loss / temperature
        correct_conf_logits = correct_conf_logits.masked_fill(flat_correct <= 0, -1e9)
        correct_conf_target = torch.softmax(correct_conf_logits, dim=-1)
        correct_conf_target = torch.where(
            correct_available,
            correct_conf_target,
            zero_target,
        )
        if str(joint_loss) != "self_preserving_correct_conf_ce":
            return correct_conf_target.view(loss_matrix.size(0), num_tasks, num_tasks)

        preserve_weight = min(max(float(self_preserve_weight), 0.0), 1.0)
        task_ids_device = task_ids.to(device=pair_logits.device, dtype=torch.long)
        valid_self_task = (task_ids_device >= 0) & (task_ids_device < num_tasks)
        self_pair_ids = task_ids_device.clamp(min=0, max=max(num_tasks - 1, 0)) * num_tasks + task_ids_device.clamp(
            min=0, max=max(num_tasks - 1, 0)
        )
        self_correct = torch.zeros(loss_matrix.size(0), dtype=torch.bool, device=pair_logits.device)
        self_correct[valid_self_task] = flat_correct.bool().gather(
            1, self_pair_ids.unsqueeze(1)
        ).squeeze(1)[valid_self_task]
        self_target = torch.zeros_like(correct_conf_target)
        self_target.scatter_(1, self_pair_ids.unsqueeze(1), 1.0)
        soft_self_target = preserve_weight * self_target + (1.0 - preserve_weight) * correct_conf_target
        self_preserving_target = torch.where(
            (valid_self_task & self_correct).unsqueeze(1),
            soft_self_target,
            correct_conf_target,
        )
        return self_preserving_target.view(loss_matrix.size(0), num_tasks, num_tasks)

    return None


def load_cache_feature_contract(root: str, manifest: Dict) -> Dict:
    config_path = os.path.join(root, "cache_config.json")
    cache_config = {}
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            cache_config = json.load(f)
    contract = {
        key: manifest.get(key, cache_config.get(key))
        for key in FEATURE_CONTRACT_KEYS
    }
    missing = [key for key, value in contract.items() if value is None and key != "llama_hidden_size"]
    if missing:
        raise ValueError(
            f"Missing cached feature contract fields {missing} under {root}. "
            "Rebuild this cache with build_cached_router_pair_dataset.py."
        )
    contract["first_layer_idx"] = int(contract["first_layer_idx"])
    contract["middle_layer_idx"] = int(contract["middle_layer_idx"])
    contract["router_pooling"] = str(contract["router_pooling"])
    contract["router_pooling_last_k"] = int(contract["router_pooling_last_k"])
    if contract["llama_hidden_size"] is not None:
        contract["llama_hidden_size"] = int(contract["llama_hidden_size"])
    return contract


def task_option_labels(task_name: str) -> Optional[List[str]]:
    return router_option_labels(task_name)


def normalize_task_label(task_name: str, target: str) -> str:
    return normalize_router_option_label(task_name, target)


def batch_gold_option_indices(tasks: Sequence[str], targets: Sequence[str], device) -> tuple[torch.Tensor, torch.Tensor]:
    gold_indices = []
    valid = []
    for task, target in zip(tasks, targets):
        labels = task_option_labels(task)
        gold = normalize_task_label(task, target)
        if labels and gold in labels:
            gold_indices.append(labels.index(gold))
            valid.append(True)
        else:
            gold_indices.append(0)
            valid.append(False)
    return (
        torch.tensor(gold_indices, dtype=torch.long, device=device),
        torch.tensor(valid, dtype=torch.bool, device=device),
    )

#####它會讀 feature_root/<split>/manifest.json，再把 chunk 檔載進來。重要的是它在 66-81 行 (line 66) 做了兩件事：
#####如果你只想訓練部分 task，它會先把 loss_matrix slice 成較小的子矩陣
####再從 slice 後的 loss_matrix 重新算一次 pair_label / first_label / mid_label
####所以 cache 可以先建大，再在 trainer 端選 task 子集
class CachedLossMatrixDataset(Dataset):
    def __init__(
        self,
        feature_root: str | Sequence[str],
        split: str,
        selected_sample_task_names: Optional[Sequence[str]] = None,
        selected_expert_names: Optional[Sequence[str]] = None,
    ):
        feature_roots = [feature_root] if isinstance(feature_root, str) else [str(root) for root in feature_root]
        manifests = []
        source_sample_task_names = []
        source_expert_names = None
        expected_manifest_fields = None
        expected_feature_contract = None
        for root in feature_roots:
            split_dir = os.path.join(root, split)
            manifest_path = os.path.join(split_dir, "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            current_expert_names = list(manifest.get("expert_names") or manifest.get("task_names") or [])
            if not current_expert_names:
                raise ValueError(f"Missing expert_names in {manifest_path}")
            if source_expert_names is None:
                source_expert_names = current_expert_names
            elif current_expert_names != source_expert_names:
                raise ValueError(
                    f"Incompatible expert_names across cache roots: {manifest_path} has {current_expert_names}, "
                    f"expected {source_expert_names}"
                )
            current_fields = {
                key: manifest.get(key)
                for key in ("score_mode", "option_prob_normalization", "sst2_option_labels")
            }
            if expected_manifest_fields is None:
                expected_manifest_fields = current_fields
            elif current_fields != expected_manifest_fields:
                raise ValueError(
                    f"Incompatible scoring fields across cache roots: {manifest_path} has {current_fields}, "
                    f"expected {expected_manifest_fields}"
                )
            current_feature_contract = load_cache_feature_contract(root, manifest)
            if expected_feature_contract is None:
                expected_feature_contract = current_feature_contract
            elif current_feature_contract != expected_feature_contract:
                raise ValueError(
                    f"Incompatible feature contract across cache roots: {manifest_path} has "
                    f"{current_feature_contract}, expected {expected_feature_contract}"
                )
            for name in manifest.get("task_names") or []:
                if name not in source_sample_task_names:
                    source_sample_task_names.append(name)
            manifests.append((root, manifest))

        if not source_sample_task_names:
            raise ValueError(f"Missing task_names in cache roots: {feature_roots}")

        if selected_sample_task_names:
            resolved_sample_task_names = [str(name) for name in selected_sample_task_names]
        else:
            resolved_sample_task_names = list(source_sample_task_names)

        if selected_expert_names:
            resolved_expert_names = [str(name) for name in selected_expert_names]
        else:
            resolved_expert_names = list(source_expert_names)

        missing_samples = [name for name in resolved_sample_task_names if name not in source_sample_task_names]
        if missing_samples:
            raise ValueError(
                f"Requested sample_task_names={missing_samples} are not present in cached feature manifest "
                f"{manifest_path}. Available={source_sample_task_names}"
            )
        missing_experts = [name for name in resolved_expert_names if name not in source_expert_names]
        if missing_experts:
            raise ValueError(
                f"Requested expert_names={missing_experts} are not present in cached feature manifest "
                f"{manifest_path}. Available={source_expert_names}"
            )

        selected_expert_indices = [source_expert_names.index(name) for name in resolved_expert_names]
        expert2id = {name: idx for idx, name in enumerate(resolved_expert_names)}

        self.items = []
        for root, manifest in manifests:
            split_dir = os.path.join(root, split)
            for fn in manifest["files"]:
                payload = torch.load(os.path.join(split_dir, fn), map_location="cpu")
                for item in payload["items"]:
                    task_name = str(item["task"])
                    if task_name not in resolved_sample_task_names:
                        continue

                    loss_matrix = item["loss_matrix"]
                    sliced_loss_matrix = loss_matrix.index_select(0, torch.tensor(selected_expert_indices)).index_select(
                        1, torch.tensor(selected_expert_indices)
                    )
                    correct_matrix = item.get("correct_matrix")
                    has_correct_matrix = correct_matrix is not None
                    if has_correct_matrix:
                        sliced_correct_matrix = correct_matrix.index_select(
                            0, torch.tensor(selected_expert_indices)
                        ).index_select(1, torch.tensor(selected_expert_indices))
                    else:
                        sliced_correct_matrix = torch.zeros_like(sliced_loss_matrix, dtype=torch.bool)
                    flat_loss = sliced_loss_matrix.view(-1)
                    best_pair = int(flat_loss.argmin().item())
                    num_tasks = len(resolved_expert_names)
                    best_first = best_pair // num_tasks
                    best_mid = best_pair % num_tasks
                    remapped = dict(item)
                    ####把 sample 的 task name 對應到 expert id。
                    remapped["task_id"] = int(expert2id.get(task_name, -1))
                    remapped["loss_matrix"] = sliced_loss_matrix
                    remapped["correct_matrix"] = sliced_correct_matrix.to(torch.bool)
                    option_prob_matrix = item.get("option_prob_matrix")
                    if option_prob_matrix is not None:
                        remapped["option_prob_matrix"] = option_prob_matrix.index_select(
                            0, torch.tensor(selected_expert_indices)
                        ).index_select(1, torch.tensor(selected_expert_indices))
                    else:
                        remapped["option_prob_matrix"] = None
                    remapped["has_correct_matrix"] = bool(has_correct_matrix)
                    remapped["pair_label"] = best_pair
                    remapped["first_label"] = best_first
                    remapped["mid_label"] = best_mid
                    remapped["item_id"] = f"{os.path.basename(root)}:{split}:{len(self.items):08d}"
                    self.items.append(remapped)
        if not self.items:
            raise ValueError(f"No items loaded from cache roots: {feature_roots}")
        self.sample_task_names = resolved_sample_task_names
        self.expert_names = resolved_expert_names
        self.task_names = resolved_expert_names
        self.source_sample_task_names = source_sample_task_names
        self.source_expert_names = source_expert_names
        self.feature_roots = feature_roots
        self.feature_contract = expected_feature_contract

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "text": item["text"],
            "prompt_text": item.get("prompt_text", item["text"]),
            "target": item.get("target", ""),
            "item_id": item.get("item_id", str(idx)),
            "task": item["task"],
            "task_id": int(item["task_id"]),
            "first_vec": item["first_vec"],
            "mid_vec": item["mid_vec"],
            "loss_matrix": item["loss_matrix"],
            "correct_matrix": item["correct_matrix"],
            "option_prob_matrix": item.get("option_prob_matrix"),
            "prediction_matrix": item.get("prediction_matrix"),
            "has_correct_matrix": bool(item.get("has_correct_matrix", False)),
            "pair_label": int(item["pair_label"]),
            "first_label": int(item["first_label"]),
            "mid_label": int(item["mid_label"]),
        }


@dataclass
class Batch:
    texts: List[str]
    prompt_texts: List[str]
    targets: List[str]
    item_ids: List[str]
    tasks: List[str]
    task_ids: torch.Tensor
    first_vec: torch.Tensor
    mid_vec: torch.Tensor
    loss_matrix: torch.Tensor
    correct_matrix: torch.Tensor
    option_prob_matrix: torch.Tensor
    prediction_matrices: List[Optional[List[List[str]]]]
    option_prob_available: torch.Tensor
    correct_available: torch.Tensor
    pair_labels: torch.Tensor
    first_labels: torch.Tensor
    mid_labels: torch.Tensor


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        max_num_options = 1
        for item in batch:
            option_prob_matrix = item.get("option_prob_matrix")
            if option_prob_matrix is not None:
                max_num_options = max(max_num_options, int(option_prob_matrix.size(-1)))
        option_prob_tensors = []
        option_prob_available = []
        for item in batch:
            option_prob_matrix = item.get("option_prob_matrix")
            if option_prob_matrix is None:
                num_tasks = item["loss_matrix"].size(0)
                option_prob_tensors.append(torch.zeros(num_tasks, num_tasks, max_num_options, dtype=torch.float32))
                option_prob_available.append(False)
            else:
                padded = torch.zeros(
                    option_prob_matrix.size(0),
                    option_prob_matrix.size(1),
                    max_num_options,
                    dtype=torch.float32,
                )
                padded[..., : option_prob_matrix.size(-1)] = option_prob_matrix.to(torch.float32)
                option_prob_tensors.append(padded)
                option_prob_available.append(True)
        return Batch(
            texts=[x["text"] for x in batch],
            prompt_texts=[str(x["prompt_text"]) for x in batch],
            targets=[str(x["target"]) for x in batch],
            item_ids=[str(x["item_id"]) for x in batch],
            tasks=[str(x["task"]) for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            first_vec=torch.stack([x["first_vec"] for x in batch], dim=0).to(torch.float32),
            mid_vec=torch.stack([x["mid_vec"] for x in batch], dim=0).to(torch.float32),
            loss_matrix=torch.stack([x["loss_matrix"] for x in batch], dim=0).to(torch.float32),
            correct_matrix=torch.stack([x["correct_matrix"] for x in batch], dim=0).to(torch.bool),
            option_prob_matrix=torch.stack(option_prob_tensors, dim=0).to(torch.float32),
            prediction_matrices=[x.get("prediction_matrix") for x in batch],
            option_prob_available=torch.tensor(option_prob_available, dtype=torch.bool),
            correct_available=torch.tensor([bool(x["has_correct_matrix"]) for x in batch], dtype=torch.bool),
            pair_labels=torch.tensor([x["pair_label"] for x in batch], dtype=torch.long),
            first_labels=torch.tensor([x["first_label"] for x in batch], dtype=torch.long),
            mid_labels=torch.tensor([x["mid_label"] for x in batch], dtype=torch.long),
        )

#### 沒有base llm
class InternalTwoRouterCachedJointModel(nn.Module):
    def __init__(self, bert_init: str, llama_hidden_size: int, router_dim: int, num_pairs: int):
        super().__init__()
        self.bert = BertExternalEncoder(bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        self.router_first = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.router_mid = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, num_pairs),
        )

    def forward(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, first_vec, mid_vec):
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        first_feat = self.router_first(first_vec, bert_prev, bert_last, bert_attention_mask)
        mid_feat = self.router_mid(mid_vec, bert_prev, bert_last, bert_attention_mask)
        pair_input = torch.cat([first_feat, mid_feat], dim=-1)
        pair_logits = self.pair_classifier(pair_input)
        return pair_logits


def set_trainable(model, freeze_bert: bool = True):
    """Train router heads and optionally fine-tune the shared BERT encoder."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.bert.parameters():
        p.requires_grad = not freeze_bert
    for p in model.router_first.parameters():
        p.requires_grad = True
    for p in model.router_mid.parameters():
        p.requires_grad = True
    for p in model.pair_classifier.parameters():
        p.requires_grad = True

####比回自己的task, 等等可能要再看一下
def compute_self_pair_ce(pair_logits: torch.Tensor, task_ids: torch.Tensor, num_tasks: int):
    valid = (task_ids >= 0) & (task_ids < int(num_tasks))
    if not bool(valid.any().item()):
        return torch.tensor(0.0, device=pair_logits.device), torch.empty(0, dtype=torch.long, device=pair_logits.device)
    self_pair = task_ids[valid] * int(num_tasks) + task_ids[valid]
    loss = nn.functional.cross_entropy(pair_logits[valid], self_pair)
    return loss, self_pair


def extract_pair_prediction(
    prediction_matrix: Optional[List[List[str]]],
    first_idx: int,
    mid_idx: int,
) -> Optional[str]:
    if prediction_matrix is None:
        return None
    if not (0 <= int(first_idx) < len(prediction_matrix)):
        return None
    row = prediction_matrix[int(first_idx)]
    if not (0 <= int(mid_idx) < len(row)):
        return None
    value = row[int(mid_idx)]
    return None if value is None else str(value)


def extract_pair_option_probs(
    option_prob_matrix: torch.Tensor,
    first_idx: int,
    mid_idx: int,
    task_name: str,
) -> List[float]:
    probs = option_prob_matrix[int(first_idx), int(mid_idx)].detach().cpu().to(torch.float32).tolist()
    labels = task_option_labels(task_name)
    if labels:
        return [float(probs[idx]) for idx in range(min(len(labels), len(probs)))]
    return [float(value) for value in probs]


@torch.no_grad()
def evaluate(
    model,
    loader,
    bert_tokenizer,
    device,
    max_bert_len,
    task_names,
    joint_loss,
    pseudo_ce_weight,
    pseudo_ce_margin,
    pair_loss_normalization,
    supervision_mode,
    correct_soft_ce_temperature,
    self_preserve_weight,
    topk_weighted_temperatures,
    collect_route_records: bool = False,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}
    pred_first_all: List[int] = []
    pred_mid_all: List[int] = []
    best_first_all: List[int] = []
    best_mid_all: List[int] = []
    task_ids_all: List[int] = []
    route_records: List[Dict] = []
    oracle_debug_acc = init_oracle_debug_accumulator(task_names)

    for batch in loader:
        bert_enc = bert_tokenizer(
            batch.prompt_texts,
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

        pair_logits = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=batch.first_vec.to(device),
            mid_vec=batch.mid_vec.to(device),
        )
        oracle_loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
            pair_logits=pair_logits,
            loss_matrix=batch.loss_matrix.to(device),
            correct_matrix=batch.correct_matrix.to(device),
            task_ids=batch.task_ids.to(device),
            joint_loss=joint_loss,
            pseudo_ce_weight=pseudo_ce_weight,
            margin=pseudo_ce_margin,
            loss_normalization=pair_loss_normalization,
            correct_soft_ce_temperature=correct_soft_ce_temperature,
            self_preserve_weight=self_preserve_weight,
        )
        pair_prob_matrix = torch.softmax(pair_logits, dim=-1).view(
            batch.loss_matrix.size(0),
            batch.loss_matrix.size(1),
            batch.loss_matrix.size(2),
        )
        router_target_matrix = build_router_target_distribution(
            pair_logits=pair_logits,
            loss_matrix=batch.loss_matrix.to(device),
            correct_matrix=batch.correct_matrix.to(device),
            task_ids=batch.task_ids.to(device),
            joint_loss=joint_loss,
            loss_normalization=pair_loss_normalization,
            correct_soft_ce_temperature=correct_soft_ce_temperature,
            self_preserve_weight=self_preserve_weight,
        )
        if supervision_mode == "self_pair_ce":
            loss, _ = compute_self_pair_ce(pair_logits, batch.task_ids.to(device), batch.loss_matrix.size(2))
            metrics["self_pair_ce"] = float(loss.detach().item())
            metrics["oracle_objective_loss"] = float(oracle_loss.detach().item())
        else:
            loss = oracle_loss
        pred_pair = pair_logits.argmax(dim=-1)
        num_tasks = batch.loss_matrix.size(2)
        pred_first = pred_pair // num_tasks
        pred_mid = pred_pair % num_tasks
        batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
        loss_matrix_device = batch.loss_matrix.to(device)
        score_stats = compute_route_score_stats(
            loss_matrix_device,
            pred_pair,
            batch.task_ids,
            loss_normalization=pair_loss_normalization,
        )
        correct_matrix_device = batch.correct_matrix.to(device)
        correct_available = batch.correct_available.to(device)
        batch_idx = torch.arange(batch.task_ids.size(0), device=device)
        pred_correct = correct_matrix_device[batch_idx, pred_first, pred_mid]
        oracle_correct = correct_matrix_device[batch_idx, best_first, best_mid]
        flat_correct_matrix = correct_matrix_device.view(correct_matrix_device.size(0), -1)
        any_correct = flat_correct_matrix.any(dim=-1)
        max_topk = min(10, pair_logits.size(-1))
        topk_pair_ids = pair_logits.topk(k=max_topk, dim=-1).indices
        valid_self = (batch.task_ids.to(device) >= 0) & (batch.task_ids.to(device) < num_tasks)
        self_ids = batch.task_ids.to(device).clamp_min(0)
        self_correct = correct_matrix_device[batch_idx, self_ids, self_ids]
        correctness_mask = correct_available
        correctness_stats: Dict[str, float] = {
            "correct_matrix_available_ratio": float(correctness_mask.float().mean().item()),
        }
        if bool(correctness_mask.any().item()):
            correctness_stats.update(
                {
                    "route_correct_acc": float(pred_correct[correctness_mask].float().mean().item()),
                    "oracle_correct_acc": float(oracle_correct[correctness_mask].float().mean().item()),
                    "any_pair_correct_rate": float(any_correct[correctness_mask].float().mean().item()),
                }
            )
            for k in (1, 3, 5, 10):
                if k <= max_topk:
                    topk_correct = flat_correct_matrix.gather(1, topk_pair_ids[:, :k]).any(dim=-1)
                    correctness_stats[f"route_top{k}_correct_acc"] = float(
                        topk_correct[correctness_mask].float().mean().item()
                    )
            option_prob_available = batch.option_prob_available.to(device)
            option_conf_mask = correctness_mask & option_prob_available
            if bool(option_conf_mask.any().item()):
                flat_option_prob = batch.option_prob_matrix.to(device).view(
                    batch.option_prob_matrix.size(0),
                    -1,
                    batch.option_prob_matrix.size(-1),
                )
                pair_confidence = flat_option_prob.max(dim=-1).values
                gold_option_idx, gold_option_valid = batch_gold_option_indices(batch.tasks, batch.targets, device)
                weighted_mask = option_conf_mask & gold_option_valid
                for k in (1, 3, 5, 10):
                    if k <= max_topk:
                        topk_ids = topk_pair_ids[:, :k]
                        topk_conf = pair_confidence.gather(1, topk_ids)
                        rerank_choice = topk_ids.gather(1, topk_conf.argmax(dim=-1, keepdim=True)).squeeze(-1)
                        rerank_correct = flat_correct_matrix.gather(1, rerank_choice.unsqueeze(1)).squeeze(1)
                        correctness_stats[f"route_top{k}_confidence_rerank_correct_acc"] = float(
                            rerank_correct[option_conf_mask].float().mean().item()
                        )
                        if bool(weighted_mask.any().item()):
                            gather_idx = topk_ids.unsqueeze(-1).expand(-1, -1, flat_option_prob.size(-1))
                            topk_option_prob = flat_option_prob.gather(1, gather_idx)
                            topk_logits = pair_logits.gather(1, topk_ids)
                            for temperature in topk_weighted_temperatures:
                                tag = metric_float_tag(float(temperature))
                                weights = torch.softmax(topk_logits / float(temperature), dim=-1)
                                weighted_option_prob = (topk_option_prob * weights.unsqueeze(-1)).sum(dim=1)
                                weighted_pred = weighted_option_prob.argmax(dim=-1)
                                weighted_correct = weighted_pred == gold_option_idx
                                correctness_stats[f"route_top{k}_weighted_t{tag}_answer_acc"] = float(
                                    weighted_correct[weighted_mask].float().mean().item()
                                )
                correctness_stats["option_prob_available_ratio"] = float(option_prob_available.float().mean().item())
                correctness_stats["option_gold_available_ratio"] = float(gold_option_valid.float().mean().item())
            else:
                correctness_stats["option_prob_available_ratio"] = 0.0
                correctness_stats["option_gold_available_ratio"] = 0.0
            self_mask = correctness_mask & valid_self
            if bool(self_mask.any().item()):
                correctness_stats["fixed_self_correct_acc"] = float(self_correct[self_mask].float().mean().item())
            else:
                correctness_stats["fixed_self_correct_acc"] = 0.0
        else:
            correctness_stats.update(
                {
                    "route_correct_acc": 0.0,
                    "route_top1_correct_acc": 0.0,
                    "route_top3_correct_acc": 0.0,
                    "route_top5_correct_acc": 0.0,
                    "route_top10_correct_acc": 0.0,
                    "route_top1_confidence_rerank_correct_acc": 0.0,
                    "route_top3_confidence_rerank_correct_acc": 0.0,
                    "route_top5_confidence_rerank_correct_acc": 0.0,
                    "route_top10_confidence_rerank_correct_acc": 0.0,
                    "option_prob_available_ratio": 0.0,
                    "oracle_correct_acc": 0.0,
                    "any_pair_correct_rate": 0.0,
                    "fixed_self_correct_acc": 0.0,
                }
            )
        update_oracle_debug_accumulator(
            oracle_debug_acc,
            batch.loss_matrix.to(device),
            batch.task_ids,
        )

        bs = batch.task_ids.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        for key, value in batch_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        for key, value in score_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        for key, value in correctness_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        pred_first_all.extend(pred_first.cpu().tolist())
        pred_mid_all.extend(pred_mid.cpu().tolist())
        best_first_all.extend(best_first.cpu().tolist())
        best_mid_all.extend(best_mid.cpu().tolist())
        task_ids_all.extend(batch.task_ids.cpu().tolist())
        pred_pair_cpu = pred_pair.detach().cpu().tolist()
        flat_best_cpu = flat_best.detach().cpu().tolist()
        pred_first_cpu = pred_first.detach().cpu().tolist()
        pred_mid_cpu = pred_mid.detach().cpu().tolist()
        best_first_cpu = best_first.detach().cpu().tolist()
        best_mid_cpu = best_mid.detach().cpu().tolist()
        pair_prob_matrix_cpu = pair_prob_matrix.detach().cpu()
        router_target_matrix_cpu = router_target_matrix.detach().cpu() if router_target_matrix is not None else None
        raw_flat_loss = batch.loss_matrix.view(batch.loss_matrix.size(0), -1)
        for idx, item_id in enumerate(batch.item_ids):
            pred_pair_name = f"{task_names[pred_first_cpu[idx]]}->{task_names[pred_mid_cpu[idx]]}"
            gold_pair_name = f"{task_names[best_first_cpu[idx]]}->{task_names[best_mid_cpu[idx]]}"
            pred_answer = extract_pair_prediction(
                batch.prediction_matrices[idx],
                pred_first_cpu[idx],
                pred_mid_cpu[idx],
            )
            gold_answer = extract_pair_prediction(
                batch.prediction_matrices[idx],
                best_first_cpu[idx],
                best_mid_cpu[idx],
            )
            pred_option_probs = None
            pred_option_confidence = None
            gold_option_probs = None
            gold_option_confidence = None
            if bool(batch.option_prob_available[idx].item()):
                pred_option_probs = extract_pair_option_probs(
                    batch.option_prob_matrix[idx],
                    pred_first_cpu[idx],
                    pred_mid_cpu[idx],
                    batch.tasks[idx],
                )
                gold_option_probs = extract_pair_option_probs(
                    batch.option_prob_matrix[idx],
                    best_first_cpu[idx],
                    best_mid_cpu[idx],
                    batch.tasks[idx],
                )
                pred_option_confidence = max(pred_option_probs) if pred_option_probs else None
                gold_option_confidence = max(gold_option_probs) if gold_option_probs else None
            route_records.append(
                {
                    "item_id": item_id,
                    "task": batch.tasks[idx],
                    "pred_pair_id": int(pred_pair_cpu[idx]),
                    "gold_pair_id": int(flat_best_cpu[idx]),
                    "pred_pair": pred_pair_name,
                    "gold_pair": gold_pair_name,
                    "match": bool(pred_pair_cpu[idx] == flat_best_cpu[idx]),
                    "pred_correct": bool(pred_correct.detach().cpu().tolist()[idx]),
                    "gold_correct": bool(oracle_correct.detach().cpu().tolist()[idx]),
                    "any_pair_correct": bool(any_correct.detach().cpu().tolist()[idx]),
                    "correct_matrix_available": bool(batch.correct_available.detach().cpu().tolist()[idx]),
                    "option_prob_available": bool(batch.option_prob_available[idx].item()),
                    "pred_raw_loss": float(raw_flat_loss[idx, pred_pair_cpu[idx]].item()),
                    "gold_raw_loss": float(raw_flat_loss[idx, flat_best_cpu[idx]].item()),
                    "pred_pair_prob": float(pair_prob_matrix_cpu[idx, pred_first_cpu[idx], pred_mid_cpu[idx]].item()),
                    "gold_pair_prob": float(pair_prob_matrix_cpu[idx, best_first_cpu[idx], best_mid_cpu[idx]].item()),
                    "pair_prob_matrix": pair_prob_matrix_cpu[idx].tolist(),
                    "router_target_type": str(joint_loss),
                    "router_target_matrix": (
                        router_target_matrix_cpu[idx].tolist() if router_target_matrix_cpu is not None else None
                    ),
                    "pred_answer": pred_answer,
                    "gold_answer": gold_answer,
                    "pred_option_probs": pred_option_probs,
                    "gold_option_probs": gold_option_probs,
                    "pred_option_confidence": pred_option_confidence,
                    "gold_option_confidence": gold_option_confidence,
                    "prompt_sha1": prompt_hash(batch.prompt_texts[idx]),
                    "text": batch.texts[idx],
                    "prompt_text": batch.prompt_texts[idx],
                    "target": batch.targets[idx],
                }
            )

    denom = max(total_samples, 1)
    result = {"loss": total_loss / denom}
    for key, value in metric_totals.items():
        result[key] = value / denom
    result["routing_summary"] = build_routing_summary(
        pred_first_all, pred_mid_all, best_first_all, best_mid_all, task_ids_all, task_names
    )
    result["oracle_debug_summary"] = build_oracle_debug_summary(oracle_debug_acc)
    result["route_records"] = route_records if collect_route_records else []
    return result


def save_ckpt(
    model,
    out_dir,
    expert_names,
    sample_task_names,
    max_bert_len,
    metrics,
    epoch,
    joint_loss,
    pseudo_ce_weight,
    correct_soft_ce_temperature,
    self_preserve_weight,
    pair_loss_normalization,
    supervision_mode,
    best_metric,
    best_metric_value,
    freeze_bert,
    feature_contract,
    llama_hidden_size,
    router_dim,
):
    os.makedirs(out_dir, exist_ok=True)
    model.bert.encoder.save_pretrained(os.path.join(out_dir, "encoder"))
    torch.save(
        {
            "bert_encoder": model.bert.state_dict(),
            "router_first": model.router_first.state_dict(),
            "router_mid": model.router_mid.state_dict(),
            "pair_classifier": model.pair_classifier.state_dict(),
        },
        os.path.join(out_dir, "router_heads.pt"),
    )
    save_json(
        {
            "task_names": list(expert_names),
            "expert_names": list(expert_names),
            "train_task_names": list(sample_task_names),
            "num_pairs": len(expert_names) * len(expert_names),
            "router_max_len": max_bert_len,
            "router_feature_type": "cached_prompt_vectors_with_loss_matrix",
            "sample_feature_mode": "none",
            "sample_feature_dim": 0,
            "freeze_bert": bool(freeze_bert),
            "first_layer_idx": int(feature_contract["first_layer_idx"]),
            "middle_layer_idx": int(feature_contract["middle_layer_idx"]),
            "router_pooling": str(feature_contract["router_pooling"]),
            "router_pooling_last_k": int(feature_contract["router_pooling_last_k"]),
            "llama_hidden_size": int(llama_hidden_size),
            "router_dim": int(router_dim),
            "feature_contract_version": 1,
            "supervision_type": "cached_pair_ce_main",
            "supervision_mode": str(supervision_mode),
            "joint_loss": str(joint_loss),
            "pseudo_ce_weight": float(pseudo_ce_weight),
            "correct_soft_ce_temperature": float(correct_soft_ce_temperature),
            "self_preserve_weight": float(self_preserve_weight),
            "pair_loss_normalization": str(pair_loss_normalization),
            "best_epoch": epoch,
            "best_metric": str(best_metric),
            "best_metric_value": float(best_metric_value),
            "best_router_argmax_score": metrics.get("router_argmax_score"),
            "best_route_correct_acc": metrics.get("route_correct_acc"),
            "best_val_loss": metrics.get("loss"),
        },
        os.path.join(out_dir, "router_config.json"),
    )
    metrics_without_records = {key: value for key, value in metrics.items() if key != "route_records"}
    save_json({"best_epoch": epoch, "metrics": metrics_without_records}, os.path.join(out_dir, "best_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, default=None, help="Single cached feature root.")
    parser.add_argument("--feature_roots", type=str, default=None, help="Comma-separated cached feature roots to combine.")
    parser.add_argument("--bert_init", type=str, required=True)
    ###輸出 router checkpoint。
    parser.add_argument("--out_dir", type=str, required=True)
    ###舊參數，相容用。現在不建議主用，因為它會同時當 sample task 和 expert names 的 fallback。所以應該要找機會註解調對吧?
    parser.add_argument("--task_names", type=str, default=None, help="backward-compatible alias used for both sample and expert names when the explicit args are omitted")
    ###訓練資料來自哪些 task
    parser.add_argument("--sample_task_names", type=str, default=None, help="comma-separated dataset tasks to train/evaluate on")
    ###router 可以選哪些 expert
    parser.add_argument("--expert_names", type=str, default=None, help="comma-separated expert axis names for the router output")
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument(
        "--sample_feature_mode",
        type=str,
        default="none",
        choices=["none"],
        help="Compatibility argument; extra base-option sample features are no longer supported.",
    )
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        default=True,
        help="Keep the BERT encoder frozen.",
    )
    parser.add_argument(
        "--train_bert",
        action="store_true",
        help="Fine-tune the BERT encoder together with the router heads.",
    )
    add_joint_loss_arguments(parser)
    parser.add_argument(
        "--topk_weighted_temperatures",
        type=str,
        default="0.5,1.0",
        help="Comma-separated router-logit temperatures for top-k weighted option-prob voting metrics.",
    )
    '''
    oracle_loss：使用 cache 裡的 loss_matrix 訓練。
    也就是讓 router 學「哪個 expert pair 實際 loss 比較低」。

    self_pair_ce：直接把每筆 sample 訓練成回自己的 expert。
    '''
    parser.add_argument(
        "--supervision_mode",
        type=str,
        default="oracle_loss",
        choices=["oracle_loss", "self_pair_ce"],
        help="oracle_loss uses the cached loss_matrix objective; self_pair_ce trains each sample to route task->task.",
    )
    parser.add_argument(
        "--best_metric",
        type=str,
        default="route_correct_acc",
        choices=[
            "route_correct_acc",
            "route_top1_correct_acc",
            "route_top3_correct_acc",
            "route_top5_correct_acc",
            "route_top10_correct_acc",
            "route_top1_confidence_rerank_correct_acc",
            "route_top3_confidence_rerank_correct_acc",
            "route_top5_confidence_rerank_correct_acc",
            "route_top10_confidence_rerank_correct_acc",
            "route_top1_weighted_t0p5_answer_acc",
            "route_top3_weighted_t0p5_answer_acc",
            "route_top5_weighted_t0p5_answer_acc",
            "route_top10_weighted_t0p5_answer_acc",
            "route_top1_weighted_t1p0_answer_acc",
            "route_top3_weighted_t1p0_answer_acc",
            "route_top5_weighted_t1p0_answer_acc",
            "route_top10_weighted_t1p0_answer_acc",
            "router_argmax_score",
            "pair_acc",
            "joint_acc",
            "first_acc",
            "mid_acc",
            "loss",
        ],
        help="Validation metric for checkpointing/early stopping. All choices are maximized except loss.",
    )
    ####validation router score 連續幾個 epoch 沒進步就停。
    parser.add_argument("--early_stop_patience", type=int, default=2)
    ####要超過多少才算進步。
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument(
        "--eval_train_each_epoch",
        action="store_true",
        help="After each epoch, run model.eval() on the train split to measure same-sample routing agreement.",
    )
    parser.add_argument(
        "--save_route_records",
        action="store_true",
        help="Save per-sample predicted/gold route records for train-eval and validation.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Load --load_from and only evaluate cached train/validation splits without updating weights.",
    )
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="router_answer_supervision")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    args = parser.parse_args()
    if args.train_bert:
        args.freeze_bert = False

    os.makedirs(args.out_dir, exist_ok=True)
    feature_roots = parse_feature_roots(args.feature_root, args.feature_roots)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise ImportError("--wandb was set but wandb is not installed") from e
        wandb_tags = [part.strip() for part in str(args.wandb_tags).split(",") if part.strip()] if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=wandb_tags,
            config=vars(args),
            dir=args.out_dir,
        )
        print(f"[INFO] wandb enabled project={args.wandb_project}")
    
    ###先讀 manifest 看有哪些 task/expert。
    probe_train_ds = CachedLossMatrixDataset(feature_roots, "train")
    probe_val_ds = CachedLossMatrixDataset(feature_roots, "validation")
    
    ###要拿哪些 sample, 要保留哪些 expert
    alias_task_names = parse_optional_task_names(args.task_names)
    sample_task_names = parse_task_names(
        args.sample_task_names,
        fallback=alias_task_names or probe_train_ds.sample_task_names or probe_val_ds.sample_task_names,
    )
    expert_names = parse_task_names(
        args.expert_names,
        fallback=alias_task_names or probe_train_ds.expert_names or probe_val_ds.expert_names,
    )
    
    ####然後真正建立train_ds跟val_ds
    train_ds = CachedLossMatrixDataset(
        feature_roots,
        "train",
        selected_sample_task_names=sample_task_names,
        selected_expert_names=expert_names,
    )
    val_ds = CachedLossMatrixDataset(
        feature_roots,
        "validation",
        selected_sample_task_names=sample_task_names,
        selected_expert_names=expert_names,
    )
    if train_ds.feature_contract != val_ds.feature_contract:
        raise ValueError(
            f"Training and validation caches use different feature contracts: "
            f"train={train_ds.feature_contract}, validation={val_ds.feature_contract}"
        )
    print(f"[INFO] sample_task_names={sample_task_names}")
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] feature_roots={feature_roots}")
    print(f"[INFO] num_pairs={len(expert_names) * len(expert_names)}")
    print(
        f"[INFO] supervision_mode={args.supervision_mode} joint_loss={args.joint_loss} "
        f"pseudo_ce_weight={args.pseudo_ce_weight} pseudo_ce_margin={args.pseudo_ce_margin} "
        f"correct_soft_ce_temperature={args.correct_soft_ce_temperature} "
        f"self_preserve_weight={args.self_preserve_weight} "
        f"pair_loss_normalization={args.pair_loss_normalization} "
        f"best_metric={args.best_metric} "
        f"sample_feature_mode={args.sample_feature_mode}"
    )
    train_cfg = vars(args).copy()
    train_cfg["resolved_feature_roots"] = feature_roots
    train_cfg["resolved_sample_task_names"] = sample_task_names
    train_cfg["resolved_expert_names"] = expert_names
    train_cfg["feature_contract"] = train_ds.feature_contract
    topk_weighted_temperatures = parse_float_list(args.topk_weighted_temperatures, default=[0.5, 1.0])
    train_cfg["resolved_topk_weighted_temperatures"] = topk_weighted_temperatures
    print(f"[INFO] topk_weighted_temperatures={topk_weighted_temperatures}")
    save_json(train_cfg, os.path.join(args.out_dir, "train_config.json"))

    ###Collator() 會把 list of item 組成 batch
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    train_eval_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,
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

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
    llama_hidden_size = int(train_ds[0]["first_vec"].numel())
    expected_llama_hidden_size = train_ds.feature_contract.get("llama_hidden_size")
    if expected_llama_hidden_size is not None and llama_hidden_size != expected_llama_hidden_size:
        raise ValueError(
            f"Cached vector dimension mismatch: item vectors have {llama_hidden_size}, "
            f"feature contract declares {expected_llama_hidden_size}"
        )
    '''
    這個model只有這幾個, BERT, router_first, router_mid, pair_classifier
    '''
    model = InternalTwoRouterCachedJointModel(
        bert_init=args.bert_init,
        llama_hidden_size=llama_hidden_size,
        router_dim=args.router_dim,
        num_pairs=len(expert_names) * len(expert_names),
    ).to(device)

    if args.load_from is not None:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        if "router_first" in state:
            model.router_first.load_state_dict(state["router_first"], strict=False)
        if "router_mid" in state:
            model.router_mid.load_state_dict(state["router_mid"], strict=False)
        if "pair_classifier" in state:
            model.pair_classifier.load_state_dict(state["pair_classifier"], strict=False)
        if "bert_encoder" in state:
            model.bert.load_state_dict(state["bert_encoder"], strict=False)
        print(f"[LOAD] loaded from {args.load_from}")

    if args.eval_only:
        if args.load_from is None:
            raise ValueError("--eval_only requires --load_from")
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            task_names=expert_names,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            pair_loss_normalization=args.pair_loss_normalization,
            supervision_mode=args.supervision_mode,
            correct_soft_ce_temperature=args.correct_soft_ce_temperature,
            self_preserve_weight=args.self_preserve_weight,
            topk_weighted_temperatures=topk_weighted_temperatures,
            collect_route_records=args.save_route_records,
        )
        print(
            f"[EVAL_ONLY][VAL] loss={val_metrics['loss']:.4f} "
            f"route_correct={val_metrics.get('route_correct_acc', 0.0):.4f} "
            f"top3_correct={val_metrics.get('route_top3_correct_acc', 0.0):.4f} "
            f"top5_correct={val_metrics.get('route_top5_correct_acc', 0.0):.4f} "
            f"self_correct={val_metrics.get('fixed_self_correct_acc', 0.0):.4f} "
            f"any_correct={val_metrics.get('any_pair_correct_rate', 0.0):.4f} "
            f"self_pair={val_metrics['self_pair_acc']:.4f}"
        )
        print_routing_summary("EVAL-ONLY-VAL", val_metrics["routing_summary"])
        save_json(val_metrics["routing_summary"], os.path.join(args.out_dir, "routing_summary_val_eval_only.json"))
        save_json(val_metrics["oracle_debug_summary"], os.path.join(args.out_dir, "oracle_debug_val_eval_only.json"))
        if args.save_route_records:
            save_json(
                {"epoch": 0, "records": val_metrics["route_records"]},
                os.path.join(args.out_dir, "route_records_val_eval_only.json"),
            )
        train_eval_metrics = evaluate(
            model=model,
            loader=train_eval_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            task_names=expert_names,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            pair_loss_normalization=args.pair_loss_normalization,
            supervision_mode=args.supervision_mode,
            correct_soft_ce_temperature=args.correct_soft_ce_temperature,
            self_preserve_weight=args.self_preserve_weight,
            topk_weighted_temperatures=topk_weighted_temperatures,
            collect_route_records=args.save_route_records,
        )
        print(
            f"[EVAL_ONLY][TRAIN] loss={train_eval_metrics['loss']:.4f} "
            f"route_correct={train_eval_metrics.get('route_correct_acc', 0.0):.4f} "
            f"self_correct={train_eval_metrics.get('fixed_self_correct_acc', 0.0):.4f} "
            f"any_correct={train_eval_metrics.get('any_pair_correct_rate', 0.0):.4f} "
            f"self_pair={train_eval_metrics['self_pair_acc']:.4f}"
        )
        print_routing_summary("EVAL-ONLY-TRAIN", train_eval_metrics["routing_summary"])
        save_json(train_eval_metrics["routing_summary"], os.path.join(args.out_dir, "routing_summary_train_eval_only.json"))
        save_json(train_eval_metrics["oracle_debug_summary"], os.path.join(args.out_dir, "oracle_debug_train_eval_only.json"))
        if args.save_route_records:
            save_json(
                {"epoch": 0, "records": train_eval_metrics["route_records"]},
                os.path.join(args.out_dir, "route_records_train_eval_only.json"),
            )
        if wandb_run is not None:
            wandb_run.finish()
        print("[DONE] eval_only finished")
        return

    ###設定可訓練參數
    set_trainable(model, freeze_bert=args.freeze_bert)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_metric_value = float("inf") if args.best_metric == "loss" else float("-inf")
    best_router_score = float("-inf")
    best_epoch = -1
    no_improve_epochs = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        running_metric_totals: Dict[str, float] = {}
        train_pred_first_all: List[int] = []
        train_pred_mid_all: List[int] = []
        train_best_first_all: List[int] = []
        train_best_mid_all: List[int] = []
        train_task_ids_all: List[int] = []
        train_oracle_debug_acc = init_oracle_debug_accumulator(expert_names)

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
            ###把文字餵給 BERT。
            bert_enc = bert_tokenizer(
                batch.prompt_texts,
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

            ####router forward。
            ####這個logits是甚麼意思?
            pair_logits = model(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=batch.first_vec.to(device),
                mid_vec=batch.mid_vec.to(device),
            )
            ###if --supervision_mode oracle_loss, loss=orcale_loss
            ###compute_pair_losses 應該只是算loss的方向而已
            oracle_loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
                pair_logits=pair_logits,
                loss_matrix=batch.loss_matrix.to(device),
                correct_matrix=batch.correct_matrix.to(device),
                task_ids=batch.task_ids.to(device),
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                margin=args.pseudo_ce_margin,
                loss_normalization=args.pair_loss_normalization,
                correct_soft_ce_temperature=args.correct_soft_ce_temperature,
                self_preserve_weight=args.self_preserve_weight,
            )
            ####--supervision_mode self_pair_ce, loss=自己的task
            if args.supervision_mode == "self_pair_ce":
                loss, _ = compute_self_pair_ce(pair_logits, batch.task_ids.to(device), batch.loss_matrix.size(2))
                metrics["self_pair_ce"] = float(loss.detach().item())
                metrics["oracle_objective_loss"] = float(oracle_loss.detach().item())
            else:
                loss = oracle_loss

            if epoch == 1 and step == 1:
                print("pair_logits[0] =", pair_logits[0].detach().cpu())
                print("pair_prob[0] =", torch.softmax(pair_logits[0], dim=-1).detach().cpu())
                print("loss_matrix[0] =", batch.loss_matrix[0])
                num_tasks = batch.loss_matrix.size(2)
                pair_prob = torch.softmax(pair_logits, dim=-1).view(-1, num_tasks, num_tasks)
                print("router_prob_matrix[0] =", pair_prob[0].detach().cpu())
                print("loss_matrix[0] =", batch.loss_matrix[0].detach().cpu())


            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_pair = pair_logits.argmax(dim=-1)
            num_tasks = batch.loss_matrix.size(2)
            pred_first = pred_pair // num_tasks
            pred_mid = pred_pair % num_tasks
            batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
            '''
            router_argmax_score:
            router 選的 pair score

            oracle_best_pair_score:
            loss_matrix 最佳 pair score

            fixed_self_score:
            固定 task->task 的 score

            '''
            score_stats = compute_route_score_stats(
                batch.loss_matrix.to(device),
                pred_pair,
                batch.task_ids,
                loss_normalization=args.pair_loss_normalization,
            )

            bs = batch.task_ids.size(0)
            running_loss += loss.item() * bs
            running_count += bs
            for key, value in metrics.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            for key, value in batch_stats.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            for key, value in score_stats.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            train_pred_first_all.extend(pred_first.detach().cpu().tolist())
            train_pred_mid_all.extend(pred_mid.detach().cpu().tolist())
            train_best_first_all.extend(best_first.detach().cpu().tolist())
            train_best_mid_all.extend(best_mid.detach().cpu().tolist())
            train_task_ids_all.extend(batch.task_ids.cpu().tolist())
            update_oracle_debug_accumulator(
                train_oracle_debug_acc,
                batch.loss_matrix.to(device),
                batch.task_ids,
            )

            if step % args.log_every == 0 or step == len(train_loader):
                avg_loss = running_loss / max(running_count, 1)
                avg_metrics = {
                    key: value / max(running_count, 1) for key, value in running_metric_totals.items()
                }
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} pair_ce={avg_metrics['main_pair_ce']:.4f} expected={avg_metrics['expected_loss']:.4f} "
                    f"best_pair={avg_metrics['best_pair_loss']:.4f} "
                    f"router_score={avg_metrics['router_argmax_score']:.2f} "
                    f"self_score={avg_metrics['fixed_self_score']:.2f} "
                    f"first_acc={avg_metrics['first_acc']:.4f} mid_acc={avg_metrics['mid_acc']:.4f} "
                    f"pair_acc={avg_metrics['pair_acc']:.4f} "
                    f"self_first={avg_metrics['self_first_acc']:.4f} self_mid={avg_metrics['self_mid_acc']:.4f} "
                    f"oracle_self_pair={avg_metrics['oracle_self_pair_acc']:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train/loss": avg_loss,
                            "train/main_pair_ce": avg_metrics["main_pair_ce"],
                            "train/expected_loss": avg_metrics["expected_loss"],
                            "train/correct_soft_ce": avg_metrics.get("correct_soft_ce", 0.0),
                            "train/correct_conf_ce": avg_metrics.get("correct_conf_ce", 0.0),
                            "train/correct_max_margin": avg_metrics.get("correct_max_margin", 0.0),
                            "train/correct_margin_available_ratio": avg_metrics.get("correct_margin_available_ratio", 0.0),
                            "train/correct_target_available_ratio": avg_metrics.get("correct_target_available_ratio", 0.0),
                            "train/avg_correct_pairs": avg_metrics.get("avg_correct_pairs", 0.0),
                            "train/best_pair_loss": avg_metrics["best_pair_loss"],
                            "train/pseudo_ce_pair": avg_metrics["pseudo_ce_pair"],
                            "train/margin_active_ratio": avg_metrics["margin_active_ratio"],
                            "train/router_argmax_score": avg_metrics["router_argmax_score"],
                            "train/fixed_self_score": avg_metrics["fixed_self_score"],
                            "train/oracle_best_pair_score": avg_metrics["oracle_best_pair_score"],
                            "train/first_acc": avg_metrics["first_acc"],
                            "train/mid_acc": avg_metrics["mid_acc"],
                            "train/joint_acc": avg_metrics["joint_acc"],
                            "train/pair_acc": avg_metrics["pair_acc"],
                            "train/self_first_acc": avg_metrics["self_first_acc"],
                            "train/self_mid_acc": avg_metrics["self_mid_acc"],
                            "train/self_joint_acc": avg_metrics["self_joint_acc"],
                            "train/self_pair_acc": avg_metrics["self_pair_acc"],
                            "train/oracle_self_pair_acc": avg_metrics["oracle_self_pair_acc"],
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

        train_summary = build_routing_summary(
            train_pred_first_all, train_pred_mid_all, train_best_first_all, train_best_mid_all, train_task_ids_all, expert_names
        )
        print_routing_summary(f"TRAIN-EPOCH{epoch}", train_summary)
        save_json(train_summary, os.path.join(args.out_dir, f"routing_summary_train_epoch{epoch}.json"))
        train_oracle_summary = build_oracle_debug_summary(train_oracle_debug_acc)
        print_oracle_debug_summary(f"TRAIN-EPOCH{epoch}", train_oracle_summary)
        save_json(train_oracle_summary, os.path.join(args.out_dir, f"oracle_debug_train_epoch{epoch}.json"))
        if wandb_run is not None:
            wandb_payload = {
                "train_epoch/epoch": epoch,
                "train_epoch/loss": running_loss / max(running_count, 1),
                "train_epoch/oracle_avg_gap": train_oracle_summary["avg_gap"],
                "train_epoch/oracle_p50_gap": train_oracle_summary["p50_gap"],
                "train_epoch/oracle_avg_self_minus_oracle": train_oracle_summary["avg_self_minus_oracle"],
            }
            wandb_payload.update(flatten_routing_summary(train_summary, prefix="train_epoch_route"))
            wandb_run.log(wandb_payload, step=global_step)

        ####這個val的部分不用開torch.nograd()嗎?
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            task_names=expert_names,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            pair_loss_normalization=args.pair_loss_normalization,
            supervision_mode=args.supervision_mode,
            correct_soft_ce_temperature=args.correct_soft_ce_temperature,
            self_preserve_weight=args.self_preserve_weight,
            topk_weighted_temperatures=topk_weighted_temperatures,
            collect_route_records=args.save_route_records,
        )
        print(
            f"[VAL] epoch={epoch} loss={val_metrics['loss']:.4f} "
            f"router_score={val_metrics['router_argmax_score']:.2f} "
            f"self_score={val_metrics['fixed_self_score']:.2f} "
            f"oracle_score={val_metrics['oracle_best_pair_score']:.2f} "
            f"first_acc={val_metrics['first_acc']:.4f} "
            f"mid_acc={val_metrics['mid_acc']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.4f} "
            f"route_correct={val_metrics.get('route_correct_acc', 0.0):.4f} "
            f"top3_correct={val_metrics.get('route_top3_correct_acc', 0.0):.4f} "
            f"top5_correct={val_metrics.get('route_top5_correct_acc', 0.0):.4f} "
            f"top3_conf={val_metrics.get('route_top3_confidence_rerank_correct_acc', 0.0):.4f} "
            f"top5_conf={val_metrics.get('route_top5_confidence_rerank_correct_acc', 0.0):.4f} "
            f"top3_wt05={val_metrics.get('route_top3_weighted_t0p5_answer_acc', 0.0):.4f} "
            f"top5_wt05={val_metrics.get('route_top5_weighted_t0p5_answer_acc', 0.0):.4f} "
            f"top3_wt1={val_metrics.get('route_top3_weighted_t1p0_answer_acc', 0.0):.4f} "
            f"top5_wt1={val_metrics.get('route_top5_weighted_t1p0_answer_acc', 0.0):.4f} "
            f"self_correct={val_metrics.get('fixed_self_correct_acc', 0.0):.4f} "
            f"any_correct={val_metrics.get('any_pair_correct_rate', 0.0):.4f} "
            f"self_first={val_metrics['self_first_acc']:.4f} "
            f"self_mid={val_metrics['self_mid_acc']:.4f} "
            f"oracle_self_pair={val_metrics['oracle_self_pair_acc']:.4f}"
        )
        print_routing_summary(f"VAL-EPOCH{epoch}", val_metrics["routing_summary"])
        save_json(val_metrics["routing_summary"], os.path.join(args.out_dir, f"routing_summary_val_epoch{epoch}.json"))
        if args.save_route_records:
            save_json(
                {"epoch": epoch, "records": val_metrics["route_records"]},
                os.path.join(args.out_dir, f"route_records_val_epoch{epoch}.json"),
            )
        print_oracle_debug_summary(f"VAL-EPOCH{epoch}", val_metrics["oracle_debug_summary"])
        save_json(val_metrics["oracle_debug_summary"], os.path.join(args.out_dir, f"oracle_debug_val_epoch{epoch}.json"))
        train_eval_metrics = None
        if args.eval_train_each_epoch:
            train_eval_metrics = evaluate(
                model=model,
                loader=train_eval_loader,
                bert_tokenizer=bert_tokenizer,
                device=device,
                max_bert_len=args.max_bert_len,
                task_names=expert_names,
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                pseudo_ce_margin=args.pseudo_ce_margin,
                pair_loss_normalization=args.pair_loss_normalization,
                supervision_mode=args.supervision_mode,
                correct_soft_ce_temperature=args.correct_soft_ce_temperature,
                self_preserve_weight=args.self_preserve_weight,
                topk_weighted_temperatures=topk_weighted_temperatures,
                collect_route_records=args.save_route_records,
            )
            print(
                f"[TRAIN-EVAL] epoch={epoch} loss={train_eval_metrics['loss']:.4f} "
                f"router_score={train_eval_metrics['router_argmax_score']:.2f} "
                f"self_score={train_eval_metrics['fixed_self_score']:.2f} "
                f"oracle_score={train_eval_metrics['oracle_best_pair_score']:.2f} "
                f"first_acc={train_eval_metrics['first_acc']:.4f} "
                f"mid_acc={train_eval_metrics['mid_acc']:.4f} "
                f"pair_acc={train_eval_metrics['pair_acc']:.4f} "
                f"route_correct={train_eval_metrics.get('route_correct_acc', 0.0):.4f} "
                f"top3_correct={train_eval_metrics.get('route_top3_correct_acc', 0.0):.4f} "
                f"top5_correct={train_eval_metrics.get('route_top5_correct_acc', 0.0):.4f} "
                f"top3_conf={train_eval_metrics.get('route_top3_confidence_rerank_correct_acc', 0.0):.4f} "
                f"top5_conf={train_eval_metrics.get('route_top5_confidence_rerank_correct_acc', 0.0):.4f} "
                f"top3_wt05={train_eval_metrics.get('route_top3_weighted_t0p5_answer_acc', 0.0):.4f} "
                f"top5_wt05={train_eval_metrics.get('route_top5_weighted_t0p5_answer_acc', 0.0):.4f} "
                f"top3_wt1={train_eval_metrics.get('route_top3_weighted_t1p0_answer_acc', 0.0):.4f} "
                f"top5_wt1={train_eval_metrics.get('route_top5_weighted_t1p0_answer_acc', 0.0):.4f} "
                f"self_correct={train_eval_metrics.get('fixed_self_correct_acc', 0.0):.4f} "
                f"any_correct={train_eval_metrics.get('any_pair_correct_rate', 0.0):.4f} "
                f"self_first={train_eval_metrics['self_first_acc']:.4f} "
                f"self_mid={train_eval_metrics['self_mid_acc']:.4f} "
                f"oracle_self_pair={train_eval_metrics['oracle_self_pair_acc']:.4f}"
            )
            print_routing_summary(f"TRAIN-EVAL-EPOCH{epoch}", train_eval_metrics["routing_summary"])
            save_json(
                train_eval_metrics["routing_summary"],
                os.path.join(args.out_dir, f"routing_summary_train_eval_epoch{epoch}.json"),
            )
            if args.save_route_records:
                save_json(
                    {"epoch": epoch, "records": train_eval_metrics["route_records"]},
                    os.path.join(args.out_dir, f"route_records_train_eval_epoch{epoch}.json"),
                )
        current_best_metric = float(val_metrics.get(args.best_metric, float("nan")))
        if not math.isfinite(current_best_metric):
            raise ValueError(
                f"best_metric={args.best_metric} is not available or is not finite in validation metrics. "
                f"Available metrics include: {sorted(val_metrics.keys())}"
            )
        if wandb_run is not None:
            wandb_payload = {
                "val/epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/main_pair_ce": val_metrics.get("main_pair_ce", 0.0),
                "val/expected_loss": val_metrics.get("expected_loss", 0.0),
                "val/correct_soft_ce": val_metrics.get("correct_soft_ce", 0.0),
                "val/correct_conf_ce": val_metrics.get("correct_conf_ce", 0.0),
                "val/correct_max_margin": val_metrics.get("correct_max_margin", 0.0),
                "val/correct_margin_available_ratio": val_metrics.get("correct_margin_available_ratio", 0.0),
                "val/correct_target_available_ratio": val_metrics.get("correct_target_available_ratio", 0.0),
                "val/avg_correct_pairs": val_metrics.get("avg_correct_pairs", 0.0),
                "val/best_pair_loss": val_metrics.get("best_pair_loss", 0.0),
                "val/pseudo_ce_pair": val_metrics.get("pseudo_ce_pair", 0.0),
                "val/router_argmax_score": val_metrics["router_argmax_score"],
                "val/fixed_self_score": val_metrics["fixed_self_score"],
                "val/oracle_best_pair_score": val_metrics["oracle_best_pair_score"],
                "val/oracle_avg_gap": val_metrics["oracle_debug_summary"]["avg_gap"],
                "val/oracle_p50_gap": val_metrics["oracle_debug_summary"]["p50_gap"],
                "val/oracle_avg_self_minus_oracle": val_metrics["oracle_debug_summary"]["avg_self_minus_oracle"],
                "val/first_acc": val_metrics["first_acc"],
                "val/mid_acc": val_metrics["mid_acc"],
                "val/joint_acc": val_metrics["joint_acc"],
                "val/pair_acc": val_metrics["pair_acc"],
                "val/route_correct_acc": val_metrics.get("route_correct_acc", 0.0),
                "val/route_top1_correct_acc": val_metrics.get("route_top1_correct_acc", 0.0),
                "val/route_top3_correct_acc": val_metrics.get("route_top3_correct_acc", 0.0),
                "val/route_top5_correct_acc": val_metrics.get("route_top5_correct_acc", 0.0),
                "val/route_top10_correct_acc": val_metrics.get("route_top10_correct_acc", 0.0),
                "val/route_top1_confidence_rerank_correct_acc": val_metrics.get("route_top1_confidence_rerank_correct_acc", 0.0),
                "val/route_top3_confidence_rerank_correct_acc": val_metrics.get("route_top3_confidence_rerank_correct_acc", 0.0),
                "val/route_top5_confidence_rerank_correct_acc": val_metrics.get("route_top5_confidence_rerank_correct_acc", 0.0),
                "val/route_top10_confidence_rerank_correct_acc": val_metrics.get("route_top10_confidence_rerank_correct_acc", 0.0),
                "val/option_prob_available_ratio": val_metrics.get("option_prob_available_ratio", 0.0),
                "val/option_gold_available_ratio": val_metrics.get("option_gold_available_ratio", 0.0),
                "val/fixed_self_correct_acc": val_metrics.get("fixed_self_correct_acc", 0.0),
                "val/any_pair_correct_rate": val_metrics.get("any_pair_correct_rate", 0.0),
                "val/correct_matrix_available_ratio": val_metrics.get("correct_matrix_available_ratio", 0.0),
                "val/self_first_acc": val_metrics["self_first_acc"],
                "val/self_mid_acc": val_metrics["self_mid_acc"],
                "val/self_joint_acc": val_metrics["self_joint_acc"],
                "val/self_pair_acc": val_metrics["self_pair_acc"],
                "val/oracle_self_pair_acc": val_metrics["oracle_self_pair_acc"],
                "val/best_metric_value": current_best_metric
                if best_epoch < 0
                else (min(best_metric_value, current_best_metric) if args.best_metric == "loss" else max(best_metric_value, current_best_metric)),
                "val/best_router_argmax_score": max(best_router_score, val_metrics["router_argmax_score"]),
            }
            for temperature in topk_weighted_temperatures:
                tag = metric_float_tag(float(temperature))
                for k in (1, 3, 5, 10):
                    metric_name = f"route_top{k}_weighted_t{tag}_answer_acc"
                    wandb_payload[f"val/{metric_name}"] = val_metrics.get(metric_name, 0.0)
            wandb_payload[f"val/{args.best_metric}_for_checkpoint"] = current_best_metric
            wandb_payload.update(flatten_routing_summary(val_metrics["routing_summary"], prefix="val_route"))
            if train_eval_metrics is not None:
                wandb_payload.update(
                    {
                        "train_eval/loss": train_eval_metrics["loss"],
                        "train_eval/router_argmax_score": train_eval_metrics["router_argmax_score"],
                        "train_eval/fixed_self_score": train_eval_metrics["fixed_self_score"],
                        "train_eval/pair_acc": train_eval_metrics["pair_acc"],
                        "train_eval/route_correct_acc": train_eval_metrics.get("route_correct_acc", 0.0),
                        "train_eval/route_top1_correct_acc": train_eval_metrics.get("route_top1_correct_acc", 0.0),
                        "train_eval/route_top3_correct_acc": train_eval_metrics.get("route_top3_correct_acc", 0.0),
                        "train_eval/route_top5_correct_acc": train_eval_metrics.get("route_top5_correct_acc", 0.0),
                        "train_eval/route_top10_correct_acc": train_eval_metrics.get("route_top10_correct_acc", 0.0),
                        "train_eval/route_top1_confidence_rerank_correct_acc": train_eval_metrics.get("route_top1_confidence_rerank_correct_acc", 0.0),
                        "train_eval/route_top3_confidence_rerank_correct_acc": train_eval_metrics.get("route_top3_confidence_rerank_correct_acc", 0.0),
                        "train_eval/route_top5_confidence_rerank_correct_acc": train_eval_metrics.get("route_top5_confidence_rerank_correct_acc", 0.0),
                        "train_eval/route_top10_confidence_rerank_correct_acc": train_eval_metrics.get("route_top10_confidence_rerank_correct_acc", 0.0),
                        "train_eval/option_prob_available_ratio": train_eval_metrics.get("option_prob_available_ratio", 0.0),
                        "train_eval/option_gold_available_ratio": train_eval_metrics.get("option_gold_available_ratio", 0.0),
                        "train_eval/fixed_self_correct_acc": train_eval_metrics.get("fixed_self_correct_acc", 0.0),
                        "train_eval/any_pair_correct_rate": train_eval_metrics.get("any_pair_correct_rate", 0.0),
                        "train_eval/correct_matrix_available_ratio": train_eval_metrics.get("correct_matrix_available_ratio", 0.0),
                        "train_eval/first_acc": train_eval_metrics["first_acc"],
                        "train_eval/mid_acc": train_eval_metrics["mid_acc"],
                    }
                )
                for temperature in topk_weighted_temperatures:
                    tag = metric_float_tag(float(temperature))
                    for k in (1, 3, 5, 10):
                        metric_name = f"route_top{k}_weighted_t{tag}_answer_acc"
                        wandb_payload[f"train_eval/{metric_name}"] = train_eval_metrics.get(metric_name, 0.0)
                wandb_payload.update(
                    flatten_routing_summary(train_eval_metrics["routing_summary"], prefix="train_eval_route")
                )
            wandb_run.log(wandb_payload, step=global_step)
        if args.best_metric == "loss":
            improved = current_best_metric < (best_metric_value - args.early_stop_min_delta)
        else:
            improved = current_best_metric > (best_metric_value + args.early_stop_min_delta)
        if improved:
            best_metric_value = current_best_metric
            best_router_score = val_metrics["router_argmax_score"]
            best_epoch = epoch
            no_improve_epochs = 0
            save_ckpt(
                model=model,
                out_dir=args.out_dir,
                expert_names=expert_names,
                sample_task_names=sample_task_names,
                max_bert_len=args.max_bert_len,
                metrics={"epoch": epoch, **val_metrics},
                epoch=epoch,
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                correct_soft_ce_temperature=args.correct_soft_ce_temperature,
                self_preserve_weight=args.self_preserve_weight,
                pair_loss_normalization=args.pair_loss_normalization,
                supervision_mode=args.supervision_mode,
                best_metric=args.best_metric,
                best_metric_value=best_metric_value,
                freeze_bert=args.freeze_bert,
                feature_contract=train_ds.feature_contract,
                llama_hidden_size=llama_hidden_size,
                router_dim=args.router_dim,
            )
            print(
                f"[SAVE] best checkpoint updated at epoch={epoch} "
                f"{args.best_metric}={best_metric_value:.4f} router_score={best_router_score:.2f}"
            )
        else:
            no_improve_epochs += 1
            print(
                f"[EARLY_STOP] no improvement for {no_improve_epochs} epoch(s). "
                f"best_{args.best_metric}={best_metric_value:.4f} at epoch={best_epoch}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(f"[EARLY_STOP] stop training because patience={args.early_stop_patience} is reached.")
                break

    if wandb_run is not None:
        wandb_run.summary["best_metric"] = args.best_metric
        wandb_run.summary["best_metric_value"] = best_metric_value
        wandb_run.summary["best_router_argmax_score"] = best_router_score
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
    print(
        f"[DONE] best_metric={args.best_metric} best_metric_value={best_metric_value:.4f} "
        f"best_router_argmax_score={best_router_score:.2f} best_epoch={best_epoch}"
    )


if __name__ == "__main__":
    main()
