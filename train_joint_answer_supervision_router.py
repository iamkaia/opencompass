import argparse
import importlib
import importlib.util
import json
import math
import os
import random
from collections import Counter, defaultdict
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

from opencompass.utils.text_postprocessors import (
    general_postprocess,
)
from model_backbone_specs import get_decoder_layers, get_pre_attn_norm, infer_backbone_spec
from task_eval_specs import (
    TASK_EVAL_SPECS,
    TaskEvalSpec,
    apply_postprocessor,
    normalize_boolq_label,
    normalize_qa_text,
    normalize_sst2_label,
)


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
MCQ_STYLE_TASKS = {"race", "medmcqa", "hellaswag", "piqa", "copa", "siqa"}
BINARY_STYLE_TASKS = {"sst2", "boolq"}
QA_STYLE_TASKS = {"squad2", "squad20", "squad2.0"}
TRANSLATION_STYLE_TASKS = {"iwslt2017"}

_OPENCOMPASS_EVAL_RUNTIME = None


def _get_opencompass_eval_runtime():
    global _OPENCOMPASS_EVAL_RUNTIME
    if _OPENCOMPASS_EVAL_RUNTIME is not None:
        return _OPENCOMPASS_EVAL_RUNTIME

    try:
        icl_eval_mod = importlib.import_module("opencompass.openicl.icl_evaluator")
        medmcqa_mod = importlib.import_module("opencompass.datasets.medmcqa")
        squad20_mod = importlib.import_module("opencompass.datasets.squad20")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import OpenCompass evaluator runtime. "
            "Please ensure the training environment can import the same "
            "OpenCompass package used for evaluation."
        ) from exc

    _OPENCOMPASS_EVAL_RUNTIME = {
        "acc": icl_eval_mod.AccEvaluator(),
        "acc_with_details": icl_eval_mod.AccwithDetailsEvaluator(),
        "bleu": icl_eval_mod.BleuEvaluator(),
        "edacc": icl_eval_mod.EDAccEvaluator(),
        "medmcqa": medmcqa_mod.MedmcqaEvaluator(),
        "squad20": squad20_mod.SQuAD20Evaluator(),
    }
    return _OPENCOMPASS_EVAL_RUNTIME


class CompactRouterFeatureEncoder(nn.Module):
    def __init__(self, llama_hidden_size: int, bert_hidden_size: int, router_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(llama_hidden_size, router_dim)
        self.k_proj = nn.Linear(bert_hidden_size, router_dim)
        self.v_proj = nn.Linear(bert_hidden_size, router_dim)
        self.out_norm = nn.LayerNorm(router_dim * 2)

    def forward(
        self,
        llama_vec: torch.Tensor,
        bert_prev: torch.Tensor,
        bert_last: torch.Tensor,
        bert_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        router_dtype = self.q_proj.weight.dtype
        router_device = self.q_proj.weight.device
        llama_vec = llama_vec.to(device=router_device, dtype=router_dtype)
        bert_prev = bert_prev.to(device=router_device, dtype=router_dtype)
        bert_last = bert_last.to(device=router_device, dtype=router_dtype)

        q = self.q_proj(llama_vec).unsqueeze(1)
        mem = torch.cat([bert_prev, bert_last], dim=1)
        k = self.k_proj(mem)
        v = self.v_proj(mem)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (q.size(-1) ** 0.5)
        if bert_attention_mask is not None:
            mask = torch.cat([bert_attention_mask, bert_attention_mask], dim=1)
            mask = (mask == 0).unsqueeze(1).to(device=router_device)
            scores = scores.masked_fill(mask, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        ctx = torch.matmul(attn, v).squeeze(1)
        qv = q.squeeze(1)
        feat = torch.cat([qv, ctx], dim=-1)
        return self.out_norm(feat)


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
                        "meta": dict(row),
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
    metas: List[Dict]
    task_ids: torch.Tensor
    task_names: List[str]


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            source_texts=[x["source_text"] for x in batch],
            targets=[x["target"] for x in batch],
            metas=[x.get("meta", {}) for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            task_names=[x["task"] for x in batch],
        )


class PromptVectorExtractor(nn.Module):
    def __init__(
        self,
        model: AutoModelForCausalLM,
        first_layer_idx: int,
        middle_layer_idx: int,
        pooling: str = "last_token",
        pooling_last_k: int = 4,
    ):
        super().__init__()
        self.model = model
        self.backbone_spec = infer_backbone_spec(model)
        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.pooling = str(pooling)
        self.pooling_last_k = int(pooling_last_k)
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

        layers = get_decoder_layers(self.model, spec=self.backbone_spec)
        get_pre_attn_norm(layers[self.first_layer_idx], self.backbone_spec).register_forward_pre_hook(
            first_pre_hook)
        get_pre_attn_norm(layers[self.middle_layer_idx], self.backbone_spec).register_forward_pre_hook(
            mid_pre_hook)

    @staticmethod
    def gather_last_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        last_idx = attention_mask.sum(dim=1) - 1
        last_idx = last_idx.clamp(min=0)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx, :]

    @staticmethod
    def gather_mean_valid(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (hidden_states * mask).sum(dim=1) / denom

    @staticmethod
    def gather_last_k_mean(hidden_states: torch.Tensor, attention_mask: torch.Tensor, k: int) -> torch.Tensor:
        k = max(int(k), 1)
        outputs = []
        lengths = attention_mask.sum(dim=1)
        for i in range(hidden_states.size(0)):
            valid_len = int(lengths[i].item())
            if valid_len <= 0:
                outputs.append(hidden_states[i, 0])
                continue
            start = max(0, valid_len - k)
            outputs.append(hidden_states[i, start:valid_len].mean(dim=0))
        return torch.stack(outputs, dim=0)

    def gather_pooled(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "last_token":
            return self.gather_last_valid(hidden_states, attention_mask)
        if self.pooling == "mean":
            return self.gather_mean_valid(hidden_states, attention_mask)
        if self.pooling == "lastk_mean":
            return self.gather_last_k_mean(hidden_states, attention_mask, self.pooling_last_k)
        raise ValueError(f"Unknown pooling mode: {self.pooling}")
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
        first_vec = self.gather_pooled(self.cached_first, attention_mask)
        mid_vec = self.gather_pooled(self.cached_mid, attention_mask)
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
        router_pooling: str,
        router_pooling_last_k: int,
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
        for p in self.model.parameters():
            p.requires_grad = False

        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))
        self.task_to_expert_id = {task: self.expert2id[task] + 1 for task in self.expert_names}

        for task in self.expert_names:
            if task not in lora_paths:
                raise KeyError(f"Missing LoRA path for task: {task}")
            load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])

        self.vector_extractor = PromptVectorExtractor(
            model=self.model,
            first_layer_idx=self.first_layer_idx,
            middle_layer_idx=self.middle_layer_idx,
            pooling=router_pooling,
            pooling_last_k=router_pooling_last_k,
        )

        self.bert = BertExternalEncoder(router_bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        llama_hidden_size = self.model.config.hidden_size

        self.num_pairs = len(self.expert_names) * len(self.expert_names)
        self.router_first = CompactRouterFeatureEncoder(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
        )
        self.router_mid = CompactRouterFeatureEncoder(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
        )
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, self.num_pairs),
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

        if "pair_first_encoder" in state:
            self.router_first.load_state_dict(state["pair_first_encoder"])
        else:
            _load_router_with_task_remap(self.router_first, state["router_first"], which="router_first")
        if "pair_mid_encoder" in state:
            self.router_mid.load_state_dict(state["pair_mid_encoder"])
        else:
            _load_router_with_task_remap(self.router_mid, state["router_mid"], which="router_mid")
        if "pair_classifier" in state:
            self.pair_classifier.load_state_dict(state["pair_classifier"], strict=False)
        if "bert_encoder" in state:
            self.bert.load_state_dict(state["bert_encoder"], strict=False)

    def set_trainable(self, freeze_bert: bool, freeze_router_first: bool = False, freeze_router_mid: bool = False):
        for p in self.router_first.parameters():
            p.requires_grad = not freeze_router_first
        for p in self.router_mid.parameters():
            p.requires_grad = not freeze_router_mid
        for p in self.pair_classifier.parameters():
            p.requires_grad = not (freeze_router_first and freeze_router_mid)
        for p in self.bert.parameters():
            p.requires_grad = not freeze_bert

    @torch.no_grad()
    def extract_prompt_vectors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        set_all_experts(self.model, NULL_EXPERT_ID)
        return self.vector_extractor.extract(input_ids=input_ids, attention_mask=attention_mask)

    @torch.no_grad()
    def score_all_route_pairs(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        targets: Sequence[str],
        source_texts: Sequence[str],
        task_names: Sequence[str],
        llm_tokenizer,
        score_mode: str,
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

                if score_mode == "token_nll":
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits
                    combo_loss = compute_dataset_aware_score(
                        logits=logits,
                        labels=labels,
                        prompt_attention_mask=prompt_attention_mask,
                        targets=targets,
                        task_names=task_names,
                        tokenizer=llm_tokenizer,
                        score_mode=score_mode,
                    )
                else:
                    generated_texts = self.generate_under_current_pair(
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        tokenizer=llm_tokenizer,
                        task_names=task_names,
                    )
                    combo_loss = compute_generated_dataset_scores(
                        predictions=generated_texts,
                        targets=targets,
                        task_names=task_names,
                        source_texts=source_texts,
                    ).to(device=input_ids.device, dtype=torch.float32)
                loss_matrix[:, first_tid, mid_tid] = combo_loss

        set_all_experts(self.model, NULL_EXPERT_ID)
        return loss_matrix

    @torch.no_grad()
    def generate_under_current_pair(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        tokenizer,
        task_names: Sequence[str],
    ) -> List[str]:
        max_new_tokens = max(task_max_new_tokens(task) for task in task_names)
        outputs = self.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_seq_len = prompt_input_ids.size(1)
        texts = []
        for idx in range(outputs.size(0)):
            new_tokens = outputs[idx, prompt_seq_len:]
            texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return texts

    def forward_router(
        self,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
        bert_token_type_ids: Optional[torch.Tensor],
        first_vec: torch.Tensor,
        mid_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        first_feat = self.router_first(
            llama_vec=first_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        mid_feat = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        pair_feat = torch.cat([first_feat, mid_feat], dim=-1)
        pair_logits = self.pair_classifier(pair_feat)
        pair_prob = torch.softmax(pair_logits, dim=-1).view(pair_logits.size(0), len(self.expert_names), len(self.expert_names))
        first_logits = torch.log(pair_prob.sum(dim=2).clamp_min(1e-12))
        mid_logits = torch.log(pair_prob.sum(dim=1).clamp_min(1e-12))
        return pair_logits, first_logits, mid_logits


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
        prompt_input_ids.append([pad_id] * prompt_pad + prompt_ids)
        prompt_attention_masks.append([0] * prompt_pad + [1] * len(prompt_ids))

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


def _first_token_id(tokenizer, text: str) -> Optional[int]:
    ids = tokenizer.encode(text, add_special_tokens=False)
    return ids[0] if ids else None


def compute_mcq_accuracy_proxy(
    logits: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    targets: Sequence[str],
    tokenizer,
) -> torch.Tensor:
    option_token_ids = []
    for option in ["A", "B", "C", "D"]:
        token_id = _first_token_id(tokenizer, option)
        if token_id is None:
            raise ValueError(f"Tokenizer cannot encode MCQ option {option!r}")
        option_token_ids.append(token_id)
    option_token_ids_tensor = torch.tensor(option_token_ids, dtype=torch.long, device=logits.device)

    prompt_lens = prompt_attention_mask.sum(dim=1).clamp_min(1)
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    answer_logits = logits[batch_idx, prompt_lens - 1, :]
    option_logits = answer_logits.index_select(dim=-1, index=option_token_ids_tensor)
    pred_idx = option_logits.argmax(dim=-1)

    target_indices = []
    for target in targets:
        clean = str(target).strip().upper()[:1]
        if clean not in {"A", "B", "C", "D"}:
            target_indices.append(0)
        else:
            target_indices.append(ord(clean) - ord("A"))
    target_idx_tensor = torch.tensor(target_indices, dtype=torch.long, device=logits.device)
    correct = (pred_idx == target_idx_tensor).to(torch.float32)

    log_probs = torch.log_softmax(option_logits, dim=-1)
    correct_nll = -log_probs.gather(dim=-1, index=target_idx_tensor.unsqueeze(-1)).squeeze(-1)
    return (1.0 - correct) + 1e-3 * correct_nll


def compute_official_evaluator_sample_score(
    prediction: str,
    target: str,
    task_name: str,
    source_text: Optional[str] = None,
) -> float:
    runtime = _get_opencompass_eval_runtime()
    task = str(task_name)
    spec: Optional[TaskEvalSpec] = TASK_EVAL_SPECS.get(task)
    if spec is None:
        pred = normalize_qa_text(prediction)
        gold = normalize_qa_text(target)
        result = runtime["acc"].score([pred], [gold])
        return 1.0 - float(result["accuracy"]) / 100.0

    processed_prediction = apply_postprocessor([prediction],
                                               spec.pred_postprocessor)[0]
    prediction_for_eval = prediction if spec.use_raw_prediction else processed_prediction
    if spec.reference_adapter is not None:
        reference = spec.reference_adapter(str(source_text or ""), target)
    elif task == "sst2":
        reference = normalize_sst2_label(target)
    elif task == "boolq":
        reference = normalize_boolq_label(target)
    else:
        reference = target

    score_kwargs = {}
    if spec.extra_score_kwargs_builder is not None:
        score_kwargs.update(
            spec.extra_score_kwargs_builder(str(source_text or ""), target))
    if "references" not in score_kwargs:
        score_kwargs["references"] = [reference]

    if spec.dataset_postprocessor is not None and "references" in score_kwargs:
        refs = score_kwargs["references"]
        if refs and isinstance(refs[0], list):
            refs = [apply_postprocessor(ref_list, spec.dataset_postprocessor) for ref_list in refs]
        else:
            refs = apply_postprocessor(refs, spec.dataset_postprocessor)
        score_kwargs["references"] = refs

    result = runtime[spec.evaluator_key].score(
        predictions=[prediction_for_eval], **score_kwargs)
    metric_name = spec.score_family if spec.score_family in result else None
    if metric_name is None:
        metric_name = "accuracy" if "accuracy" in result else "score"
    return 1.0 - float(result[metric_name]) / 100.0


def task_max_new_tokens(task_name: str) -> int:
    task_name = str(task_name)
    if task_name in MCQ_STYLE_TASKS:
        return 4
    if task_name in BINARY_STYLE_TASKS:
        return 4
    if task_name in QA_STYLE_TASKS:
        return 32
    if task_name in TRANSLATION_STYLE_TASKS:
        return 128
    return 32


def compute_generated_dataset_scores(
    predictions: Sequence[str],
    targets: Sequence[str],
    task_names: Sequence[str],
    source_texts: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    scores = []
    if source_texts is None:
        source_texts = [""] * len(predictions)
    for pred, target, task, source_text in zip(predictions, targets, task_names, source_texts):
        scores.append(
            compute_official_evaluator_sample_score(
                prediction=pred,
                target=target,
                task_name=task,
                source_text=source_text,
            ))
    return torch.tensor(scores, dtype=torch.float32)


def compute_dataset_aware_score(
    logits: torch.Tensor,
    labels: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    targets: Sequence[str],
    task_names: Sequence[str],
    tokenizer,
    score_mode: str,
) -> torch.Tensor:
    score_mode = str(score_mode)
    base_nll = compute_sequence_nll(logits=logits, labels=labels)
    if score_mode == "token_nll":
        return base_nll

    batch_scores = []
    mcq_proxy = compute_mcq_accuracy_proxy(
        logits=logits,
        prompt_attention_mask=prompt_attention_mask,
        targets=targets,
        tokenizer=tokenizer,
    )
    for idx, task in enumerate(task_names):
        task = str(task)
        if score_mode == "dataset_auto":
            if task in MCQ_STYLE_TASKS:
                batch_scores.append(mcq_proxy[idx])
            elif task == "sst2":
                batch_scores.append(base_nll[idx])
            elif task in {"squad2", "iwslt2017"}:
                batch_scores.append(base_nll[idx])
            else:
                batch_scores.append(base_nll[idx])
        else:
            raise ValueError(f"Unknown score_mode: {score_mode}")
    return torch.stack(batch_scores, dim=0)


def compute_pair_losses(
    pair_logits: torch.Tensor,
    logits_first: torch.Tensor,
    logits_mid: torch.Tensor,
    loss_matrix: torch.Tensor,
    mode: str,
    pseudo_ce_weight: float,
    margin: float,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = str(mode)
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    flat_best = flat_loss.argmin(dim=-1)
    best_first = flat_best // loss_matrix.size(2)
    best_mid = flat_best % loss_matrix.size(2)
    pair_prob = torch.softmax(pair_logits, dim=-1)
    expected_loss = (pair_prob * flat_loss).sum(dim=-1).mean()

    sorted_loss, _ = flat_loss.sort(dim=-1)
    if flat_loss.size(1) > 1:
        margin_mask = (sorted_loss[:, 1] - sorted_loss[:, 0]) >= float(margin)
    else:
        margin_mask = torch.ones_like(flat_best, dtype=torch.bool)

    ce_pair_all = nn.functional.cross_entropy(pair_logits, flat_best, reduction="none")
    ce_first_all = nn.functional.cross_entropy(logits_first, best_first, reduction="none")
    ce_mid_all = nn.functional.cross_entropy(logits_mid, best_mid, reduction="none")
    ce_pair = ce_pair_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=pair_logits.device)
    ce_first = ce_first_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=logits_first.device)
    ce_mid = ce_mid_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=logits_first.device)
    margin_active = float(margin_mask.float().mean().item())

    if mode == "stage1":
        total_loss = ce_first
    elif mode == "stage2":
        total_loss = ce_mid
    elif mode == "joint":
        total_loss = expected_loss
        if pseudo_ce_weight > 0 and margin_mask.any():
            total_loss = total_loss + pseudo_ce_weight * ce_pair
        metrics = {
            "expected_loss": float(expected_loss.detach().item()),
            "pseudo_ce_pair": float(ce_pair.detach().item()),
            "pseudo_ce_first": float(ce_first.detach().item()),
            "pseudo_ce_mid": float(ce_mid.detach().item()),
            "best_pair_loss": float(sorted_loss[:, 0].mean().item()),
            "margin_active_ratio": margin_active,
        }
        return total_loss, metrics, best_first, best_mid, flat_best
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    metrics = {
        "expected_loss": float(expected_loss.detach().item()),
        "pseudo_ce_pair": float(ce_pair.detach().item()),
        "pseudo_ce_first": float(ce_first.detach().item()),
        "pseudo_ce_mid": float(ce_mid.detach().item()),
        "best_pair_loss": float(sorted_loss[:, 0].mean().item()),
        "margin_active_ratio": margin_active,
    }
    return total_loss, metrics, best_first, best_mid, flat_best


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


def build_routing_summary(
    pred_first_all: Sequence[int],
    pred_mid_all: Sequence[int],
    best_first_all: Sequence[int],
    best_mid_all: Sequence[int],
    task_ids_all: Sequence[int],
    expert_names: Sequence[str],
    top_k: int = 5,
) -> Dict:
    if not pred_first_all:
        return {"num_samples": 0, "top_pred_pairs": [], "per_task": []}

    stats = compute_routing_accuracy_stats(
        pred_first=torch.tensor(pred_first_all, dtype=torch.long),
        pred_mid=torch.tensor(pred_mid_all, dtype=torch.long),
        best_first=torch.tensor(best_first_all, dtype=torch.long),
        best_mid=torch.tensor(best_mid_all, dtype=torch.long),
        task_ids=torch.tensor(task_ids_all, dtype=torch.long),
    )

    num_samples = len(pred_first_all)
    pred_pair_counter = Counter()
    gold_pair_counter = Counter()
    task_bucket: Dict[int, Dict[str, Counter]] = defaultdict(
        lambda: {
            "pred_first": Counter(),
            "pred_mid": Counter(),
            "pred_pair": Counter(),
            "gold_pair": Counter(),
        }
    )

    for pred_first, pred_mid, best_first, best_mid, task_id in zip(
        pred_first_all, pred_mid_all, best_first_all, best_mid_all, task_ids_all
    ):
        pred_pair_name = f"{expert_names[pred_first]}->{expert_names[pred_mid]}"
        gold_pair_name = f"{expert_names[best_first]}->{expert_names[best_mid]}"
        pred_pair_counter[pred_pair_name] += 1
        gold_pair_counter[gold_pair_name] += 1

        bucket = task_bucket[int(task_id)]
        bucket["pred_first"][expert_names[pred_first]] += 1
        bucket["pred_mid"][expert_names[pred_mid]] += 1
        bucket["pred_pair"][pred_pair_name] += 1
        bucket["gold_pair"][gold_pair_name] += 1

    def _counter_rows(counter: Counter, denom: int, limit: int) -> List[Dict]:
        rows = []
        for name, count in counter.most_common(limit):
            rows.append(
                {
                    "name": name,
                    "count": int(count),
                    "rate": float(count / max(denom, 1)),
                }
            )
        return rows

    per_task = []
    for task_id, task_name in enumerate(expert_names):
        mask_count = sum(1 for x in task_ids_all if x == task_id)
        if mask_count == 0:
            continue

        pred_self_first = sum(
            1 for pf, tid in zip(pred_first_all, task_ids_all) if tid == task_id and pf == task_id
        )
        pred_self_mid = sum(
            1 for pm, tid in zip(pred_mid_all, task_ids_all) if tid == task_id and pm == task_id
        )
        gold_self_pair = sum(
            1
            for bf, bm, tid in zip(best_first_all, best_mid_all, task_ids_all)
            if tid == task_id and bf == task_id and bm == task_id
        )

        bucket = task_bucket[task_id]
        per_task.append(
            {
                "task": task_name,
                "count": int(mask_count),
                "pred_self_first_rate": float(pred_self_first / mask_count),
                "pred_self_mid_rate": float(pred_self_mid / mask_count),
                "gold_self_pair_rate": float(gold_self_pair / mask_count),
                "top_pred_first": _counter_rows(bucket["pred_first"], mask_count, limit=3),
                "top_pred_mid": _counter_rows(bucket["pred_mid"], mask_count, limit=3),
                "top_pred_pairs": _counter_rows(bucket["pred_pair"], mask_count, limit=3),
                "top_gold_pairs": _counter_rows(bucket["gold_pair"], mask_count, limit=3),
            }
        )

    summary = {
        "num_samples": int(num_samples),
        "first_acc": float(stats["first_acc"]),
        "mid_acc": float(stats["mid_acc"]),
        "pair_acc": float(stats["pair_acc"]),
        "self_first_acc": float(stats["self_first_acc"]),
        "self_mid_acc": float(stats["self_mid_acc"]),
        "self_pair_acc": float(stats["self_pair_acc"]),
        "oracle_self_pair_acc": float(stats["oracle_self_pair_acc"]),
        "top_pred_pairs": _counter_rows(pred_pair_counter, num_samples, limit=top_k),
        "top_gold_pairs": _counter_rows(gold_pair_counter, num_samples, limit=top_k),
        "per_task": per_task,
    }
    return summary


def print_routing_summary(tag: str, summary: Dict):
    if int(summary.get("num_samples", 0)) <= 0:
        print(f"[ROUTE][{tag}] no samples")
        return

    top_pairs = ", ".join(
        f"{row['name']}:{row['rate']:.2%}" for row in summary.get("top_pred_pairs", [])[:3]
    )
    print(
        f"[ROUTE][{tag}] pair_acc={summary.get('pair_acc', 0.0):.4f} "
        f"self_pair={summary.get('self_pair_acc', 0.0):.4f} "
        f"oracle_self_pair={summary.get('oracle_self_pair_acc', 0.0):.4f} "
        f"top_pred_pairs={top_pairs}"
    )
    for row in summary.get("per_task", []):
        top_first = row.get("top_pred_first", [])
        top_mid = row.get("top_pred_mid", [])
        top_pair = row.get("top_pred_pairs", [])
        first_name = top_first[0]["name"] if top_first else "-"
        mid_name = top_mid[0]["name"] if top_mid else "-"
        pair_name = top_pair[0]["name"] if top_pair else "-"
        print(
            f"[ROUTE][{tag}][{row['task']}] n={row['count']} "
            f"self_first={row['pred_self_first_rate']:.2%} "
            f"self_mid={row['pred_self_mid_rate']:.2%} "
            f"oracle_self_pair={row['gold_self_pair_rate']:.2%} "
            f"top_first={first_name} top_mid={mid_name} top_pair={pair_name}"
        )


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
    train_mode: str,
    pseudo_ce_weight: float,
    pseudo_ce_margin: float,
    score_mode: str,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}
    pred_first_all: List[int] = []
    pred_mid_all: List[int] = []
    best_first_all: List[int] = []
    best_mid_all: List[int] = []
    task_ids_all: List[int] = []

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
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            targets=batch.targets,
            source_texts=batch.source_texts,
            task_names=batch.task_names,
            llm_tokenizer=llm_tokenizer,
            score_mode=score_mode,
        )
        pair_logits, logits_first, logits_mid = model.forward_router(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=first_vec.to(torch.float32),
            mid_vec=mid_vec.to(torch.float32),
        )
        loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
            pair_logits=pair_logits,
            logits_first=logits_first,
            logits_mid=logits_mid,
            loss_matrix=loss_matrix,
            mode=train_mode,
            pseudo_ce_weight=pseudo_ce_weight,
            margin=pseudo_ce_margin,
        )

        pred_pair = pair_logits.argmax(dim=-1)
        pred_first = pred_pair // loss_matrix.size(2)
        pred_mid = pred_pair % loss_matrix.size(2)
        batch_size = pred_first.size(0)
        batch_stats = compute_routing_accuracy_stats(
            pred_first=pred_first,
            pred_mid=pred_mid,
            best_first=best_first,
            best_mid=best_mid,
            task_ids=batch.task_ids,
        )
        pred_first_all.extend(pred_first.cpu().tolist())
        pred_mid_all.extend(pred_mid.cpu().tolist())
        best_first_all.extend(best_first.cpu().tolist())
        best_mid_all.extend(best_mid.cpu().tolist())
        task_ids_all.extend(batch.task_ids.cpu().tolist())

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for key, value in batch_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size
        for key in ["expected_loss", "pseudo_ce_pair", "pseudo_ce_first", "pseudo_ce_mid", "best_pair_loss", "margin_active_ratio"]:
            if key in metrics:
                metric_totals[key] = metric_totals.get(key, 0.0) + metrics[key] * batch_size

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
    result["routing_summary"] = build_routing_summary(
        pred_first_all=pred_first_all,
        pred_mid_all=pred_mid_all,
        best_first_all=best_first_all,
        best_mid_all=best_mid_all,
        task_ids_all=task_ids_all,
        expert_names=model.expert_names,
    )
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
    parser.add_argument("--lora_hellaswag", type=str, default=None)
    parser.add_argument("--lora_boolq", type=str, default=None)
    parser.add_argument("--lora_siqa", type=str, default=None)
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
    parser.add_argument(
        "--router_pooling",
        type=str,
        default="last_token",
        choices=["last_token", "mean", "lastk_mean"],
    )
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--freeze_router_first", action="store_true")
    parser.add_argument("--freeze_router_mid", action="store_true")
    parser.add_argument("--train_mode", type=str, default="joint", choices=["stage1", "stage2", "joint"])
    parser.add_argument("--pseudo_ce_weight", type=float, default=0.5)
    parser.add_argument("--pseudo_ce_margin", type=float, default=0.0)
    parser.add_argument("--score_mode", type=str, default="dataset_auto", choices=["dataset_auto", "token_nll"])
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
        "hellaswag": args.lora_hellaswag,
        "boolq": args.lora_boolq,
        "siqa": args.lora_siqa,
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
    llm_tokenizer.padding_side = "left"
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
        router_pooling=args.router_pooling,
        router_pooling_last_k=args.router_pooling_last_k,
    )
    if args.load_router_ckpt_dir:
        model.load_router_weights(args.load_router_ckpt_dir)
        print(f"[INFO] loaded router weights from {args.load_router_ckpt_dir}")

    freeze_router_first = args.freeze_router_first or (args.train_mode == "stage2")
    freeze_router_mid = args.freeze_router_mid or (args.train_mode == "stage1")
    model.set_trainable(
        freeze_bert=args.freeze_bert,
        freeze_router_first=freeze_router_first,
        freeze_router_mid=freeze_router_mid,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(
        "[INFO] trainable_parts="
        f"bert:{not args.freeze_bert} "
        f"router_first:{not freeze_router_first} "
        f"router_mid:{not freeze_router_mid} "
        f"num_params={trainable_param_count}"
    )
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
            train_mode=args.train_mode,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            score_mode=args.score_mode,
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
        print_routing_summary(tag=f"EVAL-{args.eval_split}", summary=eval_metrics["routing_summary"])
        save_json(
            eval_metrics["routing_summary"],
            os.path.join(args.output_dir, f"routing_summary_{args.eval_split}.json"),
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval/{k}": v
                    for k, v in eval_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
            wandb_run.finish()
        return

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
        train_pred_first_all: List[int] = []
        train_pred_mid_all: List[int] = []
        train_best_first_all: List[int] = []
        train_best_mid_all: List[int] = []
        train_task_ids_all: List[int] = []

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
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    targets=batch.targets,
                    source_texts=batch.source_texts,
                    task_names=batch.task_names,
                    llm_tokenizer=llm_tokenizer,
                    score_mode=args.score_mode,
                )

            pair_logits, logits_first, logits_mid = model.forward_router(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=first_vec.to(torch.float32),
                mid_vec=mid_vec.to(torch.float32),
            )
            loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
                pair_logits=pair_logits,
                logits_first=logits_first,
                logits_mid=logits_mid,
                loss_matrix=loss_matrix,
                mode=args.train_mode,
                pseudo_ce_weight=args.pseudo_ce_weight,
                margin=args.pseudo_ce_margin,
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            batch_size = len(batch.texts)
            running_loss += loss.item() * batch_size
            running_count += batch_size
            pred_pair = pair_logits.argmax(dim=-1)
            pred_first = pred_pair // loss_matrix.size(2)
            pred_mid = pred_pair % loss_matrix.size(2)
            train_pred_first_all.extend(pred_first.detach().cpu().tolist())
            train_pred_mid_all.extend(pred_mid.detach().cpu().tolist())
            train_best_first_all.extend(best_first.detach().cpu().tolist())
            train_best_mid_all.extend(best_mid.detach().cpu().tolist())
            train_task_ids_all.extend(batch.task_ids.cpu().tolist())

            if tqdm is not None:
                batch_stats = compute_routing_accuracy_stats(
                    pred_first=pred_first,
                    pred_mid=pred_mid,
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
                            "train/margin_active_ratio": metrics.get("margin_active_ratio", 0.0),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                    )
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} expected={metrics['expected_loss']:.4f} "
                    f"best_pair={metrics['best_pair_loss']:.4f} "
                    f"first_acc={batch_stats['first_acc']:.4f} mid_acc={batch_stats['mid_acc']:.4f} "
                    f"pair_acc={batch_stats['pair_acc']:.4f} "
                    f"self_first={batch_stats['self_first_acc']:.4f} self_mid={batch_stats['self_mid_acc']:.4f} "
                    f"oracle_self_pair={batch_stats['oracle_self_pair_acc']:.4f}"
                )

        train_routing_summary = build_routing_summary(
            pred_first_all=train_pred_first_all,
            pred_mid_all=train_pred_mid_all,
            best_first_all=train_best_first_all,
            best_mid_all=train_best_mid_all,
            task_ids_all=train_task_ids_all,
            expert_names=expert_names,
        )
        print_routing_summary(tag=f"TRAIN-EPOCH{epoch}", summary=train_routing_summary)
        save_json(
            train_routing_summary,
            os.path.join(args.output_dir, f"routing_summary_train_epoch{epoch}.json"),
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
            train_mode=args.train_mode,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            score_mode=args.score_mode,
        )
        print(
            f"[VAL] epoch={epoch} loss={val_metrics['loss']:.4f} "
            f"first_acc={val_metrics['first_acc']:.4f} "
            f"mid_acc={val_metrics['mid_acc']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.4f} "
            f"self_first={val_metrics['self_first_acc']:.4f} "
            f"self_mid={val_metrics['self_mid_acc']:.4f} "
            f"oracle_self_pair={val_metrics['oracle_self_pair_acc']:.4f}"
        )
        print_routing_summary(tag=f"VAL-EPOCH{epoch}", summary=val_metrics["routing_summary"])
        save_json(
            val_metrics["routing_summary"],
            os.path.join(args.output_dir, f"routing_summary_val_epoch{epoch}.json"),
        )
        if wandb_run is not None:
            payload = {
                "val/epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/first_acc": val_metrics["first_acc"],
                "val/mid_acc": val_metrics["mid_acc"],
                "val/pair_acc": val_metrics["pair_acc"],
                "val/joint_acc": val_metrics["joint_acc"],
                "val/best_val": min(best_val, val_metrics["loss"]),
            }
            wandb_run.log(payload)

        state = {
            "pair_first_encoder": model.router_first.state_dict(),
            "pair_mid_encoder": model.router_mid.state_dict(),
            "pair_classifier": model.pair_classifier.state_dict(),
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
                    "train_mode": args.train_mode,
                    "score_mode": args.score_mode,
                    "router_pooling": args.router_pooling,
                    "router_pooling_last_k": args.router_pooling_last_k,
                    "pseudo_ce_margin": args.pseudo_ce_margin,
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
