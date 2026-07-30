#!/usr/bin/env python3
"""End-to-end SFT-supervised training for the two-layer LoRA router.

This trainer does not build or consume cached oracle matrices.  It freezes the
base LLM and fixed LoRA experts, lets the router produce soft pair weights, and
optimizes the answer-token causal LM loss directly.
"""

import argparse
import json
import os
import random
from contextlib import nullcontext
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

from model_backbone_specs import get_decoder_layers, infer_backbone_spec
from opencompass.models.router_moe_components import BertExternalEncoder, CompactRouterFeatureEncoder
from opencompass.models.router_moe_shared import (
    HardRoutedLoRALinear,
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
)
from router_answer_supervision_core import (
    Collator,
    build_dataset,
    make_progress,
    parse_csv_arg,
    save_json,
)


def parse_lora_paths(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in str(raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"Invalid --lora_paths item {part!r}; expected task=path")
        task, path = part.split("=", 1)
        task = task.strip()
        path = path.strip()
        if not task or not path:
            raise ValueError(f"Invalid --lora_paths item {part!r}; expected task=path")
        out[task] = path
    if not out:
        raise ValueError("--lora_paths must provide at least one task=path entry")
    return out


def dtype_from_name(name: str):
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype={name!r}")


def set_layer_range_expert_weights_grad(model, start_idx: int, end_idx: int, weights: torch.Tensor):
    """Set weighted LoRA routing without detaching router-produced weights."""
    if end_idx < start_idx:
        return
    layers = get_decoder_layers(model)
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), len(layers) - 1)
    for layer_idx in range(start_idx, end_idx + 1):
        layer = layers[layer_idx]
        while hasattr(layer, "base_layer"):
            layer = layer.base_layer
        modules = []
        for name in ["gate_proj", "up_proj", "down_proj"]:
            modules.append(getattr(layer.mlp, name, None))
        for attn_name in ["self_attn", "linear_attn"]:
            attn = getattr(layer, attn_name, None)
            if attn is None:
                continue
            for name in ["q_proj", "k_proj", "v_proj", "o_proj", "in_proj_qkv", "in_proj_a", "in_proj_b", "in_proj_z", "out_proj"]:
                modules.append(getattr(attn, name, None))
        for module in modules:
            if isinstance(module, HardRoutedLoRALinear):
                module.active_expert = NULL_EXPERT_ID
                module.active_weights = weights


def pool_prompt_vector(hidden_states: torch.Tensor, attention_mask: torch.Tensor, pooling: str, last_k: int):
    if pooling == "last_token":
        last_idx = attention_mask.sum(dim=1).clamp(min=1) - 1
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, last_idx, :]
    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
    if pooling == "mean":
        return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    if pooling == "lastk_mean":
        k = max(int(last_k), 1)
        outputs = []
        for sample_hidden, sample_mask in zip(hidden_states, attention_mask):
            valid = sample_hidden[sample_mask.to(dtype=torch.bool)]
            if valid.numel() == 0:
                outputs.append(sample_hidden[0])
            else:
                outputs.append(valid[-k:].mean(dim=0))
        return torch.stack(outputs, dim=0)
    raise ValueError(f"Unknown router_pooling={pooling!r}")


def build_sft_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    add_eos_to_target: bool,
) -> Dict[str, torch.Tensor]:
    """Build prompt+target labels while preserving answer tokens.

    The cached-router helper truncates `(prompt + target)` from the right.  That
    is fine for scoring paths that do not rely on teacher-forcing labels, but it
    can drop the target entirely for long prompts and make CE loss NaN.  End to
    end SFT must reserve room for the target, so prompt tokens are left-truncated
    per sample before concatenation.
    """
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is required.")

    prompt_ids_list = []
    target_ids_list = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(str(prompt), add_special_tokens=False)
        target_ids = tokenizer.encode(str(target), add_special_tokens=False)
        if add_eos_to_target and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        if not target_ids:
            raise ValueError(f"Empty target after tokenization: {target!r}")
        if len(target_ids) >= max_length:
            target_ids = target_ids[:max_length]
            prompt_ids = []
        else:
            prompt_budget = max_length - len(target_ids)
            prompt_ids = prompt_ids[-prompt_budget:]
        prompt_ids_list.append(prompt_ids)
        target_ids_list.append(target_ids)

    max_prompt_len = max(max((len(ids) for ids in prompt_ids_list), default=1), 1)
    max_full_len = max(len(p) + len(t) for p, t in zip(prompt_ids_list, target_ids_list))

    prompt_input_ids = []
    prompt_attention_masks = []
    input_ids = []
    attention_masks = []
    labels = []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        prompt_pad = max_prompt_len - len(prompt_ids)
        prompt_input_ids.append([pad_id] * prompt_pad + prompt_ids)
        prompt_attention_masks.append([0] * prompt_pad + [1] * len(prompt_ids))

        full_ids = prompt_ids + target_ids
        seq_labels = [-100] * len(prompt_ids) + target_ids
        pad_len = max_full_len - len(full_ids)
        input_ids.append(full_ids + [pad_id] * pad_len)
        attention_masks.append([1] * len(full_ids) + [0] * pad_len)
        labels.append(seq_labels + [-100] * pad_len)

    labels_tensor = torch.tensor(labels, dtype=torch.long)
    if not bool((labels_tensor != -100).any().item()):
        raise RuntimeError("No supervised target tokens in this batch; check max_llm_len and target formatting.")

    return {
        "prompt_input_ids": torch.tensor(prompt_input_ids, dtype=torch.long),
        "prompt_attention_mask": torch.tensor(prompt_attention_masks, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": labels_tensor,
    }


class End2EndSFTRouter(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        router_bert_init: str,
        lora_paths: Dict[str, str],
        expert_names: Sequence[str],
        first_layer_idx: int,
        middle_layer_idx: int,
        router_dim: int,
        router_pooling: str,
        router_pooling_last_k: int,
        dtype: str,
        r: int,
        alpha: int,
        local_files_only: bool,
        gradient_checkpointing: bool,
    ):
        super().__init__()
        self.expert_names = list(expert_names)
        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.router_pooling = str(router_pooling)
        self.router_pooling_last_k = int(router_pooling_last_k)
        self.router_dim = int(router_dim)

        torch_dtype = dtype_from_name(dtype)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
            local_files_only=local_files_only,
        )
        self.model.eval()
        if gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
        for p in self.model.parameters():
            p.requires_grad = False

        self.backbone_spec = infer_backbone_spec(self.model)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))
        if not (0 <= self.first_layer_idx < self.num_layers):
            raise ValueError(f"first_layer_idx={self.first_layer_idx} outside num_layers={self.num_layers}")
        if not (0 <= self.middle_layer_idx < self.num_layers):
            raise ValueError(f"middle_layer_idx={self.middle_layer_idx} outside num_layers={self.num_layers}")
        if self.middle_layer_idx <= self.first_layer_idx:
            raise ValueError("middle_layer_idx must be greater than first_layer_idx")

        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.expert_names),
            r=r,
            alpha=alpha,
        )
        for task_idx, task in enumerate(self.expert_names, start=1):
            if task not in lora_paths:
                raise KeyError(f"Missing LoRA path for expert {task!r}")
            load_lora_into_expert(self.model, lora_paths[task], task_idx)
        for p in self.model.parameters():
            p.requires_grad = False

        self.bert = BertExternalEncoder(router_bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        llama_hidden_size = int(self.model.config.hidden_size)
        self.router_first = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.router_mid = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, len(self.expert_names) * len(self.expert_names)),
        )

    def set_trainable(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.bert.parameters():
            p.requires_grad = False
        for p in self.router_first.parameters():
            p.requires_grad = True
        for p in self.router_mid.parameters():
            p.requires_grad = True
        for p in self.pair_classifier.parameters():
            p.requires_grad = True

    def route(self, prompt_input_ids, prompt_attention_mask, bert_input_ids, bert_attention_mask, bert_token_type_ids, sharpness: float, topk: Optional[int]):
        set_all_experts(self.model, NULL_EXPERT_ID)
        with torch.no_grad():
            prepass = self.model(
                input_ids=prompt_input_ids,
                attention_mask=prompt_attention_mask,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            first_vec = pool_prompt_vector(
                prepass.hidden_states[self.first_layer_idx],
                prompt_attention_mask,
                self.router_pooling,
                self.router_pooling_last_k,
            )
            mid_vec = pool_prompt_vector(
                prepass.hidden_states[self.middle_layer_idx],
                prompt_attention_mask,
                self.router_pooling,
                self.router_pooling_last_k,
            )

        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        first_feat = self.router_first(first_vec, bert_prev, bert_last, bert_attention_mask)
        mid_feat = self.router_mid(mid_vec, bert_prev, bert_last, bert_attention_mask)
        pair_logits = self.pair_classifier(torch.cat([first_feat, mid_feat], dim=-1))
        routed_logits = pair_logits.float() * float(sharpness)
        if topk is not None and int(topk) < routed_logits.size(-1):
            topk_ids = routed_logits.topk(k=int(topk), dim=-1).indices
            keep = torch.zeros_like(routed_logits, dtype=torch.bool)
            keep.scatter_(1, topk_ids, True)
            routed_logits = routed_logits.masked_fill(~keep, float("-inf"))
        pair_prob = torch.softmax(routed_logits, dim=-1).view(-1, len(self.expert_names), len(self.expert_names))
        first_real = pair_prob.sum(dim=2)
        mid_real = pair_prob.sum(dim=1)
        first_weights = torch.cat([first_real.new_zeros(first_real.size(0), 1), first_real], dim=1)
        mid_weights = torch.cat([mid_real.new_zeros(mid_real.size(0), 1), mid_real], dim=1)
        return pair_logits, first_weights, mid_weights

    def forward(self, lm_batch: Dict[str, torch.Tensor], bert_batch: Dict[str, torch.Tensor], sharpness: float, topk: Optional[int]):
        pair_logits, first_weights, mid_weights = self.route(
            prompt_input_ids=lm_batch["prompt_input_ids"],
            prompt_attention_mask=lm_batch["prompt_attention_mask"],
            bert_input_ids=bert_batch["input_ids"],
            bert_attention_mask=bert_batch["attention_mask"],
            bert_token_type_ids=bert_batch.get("token_type_ids"),
            sharpness=sharpness,
            topk=topk,
        )
        set_layer_range_expert_weights_grad(
            self.model,
            self.first_layer_idx,
            self.middle_layer_idx - 1,
            first_weights,
        )
        set_layer_range_expert_weights_grad(
            self.model,
            self.middle_layer_idx,
            self.num_layers - 1,
            mid_weights,
        )
        out = self.model(
            input_ids=lm_batch["input_ids"],
            attention_mask=lm_batch["attention_mask"],
            labels=lm_batch["labels"],
            use_cache=False,
            return_dict=True,
        )
        return out.loss, pair_logits, first_weights, mid_weights


def save_checkpoint(
    model: End2EndSFTRouter,
    tokenizer,
    out_dir: str,
    args,
    expert_names: Sequence[str],
    train_task_names: Sequence[str],
    best_val_loss: float,
    epoch: int,
):
    os.makedirs(out_dir, exist_ok=True)
    encoder_dir = os.path.join(out_dir, "encoder")
    model.bert.encoder.save_pretrained(encoder_dir)
    tokenizer.save_pretrained(encoder_dir)
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
            "train_task_names": list(train_task_names),
            "num_pairs": len(expert_names) * len(expert_names),
            "router_max_len": int(args.max_bert_len),
            "router_feature_type": "end2end_sft_prompt_hidden",
            "router_architecture": "pair_joint",
            "sample_feature_mode": "none",
            "sample_feature_dim": 0,
            "freeze_bert": True,
            "first_layer_idx": int(args.first_layer_idx),
            "middle_layer_idx": int(args.middle_layer_idx),
            "router_pooling": str(args.router_pooling),
            "router_pooling_last_k": int(args.router_pooling_last_k),
            "llama_hidden_size": int(model.model.config.hidden_size),
            "router_dim": int(args.router_dim),
            "feature_contract_version": 1,
            "supervision_type": "end2end_answer_ce",
            "supervision_mode": "sft_answer_loss",
            "joint_loss": "answer_token_ce",
            "routing_mode_for_training": "weighted_sum",
            "routing_sharpness": float(args.routing_sharpness),
            "routing_topk": None if args.routing_topk is None else int(args.routing_topk),
            "best_metric": "val_loss",
            "best_val_loss": float(best_val_loss),
            "best_epoch": int(epoch),
        },
        os.path.join(out_dir, "router_config.json"),
    )


def batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def run_epoch(model, loader, llm_tokenizer, bert_tokenizer, device, args, optimizer=None, scheduler=None, desc="train"):
    is_train = optimizer is not None
    model.train(is_train)
    # The base LLM and fixed LoRA experts are frozen; keep them deterministic
    # while allowing only the router heads to follow the requested train/eval mode.
    model.model.eval()
    model.bert.eval()
    if not is_train:
        model.eval()
    total_loss = 0.0
    total_samples = 0
    pair_counts = torch.zeros(len(model.expert_names), len(model.expert_names), dtype=torch.long)
    progress = make_progress(loader, total=len(loader), desc=desc)
    context = nullcontext() if is_train else torch.no_grad()
    with context:
        for step, batch in enumerate(progress, start=1):
            lm_batch = build_sft_lm_batch(
                tokenizer=llm_tokenizer,
                prompts=batch.texts,
                targets=batch.targets,
                max_length=args.max_llm_len,
                add_eos_to_target=args.add_eos_to_target,
            )
            lm_batch = batch_to_device(lm_batch, device)
            bert_batch = bert_tokenizer(
                batch.source_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_bert_len,
            )
            bert_batch = batch_to_device(bert_batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
            loss, pair_logits, _, _ = model(
                lm_batch=lm_batch,
                bert_batch=bert_batch,
                sharpness=args.routing_sharpness,
                topk=args.routing_topk,
            )
            if not bool(torch.isfinite(loss).item()):
                valid_labels = int((lm_batch["labels"] != -100).sum().detach().cpu().item())
                raise FloatingPointError(
                    f"Non-finite SFT loss at {desc} step={step}; "
                    f"valid_label_tokens={valid_labels}, max_llm_len={args.max_llm_len}."
                )
            if is_train:
                loss.backward()
                if args.max_grad_norm > 0:
                    trainable_params = [p for p in model.parameters() if p.requires_grad]
                    try:
                        grad_norm = torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            args.max_grad_norm,
                            error_if_nonfinite=True,
                        )
                    except TypeError:
                        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                        if not bool(torch.isfinite(torch.as_tensor(grad_norm)).item()):
                            raise FloatingPointError(f"Non-finite gradient norm at {desc} step={step}.")
                optimizer.step()
                for name, param in model.named_parameters():
                    if param.requires_grad and not bool(torch.isfinite(param).all().item()):
                        raise FloatingPointError(f"Non-finite trainable parameter after optimizer step: {name}")
                if scheduler is not None:
                    scheduler.step()
            batch_size = len(batch.texts)
            total_loss += float(loss.detach().cpu()) * batch_size
            total_samples += batch_size
            pred_pair = pair_logits.detach().argmax(dim=-1).cpu()
            for pair_idx in pred_pair.tolist():
                first = int(pair_idx) // len(model.expert_names)
                mid = int(pair_idx) % len(model.expert_names)
                pair_counts[first, mid] += 1
            if is_train and (step % args.log_every == 0 or step == len(loader)):
                print(
                    f"[TRAIN] step={step}/{len(loader)} loss={float(loss.detach().cpu()):.6f}",
                    flush=True,
                )
    avg_loss = total_loss / max(total_samples, 1)
    return {"loss": avg_loss, "samples": total_samples, "pair_counts": pair_counts.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="0602_router_train_dataset")
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--expert_names", type=str, required=True)
    parser.add_argument("--lora_paths", type=str, required=True, help="comma-separated task=path entries")
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=800)
    parser.add_argument("--max_val_samples", type=int, default=200)
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=16)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--router_pooling", type=str, default="mean", choices=["last_token", "mean", "lastk_mean"])
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--routing_sharpness", type=float, default=1.0)
    parser.add_argument("--routing_topk", type=int, default=None)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--add_eos_to_target", action="store_true")
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--local_files_only", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="opencompass")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    args = parser.parse_args()

    if args.disable_tqdm:
        import router_answer_supervision_core

        router_answer_supervision_core.tqdm = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    lora_paths = parse_lora_paths(args.lora_paths)
    expert_names = parse_csv_arg(args.expert_names) or []
    missing = [task for task in expert_names if task not in lora_paths]
    if missing:
        raise ValueError(f"Missing lora_paths for expert_names={missing}")
    task_names = parse_csv_arg(args.task_names)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] lora_paths={lora_paths}")

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, local_files_only=args.local_files_only)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    llm_tokenizer.truncation_side = "left"
    bert_tokenizer = AutoTokenizer.from_pretrained(
        args.router_bert_init,
        local_files_only=args.local_files_only,
        use_fast=False,
    )

    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as exc:
            raise ImportError("--wandb was set but wandb is not installed") from exc
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
        print(f"[INFO] wandb enabled project={args.wandb_project} name={args.wandb_name}")

    train_ds, train_task_names = build_dataset(
        data_root=args.data_root,
        split="train",
        requested_tasks=task_names,
        expert_names=expert_names,
        max_samples=args.max_train_samples,
        seed=args.seed,
    )
    val_ds, val_task_names = build_dataset(
        data_root=args.data_root,
        split="validation",
        requested_tasks=task_names,
        expert_names=expert_names,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(f"[INFO] train_task_names={train_task_names}")
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

    model = End2EndSFTRouter(
        base_model_path=args.base_model_path,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        expert_names=expert_names,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        router_dim=args.router_dim,
        router_pooling=args.router_pooling,
        router_pooling_last_k=args.router_pooling_last_k,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        local_files_only=args.local_files_only,
        gradient_checkpointing=args.gradient_checkpointing,
    )
    model.set_trainable()
    model.to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[INFO] trainable_params={sum(p.numel() for p in trainable)} freeze_bert=True")

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_steps * args.warmup_ratio),
        num_training_steps=total_steps,
    )

    best_val_loss = float("inf")
    best_epoch = -1
    epochs_without_improvement = 0
    history: List[Dict] = []
    save_json(vars(args), os.path.join(args.output_dir, "train_args.json"))
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            llm_tokenizer,
            bert_tokenizer,
            device,
            args,
            optimizer=optimizer,
            scheduler=scheduler,
            desc=f"train epoch {epoch}/{args.epochs}",
        )
        val_metrics = run_epoch(
            model,
            val_loader,
            llm_tokenizer,
            bert_tokenizer,
            device,
            args,
            optimizer=None,
            scheduler=None,
            desc=f"val epoch {epoch}/{args.epochs}",
        )
        row = {"epoch": epoch, "train": train_metrics, "validation": val_metrics}
        history.append(row)
        save_json({"history": history}, os.path.join(args.output_dir, "metrics.json"))
        print(
            f"[EPOCH] epoch={epoch} train_loss={train_metrics['loss']:.6f} "
            f"val_loss={val_metrics['loss']:.6f}",
            flush=True,
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/samples": train_metrics["samples"],
                    "validation/loss": val_metrics["loss"],
                    "validation/samples": val_metrics["samples"],
                    "best/val_loss": min(best_val_loss, val_metrics["loss"]),
                },
                step=epoch,
            )
        epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
        save_checkpoint(model, bert_tokenizer, epoch_dir, args, expert_names, train_task_names, val_metrics["loss"], epoch)
        improved = val_metrics["loss"] < (best_val_loss - args.early_stop_min_delta)
        if improved:
            best_val_loss = val_metrics["loss"]
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(model, bert_tokenizer, args.output_dir, args, expert_names, train_task_names, best_val_loss, best_epoch)
            print(f"[BEST] epoch={best_epoch} val_loss={best_val_loss:.6f}", flush=True)
        else:
            epochs_without_improvement += 1
            print(
                f"[EARLY_STOP] epoch={epoch} no_improve={epochs_without_improvement}/"
                f"{args.early_stop_patience} best_epoch={best_epoch} best_val_loss={best_val_loss:.6f}",
                flush=True,
            )
            if args.early_stop_patience > 0 and epochs_without_improvement >= args.early_stop_patience:
                print(f"[EARLY_STOP] stopping at epoch={epoch}", flush=True)
                break

    print(f"[DONE] best_epoch={best_epoch} best_val_loss={best_val_loss:.6f} output_dir={args.output_dir}")
    if wandb_run is not None:
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.summary["best_val_loss"] = best_val_loss
        wandb_run.summary["output_dir"] = args.output_dir
        wandb_run.finish()


if __name__ == "__main__":
    main()
