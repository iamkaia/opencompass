import json
import os
from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from opencompass.models.unified_moe_core_internal_router_compact import (
    NULL_EXPERT_ID,
    BertExternalEncoder,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
)


class CompactRouterFeatureEncoder(nn.Module):
    def __init__(self, llama_hidden_size: int, bert_hidden_size: int, router_dim: int):
        super().__init__()
        self.q_proj = nn.Linear(llama_hidden_size, router_dim)
        self.k_proj = nn.Linear(bert_hidden_size, router_dim)
        self.v_proj = nn.Linear(bert_hidden_size, router_dim)
        self.out_norm = nn.LayerNorm(router_dim * 2)

    def forward(self, llama_vec, bert_prev, bert_last, bert_attention_mask=None):
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
        feat = torch.cat([q.squeeze(1), ctx], dim=-1)
        return self.out_norm(feat)


class UnifiedMoECoreInternalRouterCompactCachedJoint:
    def __init__(
        self,
        base_model_path: str,
        router_ckpt_dir: str,
        router_bert_init: str,
        lora_paths: Dict[str, str],
        dtype: str = "float16",
        r: int = 8,
        alpha: int = 32,
        router_dim: int = 512,
        device_map: str = "auto",
        max_seq_len: int = 2048,
        first_layer_idx: int = 0,
        middle_layer_idx: int = 15,
        force_first_task: Optional[str] = None,
        force_mid_task: Optional[str] = None,
    ):
        self.max_seq_len = int(max_seq_len)
        self.route_counter = Counter()
        self.force_first_task = force_first_task
        self.force_mid_task = force_mid_task
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        with open(os.path.join(router_ckpt_dir, "router_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.task_names = cfg["task_names"]
        self.task_to_eid = {task: i + 1 for i, task in enumerate(self.task_names)}
        self.first_layer_idx = int(cfg.get("first_layer_idx", first_layer_idx))
        self.middle_layer_idx = int(cfg.get("middle_layer_idx", middle_layer_idx))
        self.router_max_len = int(cfg.get("router_max_len", 512))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.num_layers = len(self.model.model.layers)
        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.task_names),
            r=r,
            alpha=alpha,
        )

        for task, adapter_dir in lora_paths.items():
            task_id = self.task_names.index(task)
            expert_id = task_id + 1
            load_lora_into_expert(self.model, adapter_dir, expert_id)

        encoder_dir = os.path.join(router_ckpt_dir, "encoder")
        self.router_tokenizer = AutoTokenizer.from_pretrained(router_bert_init)
        self.bert_encoder = BertExternalEncoder(encoder_dir)

        bert_hidden = self.bert_encoder.encoder.config.hidden_size
        llama_hidden = self.model.config.hidden_size
        self.router_first = CompactRouterFeatureEncoder(llama_hidden, bert_hidden, router_dim)
        self.router_mid = CompactRouterFeatureEncoder(llama_hidden, bert_hidden, router_dim)
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, len(self.task_names) * len(self.task_names)),
        )

        state = torch.load(os.path.join(router_ckpt_dir, "router_heads.pt"), map_location="cpu")
        first_state = state.get("router_first") or state.get("pair_first_encoder")
        mid_state = state.get("router_mid") or state.get("pair_mid_encoder")
        if first_state is None or mid_state is None:
            raise KeyError(
                "router_heads.pt must contain either router_first/router_mid "
                "or pair_first_encoder/pair_mid_encoder."
            )
        self.router_first.load_state_dict(first_state)
        self.router_mid.load_state_dict(mid_state)
        self.pair_classifier.load_state_dict(state["pair_classifier"])
        self.bert_encoder.load_state_dict(state["bert_encoder"], strict=False)

        self.router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_encoder.to(self.router_device).eval()
        self.router_first.to(self.router_device).eval()
        self.router_mid.to(self.router_device).eval()
        self.pair_classifier.to(self.router_device).eval()

        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None

    def _reset_runtime_cache(self):
        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None
        set_all_experts(self.model, NULL_EXPERT_ID)

    def _encode_bert_memory(self, prompt: str):
        rt = self.router_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.router_max_len,
        )
        rt = {
            k: v.to(self.router_device)
            for k, v in rt.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        bert_prev, bert_last = self.bert_encoder(**rt)
        self.cached_bert_prev = bert_prev
        self.cached_bert_last = bert_last
        self.cached_bert_mask = rt["attention_mask"]

    def _forced_task_to_eid(self, task_name: Optional[str]) -> Optional[int]:
        if task_name is None:
            return None
        if task_name not in self.task_to_eid:
            raise ValueError(f"Unknown forced task: {task_name}. Available tasks: {self.task_names}")
        return self.task_to_eid[task_name]

    @torch.no_grad()
    def _route_pair_from_vecs(self, first_vec: torch.Tensor, mid_vec: torch.Tensor):
        forced_first_eid = self._forced_task_to_eid(self.force_first_task)
        forced_mid_eid = self._forced_task_to_eid(self.force_mid_task)
        num_tasks = len(self.task_names)

        first_feat = self.router_first(
            llama_vec=first_vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        mid_feat = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        pair_logits = self.pair_classifier(torch.cat([first_feat, mid_feat], dim=-1))

        if forced_first_eid is not None or forced_mid_eid is not None:
            mask = torch.ones_like(pair_logits, dtype=torch.bool)
            for pair_idx in range(pair_logits.size(-1)):
                first_idx = pair_idx // num_tasks
                mid_idx = pair_idx % num_tasks
                keep = True
                if forced_first_eid is not None:
                    keep = keep and (first_idx == (forced_first_eid - 1))
                if forced_mid_eid is not None:
                    keep = keep and (mid_idx == (forced_mid_eid - 1))
                mask[..., pair_idx] = keep
            pair_logits = pair_logits.masked_fill(~mask, float("-inf"))

        pred_pair = int(pair_logits.argmax(dim=-1).item())
        first_eid = (pred_pair // num_tasks) + 1
        mid_eid = (pred_pair % num_tasks) + 1

        self.route_counter[f"first::{self.task_names[first_eid - 1]}"] += 1
        self.route_counter[f"mid::{self.task_names[mid_eid - 1]}"] += 1
        self.route_counter[f"pair::{self.task_names[first_eid - 1]}->{self.task_names[mid_eid - 1]}"] += 1
        return first_eid, mid_eid

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=None, gen_kwargs=None, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]
        if gen_kwargs is None:
            gen_kwargs = {}

        outputs = []
        for prompt in prompts:
            self._reset_runtime_cache()
            self._encode_bert_memory(prompt)

            inp = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.model.device)

            prepass = self.model(
                **inp,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            hidden_states = prepass.hidden_states
            first_vec = hidden_states[self.first_layer_idx][:, -1, :]
            mid_vec = hidden_states[self.middle_layer_idx][:, -1, :]

            first_eid, mid_eid = self._route_pair_from_vecs(first_vec, mid_vec)
            self.cached_first_eid = first_eid
            self.cached_mid_eid = mid_eid

            set_layer_range_expert(
                self.model,
                self.first_layer_idx,
                self.middle_layer_idx - 1,
                first_eid,
            )
            set_layer_range_expert(
                self.model,
                self.middle_layer_idx,
                self.num_layers - 1,
                mid_eid,
            )

            args = dict(gen_kwargs)
            if "max_new_tokens" not in args and max_new_tokens is not None:
                args["max_new_tokens"] = int(max_new_tokens)
            args.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            args.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            args["do_sample"] = False
            args["temperature"] = 0.0
            args["top_p"] = 1.0
            args["num_beams"] = 1

            out = self.model.generate(**inp, **args)
            prompt_len = inp["input_ids"].shape[1]
            gen_ids = out[0][prompt_len:]
            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outputs.append(text)

        return outputs
