import json
import os
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from model_backbone_specs import (
    get_decoder_layers,
    infer_backbone_spec,
    set_decoder_layer,
)


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]
NULL_EXPERT_ID = 0


class HardRoutedLoRALinear(nn.Module):
    def __init__(self, base_linear: nn.Linear, num_experts: int, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.num_experts = int(num_experts)
        self.r = int(r)
        self.scale = float(alpha) / float(r)

        self.A = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.r, base_linear.in_features)) for _ in range(self.num_experts)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.zeros(base_linear.out_features, self.r)) for _ in range(self.num_experts)]
        )

        for e in range(1, self.num_experts):
            nn.init.kaiming_uniform_(self.A[e], a=5**0.5)
            nn.init.zeros_(self.B[e])

        self.active_expert = 0

    def set_expert(self, eid: int):
        self.active_expert = int(eid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        eid = int(self.active_expert)
        A = self.A[eid].to(device=x.device, dtype=x.dtype)
        B = self.B[eid].to(device=x.device, dtype=x.dtype)
        z = x @ A.t()
        d = z @ B.t()
        return y + self.scale * d


def patch_causal_lm_with_hard_routed_lora(model, num_experts: int, r: int = 8, alpha: int = 16):
    layers = get_decoder_layers(model)
    for layer in layers:
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(mlp, name)
            if isinstance(base, nn.Linear):
                setattr(mlp, name, HardRoutedLoRALinear(base, num_experts=num_experts, r=r, alpha=alpha))

        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(attn, name)
            if isinstance(base, nn.Linear):
                setattr(attn, name, HardRoutedLoRALinear(base, num_experts=num_experts, r=r, alpha=alpha))
    return model


def patch_llama_with_hard_routed_lora(model, num_experts: int, r: int = 8, alpha: int = 16):
    return patch_causal_lm_with_hard_routed_lora(model, num_experts=num_experts, r=r, alpha=alpha)


def set_all_experts(model, eid: int):
    for m in model.modules():
        if isinstance(m, HardRoutedLoRALinear):
            m.set_expert(eid)


def _unwrap_layer(layer):
    while hasattr(layer, "base_layer"):
        layer = layer.base_layer
    return layer


def set_layer_expert(model, layer_idx: int, eid: int):
    layer = get_decoder_layers(model)[layer_idx]
    layer = _unwrap_layer(layer)

    for name in ["gate_proj", "up_proj", "down_proj"]:
        mod = getattr(layer.mlp, name)
        if isinstance(mod, HardRoutedLoRALinear):
            mod.set_expert(eid)

    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        mod = getattr(layer.self_attn, name)
        if isinstance(mod, HardRoutedLoRALinear):
            mod.set_expert(eid)


def set_layer_range_expert(model, start_idx: int, end_idx: int, eid: int):
    if end_idx < start_idx:
        return
    num_layers = len(get_decoder_layers(model))
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), num_layers - 1)
    for li in range(start_idx, end_idx + 1):
        set_layer_expert(model, li, eid)


def _normalize_key(k: str) -> str:
    if "model.layers." in k:
        return k[k.index("model.layers."):]
    if "base_model.model.model.layers." in k:
        return k[k.index("model.layers."):]
    return k


@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int):
    sd = safe_load(os.path.join(adapter_dir, "adapter_model.safetensors"))
    sd = {_normalize_key(k): v for k, v in sd.items()}

    def _copy(mod, kA, kB):
        if kA not in sd or kB not in sd:
            raise KeyError(f"Missing LoRA keys: {kA} / {kB} in {adapter_dir}")
        mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device, dtype=mod.A[expert_id].dtype))
        mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device, dtype=mod.B[expert_id].dtype))

    for li, layer in enumerate(get_decoder_layers(model)):
        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                _copy(mod, f"model.layers.{li}.mlp.{proj}.lora_A.weight", f"model.layers.{li}.mlp.{proj}.lora_B.weight")

        attn = layer.self_attn
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(attn, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                _copy(mod, f"model.layers.{li}.self_attn.{proj}.lora_A.weight", f"model.layers.{li}.self_attn.{proj}.lora_B.weight")


class BertExternalEncoder(nn.Module):
    def __init__(self, bert_name_or_path: str):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(bert_name_or_path)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "output_hidden_states": True,
            "return_dict": True,
        }
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        return out.hidden_states[-2], out.hidden_states[-1]


class CompactCrossAttentionRouter(nn.Module):
    def __init__(self, llama_hidden_size: int, bert_hidden_size: int, router_dim: int, num_tasks: int):
        super().__init__()
        self.q_proj = nn.Linear(llama_hidden_size, router_dim)
        self.k_proj = nn.Linear(bert_hidden_size, router_dim)
        self.v_proj = nn.Linear(bert_hidden_size, router_dim)
        self.out_norm = nn.LayerNorm(router_dim * 2)
        self.classifier = nn.Linear(router_dim * 2, num_tasks)

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
        qv = q.squeeze(1)

        feat = torch.cat([qv, ctx], dim=-1)
        feat = self.out_norm(feat)
        logits = self.classifier(feat)
        return logits


class BeforeAttentionRouterWrapper(nn.Module):
    def __init__(self, base_layer, core_ref, which: str):
        super().__init__()
        self.base_layer = base_layer
        self.core = core_ref
        self.which = which

    @staticmethod
    def gather_last_valid(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, -1, :]

    def forward(self, hidden_states, *args, **kwargs):
        if self.which == "first":
            if self.core._should_route_first(hidden_states):
                vec = self.gather_last_valid(hidden_states)
                eid = self.core._route_first_from_vec(vec)
                self.core.cached_first_eid = eid
                set_layer_range_expert(
                    self.core.model,
                    self.core.first_layer_idx,
                    self.core.middle_layer_idx - 1,
                    eid,
                )
        elif self.which == "mid":
            if self.core._should_route_mid(hidden_states):
                vec = self.gather_last_valid(hidden_states)
                eid = self.core._route_mid_from_vec(vec)
                self.core.cached_mid_eid = eid
                set_layer_range_expert(
                    self.core.model,
                    self.core.middle_layer_idx,
                    self.core.num_layers - 1,
                    eid,
                )

        return self.base_layer(hidden_states, *args, **kwargs)


class UnifiedMoECoreInternalRouterCompact:
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
        self.first_layer_idx = int(cfg["first_layer_idx"])
        self.middle_layer_idx = int(cfg["middle_layer_idx"])
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

        self.backbone_spec = infer_backbone_spec(self.model)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))

        self.model = patch_causal_lm_with_hard_routed_lora(
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

        self.router_first = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden,
            bert_hidden_size=bert_hidden,
            router_dim=router_dim,
            num_tasks=len(self.task_names),
        )
        self.router_mid = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden,
            bert_hidden_size=bert_hidden,
            router_dim=router_dim,
            num_tasks=len(self.task_names),
        )

        state = torch.load(os.path.join(router_ckpt_dir, "router_heads.pt"), map_location="cpu")
        self.router_first.load_state_dict(state["router_first"])
        self.router_mid.load_state_dict(state["router_mid"])
        self.bert_encoder.load_state_dict(state["bert_encoder"], strict=False)

        self.router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_encoder.to(self.router_device).eval()
        self.router_first.to(self.router_device).eval()
        self.router_mid.to(self.router_device).eval()

        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None

        base_first = get_decoder_layers(self.model, spec=self.backbone_spec)[self.first_layer_idx]
        set_decoder_layer(self.model, self.first_layer_idx, BeforeAttentionRouterWrapper(
            base_first, self, which="first"
        ), spec=self.backbone_spec)

        base_mid = get_decoder_layers(self.model, spec=self.backbone_spec)[self.middle_layer_idx]
        set_decoder_layer(self.model, self.middle_layer_idx, BeforeAttentionRouterWrapper(
            base_mid, self, which="mid"
        ), spec=self.backbone_spec)

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

    def _should_route_first(self, hidden_states: torch.Tensor) -> bool:
        return (hidden_states.size(1) > 1) and (self.cached_first_eid is None)

    def _should_route_mid(self, hidden_states: torch.Tensor) -> bool:
        return (hidden_states.size(1) > 1) and (self.cached_mid_eid is None)

    def _forced_task_to_eid(self, task_name: Optional[str]) -> Optional[int]:
        if task_name is None:
            return None
        if task_name not in self.task_to_eid:
            raise ValueError(
                f"Unknown forced task: {task_name}. "
                f"Available tasks: {self.task_names}"
            )
        return self.task_to_eid[task_name]

    '''
    @torch.no_grad()
    def _route_first_from_vec(self, vec: torch.Tensor) -> int:
        logits = self.router_first(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"first::{task}"] += 1
        return eid

    @torch.no_grad()
    def _route_mid_from_vec(self, vec: torch.Tensor) -> int:
        logits = self.router_mid(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"mid::{task}"] += 1
        return eid
    '''
    @torch.no_grad()
    def _route_first_from_vec(self, vec: torch.Tensor) -> int:
        forced_eid = self._forced_task_to_eid(self.force_first_task)
        if forced_eid is not None:
            task = self.task_names[forced_eid - 1]
            self.route_counter[f"first_forced::{task}"] += 1
            return forced_eid

        logits = self.router_first(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"first::{task}"] += 1
        return eid

    @torch.no_grad()
    def _route_mid_from_vec(self, vec: torch.Tensor) -> int:
        forced_eid = self._forced_task_to_eid(self.force_mid_task)
        if forced_eid is not None:
            task = self.task_names[forced_eid - 1]
            self.route_counter[f"mid_forced::{task}"] += 1
            return forced_eid

        logits = self.router_mid(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"mid::{task}"] += 1
        return eid

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
