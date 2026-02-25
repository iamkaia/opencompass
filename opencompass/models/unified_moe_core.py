# opencompass/models/unified_moe_core.py
import os
import time
import json
from collections import Counter
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)

# =========================
# Label space (must match experts)
# =========================
ID2LABEL = {
    0: "iwslt2017",
    1: "medmcqa",
    2: "race",
    3: "squad2",
    4: "sst2",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# =========================
# MoE-LoRA Linear (hard routing)
# =========================
class HardRoutedLoRALinear(nn.Module):
    """A Linear layer + (selected expert) LoRA delta."""

    def __init__(self, base_linear: nn.Linear, num_experts=5, r=8, alpha=16):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.num_experts = int(num_experts)
        self.r = int(r)
        self.scale = float(alpha) / float(r)

        self.A = nn.ParameterList(
            [nn.Parameter(torch.empty(self.r, base_linear.in_features))
             for _ in range(self.num_experts)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.empty(base_linear.out_features, self.r))
             for _ in range(self.num_experts)]
        )
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A[e], a=5**0.5)
            nn.init.zeros_(self.B[e])

        self.expert_id = 0

    def set_expert(self, eid: int):
        self.expert_id = int(eid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        e = self.expert_id
        A = self.A[e].to(device=x.device, dtype=x.dtype)
        B = self.B[e].to(device=x.device, dtype=x.dtype)
        z = x @ A.t()
        d = z @ B.t()
        return y + self.scale * d


# =========================
# Patch & load helpers
# =========================
def patch_llama_mlp(model, num_experts=5, r=8, alpha=16):
    """Replace LLaMA MLP projections with HardRoutedLoRALinear."""
    for layer in model.model.layers:
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(mlp, name)
            if isinstance(base, nn.Linear):
                setattr(mlp, name, HardRoutedLoRALinear(base, num_experts, r, alpha))
    return model


def set_model_expert(model, eid: int):
    for m in model.modules():
        if isinstance(m, HardRoutedLoRALinear):
            m.set_expert(eid)


def _normalize_key(k: str) -> str:
    if "model.layers." in k:
        return k[k.index("model.layers.") :]
    return k


@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int):
    """
    Load PEFT LoRA weights into expert slot expert_id.
    Expects keys like:
      model.layers.{i}.mlp.{proj}.lora_A.weight
      model.layers.{i}.mlp.{proj}.lora_B.weight
    """
    sd = safe_load(f"{adapter_dir}/adapter_model.safetensors")
    sd = {_normalize_key(k): v for k, v in sd.items()}

    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj)
            if not isinstance(mod, HardRoutedLoRALinear):
                continue
            kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
            kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
            if kA not in sd or kB not in sd:
                raise KeyError(f"Missing LoRA keys: {kA} / {kB} in {adapter_dir}")

            mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device))
            mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device))


# =========================
# Core blackbox model
# =========================
class UnifiedMoECore:
    """
    Framework-agnostic blackbox:
      text (prompt) -> text (completion)

    Only implements external routing for now.
    """

    def __init__(
        self,
        base_model_path: str,
        cls_dir: str,
        lora_paths: Dict[str, str],
        dtype: str = "float16",
        r: int = 8,
        alpha: int = 16,
        device_map: str = "auto",
        max_seq_len: int = 2048,
    ):
        self.max_seq_len = int(max_seq_len)

        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        # --- base LM + tokenizer ---
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path, local_files_only=False)
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        # pad/eos safety
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        # --- patch MoE-LoRA ---
        self.model = patch_llama_mlp(self.model, num_experts=len(ID2LABEL), r=r, alpha=alpha)

        # --- load all experts ---
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task key in lora_paths: {task}")
            load_lora_into_expert(self.model, adir, LABEL2ID[task])

        # --- external router (classifier) ---
        self.router_tokenizer = AutoTokenizer.from_pretrained(cls_dir, local_files_only=True)
        self.router = AutoModelForSequenceClassification.from_pretrained(cls_dir, local_files_only=True)
        self.router.eval()

        self.route_counter = Counter()

    @torch.no_grad()
    def route_external(self, prompt: str) -> int:
        rt = self.router_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        rt = {k: v.to(next(self.router.parameters()).device) for k, v in rt.items()}
        eid = int(self.router(**rt).logits.argmax(dim=-1).item())
        self.route_counter[eid] += 1
        return eid

    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        # optional hook: caller can log routing decisions
        on_route: Optional[Any] = None,
    ) -> List[str]:
        if isinstance(prompts, str):
            prompts = [prompts]
        if gen_kwargs is None:
            gen_kwargs = {}

        outs: List[str] = []
        for p in prompts:
            eid = self.route_external(p)
            if on_route is not None:
                on_route(prompt=p, eid=eid)

            set_model_expert(self.model, eid)

            inp = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.model.device)

            args = dict(gen_kwargs)
            if max_new_tokens is not None:
                args["max_new_tokens"] = int(max_new_tokens)
            args.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            args.setdefault("eos_token_id", self.tokenizer.eos_token_id)

            out = self.model.generate(**inp, **args)

            input_len = inp["input_ids"].shape[1]
            gen_ids = out[0][input_len:]

            text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            outs.append(text)

        return outs
