####主要釋放lora expert機制相關的東西
from typing import Dict

import torch
import torch.nn as nn
from safetensors.torch import load_file as safe_load

from model_backbone_specs import get_decoder_layers


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

        for expert_idx in range(1, self.num_experts):
            nn.init.kaiming_uniform_(self.A[expert_idx], a=5**0.5)
            nn.init.zeros_(self.B[expert_idx])

        self.active_expert = NULL_EXPERT_ID

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

####在這幾個地方掛上router
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


####目前還沒有用到，應該是指定所有layer的要用哪一個同樣的expert
def set_all_experts(model, eid: int):
    for module in model.modules():
        if isinstance(module, HardRoutedLoRALinear):
            module.set_expert(eid)


def _unwrap_layer(layer):
    while hasattr(layer, "base_layer"):
        layer = layer.base_layer
    return layer

##控制特定layer 現在吃哪個 expert
def set_layer_expert(model, layer_idx: int, eid: int):
    layer = _unwrap_layer(get_decoder_layers(model)[layer_idx])

    for name in ["gate_proj", "up_proj", "down_proj"]:
        mod = getattr(layer.mlp, name)
        if isinstance(mod, HardRoutedLoRALinear):
            mod.set_expert(eid)

    for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
        mod = getattr(layer.self_attn, name)
        if isinstance(mod, HardRoutedLoRALinear):
            mod.set_expert(eid)

####控制哪些 layer 現在吃哪個 expert
def set_layer_range_expert(model, start_idx: int, end_idx: int, eid: int):
    if end_idx < start_idx:
        return
    num_layers = len(get_decoder_layers(model))
    start_idx = max(0, int(start_idx))
    end_idx = min(int(end_idx), num_layers - 1)
    for layer_idx in range(start_idx, end_idx + 1):
        set_layer_expert(model, layer_idx, eid)


def _normalize_key(key: str) -> str:
    if "model.layers." in key:
        return key[key.index("model.layers."):]
    if "base_model.model.model.layers." in key:
        return key[key.index("model.layers."):]
    return key

####把某個 task 的 adapter 權重載進 expert slot
@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int):
    state_dict = safe_load(f"{adapter_dir}/adapter_model.safetensors")
    state_dict = {_normalize_key(key): value for key, value in state_dict.items()}

    def _copy(mod, key_a, key_b):
        if key_a not in state_dict or key_b not in state_dict:
            raise KeyError(f"Missing LoRA keys: {key_a} / {key_b} in {adapter_dir}")
        mod.A[expert_id].copy_(state_dict[key_a].to(mod.A[expert_id].device, dtype=mod.A[expert_id].dtype))
        mod.B[expert_id].copy_(state_dict[key_b].to(mod.B[expert_id].device, dtype=mod.B[expert_id].dtype))

    for layer_idx, layer in enumerate(get_decoder_layers(model)):
        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                _copy(mod, f"model.layers.{layer_idx}.mlp.{proj}.lora_A.weight", f"model.layers.{layer_idx}.mlp.{proj}.lora_B.weight")

        attn = layer.self_attn
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(attn, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                _copy(mod, f"model.layers.{layer_idx}.self_attn.{proj}.lora_A.weight", f"model.layers.{layer_idx}.self_attn.{proj}.lora_B.weight")
