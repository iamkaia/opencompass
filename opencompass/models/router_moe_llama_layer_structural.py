# router_moe_llama_layer_structural_fastbatch.py
#
# Structural Layer Router (NO classifier, NO router_ckpt)
# Prefill-only routing (T>1 route, T==1 reuse)
# Norm-based expert selection (0 parameter router)
# Fully compatible with OpenCompass HuggingFacewithChatTemplate

import os
import json
from collections import Counter
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file as safe_load
from opencompass.models import HuggingFacewithChatTemplate


# =========================
# Label space
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
# Routed LoRA Linear (Structural)
# =========================
class HardRoutedLoRALinear(nn.Module):
    """
    Structural routing version:
    - No classifier
    - Choose expert by LoRA output norm
    - Prefill-only caching
    """

    def __init__(self, base_linear, experts_A, experts_B):
        super().__init__()
        self.base = base_linear
        self.A = nn.ParameterList(experts_A)
        self.B = nn.ParameterList(experts_B)

        self.num_experts = len(self.A)
        self._cached_eid = None  # prefill-only

    @torch.no_grad()
    def _choose_expert(self, x):
        """
        x: (B,T,H)
        score_e = || B_e(A_e(x)) ||_2
        """
        scores = []
        for i in range(self.num_experts):
            delta = torch.matmul(
                torch.matmul(x, self.A[i].T),
                self.B[i].T
            )
            s = delta.norm(p=2, dim=-1).mean(dim=-1)  # (B,)
            scores.append(s)
        scores = torch.stack(scores, dim=-1)  # (B,E)
        return scores.argmax(dim=-1)  # (B,)

    def forward(self, x):
        base_out = self.base(x)

        B, T, H = x.shape

        # Prefill stage
        if T > 1:
            eid = self._choose_expert(x)
            self._cached_eid = eid
        else:
            # Decode stage
            if self._cached_eid is None:
                eid = self._choose_expert(x)
                self._cached_eid = eid
            else:
                eid = self._cached_eid

        out = []
        for b in range(B):
            i = int(eid[b])
            delta = torch.matmul(
                torch.matmul(x[b:b+1], self.A[i].T),
                self.B[i].T
            )
            out.append(base_out[b:b+1] + delta)

        return torch.cat(out, dim=0)


# =========================
# Main Model
# =========================
class RouterMoELlama(HuggingFacewithChatTemplate):

    def __init__(
        self,
        path: str,
        lora_paths: Dict[str, str],
        abbr: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(path=path, abbr=abbr, **kwargs)

        self.is_api = False

        self.expert_names = list(lora_paths.keys())
        self.num_experts = len(self.expert_names)

        self._load_lora_experts(lora_paths)
        self._inject_structural_router()

    # -------------------------
    # Load LoRA Weights
    # -------------------------
    def _load_lora_experts(self, lora_paths):

        self.experts_A = {}
        self.experts_B = {}

        for name, adapter_dir in lora_paths.items():
            state = safe_load(os.path.join(adapter_dir, "adapter_model.safetensors"))

            for k, v in state.items():
                if "lora_A" in k:
                    self.experts_A.setdefault(k, {})[name] = v
                if "lora_B" in k:
                    self.experts_B.setdefault(k, {})[name] = v

    # -------------------------
    # Replace MLP projections
    # -------------------------
    def _inject_structural_router(self):

        for name, module in self.model.named_modules():

            if isinstance(module, nn.Linear) and "mlp" in name:

                experts_A = []
                experts_B = []

                for expert in self.expert_names:

                    keyA = f"{name}.lora_A.weight"
                    keyB = f"{name}.lora_B.weight"

                    if keyA not in self.experts_A:
                        continue

                    experts_A.append(nn.Parameter(self.experts_A[keyA][expert]))
                    experts_B.append(nn.Parameter(self.experts_B[keyB][expert]))

                if len(experts_A) == 0:
                    continue

                parent = self._get_parent_module(name)
                attr = name.split(".")[-1]

                setattr(
                    parent,
                    attr,
                    HardRoutedLoRALinear(
                        module,
                        experts_A,
                        experts_B,
                    ),
                )

    def _get_parent_module(self, module_name):
        parts = module_name.split(".")
        obj = self.model
        for p in parts[:-1]:
            obj = getattr(obj, p)
        return obj
