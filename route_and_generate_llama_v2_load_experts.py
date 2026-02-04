#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

from safetensors.torch import load_file as safe_load


# ----------------------------
# 0) Your id2label mapping
# ----------------------------
ID2LABEL = {
    0: "iwslt2017",
    1: "medmcqa",
    2: "race",
    3: "squad2",
    4: "sst2",
}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


# ----------------------------
# 1) V2 MoE-style LoRA Experts Linear (hard routing)
# ----------------------------
class HardRoutedLoRALinear(nn.Module):
    """
    y = base(x) + scale * B_e(A_e(x))  where e is chosen expert_id
    """
    def __init__(self, base_linear: nn.Linear, num_experts=5, r=8, alpha=16, dropout=0.0):
        super().__init__()
        assert isinstance(base_linear, nn.Linear)

        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)

        self.in_features = base_linear.in_features
        self.out_features = base_linear.out_features

        self.num_experts = int(num_experts)
        self.r = int(r)
        self.scale = float(alpha) / float(r) if r > 0 else 1.0
        self.drop = nn.Dropout(float(dropout)) if dropout and dropout > 0 else nn.Identity()

        # Experts: A_e: (r, in), B_e: (out, r)
        self.A = nn.ParameterList([nn.Parameter(torch.empty(self.r, self.in_features)) for _ in range(self.num_experts)])
        self.B = nn.ParameterList([nn.Parameter(torch.empty(self.out_features, self.r)) for _ in range(self.num_experts)])

        # init
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A[e], a=5 ** 0.5)
            nn.init.zeros_(self.B[e])

        self.expert_id = 0  # default

    def set_expert(self, expert_id: int):
        eid = int(expert_id)
        if not (0 <= eid < self.num_experts):
            raise ValueError(f"expert_id {eid} out of range [0, {self.num_experts-1}]")
        self.expert_id = eid
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x = self.drop(x)

        e = self.expert_id
        z = torch.matmul(x, self.A[e].t())  # (..., r)
        d = torch.matmul(z, self.B[e].t())  # (..., out)
        return y + self.scale * d
    '''
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        x = self.drop(x)

        e = self.expert_id

        # ★ ensure same device & dtype as x
        A = self.A[e].to(device=x.device, dtype=x.dtype)
        B = self.B[e].to(device=x.device, dtype=x.dtype)

        z = torch.matmul(x, A.t())
        d = torch.matmul(z, B.t())
        return y + self.scale * d



def patch_llama_mlp(model, num_experts=5, r=8, alpha=16, dropout=0.0):
    """
    Patch Llama HF:
      model.model.layers[i].mlp.gate_proj/up_proj/down_proj
    """
    for layer in model.model.layers:
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(mlp, name)
            if isinstance(base, nn.Linear):
                wrapped = HardRoutedLoRALinear(
                    base_linear=base,
                    num_experts=num_experts,
                    r=r,
                    alpha=alpha,
                    dropout=dropout,
                )
                setattr(mlp, name, wrapped)
    return model


def set_model_expert(model, expert_id: int):
    for m in model.modules():
        if isinstance(m, HardRoutedLoRALinear):
            m.set_expert(expert_id)


# ----------------------------
# 2) Load PEFT LoRA weights into a given expert slot
# ----------------------------
def _normalize_key(k: str) -> str:
    # PEFT keys often look like:
    #   base_model.model.model.layers.0.mlp.up_proj.lora_A.weight
    # or model.model.layers.0.mlp.up_proj.lora_A.weight
    # normalize to start from "model.layers..."
    if "model.layers." in k:
        return k[k.index("model.layers."):]
    return k


@torch.no_grad()
def load_lora_adapter_into_expert(
    llama_model,
    adapter_dir: str,
    expert_id: int,
    strict: bool = True,
):
    """
    Read adapter_model.safetensors and fill HardRoutedLoRALinear.A/B[expert_id].
    Only supports MLP: gate_proj/up_proj/down_proj.
    """
    st_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"Missing {st_path}")

    sd = safe_load(st_path)  # tensor dict
    # normalize keys
    norm = {_normalize_key(k): v for k, v in sd.items()}

    # build quick lookup for A/B
    # expected suffix: model.layers.{i}.mlp.{proj}.lora_A.weight / lora_B.weight
    missing = []

    for layer_idx, layer in enumerate(llama_model.model.layers):
        mlp = layer.mlp
        for proj_name in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj_name)
            if not isinstance(mod, HardRoutedLoRALinear):
                continue

            kA = f"model.layers.{layer_idx}.mlp.{proj_name}.lora_A.weight"
            kB = f"model.layers.{layer_idx}.mlp.{proj_name}.lora_B.weight"

            if kA not in norm or kB not in norm:
                missing.append((kA, kB))
                continue

            A = norm[kA].to(mod.A[expert_id].device, dtype=mod.A[expert_id].dtype)
            B = norm[kB].to(mod.B[expert_id].device, dtype=mod.B[expert_id].dtype)

            # PEFT shapes:
            #  lora_A: (r, in_features)
            #  lora_B: (out_features, r)
            if A.shape != mod.A[expert_id].shape or B.shape != mod.B[expert_id].shape:
                raise ValueError(
                    f"Shape mismatch at layer {layer_idx} {proj_name}: "
                    f"A {tuple(A.shape)} vs {tuple(mod.A[expert_id].shape)}, "
                    f"B {tuple(B.shape)} vs {tuple(mod.B[expert_id].shape)}"
                )

            mod.A[expert_id].copy_(A)
            mod.B[expert_id].copy_(B)

    if missing and strict:
        # print a small hint of what keys exist
        sample_keys = list(norm.keys())[:20]
        raise KeyError(
            "Some LoRA keys not found (showing first 3 missing pairs):\n"
            + "\n".join([f"  {a} | {b}" for a, b in missing[:3]])
            + "\n\nHint: your adapter keys may use a different prefix. "
              "I normalized from the first occurrence of 'model.layers.'. "
              "Example existing keys:\n  "
            + "\n  ".join(sample_keys)
        )

    print(f"[LOAD] expert_id={expert_id} <- {adapter_dir}  (missing_pairs={len(missing)})")


# ----------------------------
# 3) Router: bert-tiny classifier (top-1)
# ----------------------------
@torch.no_grad()
def route_expert_id(router_model, router_tok, prompt: str, device: str, max_len: int = 512) -> int:
    toks = router_tok(
        prompt,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors="pt",
    ).to(device)
    logits = router_model(**toks).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())
    return pred_id


# ----------------------------
# 4) Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls_dir", type=str, required=True, help="task_classifier_ckpt")
    ap.add_argument("--base_model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    ap.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    ap.add_argument("--r", type=int, default=8)
    ap.add_argument("--alpha", type=int, default=16)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--prompt", type=str, required=True)

    # Each task's adapter dir (PEFT format with adapter_model.safetensors)
    ap.add_argument("--lora_iwslt", type=str, required=True)
    ap.add_argument("--lora_medmcqa", type=str, required=True)
    ap.add_argument("--lora_race", type=str, required=True)
    ap.add_argument("--lora_squad2", type=str, required=True)
    ap.add_argument("--lora_sst2", type=str, required=True)

    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    # --- load router ---
    router_tok = AutoTokenizer.from_pretrained(args.cls_dir)
    router_model = AutoModelForSequenceClassification.from_pretrained(args.cls_dir).to(device).eval()

    # --- load llama + patch to V2 experts ---
    llm_tok = AutoTokenizer.from_pretrained(args.base_model, use_fast=False)
    llm = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        #device_map="auto",
    )
    llm = patch_llama_mlp(llm, num_experts=len(ID2LABEL), r=args.r, alpha=args.alpha, dropout=args.dropout)

    # --- load each task LoRA into expert slot ---
    load_lora_adapter_into_expert(llm, args.lora_iwslt,  LABEL2ID["iwslt2017"])
    load_lora_adapter_into_expert(llm, args.lora_medmcqa, LABEL2ID["medmcqa"])
    load_lora_adapter_into_expert(llm, args.lora_race,    LABEL2ID["race"])
    load_lora_adapter_into_expert(llm, args.lora_squad2,  LABEL2ID["squad2"])
    load_lora_adapter_into_expert(llm, args.lora_sst2,    LABEL2ID["sst2"])

    # --- route ---
    pred_id = route_expert_id(router_model, router_tok, args.prompt, device=device, max_len=512)
    task = ID2LABEL.get(pred_id, "UNKNOWN")
    print(f"\n[ROUTE] pred_id={pred_id} task={task}")

    # --- set expert inside llama (V2) ---
    set_model_expert(llm, pred_id)

    # --- generate ---
    inputs = llm_tok(args.prompt, return_tensors="pt").to(llm.device)
    out = llm.generate(**inputs, max_new_tokens=args.max_new_tokens)
    print("\n[OUTPUT]")
    print(llm_tok.decode(out[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
