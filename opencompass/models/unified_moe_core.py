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

def patch_llama_with_hard_routed_lora(model, num_experts=5, r=8, alpha=16):
    for layer in model.model.layers:
        # ---- MLP ----
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(mlp, name)
            if isinstance(base, nn.Linear):
                setattr(mlp, name, HardRoutedLoRALinear(base, num_experts, r, alpha))

        # ---- Attention ----
        attn = layer.self_attn
        for name in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            base = getattr(attn, name)
            if isinstance(base, nn.Linear):
                setattr(attn, name, HardRoutedLoRALinear(base, num_experts, r, alpha))

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

    Supports BOTH:
      model.layers.{i}.mlp.{gate_proj,up_proj,down_proj}.lora_A/B.weight
      model.layers.{i}.self_attn.{q_proj,k_proj,v_proj,o_proj}.lora_A/B.weight
    """
    sd = safe_load(f"{adapter_dir}/adapter_model.safetensors")
    sd = {_normalize_key(k): v for k, v in sd.items()}

    # helper to copy A/B
    def _copy(mod, kA, kB):
        if kA not in sd or kB not in sd:
            raise KeyError(f"Missing LoRA keys: {kA} / {kB} in {adapter_dir}")
        mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device))
        mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device))

    for li, layer in enumerate(model.model.layers):
        # ---- MLP ----
        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
                kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
                _copy(mod, kA, kB)

        # ---- Attention ----
        attn = layer.self_attn
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(attn, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                kA = f"model.layers.{li}.self_attn.{proj}.lora_A.weight"
                kB = f"model.layers.{li}.self_attn.{proj}.lora_B.weight"
                _copy(mod, kA, kB)

# =========================
# Core blackbox model
# =========================
class UnifiedMoECore:
    """
    Framework-agnostic blackbox:
      text (prompt) -> text (completion)

    External routing only (classifier).
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
        # optional: if cls_dir doesn't contain tokenizer files, set this to a HF id or a local tokenizer dir
        cls_tokenizer: Optional[str] = None,
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

        # --- patch MoE-LoRA (MLP + ATTN) ---
        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=len(ID2LABEL),
            r=r,
            alpha=alpha,
        )

        # --- load all experts ---
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task key in lora_paths: {task}")
            eid = LABEL2ID[task]

            load_lora_into_expert(self.model, adir, eid)

            # ✅ sanity checks right after load
            self.quick_check_one("mlp.gate_proj", eid=eid)
            self.quick_check_one("self_attn.q_proj", eid=eid)

        # --- external router (classifier) ---
        # cls_dir might not contain tokenizer files; allow override
        tok_src = cls_tokenizer if cls_tokenizer is not None else cls_dir
        try:
            self.router_tokenizer = AutoTokenizer.from_pretrained(tok_src, local_files_only=True)
        except Exception as e:
            raise RuntimeError(
                f"[Router tokenizer load failed]\n"
                f"tok_src={tok_src}\n"
                f"cls_dir={cls_dir}\n"
                f"Fix: pass cls_tokenizer='prajjwal1/bert-tiny' (if cached) "
                f"or point to a local tokenizer dir that contains tokenizer.json/tokenizer_config.json.\n"
                f"Original error: {repr(e)}"
            )

        self.router = AutoModelForSequenceClassification.from_pretrained(cls_dir, local_files_only=True)
        self.router.eval()

        self.route_counter = Counter()

    # -------------------------
    # Debug helper
    # -------------------------
    def quick_check_one(self, module_name_substr: str, eid: int):
        found = False
        for n, m in self.model.named_modules():
            if (module_name_substr in n) and isinstance(m, HardRoutedLoRALinear):
                found = True
                A = m.A[eid].detach().float().cpu()
                B = m.B[eid].detach().float().cpu()
                '''
                print(
                    "[CHECK]", n, "eid", eid,
                    "A std", float(A.std()), "B std", float(B.std()),
                    "A max", float(A.abs().max()), "B max", float(B.abs().max())
                )
                '''
                break
        if not found:
            print("[CHECK] not found:", module_name_substr, "eid", eid)

    # -------------------------
    # Routing
    # -------------------------
    @torch.no_grad()
    def route_external(self, prompt: str) -> int:
        rt = self.router_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        rt = {k: v.to(next(self.router.parameters()).device) for k, v in rt.items()}
        eid = int(self.router(**rt).logits.argmax(dim=-1).item())
        self.route_counter[eid] += 1
        return eid

    # -------------------------
    # Generate
    # -------------------------
    @torch.no_grad()
    def generate(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: Optional[int] = None,
        gen_kwargs: Optional[Dict[str, Any]] = None,
        on_route: Optional[Any] = None,
    ) -> List[str]:

        if isinstance(prompts, str):
            prompts = [prompts]
        if gen_kwargs is None:
            gen_kwargs = {}

        outs: List[str] = []

        # 你可以調：只 debug 前 N 筆，避免刷屏
        debug_first_n = int(os.environ.get("MOE_DEBUG_FIRST_N", "5"))

        def _find_param(sub: str):
            """return (name, abs_mean) or (None, None)"""
            for n, p in self.model.named_parameters():
                if sub in n:
                    try:
                        return n, float(p.detach().abs().mean())
                    except Exception:
                        return n, None
            return None, None

        def _count_expert_set(expected_eid: int) -> int:
            """count how many HardRoutedLoRALinear modules have .eid == expected_eid"""
            cnt = 0
            for m in self.model.modules():
                if isinstance(m, HardRoutedLoRALinear):
                    cur = getattr(m, "eid", None)
                    if cur == expected_eid:
                        cnt += 1
            return cnt

        for idx, p in enumerate(prompts):
            # ---------- routing ----------
            eid = self.route_external(p)
            '''
            if idx < debug_first_n:
                print(f"\n[DEBUG] sample#{idx} routed eid = {eid}")
            '''
            if on_route is not None:
                on_route(prompt=p, eid=eid)

            # ---------- set expert ----------
            set_model_expert(self.model, eid)

            '''
            # after set_model_expert(self.model, eid)
            if idx == 0:  # 只印第一筆
                for n, m in self.model.named_modules():
                    if isinstance(m, HardRoutedLoRALinear):
                        print("[DEBUG] one HardRoutedLoRALinear module =", n)
                        print("[DEBUG] module __dict__ keys =", sorted(list(m.__dict__.keys())))
                        # 嘗試印幾個常見名字
                        for key in ["eid", "expert_id", "expert_idx", "active_expert", "cur_eid", "current_expert"]:
                            if hasattr(m, key):
                                print(f"[DEBUG] {key} =", getattr(m, key))
                        break
            '''

            # ---------- verify expert set (optional but useful) ----------
            if idx < debug_first_n:
                total = 0
                for m in self.model.modules():
                    if isinstance(m, HardRoutedLoRALinear):
                        total += 1
                hit = _count_expert_set(eid)
                #print(f"[DEBUG] HardRoutedLoRALinear eid set ok: {hit}/{total}")

            # ---------- print routed expert fingerprint (A.eid/B.eid) ----------
            if idx < debug_first_n:
                # 這裡用你最關心的 module 當指紋：layer0 q_proj
                nA0, vA0 = _find_param("model.layers.0.self_attn.q_proj.A.0")
                nAe, vAe = _find_param(f"model.layers.0.self_attn.q_proj.A.{eid}")
                nB0, vB0 = _find_param("model.layers.0.self_attn.q_proj.B.0")
                nBe, vBe = _find_param(f"model.layers.0.self_attn.q_proj.B.{eid}")

                #print("[DEBUG] A.0 vs A.eid:", (nA0, vA0), (nAe, vAe))
                #print("[DEBUG] B.0 vs B.eid:", (nB0, vB0), (nBe, vBe))

                '''
                if nAe is None or nBe is None:
                    print("[DEBUG] WARNING: cannot find A/B params for routed eid (maybe num_experts mismatch?)")
                '''
            # ---------- tokenize ----------
            inp = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.model.device)

            # ---------- HF kwargs ----------
            args = dict(gen_kwargs)

            # 只有 wrapper 沒設 max_new_tokens 才補
            if "max_new_tokens" not in args and max_new_tokens is not None:
                args["max_new_tokens"] = int(max_new_tokens)

            # defaults
            args.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            args.setdefault("pad_token_id", args["eos_token_id"])

            '''
            if idx < debug_first_n:
                print("[DEBUG] HF args keys =", sorted(list(args.keys())))
                print("[DEBUG] HF max_new_tokens =", args.get("max_new_tokens"))
            '''
            # -------------------------
            # Deterministic + short output for MCQ (A/B/C/D)
            # -------------------------

            # 1) 強制 deterministic（避免 sampling / beam 的不穩定）
            args["do_sample"] = False
            args["temperature"] = 0.0
            args["top_p"] = 1.0
            args["num_beams"] = 1

            # ---------- generate ----------
            out = self.model.generate(**inp, **args)
            input_len = inp["input_ids"].shape[1]
            gen_ids = out[0][input_len:]

            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            outs.append(text)

        return outs


# =========================
# LoRA loader (IMPORTANT: MLP + ATTN)
# =========================
@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int):
    """
    Load PEFT LoRA weights into expert slot expert_id.

    Accepts keys like:
      base_model.model.model.layers.{i}.mlp.{proj}.lora_A/B.weight
      base_model.model.model.layers.{i}.self_attn.{proj}.lora_A/B.weight

    After normalize:
      model.layers.{i}.mlp.{proj}.lora_A/B.weight
      model.layers.{i}.self_attn.{proj}.lora_A/B.weight
    """
    sd = safe_load(f"{adapter_dir}/adapter_model.safetensors")
    sd = {_normalize_key(k): v for k, v in sd.items()}

    def _copy(mod: "HardRoutedLoRALinear", kA: str, kB: str):
        if kA not in sd or kB not in sd:
            raise KeyError(f"Missing LoRA keys: {kA} / {kB} in {adapter_dir}")
        mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device))
        mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device))

    for li, layer in enumerate(model.model.layers):
        # ---- MLP ----
        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
                kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
                _copy(mod, kA, kB)

        # ---- ATTN ----
        attn = layer.self_attn
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]:
            mod = getattr(attn, proj)
            if isinstance(mod, HardRoutedLoRALinear):
                kA = f"model.layers.{li}.self_attn.{proj}.lora_A.weight"
                kB = f"model.layers.{li}.self_attn.{proj}.lora_B.weight"
                _copy(mod, kA, kB)
