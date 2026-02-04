# router_moe_llama_layer.py
# Layer-level internal router + Hard-routed LoRA experts on LLaMA MLP (gate/up/down)
# - Robustly loads LoRA from PEFT safetensors (handles base_model.model.model.* prefix)
# - Robustly loads layer-router weights from HF Trainer outputs (model.safetensors in ckpt or root)
# - Writes routing logs to OC_OUTPUT_DIR (or cwd) so OpenCompass won't swallow prints

import os
import json
import time
import re
import torch
import torch.nn as nn
from collections import Counter
from transformers import AutoTokenizer, AutoModelForCausalLM
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
# Hard-routed LoRA Linear
# =========================
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

        self.A = nn.ParameterList([
            nn.Parameter(torch.empty(self.r, base_linear.in_features))
            for _ in range(self.num_experts)
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.empty(base_linear.out_features, self.r))
            for _ in range(self.num_experts)
        ])

        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A[e], a=5 ** 0.5)
            nn.init.zeros_(self.B[e])

        self.expert_id = 0

    def set_expert(self, eid: int):
        self.expert_id = int(eid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        A = self.A[self.expert_id].to(device=x.device, dtype=x.dtype)
        B = self.B[self.expert_id].to(device=x.device, dtype=x.dtype)
        return y + self.scale * ((x @ A.t()) @ B.t())


# =========================
# Patch helpers
# =========================
def patch_llama_mlp(model, num_experts: int, r: int = 8, alpha: int = 16):
    """Replace LLaMA MLP projections with HardRoutedLoRALinear: gate_proj, up_proj, down_proj"""
    for layer in model.model.layers:
        mlp = layer.mlp
        for name in ["gate_proj", "up_proj", "down_proj"]:
            base = getattr(mlp, name, None)
            if isinstance(base, nn.Linear):
                setattr(mlp, name, HardRoutedLoRALinear(base, num_experts, r=r, alpha=alpha))
    return model


def set_layer_expert(layer, eid: int):
    mlp = layer.mlp
    for name in ["gate_proj", "up_proj", "down_proj"]:
        mod = getattr(mlp, name, None)
        if isinstance(mod, HardRoutedLoRALinear):
            mod.set_expert(eid)


# =========================
# LoRA loader (robust)
# =========================
def _normalize_lora_key(k: str) -> str:
    """
    Turn keys like:
      base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight
    into:
      model.layers.0.mlp.gate_proj.lora_A.weight
    """
    if "model.layers." in k:
        return k[k.index("model.layers."):]
    if "layers." in k:
        return "model." + k[k.index("layers."):]
    return k


@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int):
    """
    Load PEFT LoRA weights into expert slot expert_id for MLP projections:
      gate_proj, up_proj, down_proj
    Expects normalized keys:
      model.layers.{i}.mlp.{proj}.lora_A.weight
      model.layers.{i}.mlp.{proj}.lora_B.weight
    """
    wpath = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"LoRA weights not found: {wpath}")

    sd_raw = safe_load(wpath)
    sd = {_normalize_lora_key(k): v for k, v in sd_raw.items()}
    keys = set(sd.keys())

    missing = []
    for li, layer in enumerate(model.model.layers):
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(layer.mlp, proj, None)
            if not isinstance(mod, HardRoutedLoRALinear):
                continue

            kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
            kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
            if kA not in keys or kB not in keys:
                missing.append((li, proj))
                continue

            mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device))
            mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device))

    if missing:
        print(f"[warn] {adapter_dir}: missing LoRA blocks (show 10): {missing[:10]}")


# =========================
# Layer-router loader (robust, safetensors)
# =========================
def _pick_router_weights_file(ckpt_dir: str) -> str:
    """
    Your Trainer output shows:
      ckpt_dir/model.safetensors
      ckpt_dir/checkpoint-25000/model.safetensors
    We prefer the latest checkpoint-*/model.safetensors, else root model.safetensors.
    """
    # 1) latest checkpoint-*/model.safetensors
    subs = [d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")]
    if subs:
        def step_of(x: str) -> int:
            try:
                return int(x.split("-")[-1])
            except Exception:
                return -1
        subs = sorted(subs, key=step_of)
        cand = os.path.join(ckpt_dir, subs[-1], "model.safetensors")
        if os.path.exists(cand):
            return cand

    # 2) root model.safetensors
    cand = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(cand):
        return cand

    raise FileNotFoundError(
        f"Cannot find layer-router weights in {ckpt_dir} "
        f"(expected checkpoint-*/model.safetensors or model.safetensors)"
    )


def _find_router_weight_key(sd_keys: set, layer_idx: int) -> str | None:
    """
    Your training code likely saved routers as:
      routers.{i}.weight
    But depending on wrappers it could be:
      model.routers.{i}.weight
      module.routers.{i}.weight
    We'll search robustly.
    """
    candidates = [
        f"routers.{layer_idx}.weight",
        f"model.routers.{layer_idx}.weight",
        f"module.routers.{layer_idx}.weight",
    ]
    for c in candidates:
        if c in sd_keys:
            return c

    # fallback: endswith
    suffix = f"routers.{layer_idx}.weight"
    for k in sd_keys:
        if k.endswith(suffix):
            return k
    return None


# =========================
# Router-MoE LLaMA (layer-level internal routing)
# =========================
class RouterMoELlama(HuggingFacewithChatTemplate):
    is_api = False

    def __init__(
        self,
        path,
        layer_router_ckpt,
        lora_paths,
        abbr="router_moe_layer",
        dtype="float16",
        r=8,
        alpha=16,
        log_every_steps=200,
        **kwargs,
    ):
        super().__init__(path=path, **kwargs)

        self.abbr = abbr
        self.step = 0
        self.route_counter = Counter()
        self.log_every_steps = int(log_every_steps)

        # ---------- output/log dir ----------
        out_dir = os.environ.get("OC_OUTPUT_DIR", os.getcwd())
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()

        self.route_log_path = os.path.join(out_dir, f"layer_route_log_{abbr}_{ts}_pid{pid}.jsonl")
        self.route_count_path = os.path.join(out_dir, f"layer_route_counts_{abbr}_{ts}_pid{pid}.json")

        # ---------- base model ----------
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # ---------- patch MLP for MoE-LoRA ----------
        self.model = patch_llama_mlp(self.model, num_experts=len(ID2LABEL), r=int(r), alpha=int(alpha))

        # ---------- load LoRA experts ----------
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task '{task}' in lora_paths. Must be one of {list(LABEL2ID.keys())}")
            load_lora_into_expert(self.model, adir, LABEL2ID[task])

        # ---------- load layer-router meta ----------
        meta_path = os.path.join(layer_router_ckpt, "router_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"router_meta.json not found in {layer_router_ckpt}")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.layer_start = int(meta.get("layer_start", 0))

        hidden = int(self.model.config.hidden_size)
        num_layers = len(self.model.model.layers)

        # ---------- init routers ----------
        self.layer_routers = nn.ModuleList([
            nn.Linear(hidden, len(ID2LABEL), bias=False)
            for _ in range(num_layers)
        ])
        self.layer_routers.eval()

        # ---- move layer routers to same device as model (GPU) ----
        router_device = next(self.model.parameters()).device
        self.layer_routers.to(router_device)

        # ---------- load router weights (safetensors) ----------
        router_wfile = _pick_router_weights_file(layer_router_ckpt)
        sd = safe_load(router_wfile)
        sd_keys = set(sd.keys())

        print(f"[router] weights file: {router_wfile}")
        print(f"[router] num_keys={len(sd_keys)}  sample_keys={list(sd_keys)[:5]}")

        loaded = 0
        for i in range(num_layers):
            k = _find_router_weight_key(sd_keys, i)
            if k is None:
                continue
            self.layer_routers[i].weight.data.copy_(sd[k].to(self.layer_routers[i].weight.dtype))
            loaded += 1
        print(f"[router] loaded router layers: {loaded}/{num_layers} (layer_start={self.layer_start})")

        # ---------- register hooks ----------
        for li, layer in enumerate(self.model.model.layers):
            def make_hook(idx: int):
                def hook_fn(module, inputs):
                    if idx < self.layer_start:
                        return
                    x = inputs[0]              # [B, T, H]
                    pooled = x.mean(dim=1)     # [B, H]

                    # Router is on CPU by default; move logits computation safely
                    # Compute in float32 for stability; weight is float32 by default too.
                    #logits = self.layer_routers[idx](pooled.float())

                    router_device = self.layer_routers[idx].weight.device
                    pooled = pooled.to(device=router_device, dtype=torch.float32)
                    logits = self.layer_routers[idx](pooled)

                    eid = int(logits.argmax(dim=-1)[0].item())

                    set_layer_expert(self.model.model.layers[idx], eid)
                    self.route_counter[eid] += 1
                    self.step += 1

                    # Append log (never swallowed)
                    try:
                        rec = {
                            "step": self.step,
                            "layer": idx,
                            "expert": eid,
                            "task": ID2LABEL.get(eid, str(eid)),
                        }
                        with open(self.route_log_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    except Exception:
                        pass

                    # Occasional print
                    if self.log_every_steps > 0 and (self.step % self.log_every_steps == 0):
                        print(f"[MoE] layer={idx:02d} expert={eid} ({ID2LABEL[eid]})")
                return hook_fn

            layer.register_forward_pre_hook(make_hook(li))

        # ---------- tokenizer safety ----------
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Ensure model config has pad/eos (avoid CUDA asserts in generation)
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

    def _to_prompt_str(self, x):
        # Minimal converter (same spirit as your previous code)
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            for k in ["prompt", "text", "input", "inputs"]:
                if k in x and isinstance(x[k], str):
                    return x[k]
            if "messages" in x and hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    x["messages"], tokenize=False, add_generation_prompt=True
                )
        if isinstance(x, (list, tuple)) and x and isinstance(x[0], dict) and hasattr(self.tokenizer, "apply_chat_template"):
            msgs = []
            for m in x:
                role = m.get("role", "user")
                if role in ["HUMAN", "USER"]:
                    role = "user"
                elif role in ["ASSISTANT", "BOT"]:
                    role = "assistant"
                elif role in ["SYSTEM"]:
                    role = "system"
                content = m.get("content", None)
                if content is None:
                    content = m.get("prompt", "")
                msgs.append({"role": role, "content": content})
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

        return str(x)

    @torch.no_grad()
    def generate(self, prompts, **gen_kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]

        outs = []
        for p in prompts:
            p = self._to_prompt_str(p)

            inp = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=getattr(self, "max_seq_len", 2048),
            ).to(self.model.device)

            #out = self.model.generate(**inp, **gen_kwargs)

            # ---- OpenCompass compatibility ----
            # OpenCompass passes max_out_len, but HF generate() does NOT accept it
            max_out_len = gen_kwargs.pop("max_out_len", None)
            if max_out_len is not None:
                gen_kwargs["max_new_tokens"] = int(max_out_len)

            # safety defaults
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

            out = self.model.generate(**inp, **gen_kwargs)

            outs.append(self.tokenizer.decode(out[0], skip_special_tokens=True))

        # Write routing counts at end of each generate() call
        try:
            with open(self.route_count_path, "w", encoding="utf-8") as f:
                json.dump(
                    {ID2LABEL.get(k, str(k)): v for k, v in self.route_counter.items()},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

        return outs
