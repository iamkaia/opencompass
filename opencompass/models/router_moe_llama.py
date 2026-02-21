import torch
import torch.nn as nn
import json, os, time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from safetensors.torch import load_file as safe_load

from collections import Counter
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
# MoE-LoRA Linear (hard routing)
# =========================
class HardRoutedLoRALinear(nn.Module):
    """
    A Linear layer + (selected expert) LoRA delta.
    For stability with device_map="auto", we align A/B to x's device/dtype in forward.
    """
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
            [nn.Parameter(torch.empty(self.r, base_linear.in_features)) for _ in range(self.num_experts)]
        )
        self.B = nn.ParameterList(
            [nn.Parameter(torch.empty(base_linear.out_features, self.r)) for _ in range(self.num_experts)]
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

        # align dtype/device
        A = self.A[e].to(device=x.device, dtype=x.dtype)
        B = self.B[e].to(device=x.device, dtype=x.dtype)

        z = x @ A.t()
        d = z @ B.t()
        return y + self.scale * d


# =========================
# Patch & load helpers
# =========================
def patch_llama_mlp(model, num_experts=5, r=8, alpha=16):
    """Replace LLaMA MLP projections with HardRoutedLoRALinear: gate_proj, up_proj, down_proj"""
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
    Load PEFT LoRA weights (adapter_model.safetensors) into expert slot expert_id.
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
# OpenCompass Model
# =========================
class RouterMoELlama(HuggingFacewithChatTemplate):
    """OpenCompass-compatible MoE-LoRA model with routing logs."""

    is_api = False

    def __init__(
        self,
        path,
        cls_dir,
        lora_paths,
        max_out_len=1024,
        batch_size=1,
        run_cfg=None,
        dtype="float16",
        r=8,
        alpha=16,
        abbr="router_moe_lora",
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_out_len=max_out_len,
            batch_size=batch_size,
            run_cfg=run_cfg,
            **kwargs,
        )

        # keep abbr for filenames
        self.abbr = abbr

        # ----- router (bert-tiny classifier) -----
        self.router_tokenizer = AutoTokenizer.from_pretrained(cls_dir, local_files_only=True)
        self.router = AutoModelForSequenceClassification.from_pretrained(cls_dir, local_files_only=True)
        self.router.eval()

        self.route_counter = Counter()

        # ----- choose a robust output dir -----
        # If OpenCompass sets OC_OUTPUT_DIR, use it; otherwise use current working directory
        out_dir = os.environ.get("OC_OUTPUT_DIR", None)
        if not out_dir:
            out_dir = os.getcwd()
        os.makedirs(out_dir, exist_ok=True)

        pid = os.getpid()
        ts = time.strftime("%Y%m%d_%H%M%S")

        self.save_route_counts_path = os.path.join(out_dir, f"routing_counts_{self.abbr}_{ts}_pid{pid}.json")
        self.save_route_log_path = os.path.join(out_dir, f"routing_log_{self.abbr}_{ts}_pid{pid}.jsonl")

        # ----- llama -----
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # Patch MLP to MoE-LoRA
        self.model = patch_llama_mlp(self.model, num_experts=len(ID2LABEL), r=r, alpha=alpha)

        # Load all experts
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task key in lora_paths: {task} (must be one of {list(LABEL2ID.keys())})")
            load_lora_into_expert(self.model, adir, LABEL2ID[task])

        # pad/eos safety
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

    def _to_prompt_str(self, x):
        if isinstance(x, str):
            return x

        if isinstance(x, dict):
            for k in ["prompt", "text", "input", "inputs"]:
                if k in x and isinstance(x[k], str):
                    return x[k]

            if "messages" in x:
                msgs = x["messages"]
                if hasattr(self.tokenizer, "apply_chat_template"):
                    return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                return "\n".join([f"{m.get('role','user')}: {m.get('content','')}" for m in msgs])

        if isinstance(x, (list, tuple)) and x and isinstance(x[0], dict):
            if hasattr(self.tokenizer, "apply_chat_template"):
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

            parts = []
            for m in x:
                role = m.get("role", "user")
                txt = m.get("content", None)
                if txt is None:
                    txt = m.get("prompt", "")
                parts.append(f"{role}: {txt}")
            return "\n".join(parts)

        raise TypeError(f"Unsupported input type for prompt: {type(x)}; value={repr(x)[:300]}")

    @torch.no_grad()
    def generate(self, prompts, **gen_kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]

        max_out_len = gen_kwargs.pop("max_out_len", None)

        outputs = []
        for item in prompts:
            p = self._to_prompt_str(item)

            # ---- route ----
            rt = self.router_tokenizer(p, return_tensors="pt", truncation=True, max_length=512)
            rt = {k: v.to(next(self.router.parameters()).device) for k, v in rt.items()}
            eid = int(self.router(**rt).logits.argmax(dim=-1).item())

            self.route_counter[eid] += 1

            # log per-sample routing (append)
            try:
                rec = {
                    "eid": eid,
                    "routed_task": ID2LABEL.get(eid, str(eid)),
                    "prompt_len_chars": len(p),
                }
                with open(self.save_route_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            except Exception:
                # logging must never break generation
                pass

            # ---- set expert for all routed layers ----
            set_model_expert(self.model, eid)

            # ---- tokenize ----
            max_len = getattr(self, "max_seq_len", 2048)
            inp = self.tokenizer(
                p,
                return_tensors="pt",
                truncation=True,
                max_length=max_len,
            ).to(self.model.device)

            gen_args = dict(gen_kwargs)
            if max_out_len is not None:
                gen_args["max_new_tokens"] = int(max_out_len)

            gen_args.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            gen_args.setdefault("eos_token_id", self.tokenizer.eos_token_id)

            out = self.model.generate(**inp, **gen_args)
            outputs.append(self.tokenizer.decode(out[0], skip_special_tokens=True))

        # write counts (overwrite OK; filename is unique)
        try:
            with open(self.save_route_counts_path, "w", encoding="utf-8") as f:
                json.dump(
                    {ID2LABEL.get(k, str(k)): v for k, v in self.route_counter.items()},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

        return outputs
