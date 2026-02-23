# opencompass/models/dmole_router_moe_llama.py
# D-MoLE: allocation-gated LoRA experts + (optional) layer-router (prefill-only) on LLaMA MLP
# Fixes:
# - OpenCompass entries may be dict/list, not str: use parse_template() to build prompt string
# - Tokenizer input type safety
# - Works with sparse LoRA safetensors (only allocated layers saved)
# - OpenCompass max_out_len compatibility

import os
import json
import time
import torch
import torch.nn as nn
from collections import Counter
from safetensors.torch import load_file as safe_load
from transformers import AutoModelForCausalLM
from opencompass.models import HuggingFacewithChatTemplate
import os, json, time
from collections import Counter

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _append_jsonl(path: str, obj: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def _norm_task(t):
    if t is None:
        return None
    t = str(t).strip()
    if t == "squad2":
        t = "squad2.0"
    return t

def _get_routing_log_path(output_json_filepath, abbr):
    if output_json_filepath:
        pred_dir = os.path.dirname(output_json_filepath)
        return os.path.join(pred_dir, "routing_log.jsonl")
    safe = abbr.replace("/", "_")
    return f"routing_logs/routing_{safe}.jsonl"



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
    """
    expert_id = -1  => disable LoRA (base only)
    expert_id >= 0  => apply expert's LoRA
    """
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

        self.expert_id = -1  # disabled by default

    def set_expert(self, eid: int):
        self.expert_id = int(eid)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)
        if self.expert_id < 0:
            return y
        A = self.A[self.expert_id].to(device=x.device, dtype=x.dtype)
        B = self.B[self.expert_id].to(device=x.device, dtype=x.dtype)
        return y + self.scale * ((x @ A.t()) @ B.t())


def patch_llama_mlp(model, num_experts: int, r: int = 8, alpha: int = 16):
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
# LoRA loader (robust + sparse)
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
def load_lora_into_expert(model, adapter_dir: str, expert_id: int, alloc_layers: set):
    """
    Load sparse LoRA (only allocated layers may exist in safetensors).
    Missing keys on non-allocated layers are fine.
    """
    wpath = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"LoRA weights not found: {wpath}")

    sd_raw = safe_load(wpath)
    sd = {_normalize_lora_key(k): v for k, v in sd_raw.items()}
    keys = set(sd.keys())

    for li, layer in enumerate(model.model.layers):
        if li not in alloc_layers:
            continue
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(layer.mlp, proj, None)
            if not isinstance(mod, HardRoutedLoRALinear):
                continue
            kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
            kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
            if (kA in keys) and (kB in keys):
                mod.A[expert_id].copy_(sd[kA].to(mod.A[expert_id].device))
                mod.B[expert_id].copy_(sd[kB].to(mod.B[expert_id].device))
            # If missing on allocated layer, we skip silently (keeps zeros). You can make this strict if desired.


# =========================
# Layer-router loader
# =========================
def _pick_router_weights_file(ckpt_dir: str) -> str:
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
    cand = os.path.join(ckpt_dir, "model.safetensors")
    if os.path.exists(cand):
        return cand
    raise FileNotFoundError(f"Cannot find router weights under: {ckpt_dir}")


def _find_router_weight_key(sd_keys: set, layer_idx: int):
    suffix = f"routers.{layer_idx}.weight"
    for k in sd_keys:
        if k.endswith(suffix):
            return k
    return None


# =========================
# D-MoLE Model Wrapper
# =========================
class DMoLERouterMoELlama(HuggingFacewithChatTemplate):
    is_api = False

    def __init__(
        self,
        path,
        allocations_json,
        lora_paths,                 # dict task->dir
        layer_router_ckpt=None,     # optional
        abbr="dmole_router_moe",
        dtype="float16",
        r=8,
        alpha=16,
        max_seq_len=2048,
        log_plans=False,
        **kwargs,
    ):
        super().__init__(path=path, **kwargs)
        self.abbr = abbr
        self.max_seq_len = int(max_seq_len)

        # log files (optional)
        self.log_plans = bool(log_plans)
        out_dir = os.environ.get("OC_OUTPUT_DIR", os.getcwd())
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.plan_log_path = os.path.join(out_dir, f"dmole_plan_{abbr}_{ts}_pid{pid}.jsonl")

        # allocations
        alloc = json.load(open(allocations_json, "r", encoding="utf-8"))
        self.alloc_layers = {t: set(v.get("layers", [])) for t, v in alloc.items()}

        # base model
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # patch MLP -> MoE-LoRA
        self.model = patch_llama_mlp(self.model, num_experts=len(ID2LABEL), r=int(r), alpha=int(alpha))

        # load LoRA experts
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task '{task}' in lora_paths. Must be one of {list(LABEL2ID.keys())}")
            load_lora_into_expert(self.model, adir, LABEL2ID[task], self.alloc_layers.get(task, set()))

        # router (optional)
        self.use_router = layer_router_ckpt is not None
        hidden = int(self.model.config.hidden_size)
        self.num_layers = len(self.model.model.layers)

        if self.use_router:
            self.layer_routers = nn.ModuleList(
                [nn.Linear(hidden, len(ID2LABEL), bias=False) for _ in range(self.num_layers)]
            )
            self.layer_routers.eval()

            router_device = next(self.model.parameters()).device
            self.layer_routers.to(router_device)

            wfile = _pick_router_weights_file(layer_router_ckpt)
            sd = safe_load(wfile)
            sd_keys = set(sd.keys())
            loaded = 0
            for i in range(self.num_layers):
                k = _find_router_weight_key(sd_keys, i)
                if k is None:
                    continue
                self.layer_routers[i].weight.data.copy_(sd[k].to(self.layer_routers[i].weight.dtype))
                loaded += 1

        # tokenizer safety
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

    # --------- prompt conversion (critical for OpenCompass) ---------
    def _entry_to_prompt_str(self, entry):
        """
        OpenCompass passes `entry` objects (dict/list) to generate_from_template().
        Best practice: use parse_template() to obtain the real prompt string.
        """
        # If already str, return directly
        if isinstance(entry, str):
            return entry

        # Prefer OpenCompass template parsing (this is the most correct)
        try:
            parsed = self.parse_template([entry], mode="gen")
            if isinstance(parsed, list) and parsed and isinstance(parsed[0], str):
                return parsed[0]
        except Exception:
            pass

        # Fallback: try common keys
        if isinstance(entry, dict):
            for k in ["prompt", "text", "input", "inputs", "sentence", "question"]:
                v = entry.get(k, None)
                if isinstance(v, str) and v.strip():
                    return v

            # messages style
            if "messages" in entry and hasattr(self.tokenizer, "apply_chat_template"):
                return self.tokenizer.apply_chat_template(
                    entry["messages"], tokenize=False, add_generation_prompt=True
                )

        # Fallback stringify
        return str(entry)

    # --------- prefill-only plan ---------
    def _build_plan_prefill(self, input_ids: torch.Tensor):
        plan = [-1] * self.num_layers
        if not self.use_router:
            return plan

        out = self.model(input_ids=input_ids, output_hidden_states=True, use_cache=True)
        hs = out.hidden_states  # len = num_layers+1, use hs[i] as input to layer i
        router_device = self.layer_routers[0].weight.device

        for i in range(self.num_layers):
            pooled = hs[i].mean(dim=1).to(router_device, dtype=torch.float32)
            logits = self.layer_routers[i](pooled)
            eid = int(logits.argmax(dim=-1)[0].item())

            task = ID2LABEL[eid]
            # D-MoLE allocation gate
            if i in self.alloc_layers.get(task, set()):
                plan[i] = eid
            else:
                plan[i] = -1
        return plan

    def _apply_plan(self, plan):
        for li, layer in enumerate(self.model.model.layers):
            set_layer_expert(layer, plan[li])

    
    # --------- generate (OpenCompass compatible) ---------
    @torch.no_grad()
    def generate(self, prompts, **gen_kwargs):

        # -------- metadata --------
        gt_task = gen_kwargs.pop("gt_task", None)
        output_json_filepath = gen_kwargs.pop("output_json_filepath", None)
        
        # ❗ 只 pop 一次（不能放在 for 裡）
        max_out_len = gen_kwargs.pop("max_out_len", None)
        if max_out_len is not None:
            gen_kwargs["max_new_tokens"] = int(max_out_len)
        
        if not isinstance(prompts, list):
            prompts = [prompts]
        
        outs = []
        
        for entry in prompts:
        
            prompt_str = self._entry_to_prompt_str(entry)
        
            enc = self.tokenizer(
                prompt_str,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.model.device)
            
            gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

            # -------- Prefill-only routing plan --------
            plan = self._build_plan_prefill(enc["input_ids"])
            self._apply_plan(plan)
        
            # ================= ROUTING LOG =================
        
            gt = _norm_task(gt_task)
            gt_eid = LABEL2ID.get(gt, None) if gt is not None else None
        
            plan_int = [int(x) for x in plan]
            valid = [x for x in plan_int if x >= 0]
        
            # LoRA coverage（非 -1 的比例）
            coverage = None
            if plan_int:
                coverage = sum(1 for x in plan_int if x != -1) / len(plan_int)
        
            # Majority vote（安全寫法）
            if valid:
                maj_eid = Counter(valid).most_common(1)[0][0]
                maj_task = ID2LABEL.get(maj_eid, None)
            else:
                maj_eid = None
                maj_task = None
        
            ok_major = (gt == maj_task) if (gt is not None and maj_task is not None) else None
        
            match_rate = None
            if gt_eid is not None and valid:
                match_rate = sum(1 for x in valid if x == gt_eid) / len(valid)
        
            rec = {
                "ts": time.time(),
                "kind": "dmole",
                "gt_task": gt,
                "major_task": maj_task,
                "major_eid": maj_eid,
                "route_ok_major": ok_major,
                "match_rate": match_rate,
                "coverage": coverage,
                "plan": plan_int,
            }
        
            log_path = _get_routing_log_path(output_json_filepath, self.abbr)
            _append_jsonl(log_path, rec)
        
            # ==============================================
        
            # (可選) 原本的 debug plan log
            if self.log_plans:
                try:
                    rec2 = {
                        "plan": plan_int,
                        "plan_tasks": [
                            ID2LABEL[i] if i >= 0 else "base"
                            for i in plan_int
                        ],
                    }
                    with open(self.plan_log_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec2, ensure_ascii=False) + "\n")
                except Exception:
                    pass

            # -------- 真正生成 --------
            out = self.model.generate(**enc, **gen_kwargs)
            text = self.tokenizer.decode(out[0], skip_special_tokens=True)
            outs.append(text)
        
        return outs
