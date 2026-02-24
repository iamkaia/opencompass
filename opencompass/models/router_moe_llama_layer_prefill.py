# router_moe_llama_layer_prefill_only_fastbatch.py
# Prefill-only layer-router + hard-routed LoRA experts on LLaMA MLP (Path A)
#
# ✅ Only modify MODEL (no dataset/template changes)
# ✅ Prefill-only routing: route when T>1, reuse cached eid when T==1
# ✅ FAST: no per-forward device moves / no fp32 casting of A/B/x
# ✅ Batch generation: tokenize ALL prompts together, call HF generate ONCE
# ✅ Reduce CPU/I/O stalls: buffer routing logs and write once per generate()
# ✅ Keep OpenCompass behavior: we do NOT override/cap max_new_tokens
#    (we only map OpenCompass max_out_len -> max_new_tokens, which is required)
#
# Put this file under: opencompass/models/
# Then in config: type='router_moe_llama_layer_prefill_only_fastbatch.RouterMoELlama'

import os
import json
import time
from collections import Counter
from typing import Dict, Optional, List

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from safetensors.torch import load_file as safe_load
from opencompass.models import HuggingFacewithChatTemplate

def _norm_task(t):
    if t is None:
        return None
    t = str(t).strip()
    if t == "squad2":
        t = "squad2.0"
    return t


def _get_routing_log_path(output_json_filepath, abbr, gt_task=None):
    """
    output_json_filepath: outputs/.../predictions/<abbr>/<dataset>.json
    gt_task: e.g. 'sst2' (recommended)
    """
    if output_json_filepath:
        pred_dir = os.path.dirname(output_json_filepath)

        # dataset stem 優先用 gt_task，其次用 output_json_filepath 的檔名
        if gt_task:
            ds = str(gt_task).strip()
            if ds.endswith(".json"):
                ds = ds[:-5]
        else:
            ds = os.path.splitext(os.path.basename(output_json_filepath))[0]  # sst2

        # 避免奇怪字元
        ds = ds.replace("/", "_")
        return os.path.join(pred_dir, f"routing_log__{ds}.jsonl")

    safe = abbr.replace("/", "_")
    ds = str(gt_task).strip() if gt_task else "unknown"
    ds = ds.replace("/", "_")
    return f"routing_logs/routing_{safe}__{ds}.jsonl"


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
# Utils
# =========================
def read_adapter_config(adapter_dir: str) -> dict:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_lora_key(k: str) -> str:
    """
    Normalize PEFT keys to:
      model.layers.{i}.mlp.{proj}.lora_A.weight
    Handles prefixes like base_model.model.model.
    """
    if "model.layers." in k:
        return k[k.index("model.layers."):]
    if "layers." in k:
        return "model." + k[k.index("layers."):]
    return k


# =========================
# Routed LoRA Linear (FAST)
# =========================
class RoutedLoRALinearBase(nn.Module):
    """
    Wrap an existing base Linear + LoRA experts.
    Expert id read from parent_mlp._cached_eid (LongTensor [B]).

    Speed rules:
    - DO NOT move A/B between devices in forward
    - DO NOT cast x/A/B to fp32 in forward
    - Only move tiny scale_per_expert if needed (small)
    """
    def __init__(self, base_linear: nn.Linear, parent_mlp: nn.Module, num_experts=5, r=8, alpha=32):
        super().__init__()
        self.base = base_linear
        self.parent_mlp = parent_mlp

        self.num_experts = int(num_experts)
        self.r = int(r)

        default_scale = float(alpha) / float(r)
        self.scale_per_expert = nn.Parameter(
            torch.full((self.num_experts,), default_scale, dtype=torch.float32),
            requires_grad=False,
        )

        self.A = nn.Parameter(torch.empty(self.num_experts, self.r, base_linear.in_features))
        self.B = nn.Parameter(torch.empty(self.num_experts, base_linear.out_features, self.r))
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(self.A[e], a=5**0.5)
            nn.init.zeros_(self.B[e])

    def set_expert_scale(self, expert_id: int, alpha: int, r: int):
        with torch.no_grad():
            s = float(alpha) / float(r)
            # clamp scale for fp16 stability (tunable)
            s = min(s, 4.0)     # 先用 4.0 很保守
            self.scale_per_expert.data[expert_id] = s

    '''
    def set_expert_scale(self, expert_id: int, alpha: int, r: int):
        with torch.no_grad():
            self.scale_per_expert.data[expert_id] = float(alpha) / float(r)
    '''

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        FAST + stable forward:
        - No per-forward device moves for A/B (must be materialized once beforehand)
        - No fp32 casting for x/A/B
        - Scale is tiny -> ok to move as fp32 then cast to x.dtype
        - Optional stability: clamp scale and check NaN/Inf (cheap when debug disabled)

        Expected attributes on self:
        - self.base : nn.Linear
        - self.parent_mlp : holds _cached_eid (LongTensor [B]) set by prefill-only router hook
        - self.A, self.B : [E, r, in] and [E, out, r]
        - self.scale_per_expert : [E] fp32
        - (optional) self.debug_nan : bool
        - (optional) self.debug_every : int
        - (optional) self._dbg_step : int
        - (optional) self.scale_clip : float  (e.g. 8.0)
        """
        # ---------- defaults for debug/stability knobs ----------
        if not hasattr(self, "debug_nan"):
            self.debug_nan = False
        if not hasattr(self, "debug_every"):
            self.debug_every = 200
        if not hasattr(self, "_dbg_step"):
            self._dbg_step = 0
        if not hasattr(self, "scale_clip"):
            self.scale_clip = 8.0  # <= change here if you want 4.0

        def _maybe_check(t: torch.Tensor, tag: str, eid_: torch.Tensor, scale_: torch.Tensor):
            if not self.debug_nan:
                return
            self._dbg_step += 1
            if (self._dbg_step % int(self.debug_every)) != 0:
                return
            if torch.isfinite(t).all():
                return
            raise RuntimeError(
                f"[NaN/Inf] {tag} | "
                f"x={x.dtype}/{x.device} "
                f"t={t.dtype}/{t.device} "
                f"A={self.A.dtype}/{self.A.device} "
                f"B={self.B.dtype}/{self.B.device} "
                f"scale={scale_.dtype}/{scale_.device} "
                f"eid_min={int(eid_.min())} eid_max={int(eid_.max())}"
            )

        y = self.base(x)

        # --------- expert ids ----------
        eid = getattr(self.parent_mlp, "_cached_eid", None)
        if eid is None:
            eid = torch.zeros((x.shape[0],), dtype=torch.long, device=x.device)
        else:
            eid = eid.to(device=x.device)
        # safe clamp
        if eid.numel() > 0:
            eid = torch.clamp(eid, 0, self.scale_per_expert.numel() - 1)

        # --------- per-expert scale ----------
        scale = self.scale_per_expert.to(device=x.device, dtype=torch.float32).index_select(0, eid)
        # clamp scale for fp16/bf16 stability (no effect if already small)
        if self.scale_clip is not None and float(self.scale_clip) > 0:
            scale = torch.clamp(scale, -float(self.scale_clip), float(self.scale_clip))
        scale = scale.to(dtype=x.dtype)

        A = self.A
        B = self.B

        # --------- 2D: [B, in] ----------
        if x.dim() == 2:
            A_sel = A.index_select(0, eid)                       # [B, r, in]
            B_sel = B.index_select(0, eid)                       # [B, out, r]
            z = torch.bmm(A_sel, x.unsqueeze(-1)).squeeze(-1)     # [B, r]
            d = torch.bmm(B_sel, z.unsqueeze(-1)).squeeze(-1)     # [B, out]
            out = y + d * scale.unsqueeze(-1)

            _maybe_check(out, "out_2d", eid, scale)
            return out

        # --------- 3D: [B, T, in] ----------
        Bsz, T, _ = x.shape
        x2 = x.reshape(Bsz * T, -1)                              # [B*T, in]
        eid2 = eid.repeat_interleave(T)                          # [B*T]
        A2 = A.index_select(0, eid2)                             # [B*T, r, in]
        B2 = B.index_select(0, eid2)                             # [B*T, out, r]
        z = torch.bmm(A2, x2.unsqueeze(-1)).squeeze(-1)          # [B*T, r]
        d = torch.bmm(B2, z.unsqueeze(-1)).squeeze(-1)           # [B*T, out]
        d = d.reshape(Bsz, T, -1)
        out = y + d * scale.view(Bsz, 1, 1)

        _maybe_check(out, "out_3d", eid, scale)
        return out



# =========================
# Patch: do NOT replace MLP; only wrap projections + add router
# =========================
def patch_llama_mlp_no_replace(model, num_experts=5, r=8, alpha=32, layer_start=0):
    hidden = int(model.config.hidden_size)

    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue

        mlp = layer.mlp

        if not hasattr(mlp, "router"):
            mlp.router = nn.Linear(hidden, num_experts, bias=False)

        mlp.gate_proj = RoutedLoRALinearBase(mlp.gate_proj, parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha)
        mlp.up_proj   = RoutedLoRALinearBase(mlp.up_proj,   parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha)
        mlp.down_proj = RoutedLoRALinearBase(mlp.down_proj, parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha)

    return model


# =========================
# Prefill-only routing hook
# =========================
def register_mlp_router_hooks_prefill_only(model, layer_start=0):
    """
    Prefill-only routing:
    - When T > 1 (prefill), compute eid per layer and cache in mlp._cached_eid
    - When decoding with T == 1, reuse cached eid (no router compute)
    """
    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue

        mlp = layer.mlp
        router = mlp.router

        hook = getattr(layer, "_hf_hook", None)
        if hook is not None and hasattr(hook, "execution_device"):
            try:
                router.to(device=hook.execution_device, dtype=torch.float32)
            except Exception:
                pass

        def make_hook(this_mlp, this_router):
            def hook_fn(module, inputs):
                x = inputs[0]  # [B, T, H]
                _, T, _ = x.shape

                # decoding step: reuse
                if T == 1 and getattr(this_mlp, "_cached_eid", None) is not None:
                    return

                pooled = x.mean(dim=1)  # [B, H]
                rdev = this_router.weight.device
                pooled = pooled.to(device=rdev, dtype=torch.float32)

                logits = this_router(pooled)  # [B, E]
                eid = logits.argmax(dim=-1).long()
                eid = torch.clamp(eid, 0, logits.size(-1) - 1)
                this_mlp._cached_eid = eid.detach()
            return hook_fn

        mlp.register_forward_pre_hook(make_hook(mlp, router))


# =========================
# One-time materialize to exec devices (critical)
# =========================
def materialize_wrappers_to_exec_device(model, layer_start=0):
    """
    Move wrapper params ONCE to each layer's accelerate execution_device.
    This avoids silent CPU usage and avoids any per-forward movement.
    """
    # pick target dtype from model parameters
    try:
        any_param = next(model.parameters())
        target_dtype = any_param.dtype
    except Exception:
        target_dtype = torch.float16

    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue

        hook = getattr(layer, "_hf_hook", None)
        if hook is not None and hasattr(hook, "execution_device"):
            dev = hook.execution_device
        else:
            dev = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        mlp = layer.mlp

        if hasattr(mlp, "router"):
            mlp.router.to(device=dev, dtype=torch.float32)

        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj, None)
            if isinstance(mod, RoutedLoRALinearBase):
                mod.A.data = mod.A.data.to(device=dev, dtype=target_dtype)
                mod.B.data = mod.B.data.to(device=dev, dtype=target_dtype)
                mod.scale_per_expert.data = mod.scale_per_expert.data.to(device=dev, dtype=torch.float32)

    print("[perf] materialized router/wrappers to execution devices", flush=True)


# =========================
# Load LoRA weights into expert slots
# =========================
@torch.no_grad()
def load_lora_into_expert(model, adapter_dir: str, expert_id: int, layer_start=0):
    wpath = os.path.join(adapter_dir, "adapter_model.safetensors")
    if not os.path.exists(wpath):
        raise FileNotFoundError(f"LoRA weights not found: {wpath}")

    sd_raw = safe_load(wpath)
    sd = {_normalize_lora_key(k): v for k, v in sd_raw.items()}
    keys = set(sd.keys())

    missing = []
    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue

        mlp = layer.mlp
        for proj in ["gate_proj", "up_proj", "down_proj"]:
            mod = getattr(mlp, proj, None)
            if not isinstance(mod, RoutedLoRALinearBase):
                continue

            kA = f"model.layers.{li}.mlp.{proj}.lora_A.weight"
            kB = f"model.layers.{li}.mlp.{proj}.lora_B.weight"
            if kA not in keys or kB not in keys:
                missing.append((li, proj))
                continue

            mod.A.data[expert_id].copy_(sd[kA].to(device=mod.A.device, dtype=mod.A.dtype))
            mod.B.data[expert_id].copy_(sd[kB].to(device=mod.B.device, dtype=mod.B.dtype))

    if missing:
        print(f"[warn] {adapter_dir}: missing LoRA blocks (show 10): {missing[:10]}", flush=True)


# =========================
# Router weights loader
# =========================
def _pick_router_weights_file(ckpt_dir: str) -> str:
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"[layer_router_ckpt] directory not found: {ckpt_dir}")

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

    raise FileNotFoundError(f"Cannot find router weights in {ckpt_dir} (model.safetensors missing)")


def _find_router_weight_key(sd_keys: set, layer_idx: int) -> Optional[str]:
    candidates = [
        f"routers.{layer_idx}.weight",
        f"model.routers.{layer_idx}.weight",
        f"module.routers.{layer_idx}.weight",
    ]
    for c in candidates:
        if c in sd_keys:
            return c
    suffix = f"routers.{layer_idx}.weight"
    for k in sd_keys:
        if k.endswith(suffix):
            return k
    return None


@torch.no_grad()
def load_layer_router_weights_into_mlp(model, layer_router_ckpt: str):
    wfile = _pick_router_weights_file(layer_router_ckpt)
    sd = safe_load(wfile)
    sd_keys = set(sd.keys())

    loaded = 0
    for li, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if not hasattr(mlp, "router"):
            continue

        k = _find_router_weight_key(sd_keys, li)
        if k is None:
            continue

        mlp.router.weight.copy_(sd[k].to(device=mlp.router.weight.device, dtype=mlp.router.weight.dtype))
        loaded += 1

    print(f"[router] weights file: {wfile}", flush=True)
    print(f"[router] loaded router layers: {loaded}/{len(model.model.layers)}", flush=True)


# =========================
# OpenCompass Model
# =========================
class RouterMoELlama(HuggingFacewithChatTemplate):
    """
    Prefill-only layer-router model (OpenCompass compatible), optimized for speed.
    - Batched generate: one call for many prompts
    - Buffered routing logs (optional)
    """
    is_api = False

    def __init__(
        self,
        path,
        layer_router_ckpt,
        lora_paths: Dict[str, str],
        abbr="router_moe_layer_prefill_only_fastbatch",
        dtype="float16",
        r=8,
        alpha=32,
        log_every_steps=0,
        enable_routing_log=False,   # default OFF for speed
        **kwargs,
    ):
        super().__init__(path=path, **kwargs)

        # TF32 speeds up matmuls on 4090 (safe for inference)
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.abbr = abbr
        self.step = 0
        self.route_counter = Counter()
        self.log_every_steps = int(log_every_steps)
        self.enable_routing_log = bool(enable_routing_log)

        # output dir
        out_dir = os.environ.get("OC_OUTPUT_DIR", os.getcwd())
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.route_log_path = os.path.join(out_dir, f"layer_route_log_{abbr}_{ts}_pid{pid}.jsonl")
        self.route_count_path = os.path.join(out_dir, f"layer_route_counts_{abbr}_{ts}_pid{pid}.json")

        # meta (layer_start)
        meta_path = os.path.join(layer_router_ckpt, "router_meta.json")
        layer_start = 0
        if os.path.exists(meta_path):
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            layer_start = int(meta.get("layer_start", 0))
        self.layer_start = layer_start
        print(f"[router] layer_start={self.layer_start}", flush=True)

        # base model
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch_dtype,
            device_map="auto",
        )
        self.model.eval()

        # patch wrappers (no replace)
        self.model = patch_llama_mlp_no_replace(
            self.model,
            num_experts=len(ID2LABEL),
            r=int(r),
            alpha=int(alpha),
            layer_start=self.layer_start,
        )

        # register prefill-only routing
        register_mlp_router_hooks_prefill_only(self.model, layer_start=self.layer_start)

        # load LoRA experts + set per-expert scales
        for task, adir in lora_paths.items():
            if task not in LABEL2ID:
                raise ValueError(f"Unknown task '{task}' in lora_paths. Must be one of {list(LABEL2ID.keys())}")

            eid = LABEL2ID[task]

            load_lora_into_expert(self.model, adir, eid, layer_start=self.layer_start)

            cfg = read_adapter_config(adir)
            r_cfg = int(cfg.get("r", r))
            a_cfg = int(cfg.get("lora_alpha", cfg.get("alpha", alpha)))

            for layer in self.model.model.layers:
                mlp = layer.mlp
                for proj in ["gate_proj", "up_proj", "down_proj"]:
                    mod = getattr(mlp, proj, None)
                    if hasattr(mod, "set_expert_scale"):
                        mod.set_expert_scale(eid, alpha=a_cfg, r=r_cfg)

            print(f"[lora] task={task} eid={eid} r={r_cfg} alpha={a_cfg} scale={a_cfg/r_cfg:.4f}", flush=True)

        # load router weights
        load_layer_router_weights_into_mlp(self.model, layer_router_ckpt)

        # materialize router + wrappers to exec devices ONCE
        materialize_wrappers_to_exec_device(self.model, layer_start=self.layer_start)

        # tokenizer safety
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

    def _to_prompt_str(self, x):
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
                content = m.get("content", None) or m.get("prompt", "")
                msgs.append({"role": role, "content": content})
            return self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return str(x)
    
    @torch.inference_mode()
    def generate(self, prompts, **gen_kwargs):

        # ---------- metadata from inferencer ----------
        gt_task = gen_kwargs.pop("gt_task", None)
        output_json_filepath = gen_kwargs.pop("output_json_filepath", None)

        if not isinstance(prompts, list):
            prompts = [prompts]

        ps: List[str] = [self._to_prompt_str(p) for p in prompts]

        # ---------- batched tokenize ----------
        enc = self.tokenizer(
            ps,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=getattr(self, "max_seq_len", 2048),
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        # ---------- OpenCompass compatibility ----------
        max_out_len = gen_kwargs.pop("max_out_len", None)
        if max_out_len is not None and "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = int(max_out_len)

        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        gen_kwargs["do_sample"] = False
        gen_kwargs["num_beams"] = 1
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)
        gen_kwargs.pop("top_k", None)

        # ---------- generate ----------
        out = self.model.generate(**enc, **gen_kwargs)
        texts = self.tokenizer.batch_decode(out, skip_special_tokens=True)

        # =====================================================
        #                    ROUTING LOG
        # =====================================================
        try:
            from collections import Counter

            gt = _norm_task(gt_task)
            gt_eid = LABEL2ID.get(gt, None) if gt is not None else None

            B = len(ps)

            for b in range(B):

                plan = []

                for li, layer in enumerate(self.model.model.layers):
                    if li < self.layer_start:
                        continue

                    mlp = layer.mlp
                    eid = getattr(mlp, "_cached_eid", None)

                    if eid is None:
                        continue

                    plan.append(int(eid[b].item()))

                valid = [x for x in plan if x >= 0]

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
                    "kind": "layer_router",
                    "gt_task": gt,
                    "major_task": maj_task,
                    "major_eid": maj_eid,
                    "route_ok_major": ok_major,
                    "match_rate": match_rate,
                    "plan": plan,
                }

                #log_path = _get_routing_log_path(output_json_filepath, self.abbr)
                log_path = _get_routing_log_path(output_json_filepath, self.abbr, gt_task=gt)
                os.makedirs(os.path.dirname(log_path), exist_ok=True)
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        except Exception:
            pass

        # =====================================================

        return texts
