# -*- coding: utf-8 -*-
"""
router_moe_llama_layer_prefill.py (NEW)

Prefill-only per-layer router + Routed LoRA experts on LLaMA MLP.

Major OOM fixes / optimizations:
(1) 3D prefill LoRA: avoid expanding A/B to [B*T, ...] (group-by-expert + scatter)
(2) Length bucketing: avoid padding whole mega-batch to the longest prompt
(3) On-demand expert cache: do NOT keep all experts A/B resident on GPU for every layer
    - keep full A/B on CPU (pinned) per layer
    - move only needed expert slices to that layer's execution_device
    - optional LRU cap per layer to limit VRAM

Notes:
- This design trades some PCIe traffic for lower VRAM.
- In many OpenCompass runs, prompts are homogeneous per task → cache quickly stabilizes.

"""

import os
import json
import time
from collections import Counter, OrderedDict
from typing import Dict, Optional, List, Tuple

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
# Utils
# =========================
def read_adapter_config(adapter_dir: str) -> dict:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_lora_key(k: str) -> str:
    """Normalize PEFT keys to 'model.layers.{i}.mlp.{proj}.lora_A.weight' style."""
    if "model.layers." in k:
        return k[k.index("model.layers.") :]
    if "layers." in k:
        return "model." + k[k.index("layers.") :]
    return k


# =========================
# Routed LoRA Linear (OOM-safe + expert cache)
# =========================
class RoutedLoRALinearBase(nn.Module):
    """
    Wrap base Linear with multiple LoRA experts.
    Expert ids are read from parent_mlp._cached_eid (LongTensor [B]).

    Memory optimization:
    - Store full A/B on CPU pinned memory by default
    - Maintain per-layer GPU cache of expert slices, capped by max_cached_experts (LRU)
    """

    def __init__(
        self,
        base_linear: nn.Linear,
        parent_mlp: nn.Module,
        num_experts: int = 5,
        r: int = 8,
        alpha: int = 32,
        store_on_cpu: bool = True,
        max_cached_experts: int = 2,   # <= num_experts
    ):
        super().__init__()
        self.base = base_linear
        self.parent_mlp = parent_mlp
        self.num_experts = int(num_experts)
        self.r = int(r)

        # default scale
        default_scale = float(alpha) / float(r)
        self.scale_per_expert = nn.Parameter(
            torch.full((self.num_experts,), default_scale, dtype=torch.float32),
            requires_grad=False,
        )

        # Full weights storage (CPU pinned by default)
        # Shape: A[E, r, in], B[E, out, r]
        A = torch.empty(self.num_experts, self.r, base_linear.in_features)
        B = torch.empty(self.num_experts, base_linear.out_features, self.r)
        for e in range(self.num_experts):
            nn.init.kaiming_uniform_(A[e], a=5**0.5)
            nn.init.zeros_(B[e])

        self.store_on_cpu = bool(store_on_cpu)
        if self.store_on_cpu and torch.cuda.is_available():
            # keep in CPU pinned memory
            A = A.contiguous().pin_memory()
            B = B.contiguous().pin_memory()
        self.A_full = nn.Parameter(A, requires_grad=False)
        self.B_full = nn.Parameter(B, requires_grad=False)

        # Per-layer GPU cache: expert_id -> (A_e, B_e) on device
        self.max_cached_experts = int(max_cached_experts)
        self._gpu_cache: "OrderedDict[int, Tuple[torch.Tensor, torch.Tensor]]" = OrderedDict()

        # scale clamp for stability in fp16/bf16
        self.scale_clip = 8.0

    def set_expert_scale(self, expert_id: int, alpha: int, r: int):
        with torch.no_grad():
            s = float(alpha) / float(r)
            s = min(s, 4.0)  # keep conservative if you want
            self.scale_per_expert.data[int(expert_id)] = s

    def _get_exec_device(self, x: torch.Tensor) -> torch.device:
        # base linear is already on the correct device per layer (device_map="auto")
        return x.device

    @torch.no_grad()
    def _get_expert_AB(self, e: int, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return (A_e, B_e) on 'device' with 'dtype'.
        Uses LRU cache to limit VRAM usage.
        """
        e = int(e)
        if not self.store_on_cpu:
            # if user chooses to store full on GPU, just slice
            Ae = self.A_full[e].to(device=device, dtype=dtype)
            Be = self.B_full[e].to(device=device, dtype=dtype)
            return Ae, Be

        # cache hit
        hit = self._gpu_cache.get(e, None)
        if hit is not None:
            self._gpu_cache.move_to_end(e)
            Ae, Be = hit
            # ensure dtype matches current compute dtype (rare if mixed)
            if Ae.dtype != dtype:
                Ae = Ae.to(dtype=dtype)
                Be = Be.to(dtype=dtype)
                self._gpu_cache[e] = (Ae, Be)
            return Ae, Be

        # cache miss: move slice CPU->GPU
        Ae = self.A_full[e].to(device=device, dtype=dtype, non_blocking=True)
        Be = self.B_full[e].to(device=device, dtype=dtype, non_blocking=True)

        self._gpu_cache[e] = (Ae, Be)
        self._gpu_cache.move_to_end(e)

        # evict LRU
        while len(self._gpu_cache) > max(1, self.max_cached_experts):
            self._gpu_cache.popitem(last=False)

        return Ae, Be

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        OOM-safe forward:
        - 2D: group-by-expert + scatter
        - 3D: reshape to [N=B*T, Din], group-by-expert + scatter
        """
        y = self.base(x)
        device = self._get_exec_device(x)

        # expert ids
        eid = getattr(self.parent_mlp, "_cached_eid", None)
        if eid is None:
            eid = torch.zeros((x.shape[0],), dtype=torch.long, device=device)
        else:
            eid = eid.to(device=device, non_blocking=True)
        eid = torch.clamp(eid, 0, self.num_experts - 1)

        # per-sample scale
        scale = self.scale_per_expert.to(device=device, dtype=torch.float32).index_select(0, eid)
        if self.scale_clip and float(self.scale_clip) > 0:
            scale = torch.clamp(scale, -float(self.scale_clip), float(self.scale_clip))
        scale = scale.to(dtype=x.dtype)

        # 2D
        if x.dim() == 2:
            out = y.clone()
            for e in range(self.num_experts):
                idx = (eid == e).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                xe = x.index_select(0, idx)  # [Ne, Din]
                Ae, Be = self._get_expert_AB(e, device=device, dtype=x.dtype)
                ze = xe @ Ae.t()            # [Ne, r]
                de = ze @ Be.t()            # [Ne, Dout]
                se = scale.index_select(0, idx).unsqueeze(-1)
                out.index_add_(0, idx, de * se)
            return out

        # 3D
        if x.dim() == 3:
            Bsz, T, Din = x.shape
            Dout = y.shape[-1]
            x2 = x.reshape(Bsz * T, Din)
            y2 = y.reshape(Bsz * T, Dout)

            eid2 = eid.repeat_interleave(T)
            scale2 = scale.repeat_interleave(T)

            out2 = y2.clone()

            # group-by-expert, no A2/B2 expansion
            for e in range(self.num_experts):
                idx = (eid2 == e).nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                xe = x2.index_select(0, idx)  # [Ne, Din]
                Ae, Be = self._get_expert_AB(e, device=device, dtype=x.dtype)
                ze = xe @ Ae.t()
                de = ze @ Be.t()
                se = scale2.index_select(0, idx).unsqueeze(-1)
                out2.index_add_(0, idx, de * se)

            return out2.reshape(Bsz, T, Dout)

        # fallback
        return y


# =========================
# Patch MLP: wrap gate/up/down + add router
# =========================
def patch_llama_mlp_no_replace(
    model,
    num_experts=5,
    r=8,
    alpha=32,
    layer_start=0,
    store_on_cpu=True,
    max_cached_experts=2,
):
    hidden = int(model.config.hidden_size)
    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue
        mlp = layer.mlp
        if not hasattr(mlp, "router"):
            mlp.router = nn.Linear(hidden, num_experts, bias=False)

        mlp.gate_proj = RoutedLoRALinearBase(
            mlp.gate_proj, parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha,
            store_on_cpu=store_on_cpu, max_cached_experts=max_cached_experts
        )
        mlp.up_proj = RoutedLoRALinearBase(
            mlp.up_proj, parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha,
            store_on_cpu=store_on_cpu, max_cached_experts=max_cached_experts
        )
        mlp.down_proj = RoutedLoRALinearBase(
            mlp.down_proj, parent_mlp=mlp, num_experts=num_experts, r=r, alpha=alpha,
            store_on_cpu=store_on_cpu, max_cached_experts=max_cached_experts
        )
    return model


# =========================
# Prefill-only routing hook
# =========================
def register_mlp_router_hooks_prefill_only(model, layer_start=0):
    """
    Prefill-only:
    - if T>1: compute per-layer eid and cache in mlp._cached_eid
    - if T==1: reuse cached eid (no router compute)
    """
    for li, layer in enumerate(model.model.layers):
        if li < int(layer_start):
            continue

        mlp = layer.mlp
        router = mlp.router

        def make_hook(this_mlp, this_router):
            def hook_fn(module, inputs):
                x = inputs[0]  # [B, T, H]
                _, T_, _ = x.shape

                if T_ == 1 and getattr(this_mlp, "_cached_eid", None) is not None:
                    return

                pooled = x.mean(dim=1)  # [B, H]
                # router in fp32 for stability
                pooled = pooled.to(device=this_router.weight.device, dtype=torch.float32)
                logits = this_router(pooled)
                eid = logits.argmax(dim=-1).long()
                this_mlp._cached_eid = torch.clamp(eid, 0, logits.size(-1) - 1).detach()
                return
            return hook_fn

        mlp.register_forward_pre_hook(make_hook(mlp, router))


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
# Load LoRA weights into expert slots (CPU full storage)
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

            # Copy into CPU full storage
            A_full = mod.A_full.data
            B_full = mod.B_full.data
            A_full[expert_id].copy_(sd[kA].to(device=A_full.device, dtype=A_full.dtype))
            B_full[expert_id].copy_(sd[kB].to(device=B_full.device, dtype=B_full.dtype))

            # Invalidate GPU cache for this expert (if any)
            if expert_id in mod._gpu_cache:
                mod._gpu_cache.pop(int(expert_id), None)

    if missing:
        print(f"[warn] {adapter_dir}: missing LoRA blocks (show 10): {missing[:10]}", flush=True)


# =========================
# Length bucketing helper (reduces padding waste)
# =========================
def make_length_buckets(
    lengths: List[int],
    max_batch_size: int,
    max_batch_tokens: int,
) -> List[List[int]]:
    """
    Given per-sample token lengths, return list of index groups (batches),
    sorted by length and chunked so that:
      - batch size <= max_batch_size
      - (max_len_in_batch * batch_size) <= max_batch_tokens  (rough VRAM proxy)
    """
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    buckets: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0

    for i in order:
        L = int(lengths[i])
        if not cur:
            cur = [i]
            cur_max = L
            continue

        next_max = max(cur_max, L)
        next_size = len(cur) + 1
        if next_size > max_batch_size or (next_max * next_size) > max_batch_tokens:
            buckets.append(cur)
            cur = [i]
            cur_max = L
        else:
            cur.append(i)
            cur_max = next_max

    if cur:
        buckets.append(cur)
    return buckets


# =========================
# OpenCompass Model
# =========================
class RouterMoELlama(HuggingFacewithChatTemplate):
    is_api = False

    def __init__(
        self,
        path,
        layer_router_ckpt,
        lora_paths: Dict[str, str],
        abbr="router_moe_layer_prefill_bucketed_cached",
        dtype="float16",
        r=8,
        alpha=32,
        enable_routing_log=False,

        # ---- OOM/VRAM controls ----
        bucket_by_length: bool = True,
        max_batch_size: int = 16,
        max_batch_tokens: int = 16000,     # proxy: max_len * batch_size
        max_seq_len: int = 2048,

        store_lora_on_cpu: bool = True,    # key optimization #2
        max_cached_experts: int = 2,       # per layer per proj cache cap (1~5)
        **kwargs,
    ):
        super().__init__(path=path, **kwargs)

        # TF32 can speed inference
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass

        self.abbr = abbr
        self.enable_routing_log = bool(enable_routing_log)

        # OOM controls
        self.bucket_by_length = bool(bucket_by_length)
        self.max_batch_size = int(max_batch_size)
        self.max_batch_tokens = int(max_batch_tokens)
        self.max_seq_len = int(max_seq_len)

        self.store_lora_on_cpu = bool(store_lora_on_cpu)
        self.max_cached_experts = int(max_cached_experts)

        # output dir
        out_dir = os.environ.get("OC_OUTPUT_DIR", os.getcwd())
        os.makedirs(out_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        pid = os.getpid()
        self.route_log_path = os.path.join(out_dir, f"layer_route_log_{abbr}_{ts}_pid{pid}.jsonl")

        self.route_counter = Counter()

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

        # patch wrappers + routers
        self.model = patch_llama_mlp_no_replace(
            self.model,
            num_experts=len(ID2LABEL),
            r=int(r),
            alpha=int(alpha),
            layer_start=self.layer_start,
            store_on_cpu=self.store_lora_on_cpu,
            max_cached_experts=self.max_cached_experts,
        )

        # prefill-only routing hooks
        register_mlp_router_hooks_prefill_only(self.model, layer_start=self.layer_start)

        # load LoRA experts (into CPU full storage)
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

        # tokenizer safety
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        try:
            self.model.config.pad_token_id = self.tokenizer.pad_token_id
            self.model.config.eos_token_id = self.tokenizer.eos_token_id
        except Exception:
            pass

        print(
            f"[oom-control] bucket_by_length={self.bucket_by_length} "
            f"max_batch_size={self.max_batch_size} max_batch_tokens={self.max_batch_tokens} "
            f"store_lora_on_cpu={self.store_lora_on_cpu} max_cached_experts={self.max_cached_experts}",
            flush=True,
        )

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
    def _generate_one_batch(self, prompts: List[str], gen_kwargs: dict) -> List[str]:
        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_seq_len,
        )
        enc = {k: v.to(self.model.device) for k, v in enc.items()}

        out = self.model.generate(**enc, **gen_kwargs)
        return self.tokenizer.batch_decode(out, skip_special_tokens=True)

    @torch.inference_mode()
    def generate(self, prompts, **gen_kwargs):
        if not isinstance(prompts, list):
            prompts = [prompts]
        ps: List[str] = [self._to_prompt_str(p) for p in prompts]
        n = len(ps)

        # OpenCompass: map max_out_len -> max_new_tokens
        max_out_len = gen_kwargs.pop("max_out_len", None)
        if max_out_len is not None and "max_new_tokens" not in gen_kwargs:
            gen_kwargs["max_new_tokens"] = int(max_out_len)

        # stable greedy defaults
        gen_kwargs.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        gen_kwargs["do_sample"] = False
        gen_kwargs["num_beams"] = 1
        gen_kwargs.pop("temperature", None)
        gen_kwargs.pop("top_p", None)
        gen_kwargs.pop("top_k", None)

        # If no bucketing: do one big batch (fast but more VRAM)
        if not self.bucket_by_length or n <= self.max_batch_size:
            texts = self._generate_one_batch(ps, gen_kwargs)
            return texts

        # ---------- Length bucketing ----------
        # Get token lengths with no padding
        lens = []
        for s in ps:
            # cheap: only length, not tensors on GPU
            ids = self.tokenizer(
                s,
                truncation=True,
                max_length=self.max_seq_len,
                add_special_tokens=True,
            )["input_ids"]
            lens.append(len(ids))

        buckets = make_length_buckets(
            lengths=lens,
            max_batch_size=self.max_batch_size,
            max_batch_tokens=self.max_batch_tokens,
        )

        outputs = [None] * n
        for b in buckets:
            batch_prompts = [ps[i] for i in b]
            batch_texts = self._generate_one_batch(batch_prompts, gen_kwargs)
            for i, t in zip(b, batch_texts):
                outputs[i] = t

        # optional routing log: lightweight
        if self.enable_routing_log:
            try:
                recs = []
                for li, layer in enumerate(self.model.model.layers):
                    if li < self.layer_start:
                        continue
                    eid = getattr(layer.mlp, "_cached_eid", None)
                    if eid is None:
                        continue
                    recs.append({"layer": li, "eids": eid[: min(n, eid.numel())].tolist()})
                with open(self.route_log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"n": n, "layers": recs}, ensure_ascii=False) + "\n")
            except Exception:
                pass

        return outputs
