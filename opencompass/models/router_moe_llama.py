# opencompass/models/router_moe_llama.py
import os
import json
import time
from typing import Any, Dict

import torch
from collections import Counter

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models.unified_moe_core import UnifiedMoECore, ID2LABEL


def _ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


def _append_jsonl(path: str, obj: dict):
    _ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _norm_task(t):
    if t is None:
        return None
    t = str(t).strip()
    # keep your normalization here if needed
    if t == "squad2":
        t = "squad2.0"
    return t


def _get_routing_log_path(output_json_filepath, abbr, gt_task=None):
    """
    Keep the same behavior you already had:
    - if output_json_filepath exists, log next to prediction file
    - filename uses gt_task if available
    """
    if output_json_filepath:
        pred_dir = os.path.dirname(output_json_filepath)
        if gt_task:
            ds = str(gt_task).strip()
            if ds.endswith(".json"):
                ds = ds[:-5]
        else:
            ds = os.path.splitext(os.path.basename(output_json_filepath))[0]
        ds = ds.replace("/", "_")
        return os.path.join(pred_dir, f"routing_log__{ds}.jsonl")

    safe = abbr.replace("/", "_")
    ds = str(gt_task).strip() if gt_task else "unknown"
    ds = ds.replace("/", "_")
    return f"routing_logs/routing_{safe}__{ds}.jsonl"


class RouterMoELlama(HuggingFacewithChatTemplate):
    """OpenCompass wrapper around UnifiedMoECore (external router only)."""

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
        max_seq_len=2048,
        **kwargs,
    ):
        super().__init__(
            path=path,
            max_out_len=max_out_len,
            batch_size=batch_size,
            run_cfg=run_cfg,
            **kwargs,
        )
        self.abbr = abbr

        # output dir for debug counts/logs (still OK to keep)
        out_dir = os.environ.get("OC_OUTPUT_DIR", None) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        pid = os.getpid()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_route_counts_path = os.path.join(out_dir, f"routing_counts_{self.abbr}_{ts}_pid{pid}.json")

        # create the *core* blackbox model
        self.core = UnifiedMoECore(
            base_model_path=path,
            cls_dir=cls_dir,
            lora_paths=lora_paths,
            dtype=dtype,
            r=r,
            alpha=alpha,
            device_map="auto",
            max_seq_len=max_seq_len,
        )

    def _to_prompt_str(self, x):
        # keep your original conversion logic (unchanged)
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
        gt_task = gen_kwargs.pop("gt_task", None)
        output_json_filepath = gen_kwargs.pop("output_json_filepath", None)

        if not isinstance(prompts, list):
            prompts = [prompts]

        max_out_len = gen_kwargs.pop("max_out_len", None)

        # Prepare routing log path
        gt = _norm_task(gt_task)
        log_path = _get_routing_log_path(output_json_filepath, self.abbr, gt_task=gt)

        # callback for core routing
        def on_route(prompt: str, eid: int):
            routed = ID2LABEL.get(eid, str(eid))
            ok = (gt == routed) if gt is not None else None
            rec = {
                "ts": time.time(),
                "kind": "external",
                "gt_task": gt,
                "routed_task": routed,
                "eid": eid,
                "route_ok": ok,
                "prompt_len_chars": len(prompt),
            }
            _append_jsonl(log_path, rec)

        # convert inputs to strings
        prompt_strs = [self._to_prompt_str(x) for x in prompts]

        # run core blackbox generate
        outputs = self.core.generate(
            prompt_strs,
            max_new_tokens=int(max_out_len) if max_out_len is not None else None,
            gen_kwargs=gen_kwargs,
            on_route=on_route,
        )

        # write counts snapshot (optional)
        try:
            with open(self.save_route_counts_path, "w", encoding="utf-8") as f:
                json.dump(
                    {ID2LABEL.get(k, str(k)): v for k, v in self.core.route_counter.items()},
                    f,
                    indent=2,
                    ensure_ascii=False,
                )
        except Exception:
            pass

        return outputs
