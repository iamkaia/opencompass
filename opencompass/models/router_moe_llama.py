# opencompass/models/router_moe_llama.py
import json
import os
import time
from typing import List, Optional

import torch

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models.unified_moe_core import UnifiedMoECore
from opencompass.registry import MODELS


def _ensure_dir(p: str):
    if p:
        os.makedirs(p, exist_ok=True)


@MODELS.register_module()
class RouterMoELlama(HuggingFacewithChatTemplate):
    """
    OpenCompass wrapper around UnifiedMoECore (two-layer router version).
    """

    is_api = False

    def __init__(
        self,
        path,
        router_ckpt_dir,
        lora_paths,
        max_out_len=128,
        batch_size=128,
        run_cfg=None,
        dtype="float16",
        r=8,
        alpha=16,
        abbr="router_moe_lora_2layer",
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

        self.batch_size = batch_size
        self.max_out_len = max_out_len

        out_dir = os.environ.get("OC_OUTPUT_DIR", None) or os.getcwd()
        os.makedirs(out_dir, exist_ok=True)
        pid = os.getpid()
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.save_route_counts_path = os.path.join(
            out_dir, f"routing_counts_{self.abbr}_{ts}_pid{pid}.json"
        )

        self.core = UnifiedMoECore(
            base_model_path=path,
            router_ckpt_dir=router_ckpt_dir,
            lora_paths=lora_paths,
            dtype=dtype,
            r=r,
            alpha=alpha,
            device_map="auto",
            max_seq_len=max_seq_len,
        )

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
                    return self.tokenizer.apply_chat_template(
                        msgs, tokenize=False, add_generation_prompt=True
                    )
                return "\n".join(
                    [f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs]
                )

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

                return self.tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )

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
    def generate(
        self,
        inputs: List[str],
        max_out_len: int,
        min_out_len: Optional[int] = None,
        stopping_criteria: List[str] = [],
        **kwargs,
    ):
        kwargs.pop("gt_task", None)
        kwargs.pop("output_json_filepath", None)

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs.update(kwargs)

        if max_out_len is not None:
            gen_kwargs["max_new_tokens"] = int(max_out_len)

        if min_out_len is not None:
            gen_kwargs["min_new_tokens"] = int(min_out_len)

        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        prompt_strs = [self._to_prompt_str(x) for x in inputs]


        outputs = self.core.generate(
            prompt_strs,
            gen_kwargs=gen_kwargs,
        )

        # 如果 core 有 route_counter 就存；沒有就略過
        try:
            route_counter = getattr(self.core, "route_counter", None)
            if route_counter is not None:
                with open(self.save_route_counts_path, "w", encoding="utf-8") as f:
                    json.dump(route_counter, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return outputs

