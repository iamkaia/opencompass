import json
import os
from typing import List, Optional
import time
import torch

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models.unified_moe_core_internal_router_compact import (
    UnifiedMoECoreInternalRouterCompact,
)
from opencompass.registry import MODELS


def _ensure_dir(path: str):
    if path:
        os.makedirs(path, exist_ok=True)


@MODELS.register_module()
class RouterMoELlamaInternalCompact(HuggingFacewithChatTemplate):
    is_api = False

    def __init__(
        self,
        path: str,
        router_ckpt_dir: str,
        router_bert_init: str,
        lora_paths: dict,
        max_out_len: int = 1024,
        batch_size: int = 1,
        run_cfg=None,
        dtype: str = "float16",
        r: int = 8,
        alpha: int = 16,
        router_dim: int = 512,
        abbr: str = "router_moe_internal_compact",
        max_seq_len: int = 2048,
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

        self.core = UnifiedMoECoreInternalRouterCompact(
            base_model_path=path,
            router_ckpt_dir=router_ckpt_dir,
            router_bert_init=router_bert_init,
            lora_paths=lora_paths,
            dtype=dtype,
            r=r,
            alpha=alpha,
            router_dim=router_dim,
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
                        msgs,
                        tokenize=False,
                        add_generation_prompt=True,
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
                    msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )

            parts = []
            for m in x:
                role = m.get("role", "user")
                txt = m.get("content", None)
                if txt is None:
                    txt = m.get("prompt", "")
                parts.append(f"{role}: {txt}")
            return "\n".join(parts)

        raise TypeError(
            f"Unsupported input type for prompt: {type(x)}; value={repr(x)[:300]}"
        )

    def _resolve_run_dir(self, output_json_filepath: Optional[str]) -> str:
        """
        從 OpenCompass 的 prediction json 路徑反推本次 run 目錄：
        .../outputs/default/<run_id>/predictions/xxx.json
                              ^^^^^^^^^^^^^^^^^^
        """
        if output_json_filepath:
            pred_dir = os.path.dirname(output_json_filepath)
            run_dir = os.path.dirname(pred_dir)
            return run_dir
        return os.getcwd()

    def _eid_to_task(self, eid: Optional[int]) -> Optional[str]:
        if eid is None:
            return None

        # unified core 裡通常 0 是 NULL_EXPERT_ID，真正 task eid 從 1 開始
        if eid <= 0:
            return f"NULL({eid})"

        task_names = getattr(self.core, "task_names", None)
        if not task_names:
            return str(eid)

        idx = eid - 1
        if 0 <= idx < len(task_names):
            return task_names[idx]

        return str(eid)

    def _append_routing_log(
        self,
        run_dir: str,
        prompt: str,
        first_eid: Optional[int],
        mid_eid: Optional[int],
    ):
        routing_root = os.path.join(run_dir, "routing")
        _ensure_dir(routing_root)

        # 固定一個 run_tag（不要每次都變！）
        if not hasattr(self, "_run_tag"):
            self._run_tag = time.strftime("%Y%m%d_%H%M%S")

        routing_dir = os.path.join(routing_root, self._run_tag)
        _ensure_dir(routing_dir)

        counter_path = os.path.join(routing_dir, "routing_counter.json")

        first_task = self._eid_to_task(first_eid)
        mid_task = self._eid_to_task(mid_eid)
        pair_key = f"{first_task}->{mid_task}"

        if not hasattr(self, "_pair_counter"):
            self._pair_counter = {}

        self._pair_counter[pair_key] = self._pair_counter.get(pair_key, 0) + 1

        '''：
        row = {
            "prompt_preview": prompt[:200],
            "first_eid": first_eid,
            "mid_eid": mid_eid,
            "first_task": first_task,
            "mid_task": mid_task,
            "pair": pair_key,
        }

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        '''
        #if sum(self._pair_counter.values()) % 50 == 0:
        with open(counter_path, "w", encoding="utf-8") as f:
            json.dump(self._pair_counter, f, ensure_ascii=False, indent=2)

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
        output_json_filepath = kwargs.pop("output_json_filepath", None)

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs.update(kwargs)

        if max_out_len is not None:
            gen_kwargs["max_new_tokens"] = int(max_out_len)
        if min_out_len is not None:
            gen_kwargs["min_new_tokens"] = int(min_out_len)

        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        prompt_strs = [self._to_prompt_str(x) for x in inputs]
        run_dir = self._resolve_run_dir(output_json_filepath)

        outputs = []

        # 關鍵：逐筆跑，這樣每筆跑完就能從 core 抓到 first/mid eid
        for prompt in prompt_strs:
            one_out = self.core.generate(
                [prompt],
                gen_kwargs=gen_kwargs,
            )

            if isinstance(one_out, list):
                outputs.extend(one_out)
            else:
                outputs.append(one_out)

            first_eid = getattr(self.core, "cached_first_eid", None)
            mid_eid = getattr(self.core, "cached_mid_eid", None)

            self._append_routing_log(
                run_dir=run_dir,
                prompt=prompt,
                first_eid=first_eid,
                mid_eid=mid_eid,
            )

        return outputs
