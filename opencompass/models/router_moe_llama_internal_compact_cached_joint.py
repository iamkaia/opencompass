import atexit
import hashlib
import json
import os
from collections import Counter
from typing import List, Optional

import torch

from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models.unified_moe_core_internal_router_compact_cached_joint import (
    UnifiedMoECoreInternalRouterCompactCachedJoint,
)
from opencompass.registry import MODELS


def _normalize_dataset_name(name: Optional[str]) -> str:
    if name is None:
        return "unknown"
    name = str(name).strip()
    if name.endswith(".json"):
        name = name[:-5]
    return name.replace("/", "_") or "unknown"


def _dataset_name_from_context(
    output_json_filepath: Optional[str],
    output_json_filename: Optional[str],
    gt_task: Optional[str],
) -> str:
    if gt_task:
        return _normalize_dataset_name(gt_task)
    if output_json_filename:
        return _normalize_dataset_name(output_json_filename)
    if output_json_filepath:
        return _normalize_dataset_name(os.path.splitext(os.path.basename(output_json_filepath))[0])
    return "unknown"


def _prompt_hash(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


@MODELS.register_module()
class RouterMoELlamaInternalCompactCachedJoint(HuggingFacewithChatTemplate):
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
        abbr: str = "router_moe_internal_compact_cached_joint",
        max_seq_len: int = 2048,
        first_layer_idx: int = 0,
        middle_layer_idx: int = 15,
        local_files_only: bool = False,
        hf_offline: bool = False,
        force_first_task: Optional[str] = None,
        force_mid_task: Optional[str] = None,
        debug_router_topk: int = 0,
        debug_router_max_prints: int = 0,
        debug_router_prompt_on_dataset_change: bool = True,
        debug_router_prompt_preview_chars: int = 320,
        debug_router_record_path: Optional[str] = None,
        routing_mode: str = "hard",
        routing_sharpness: float = 1.0,
        routing_topk: Optional[int] = None,
        share_first_weights_all_layers: bool = False,
        oracle_weight_path: Optional[str] = None,
        static_weight_path: Optional[str] = None,
        **kwargs,
    ):
        if hf_offline:
            os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_EVALUATE_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")

        tokenizer_only = kwargs.pop("tokenizer_only", True)
        super().__init__(
            path=path,
            tokenizer_only=tokenizer_only,
            max_seq_len=max_seq_len,
            run_cfg=run_cfg,
            **kwargs,
        )
        self.abbr = abbr
        self.max_out_len = max_out_len
        self.batch_size = batch_size
        self.run_cfg = run_cfg
        self.core = UnifiedMoECoreInternalRouterCompactCachedJoint(
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
            first_layer_idx=first_layer_idx,
            middle_layer_idx=middle_layer_idx,
            force_first_task=force_first_task,
            force_mid_task=force_mid_task,
            local_files_only=local_files_only or hf_offline,
            debug_router_topk=debug_router_topk,
            debug_router_max_prints=debug_router_max_prints,
            routing_mode=routing_mode,
            routing_sharpness=routing_sharpness,
            routing_topk=routing_topk,
            share_first_weights_all_layers=share_first_weights_all_layers,
            oracle_weight_path=oracle_weight_path,
            static_weight_path=static_weight_path,
        )
        self._active_dataset_name = None
        self._printed_dataset_totals = {}
        self._printed_dataset_prompt_debug = set()
        self.debug_router_prompt_on_dataset_change = bool(debug_router_prompt_on_dataset_change)
        self.debug_router_prompt_preview_chars = int(debug_router_prompt_preview_chars)
        self.debug_router_record_path = debug_router_record_path
        self._dataset_sample_counter = Counter()
        if self.debug_router_record_path:
            record_dir = os.path.dirname(os.path.abspath(self.debug_router_record_path))
            if record_dir:
                os.makedirs(record_dir, exist_ok=True)
        atexit.register(self._flush_all_dataset_routing_summaries)

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
                return "\n".join([f"{m.get('role', 'user')}: {m.get('content', '')}" for m in msgs])
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
        raise TypeError(f"Unsupported input type for prompt: {type(x)}; value={repr(x)[:300]}")

    def _resolve_run_dir(self, output_json_filepath: Optional[str]) -> str:
        if output_json_filepath:
            pred_dir = os.path.dirname(output_json_filepath)
            return os.path.dirname(pred_dir)
        return os.getcwd()

    def _eid_to_task(self, eid: Optional[int]) -> Optional[str]:
        if eid is None:
            return None
        if eid <= 0:
            return f"NULL({eid})"
        task_names = getattr(self.core, "task_names", None)
        if not task_names:
            return str(eid)
        idx = eid - 1
        if 0 <= idx < len(task_names):
            return task_names[idx]
        return str(eid)

    def _append_routing_log(self, dataset_name: str, first_eid: Optional[int], mid_eid: Optional[int]):
        first_task = self._eid_to_task(first_eid)
        mid_task = self._eid_to_task(mid_eid)
        pair_key = f"{first_task}->{mid_task}"
        if not hasattr(self, "_dataset_pair_counter"):
            self._dataset_pair_counter = {}
        if dataset_name not in self._dataset_pair_counter:
            self._dataset_pair_counter[dataset_name] = Counter()
        self._dataset_pair_counter[dataset_name][pair_key] += 1

    def _maybe_print_dataset_prompt_debug(self, dataset_name: str, prompts: List[str]):
        if not self.debug_router_prompt_on_dataset_change or not prompts:
            return
        if dataset_name in self._printed_dataset_prompt_debug:
            return
        self._printed_dataset_prompt_debug.add(dataset_name)
        prompt = str(prompts[0])
        preview = prompt.replace("\n", "\\n")
        if len(preview) > self.debug_router_prompt_preview_chars:
            preview = preview[: self.debug_router_prompt_preview_chars] + (
                f"...<truncated {len(preview) - self.debug_router_prompt_preview_chars} chars>"
            )
        print(
            f"[ROUTING_PROMPT][dataset={dataset_name}] before_router=true "
            f"prompt_sha1={_prompt_hash(prompt)} prompt_len={len(prompt)} "
            f"prompt_preview={preview}",
            flush=True,
        )

    def _append_routing_record(
        self,
        dataset_name: str,
        prompt: str,
        first_eid: Optional[int],
        mid_eid: Optional[int],
        first_weights=None,
        mid_weights=None,
    ):
        if not self.debug_router_record_path:
            return
        sample_index = self._dataset_sample_counter[dataset_name]
        self._dataset_sample_counter[dataset_name] += 1
        first_task = self._eid_to_task(first_eid)
        mid_task = self._eid_to_task(mid_eid)
        record = {
            "dataset": dataset_name,
            "sample_index": int(sample_index),
            "prompt_sha1": _prompt_hash(prompt),
            "pred_pair": f"{first_task}->{mid_task}",
            "first_task": first_task,
            "mid_task": mid_task,
            "first_eid": None if first_eid is None else int(first_eid),
            "mid_eid": None if mid_eid is None else int(mid_eid),
            "prompt_preview": prompt.replace("\n", "\\n")[:240],
            "routing_mode": getattr(self.core, "routing_mode", "hard"),
            "routing_sharpness": getattr(self.core, "routing_sharpness", 1.0),
            "routing_topk": getattr(self.core, "routing_topk", None),
            "share_first_weights_all_layers": getattr(
                self.core, "share_first_weights_all_layers", False
            ),
        }
        if first_weights is not None:
            record["first_weights"] = first_weights.detach().cpu().tolist()[1:]
        if mid_weights is not None:
            record["mid_weights"] = mid_weights.detach().cpu().tolist()[1:]
        with open(self.debug_router_record_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _flush_dataset_routing_summary(self, dataset_name: str):
        if not hasattr(self, "_dataset_pair_counter"):
            return
        counter = self._dataset_pair_counter.get(dataset_name)
        if not counter:
            return
        total = sum(counter.values())
        if self._printed_dataset_totals.get(dataset_name) == total:
            return
        self._printed_dataset_totals[dataset_name] = total
        summary = ", ".join(
            f"{pair}:{count}/{total} ({count / total:.1%})"
            for pair, count in counter.most_common()
        )
        print(f"[ROUTING][dataset={dataset_name}] sample_count={total} pair_distribution={summary}", flush=True)

    def _flush_all_dataset_routing_summaries(self):
        if not hasattr(self, "_dataset_pair_counter"):
            return
        for dataset_name in sorted(self._dataset_pair_counter):
            self._flush_dataset_routing_summary(dataset_name)

    def get_ppl(self, inputs: List[str], mask_length: Optional[List[int]] = None) -> List[float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports gen-based evaluation. "
            "Please use generation datasets or implement route-aware PPL scoring."
        )

    def get_ppl_tokenwise(
        self,
        inputs: List[str],
        label: Optional[List[List[int]]] = None,
        mask_length: Optional[List[int]] = None,
    ) -> List[float]:
        raise NotImplementedError(
            f"{self.__class__.__name__} only supports gen-based evaluation. "
            "Please use generation datasets or implement route-aware tokenwise PPL scoring."
        )

    @torch.no_grad()
    def generate(self, inputs, max_out_len, min_out_len=None, stopping_criteria=[], **kwargs):
        gt_task = kwargs.pop("gt_task", None)
        output_json_filepath = kwargs.pop("output_json_filepath", None)
        output_json_filename = kwargs.pop("output_json_filename", None)

        gen_kwargs = self.generation_kwargs.copy()
        gen_kwargs.update(kwargs)
        if max_out_len is not None:
            gen_kwargs["max_new_tokens"] = int(max_out_len)
        if min_out_len is not None:
            gen_kwargs["min_new_tokens"] = int(min_out_len)
        gen_kwargs["pad_token_id"] = self.tokenizer.pad_token_id
        gen_kwargs.setdefault("eos_token_id", self.tokenizer.eos_token_id)

        prompt_strs = [self._to_prompt_str(x) for x in inputs]
        dataset_name = _dataset_name_from_context(
            output_json_filepath=output_json_filepath,
            output_json_filename=output_json_filename,
            gt_task=gt_task,
        )
        previous_dataset_name = getattr(self, "_active_dataset_name", None)
        if previous_dataset_name is not None and previous_dataset_name != dataset_name:
            self._flush_dataset_routing_summary(previous_dataset_name)
        self._active_dataset_name = dataset_name
        self._maybe_print_dataset_prompt_debug(dataset_name, prompt_strs)

        outputs = self.core.generate(
            prompt_strs,
            gen_kwargs=gen_kwargs,
            dataset_name=dataset_name,
        )
        first_eids = getattr(self.core, "cached_first_eid", None)
        mid_eids = getattr(self.core, "cached_mid_eid", None)
        first_weights = getattr(self.core, "cached_first_weights", None)
        mid_weights = getattr(self.core, "cached_mid_weights", None)
        if first_eids is not None:
            first_eids = first_eids.detach().cpu().tolist()
        if mid_eids is not None:
            mid_eids = mid_eids.detach().cpu().tolist()
        for index, prompt in enumerate(prompt_strs):
            first_eid = None if first_eids is None else first_eids[index]
            mid_eid = None if mid_eids is None else mid_eids[index]
            self._append_routing_log(
                dataset_name=dataset_name,
                first_eid=first_eid,
                mid_eid=mid_eid,
            )
            self._append_routing_record(
                dataset_name=dataset_name,
                prompt=prompt,
                first_eid=first_eid,
                mid_eid=mid_eid,
                first_weights=None if first_weights is None else first_weights[index],
                mid_weights=None if mid_weights is None else mid_weights[index],
            )
        return outputs
