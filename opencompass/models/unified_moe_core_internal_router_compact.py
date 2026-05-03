"""
這是「舊版兩段式 router core」。流程是：

載入 base LLM 與 tokenizer。
把 LLM 每層的 attention / MLP 線性層換成 HardRoutedLoRALinear。
把每個 task 的 LoRA adapter 載進不同 expert slot。
載入外部 BERT encoder + 兩個 router head：
router_first
router_mid
在 first_layer_idx 和 middle_layer_idx 這兩層外面包一層 wrapper，等 hidden state 跑到那裡時再即時計算 route。
第一次 route 決定前半段 layer 用哪個 expert；第二次 route 決定後半段 layer 用哪個 expert；最後再正常 generate。
"""
import json
import os
from collections import Counter
from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from model_backbone_specs import get_decoder_layers, infer_backbone_spec, set_decoder_layer
from opencompass.models.router_moe_components import BertExternalEncoder, CompactCrossAttentionRouter
from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    HardRoutedLoRALinear,
    load_lora_into_expert,
    patch_causal_lm_with_hard_routed_lora,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_expert,
    set_layer_range_expert,
)


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


class BeforeAttentionRouterWrapper(nn.Module):
    def __init__(self, base_layer, core_ref, which: str):
        super().__init__()
        self.base_layer = base_layer
        self.core = core_ref
        self.which = which

    @staticmethod
    def gather_last_valid(hidden_states: torch.Tensor) -> torch.Tensor:
        return hidden_states[:, -1, :]

    def forward(self, hidden_states, *args, **kwargs):
        if self.which == "first":
            if self.core._should_route_first(hidden_states):
                vec = self.gather_last_valid(hidden_states)
                eid = self.core._route_first_from_vec(vec)
                self.core.cached_first_eid = eid
                set_layer_range_expert(
                    self.core.model,
                    self.core.first_layer_idx,
                    self.core.middle_layer_idx - 1,
                    eid,
                )
        elif self.which == "mid":
            if self.core._should_route_mid(hidden_states):
                vec = self.gather_last_valid(hidden_states)
                eid = self.core._route_mid_from_vec(vec)
                self.core.cached_mid_eid = eid
                set_layer_range_expert(
                    self.core.model,
                    self.core.middle_layer_idx,
                    self.core.num_layers - 1,
                    eid,
                )

        return self.base_layer(hidden_states, *args, **kwargs)


class UnifiedMoECoreInternalRouterCompact:
    def __init__(
        self,
        base_model_path: str,
        router_ckpt_dir: str,
        router_bert_init: str,
        lora_paths: Dict[str, str],
        dtype: str = "float16",
        r: int = 8,
        alpha: int = 32,
        router_dim: int = 512,
        device_map: str = "auto",
        max_seq_len: int = 2048,
        force_first_task: Optional[str] = None,
        force_mid_task: Optional[str] = None,
    ):
        self.max_seq_len = int(max_seq_len)
        self.route_counter = Counter()
        self.force_first_task = force_first_task
        self.force_mid_task = force_mid_task
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        with open(os.path.join(router_ckpt_dir, "router_config.json"), "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.task_names = cfg["task_names"]
        self.task_to_eid = {task: i + 1 for i, task in enumerate(self.task_names)}
        self.first_layer_idx = int(cfg["first_layer_idx"])
        self.middle_layer_idx = int(cfg["middle_layer_idx"])
        self.router_max_len = int(cfg.get("router_max_len", 512))

        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.backbone_spec = infer_backbone_spec(self.model)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))

        self.model = patch_causal_lm_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.task_names),
            r=r,
            alpha=alpha,
        )

        for task, adapter_dir in lora_paths.items():
            task_id = self.task_names.index(task)
            expert_id = task_id + 1
            load_lora_into_expert(self.model, adapter_dir, expert_id)

        encoder_dir = os.path.join(router_ckpt_dir, "encoder")

        self.router_tokenizer = AutoTokenizer.from_pretrained(router_bert_init)
        self.bert_encoder = BertExternalEncoder(encoder_dir)

        bert_hidden = self.bert_encoder.encoder.config.hidden_size
        llama_hidden = self.model.config.hidden_size

        self.router_first = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden,
            bert_hidden_size=bert_hidden,
            router_dim=router_dim,
            num_tasks=len(self.task_names),
        )
        self.router_mid = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden,
            bert_hidden_size=bert_hidden,
            router_dim=router_dim,
            num_tasks=len(self.task_names),
        )

        state = torch.load(os.path.join(router_ckpt_dir, "router_heads.pt"), map_location="cpu")
        self.router_first.load_state_dict(state["router_first"])
        self.router_mid.load_state_dict(state["router_mid"])
        self.bert_encoder.load_state_dict(state["bert_encoder"], strict=False)

        self.router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_encoder.to(self.router_device).eval()
        self.router_first.to(self.router_device).eval()
        self.router_mid.to(self.router_device).eval()

        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None

        base_first = get_decoder_layers(self.model, spec=self.backbone_spec)[self.first_layer_idx]
        set_decoder_layer(
            self.model,
            self.first_layer_idx,
            BeforeAttentionRouterWrapper(base_first, self, which="first"),
            spec=self.backbone_spec,
        )

        base_mid = get_decoder_layers(self.model, spec=self.backbone_spec)[self.middle_layer_idx]
        set_decoder_layer(
            self.model,
            self.middle_layer_idx,
            BeforeAttentionRouterWrapper(base_mid, self, which="mid"),
            spec=self.backbone_spec,
        )

    def _reset_runtime_cache(self):
        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None
        set_all_experts(self.model, NULL_EXPERT_ID)

    def _encode_bert_memory(self, prompt: str):
        rt = self.router_tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.router_max_len,
        )
        rt = {
            key: value.to(self.router_device)
            for key, value in rt.items()
            if key in ["input_ids", "attention_mask", "token_type_ids"]
        }
        bert_prev, bert_last = self.bert_encoder(**rt)
        self.cached_bert_prev = bert_prev
        self.cached_bert_last = bert_last
        self.cached_bert_mask = rt["attention_mask"]

    def _should_route_first(self, hidden_states: torch.Tensor) -> bool:
        return (hidden_states.size(1) > 1) and (self.cached_first_eid is None)

    def _should_route_mid(self, hidden_states: torch.Tensor) -> bool:
        return (hidden_states.size(1) > 1) and (self.cached_mid_eid is None)

    def _forced_task_to_eid(self, task_name: Optional[str]) -> Optional[int]:
        if task_name is None:
            return None
        if task_name not in self.task_to_eid:
            raise ValueError(
                f"Unknown forced task: {task_name}. "
                f"Available tasks: {self.task_names}"
            )
        return self.task_to_eid[task_name]

    @torch.no_grad()
    def _route_first_from_vec(self, vec: torch.Tensor) -> int:
        forced_eid = self._forced_task_to_eid(self.force_first_task)
        if forced_eid is not None:
            task = self.task_names[forced_eid - 1]
            self.route_counter[f"first_forced::{task}"] += 1
            return forced_eid

        logits = self.router_first(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"first::{task}"] += 1
        return eid

    @torch.no_grad()
    def _route_mid_from_vec(self, vec: torch.Tensor) -> int:
        forced_eid = self._forced_task_to_eid(self.force_mid_task)
        if forced_eid is not None:
            task = self.task_names[forced_eid - 1]
            self.route_counter[f"mid_forced::{task}"] += 1
            return forced_eid

        logits = self.router_mid(
            llama_vec=vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        eid = int(logits.argmax(dim=-1).item()) + 1
        task = self.task_names[eid - 1]
        self.route_counter[f"mid::{task}"] += 1
        return eid

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=None, gen_kwargs=None, **kwargs):
        if isinstance(prompts, str):
            prompts = [prompts]

        if gen_kwargs is None:
            gen_kwargs = {}

        outputs = []
        for prompt in prompts:
            self._reset_runtime_cache()
            self._encode_bert_memory(prompt)

            inp = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_seq_len,
            ).to(self.model.device)

            args = dict(gen_kwargs)
            if "max_new_tokens" not in args and max_new_tokens is not None:
                args["max_new_tokens"] = int(max_new_tokens)

            args.setdefault("eos_token_id", self.tokenizer.eos_token_id)
            args.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            args["do_sample"] = False
            args["temperature"] = 0.0
            args["top_p"] = 1.0
            args["num_beams"] = 1

            out = self.model.generate(**inp, **args)

            prompt_len = inp["input_ids"].shape[1]
            gen_ids = out[0][prompt_len:]

            text = self.tokenizer.decode(
                gen_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            outputs.append(text)

        return outputs
