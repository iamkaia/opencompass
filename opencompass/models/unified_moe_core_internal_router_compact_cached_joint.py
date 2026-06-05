'''
這是你現在較新的「joint pair router core」。它和上面最大的差別是：

不再分開做 first 分類、mid 分類
改成先抽 first_vec / mid_vec
再一次直接預測 (first_task, mid_task) 這個 pair
流程是：

載入 base LLM、tokenizer、所有 LoRA expert。
載入 BERT encoder。
載入兩個 feature encoder：
router_first
router_mid
再載入一個 pair_classifier。
每個 prompt 先做一次 prepass，拿到：
hidden_states[first_layer_idx]
hidden_states[middle_layer_idx]
用這兩個向量加上 BERT memory 算出 pair logits。
一次決定前半段 expert 和後半段 expert。
設好 layer range expert 後，再正式 generate。
所以它是「先 route、後 generate」，而且 route 的 supervision 是 pair 級別，不是兩個 task head 各自獨立。
'''
import json
import os
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from model_backbone_specs import get_decoder_layers, infer_backbone_spec

from opencompass.models.router_moe_components import BertExternalEncoder, CompactRouterFeatureEncoder
from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
    set_layer_range_expert_weights,
)


class UnifiedMoECoreInternalRouterCompactCachedJoint:
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
        first_layer_idx: int = 0,
        middle_layer_idx: int = 15,
        force_first_task: Optional[str] = None,
        force_mid_task: Optional[str] = None,
        local_files_only: bool = False,
        debug_router_topk: int = 0,
        debug_router_max_prints: int = 0,
        routing_mode: str = "hard",
        routing_sharpness: float = 1.0,
        routing_topk: Optional[int] = None,
    ):
        self.max_seq_len = int(max_seq_len)
        self.route_counter = Counter()
        self.force_first_task = force_first_task
        self.force_mid_task = force_mid_task
        self.debug_router_topk = max(0, int(debug_router_topk))
        self.debug_router_max_prints = max(0, int(debug_router_max_prints))
        self.debug_router_print_count = 0
        self.routing_mode = str(routing_mode)
        if self.routing_mode not in {"hard", "weighted_sum", "uniform"}:
            raise ValueError(
                f"Unknown routing_mode={self.routing_mode!r}; "
                "expected 'hard', 'weighted_sum', or 'uniform'."
            )
        self.routing_sharpness = float(routing_sharpness)
        if self.routing_sharpness <= 0.0:
            raise ValueError("routing_sharpness must be positive.")
        self.routing_topk = None if routing_topk is None else int(routing_topk)
        if self.routing_topk is not None and self.routing_topk <= 0:
            raise ValueError("routing_topk must be positive when set.")
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16

        cfg_path = os.path.join(router_ckpt_dir, "router_config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"Missing router_config.json under router_ckpt_dir={router_ckpt_dir!r}. "
                "Please ensure the OpenCompass run cwd matches the relative path."
            )
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        self.task_names = cfg["task_names"]
        self.task_to_eid = {task: i + 1 for i, task in enumerate(self.task_names)}
        self.first_layer_idx = int(cfg.get("first_layer_idx", first_layer_idx))
        self.middle_layer_idx = int(cfg.get("middle_layer_idx", middle_layer_idx))
        self.router_max_len = int(cfg.get("router_max_len", 512))
        self.router_pooling = str(cfg.get("router_pooling", "last_token"))
        self.router_pooling_last_k = int(cfg.get("router_pooling_last_k", 4))
        self.router_dim = int(cfg.get("router_dim", router_dim))
        self.sample_feature_mode = str(cfg.get("sample_feature_mode", "none"))
        self.sample_feature_dim = int(cfg.get("sample_feature_dim", 0))
        if self.sample_feature_mode != "none" or self.sample_feature_dim != 0:
            raise ValueError(
                "Runtime generation does not implement cached sample features: "
                f"sample_feature_mode={self.sample_feature_mode!r}, "
                f"sample_feature_dim={self.sample_feature_dim}. "
                "Train with --sample_feature_mode none for OpenCompass evaluation."
            )
        requested_lora_tasks = list(lora_paths.keys())
        missing_lora_tasks = [task for task in self.task_names if task not in lora_paths]
        extra_lora_tasks = [task for task in requested_lora_tasks if task not in self.task_names]
        if missing_lora_tasks or extra_lora_tasks:
            raise ValueError(
                "LoRA task mismatch between router checkpoint and eval config. "
                f"ckpt task_names={self.task_names}, "
                f"config lora_paths keys={requested_lora_tasks}, "
                f"missing_in_lora_paths={missing_lora_tasks}, "
                f"extra_in_lora_paths={extra_lora_tasks}"
            )
        print(
            "[INFO] loaded router ckpt config: "
            f"task_names={self.task_names}, "
            f"first_layer_idx={self.first_layer_idx}, "
            f"middle_layer_idx={self.middle_layer_idx}, "
            f"router_pooling={self.router_pooling}, "
            f"router_pooling_last_k={self.router_pooling_last_k}, "
            f"router_dim={self.router_dim}, "
            f"router_max_len={self.router_max_len}, "
            f"routing_mode={self.routing_mode}, "
            f"routing_sharpness={self.routing_sharpness}, "
            f"routing_topk={self.routing_topk}",
            flush=True,
        )

        self.local_files_only = bool(local_files_only)
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_path,
            local_files_only=self.local_files_only,
        )
        self.tokenizer.padding_side = "left"
        self.tokenizer.truncation_side = "left"

        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            local_files_only=self.local_files_only,
        )
        self.model.eval()

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id

        self.backbone_spec = infer_backbone_spec(self.model)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))
        expected_llama_hidden = cfg.get("llama_hidden_size")
        if expected_llama_hidden is not None and int(expected_llama_hidden) != int(self.model.config.hidden_size):
            raise ValueError(
                f"Router checkpoint expects llama_hidden_size={expected_llama_hidden}, "
                f"but runtime base model provides hidden_size={self.model.config.hidden_size}."
            )
        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.task_names),
            r=r,
            alpha=alpha,
        )

        for task in self.task_names:
            adapter_dir = lora_paths[task]
            task_id = self.task_names.index(task)
            expert_id = task_id + 1
            load_lora_into_expert(self.model, adapter_dir, expert_id)

        encoder_dir = os.path.join(router_ckpt_dir, "encoder")
        self.router_tokenizer = AutoTokenizer.from_pretrained(
            router_bert_init,
            local_files_only=self.local_files_only,
        )
        self.bert_encoder = BertExternalEncoder(encoder_dir)

        bert_hidden = self.bert_encoder.encoder.config.hidden_size
        llama_hidden = self.model.config.hidden_size
        self.router_first = CompactRouterFeatureEncoder(llama_hidden, bert_hidden, self.router_dim)
        self.router_mid = CompactRouterFeatureEncoder(llama_hidden, bert_hidden, self.router_dim)
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(self.router_dim * 4),
            nn.Linear(self.router_dim * 4, self.router_dim * 2),
            nn.GELU(),
            nn.Linear(self.router_dim * 2, len(self.task_names) * len(self.task_names)),
        )

        state_path = os.path.join(router_ckpt_dir, "router_heads.pt")
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"Missing router heads checkpoint: {state_path}")
        state = torch.load(state_path, map_location="cpu")
        first_state = state.get("router_first") or state.get("pair_first_encoder")
        mid_state = state.get("router_mid") or state.get("pair_mid_encoder")
        if first_state is None or mid_state is None:
            raise KeyError(
                "router_heads.pt must contain either router_first/router_mid "
                "or pair_first_encoder/pair_mid_encoder."
            )
        self.router_first.load_state_dict(first_state)
        self.router_mid.load_state_dict(mid_state)
        self.pair_classifier.load_state_dict(state["pair_classifier"])
        self.bert_encoder.load_state_dict(state["bert_encoder"], strict=False)

        self.router_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.bert_encoder.to(self.router_device).eval()
        self.router_first.to(self.router_device).eval()
        self.router_mid.to(self.router_device).eval()
        self.pair_classifier.to(self.router_device).eval()

        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_first_weights = None
        self.cached_mid_weights = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None

    def _reset_runtime_cache(self):
        self.cached_first_eid = None
        self.cached_mid_eid = None
        self.cached_first_weights = None
        self.cached_mid_weights = None
        self.cached_bert_prev = None
        self.cached_bert_last = None
        self.cached_bert_mask = None
        set_all_experts(self.model, NULL_EXPERT_ID)

    def _encode_bert_memory(self, prompts: Sequence[str]):
        rt = self.router_tokenizer(
            list(prompts),
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.router_max_len,
        )
        rt = {
            k: v.to(self.router_device)
            for k, v in rt.items()
            if k in ["input_ids", "attention_mask", "token_type_ids"]
        }
        bert_prev, bert_last = self.bert_encoder(**rt)
        self.cached_bert_prev = bert_prev
        self.cached_bert_last = bert_last
        self.cached_bert_mask = rt["attention_mask"]

    def _forced_task_to_eid(self, task_name: Optional[str]) -> Optional[int]:
        if task_name is None:
            return None
        if task_name not in self.task_to_eid:
            raise ValueError(f"Unknown forced task: {task_name}. Available tasks: {self.task_names}")
        return self.task_to_eid[task_name]

    def _pool_prompt_vector(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.router_pooling == "last_token":
            return hidden_states[:, -1, :]
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        if self.router_pooling == "mean":
            return (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        if self.router_pooling == "lastk_mean":
            k = max(self.router_pooling_last_k, 1)
            outputs = []
            for sample_hidden, sample_mask in zip(hidden_states, attention_mask):
                valid_hidden = sample_hidden[sample_mask.to(dtype=torch.bool)]
                outputs.append(valid_hidden[-k:].mean(dim=0))
            return torch.stack(outputs, dim=0)
        raise ValueError(f"Unknown router_pooling={self.router_pooling!r} in router checkpoint.")

    def _pair_idx_to_name(self, pair_idx: int) -> str:
        num_tasks = len(self.task_names)
        first_idx = int(pair_idx) // num_tasks
        mid_idx = int(pair_idx) % num_tasks
        return f"{self.task_names[first_idx]}->{self.task_names[mid_idx]}"

    def _maybe_print_topk_pair_logits(
        self,
        pair_logits: torch.Tensor,
        dataset_name: Optional[str],
        prompt_previews: Optional[Sequence[str]],
        first_weights: Optional[torch.Tensor] = None,
        mid_weights: Optional[torch.Tensor] = None,
    ):
        if self.debug_router_topk <= 0 or self.debug_router_max_prints <= 0:
            return

        topk = min(self.debug_router_topk, pair_logits.size(-1))
        probs = torch.softmax(pair_logits.float(), dim=-1)
        top_vals, top_idx = torch.topk(pair_logits.float(), k=topk, dim=-1)
        top_probs = torch.gather(probs, dim=-1, index=top_idx)
        previews = list(prompt_previews or [""] * pair_logits.size(0))
        for sample_idx in range(pair_logits.size(0)):
            if self.debug_router_print_count >= self.debug_router_max_prints:
                break
            preview = previews[sample_idx].replace("\n", "\\n")[:160]
            pieces = []
            for rank in range(topk):
                pair_name = self._pair_idx_to_name(int(top_idx[sample_idx, rank].item()))
                logit = float(top_vals[sample_idx, rank].item())
                prob = float(top_probs[sample_idx, rank].item())
                pieces.append(f"{pair_name}:logit={logit:.4f},prob={prob:.4f}")
            sample_first_weights = None if first_weights is None else first_weights[sample_idx].tolist()
            sample_mid_weights = None if mid_weights is None else mid_weights[sample_idx].tolist()
            print(
                f"[ROUTING_DEBUG][dataset={dataset_name or 'unknown'}] "
                f"mode={self.routing_mode} topk_pairs={' | '.join(pieces)} "
                f"first_weights={sample_first_weights} "
                f"mid_weights={sample_mid_weights} "
                f"prompt={preview}",
                flush=True,
            )
            self.debug_router_print_count += 1

    @torch.no_grad()
    def _route_pair_from_vecs(
        self,
        first_vec: torch.Tensor,
        mid_vec: torch.Tensor,
        dataset_name: Optional[str] = None,
        prompt_previews: Optional[Sequence[str]] = None,
    ):
        forced_first_eid = self._forced_task_to_eid(self.force_first_task)
        forced_mid_eid = self._forced_task_to_eid(self.force_mid_task)
        num_tasks = len(self.task_names)

        first_feat = self.router_first(
            llama_vec=first_vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        mid_feat = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=self.cached_bert_prev,
            bert_last=self.cached_bert_last,
            bert_attention_mask=self.cached_bert_mask,
        )
        pair_logits = self.pair_classifier(torch.cat([first_feat, mid_feat], dim=-1))

        if forced_first_eid is not None or forced_mid_eid is not None:
            mask = torch.ones_like(pair_logits, dtype=torch.bool)
            for pair_idx in range(pair_logits.size(-1)):
                first_idx = pair_idx // num_tasks
                mid_idx = pair_idx % num_tasks
                keep = True
                if forced_first_eid is not None:
                    keep = keep and (first_idx == (forced_first_eid - 1))
                if forced_mid_eid is not None:
                    keep = keep and (mid_idx == (forced_mid_eid - 1))
                mask[..., pair_idx] = keep
            pair_logits = pair_logits.masked_fill(~mask, float("-inf"))

        pred_pair = pair_logits.argmax(dim=-1)
        first_eid = (pred_pair // num_tasks) + 1
        mid_eid = (pred_pair % num_tasks) + 1

        first_weights = None
        mid_weights = None
        routed_logits = pair_logits
        if self.routing_mode == "uniform":
            first_weights = pair_logits.new_full((pair_logits.size(0), num_tasks + 1), 1.0 / num_tasks)
            mid_weights = pair_logits.new_full((pair_logits.size(0), num_tasks + 1), 1.0 / num_tasks)
            first_weights[:, 0] = 0.0
            mid_weights[:, 0] = 0.0
            routed_logits = pair_logits.new_zeros(pair_logits.shape)
        elif self.routing_mode == "weighted_sum":
            routed_logits = pair_logits.float() * self.routing_sharpness
            max_pairs = routed_logits.size(-1)
            if self.routing_topk is not None and self.routing_topk < max_pairs:
                topk_ids = routed_logits.topk(k=self.routing_topk, dim=-1).indices
                keep_mask = torch.zeros_like(routed_logits, dtype=torch.bool)
                keep_mask.scatter_(dim=-1, index=topk_ids, value=True)
                routed_logits = routed_logits.masked_fill(~keep_mask, float("-inf"))
            pair_prob = torch.softmax(routed_logits, dim=-1).view(-1, num_tasks, num_tasks)
            real_first_weights = pair_prob.sum(dim=2)
            real_mid_weights = pair_prob.sum(dim=1)
            null_weights = real_first_weights.new_zeros(real_first_weights.size(0), 1)
            first_weights = torch.cat([null_weights, real_first_weights], dim=1)
            mid_weights = torch.cat([null_weights, real_mid_weights], dim=1)

        self._maybe_print_topk_pair_logits(
            pair_logits=routed_logits,
            dataset_name=dataset_name,
            prompt_previews=prompt_previews,
            first_weights=first_weights,
            mid_weights=mid_weights,
        )

        for sample_first, sample_mid in zip(first_eid.detach().cpu().tolist(), mid_eid.detach().cpu().tolist()):
            self.route_counter[f"first::{self.task_names[sample_first - 1]}"] += 1
            self.route_counter[f"mid::{self.task_names[sample_mid - 1]}"] += 1
            self.route_counter[f"pair::{self.task_names[sample_first - 1]}->{self.task_names[sample_mid - 1]}"] += 1
        return first_eid, mid_eid, first_weights, mid_weights

    def _generation_args(self, gen_kwargs: Dict, max_new_tokens: Optional[int]) -> Dict:
        args = dict(gen_kwargs)
        if "max_new_tokens" not in args and max_new_tokens is not None:
            args["max_new_tokens"] = int(max_new_tokens)
        args.setdefault("eos_token_id", self.tokenizer.eos_token_id)
        args.setdefault("pad_token_id", self.tokenizer.pad_token_id)
        args["do_sample"] = False
        args["temperature"] = 0.0
        args["top_p"] = 1.0
        args["num_beams"] = 1
        return args

    def _decode_outputs(self, output_ids: torch.Tensor, prompt_width: int) -> List[str]:
        return self.tokenizer.batch_decode(
            output_ids[:, prompt_width:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

    @torch.no_grad()
    def generate(self, prompts, max_new_tokens=None, gen_kwargs=None, **kwargs):
        dataset_name = kwargs.pop("dataset_name", None)
        if isinstance(prompts, str):
            prompts = [prompts]
        if gen_kwargs is None:
            gen_kwargs = {}
        if not prompts:
            return []

        self._reset_runtime_cache()
        self._encode_bert_memory(prompts)
        inp = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_seq_len,
        ).to(self.model.device)

        prepass = self.model(
            **inp,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = prepass.hidden_states
        first_vec = self._pool_prompt_vector(hidden_states[self.first_layer_idx], inp["attention_mask"])
        mid_vec = self._pool_prompt_vector(hidden_states[self.middle_layer_idx], inp["attention_mask"])
        first_eid, mid_eid, first_weights, mid_weights = self._route_pair_from_vecs(
            first_vec,
            mid_vec,
            dataset_name=dataset_name,
            prompt_previews=prompts,
        )
        self.cached_first_eid = first_eid
        self.cached_mid_eid = mid_eid
        self.cached_first_weights = first_weights
        self.cached_mid_weights = mid_weights

        args = self._generation_args(gen_kwargs, max_new_tokens)
        prompt_width = inp["input_ids"].shape[1]
        if self.routing_mode in {"weighted_sum", "uniform"}:
            set_layer_range_expert_weights(
                self.model,
                self.first_layer_idx,
                self.middle_layer_idx - 1,
                first_weights,
            )
            set_layer_range_expert_weights(
                self.model,
                self.middle_layer_idx,
                self.num_layers - 1,
                mid_weights,
            )
            out = self.model.generate(**inp, **args)
            return self._decode_outputs(out, prompt_width)

        pair_groups = defaultdict(list)
        for index, pair in enumerate(zip(first_eid.detach().cpu().tolist(), mid_eid.detach().cpu().tolist())):
            pair_groups[pair].append(index)
        outputs = [None] * len(prompts)
        for (group_first_eid, group_mid_eid), indices in pair_groups.items():
            set_layer_range_expert(
                self.model,
                self.first_layer_idx,
                self.middle_layer_idx - 1,
                group_first_eid,
            )
            set_layer_range_expert(
                self.model,
                self.middle_layer_idx,
                self.num_layers - 1,
                group_mid_eid,
            )
            group_idx = torch.tensor(indices, device=inp["input_ids"].device, dtype=torch.long)
            group_inp = {key: value.index_select(0, group_idx) for key, value in inp.items()}
            group_out = self.model.generate(**group_inp, **args)
            group_text = self._decode_outputs(group_out, prompt_width)
            for index, text in zip(indices, group_text):
                outputs[index] = text
        return outputs
