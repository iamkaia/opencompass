"""Reusable raw-data, expert-pair scoring, and full-training implementation.

The command entrypoint remains train_joint_answer_supervision_router.py.
Cache generation imports this module directly instead of importing a CLI file.
"""

import argparse
import importlib
import json
import os
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from opencompass.utils.text_postprocessors import (
    general_postprocess,
)
from model_backbone_specs import (
    get_decoder_layers,
    get_hidden_size,
    get_pre_attn_norm,
    infer_backbone_spec,
)
from opencompass.models.router_moe_components import (
    BertExternalEncoder,
    CompactRouterFeatureEncoder,
    PromptVectorExtractor,
)
from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    load_lora_into_expert,
    patch_llama_with_hard_routed_lora,
    set_all_experts,
    set_layer_range_expert,
)
from task_eval_specs import (
    TASK_EVAL_SPECS,
    TaskEvalSpec,
    apply_postprocessor,
    normalize_boolq_label,
    normalize_qa_text,
    normalize_router_option_label,
    normalize_sst2_label,
    router_option_labels,
)
from router_pair_common import (
    add_joint_loss_arguments,
    build_oracle_debug_summary,
    build_routing_summary,
    compute_pair_losses,
    compute_route_score_stats,
    compute_routing_accuracy_stats,
    init_oracle_debug_accumulator,
    optional_float,
    print_oracle_debug_summary,
    print_routing_summary,
    update_oracle_debug_accumulator,
)


DEFAULT_EXPERT_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]
MCQ_STYLE_TASKS = {"race", "medmcqa", "hellaswag", "piqa", "copa", "siqa"}
BINARY_STYLE_TASKS = {"sst2", "boolq"}
QA_STYLE_TASKS = {"squad2", "squad20", "squad2.0"}
TRANSLATION_STYLE_TASKS = {"iwslt2017"}

_OPENCOMPASS_EVAL_RUNTIME = None

####只是建立 evaluator 實例，例如：AccEvaluator, BleuEvaluator, MedmcqaEvaluator, SQuAD20Evaluator, 避免每次 sample 評分都重建。
def _get_opencompass_eval_runtime():
    global _OPENCOMPASS_EVAL_RUNTIME
    if _OPENCOMPASS_EVAL_RUNTIME is not None:
        return _OPENCOMPASS_EVAL_RUNTIME

    try:
        icl_eval_mod = importlib.import_module("opencompass.openicl.icl_evaluator")
        medmcqa_mod = importlib.import_module("opencompass.datasets.medmcqa")
        squad20_mod = importlib.import_module("opencompass.datasets.squad20")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import OpenCompass evaluator runtime. "
            "Please ensure the training environment can import the same "
            "OpenCompass package used for evaluation."
        ) from exc

    _OPENCOMPASS_EVAL_RUNTIME = {
        "acc": icl_eval_mod.AccEvaluator(),
        "acc_with_details": icl_eval_mod.AccwithDetailsEvaluator(),
        "bleu": icl_eval_mod.BleuEvaluator(),
        "edacc": icl_eval_mod.EDAccEvaluator(),
        "medmcqa": medmcqa_mod.MedmcqaEvaluator(),
        "squad20": squad20_mod.SQuAD20Evaluator(),
    }
    return _OPENCOMPASS_EVAL_RUNTIME


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def discover_tasks(data_root: str, requested_tasks: Optional[Sequence[str]] = None) -> List[str]:
    if requested_tasks:
        tasks = [str(task).strip() for task in requested_tasks if str(task).strip()]
    else:
        tasks = []
        if os.path.isdir(data_root):
            for name in sorted(os.listdir(data_root)):
                full = os.path.join(data_root, name)
                if os.path.isdir(full):
                    tasks.append(name)
        if not tasks:
            tasks = list(DEFAULT_EXPERT_NAMES)
    if not tasks:
        raise ValueError(f"No tasks found under data_root={data_root}")
    return tasks

###這次要拿哪些 expert 進來玩 routing
def discover_expert_names(
    requested_experts: Optional[Sequence[str]],
    all_lora_paths: Dict[str, Optional[str]],
) -> tuple[List[str], str]:
    if requested_experts:
        expert_names = [str(name).strip() for name in requested_experts if str(name).strip()]
        source = "manual"
    else:
        expert_names = [name for name, path in all_lora_paths.items() if path]
        if not expert_names:
            expert_names = list(DEFAULT_EXPERT_NAMES)
            source = "default"
        else:
            source = "inferred_from_lora_paths"
    if not expert_names:
        raise ValueError("No expert names selected.")
    return expert_names, source



# 將各 task 的 jsonl 讀成 router 訓練 sample；每筆保留 prompt、target 與對應的
# expert index。`text`/`source_text` 刻意寫成同一個 canonical prompt，讓 BERT、
# LLM vector extraction 與 evaluator 看到完全相同的輸入字串。
class RouterTrainDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        split: str,
        tasks: Sequence[str],
        task2id: Dict[str, int],
        max_samples: Optional[int] = None,
        seed: int = 42,
    ):
        items: List[Dict] = []
        rng = random.Random(seed)
        for task in tasks:
            path = os.path.join(data_root, task, f"{split}.jsonl")
            if not os.path.exists(path):
                raise FileNotFoundError(f"Missing dataset file: {path}")
            rows = read_jsonl(path)
            rng.shuffle(rows)
            if max_samples is not None:
                rows = rows[: int(max_samples)]
            for row in rows:
                prompt = row.get("text") or row.get("source_text")
                target = row.get("target") or row.get("answer") or row.get("output")
                if not prompt or target is None:
                    raise ValueError(f"Invalid row in {path}: keys={list(row.keys())}")
                prompt = str(prompt)
                items.append(
                    {
                        "task": task,
                        "task_id": task2id[task],
                        # Keep one canonical prompt string so BERT routing,
                        # LLM prompt-vector extraction, and evaluator-side
                        # origin_prompt references all see the same prompt.
                        "text": prompt,
                        "source_text": prompt,
                        "target": str(target),
                        "meta": dict(row),
                    }
                )

        rng.shuffle(items)
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict:
        return self.items[idx]


# 根據 `requested_tasks` 決定本 split 要讀哪些資料，再以 `expert_names` 定義
# `task_id`。sample task 可以多於 expert task；沒有自己的 expert 時使用 -1。
def build_dataset(
    data_root: str,
    split: str,
    requested_tasks: Optional[Sequence[str]],
    expert_names: Sequence[str],
    max_samples: Optional[int],
    seed: int,
) -> tuple[RouterTrainDataset, List[str]]:
    task_names = discover_tasks(data_root, requested_tasks=requested_tasks)
    expert2id = {task: idx for idx, task in enumerate(expert_names)}
    # task_id 會被 self-preserving loss 當成 expert index，因此必須依 expert_names 編號；
    # 不在 expert 集合中的資料標成 -1，代表它沒有可套用的 self expert 監督。
    task2id = {task: expert2id.get(task, -1) for task in task_names}
    dataset = RouterTrainDataset(
        data_root=data_root,
        split=split,
        tasks=task_names,
        task2id=task2id,
        max_samples=max_samples,
        seed=seed,
    )
    return dataset, task_names


@dataclass
class Batch:
    texts: List[str]
    source_texts: List[str]
    targets: List[str]
    metas: List[Dict]
    task_ids: torch.Tensor
    task_names: List[str]


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            source_texts=[x["source_text"] for x in batch],
            targets=[x["target"] for x in batch],
            metas=[x.get("meta", {}) for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            task_names=[x["task"] for x in batch],
        )



####整個moe model本人, 有bert有兩個layer router，還有layer router要放哪一層
class JointAnswerSupervisionRouterModel(nn.Module):
    def __init__(
        self,
        base_model_path: str,
        router_bert_init: str,
        lora_paths: Dict[str, str],
        first_layer_idx: int,
        middle_layer_idx: int,
        router_dim: int,
        dtype: str,
        r: int,
        alpha: int,
        expert_names: Sequence[str],
        router_pooling: str,
        router_pooling_last_k: int,
    ):
        super().__init__()
        self.expert_names = list(expert_names)
        ####建立 expert name 到 index 的 mapping：注意這裡是 router output index，不是 LoRA expert id。所以要注意對齊
        self.expert2id = {task: idx for idx, task in enumerate(self.expert_names)}
        print(self.expert2id)
        #### 所以目前應該都是float16, 之後就可以用bfloat16補一下看看
        torch_dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch_dtype,
            device_map=None,
        )
        self.model.eval()
        ####凍結 base LLM。
        for p in self.model.parameters():
            p.requires_grad = False
        ####判斷這個模型是 llama/qwen/mistral/gemma 哪種架構，方便後面取得 decoder layers。
        self.backbone_spec = infer_backbone_spec(self.model)

        ####把 LLM 改造成 hard-routed LoRA 模型
        ####之後改成weighted_sum是只有在算loss的時候，而不是在建cached的時候，就是改loss而已
        self.model = patch_llama_with_hard_routed_lora(
            self.model,
            num_experts=1 + len(self.expert_names),
            r=r,
            alpha=alpha,
        )
        for p in self.model.parameters():
            p.requires_grad = False

        self.first_layer_idx = int(first_layer_idx)
        self.middle_layer_idx = int(middle_layer_idx)
        self.num_layers = len(get_decoder_layers(self.model, spec=self.backbone_spec))
        ####把 task name 對應到 LoRA expert slot。
        ####為甚麼+1, 因為0是null expert
        ####只訓練 3 個 expert 是可以的：pair output 空間會變成 3 x 3。
        ####但訓練、建 cache 與 runtime 掛載時必須使用同一份 expert_names；
        ####若載入 expert 數量不同的舊 pair classifier checkpoint，輸出維度不相容。
        '''
        self.expert2id = {
        "iwslt2017": 0,
        "medmcqa": 1,
        "race": 2,
        "squad2": 3,
        "sst2": 4,
        }
        會變成
        self.task_to_expert_id = {
        "iwslt2017": 1,
        "medmcqa": 2,
        "race": 3,
        "squad2": 4,
        "sst2": 5,
        }
        '''
        self.task_to_expert_id = {task: self.expert2id[task] + 1 for task in self.expert_names}

        for task in self.expert_names:
            if task not in lora_paths:
                raise KeyError(f"Missing LoRA path for task: {task}")
            ####把這個 task 的 LoRA adapter 權重塞進指定 expert slot。
            ####目前是 hard routing：每個 slot 載入一個固定 LoRA。
            '''
            load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])

            把每個 task 的 LoRA adapter 載進對應 expert slot
            '''
            load_lora_into_expert(self.model, lora_paths[task], self.task_to_expert_id[task])
        
        ###從 LLM 指定 layer 抓 first_vec / mid_vec, 就是hidden states
        self.vector_extractor = PromptVectorExtractor(
            model=self.model,
            first_layer_idx=self.first_layer_idx,
            middle_layer_idx=self.middle_layer_idx,
            pooling=router_pooling,
            pooling_last_k=router_pooling_last_k,
        )

        ###Question: 這個llama跟bert的hidden size會是什麼東西？我應該會看到什麼東西？
        self.bert = BertExternalEncoder(router_bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        ####名稱雖然叫 llama_hidden_size，實際意義是目前 base LLM 的 hidden size。
        ####換成 Qwen/Gemma 時會讀目前模型的 hidden size；前提是 backbone spec 與
        ####hard-routed LoRA patch 支援該模型結構，不能只靠這一行保證相容。
        llama_hidden_size = get_hidden_size(self.model)

        self.num_pairs = len(self.expert_names) * len(self.expert_names)
        ###抽出router_feature
        self.router_first = CompactRouterFeatureEncoder(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
        )
        self.router_mid = CompactRouterFeatureEncoder(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
        )
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, self.num_pairs),
        )

    ####是為了載入已訓練好的 router checkpoint，現在還沒認真看
    def load_router_weights(self, ckpt_dir: str):
        ckpt_expert_names = None
        cfg_path = os.path.join(ckpt_dir, "router_config.json")
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                ckpt_cfg = json.load(f)
            ckpt_expert_names = ckpt_cfg.get("expert_names") or ckpt_cfg.get("task_names")
        state = torch.load(os.path.join(ckpt_dir, "router_heads.pt"), map_location="cpu")

        def _load_router_with_task_remap(module: nn.Module, saved_state: Dict[str, torch.Tensor], which: str):
            current_state = module.state_dict()
            loaded = {}

            # New pair-joint checkpoints store only the shared encoder weights
            # for router_first/router_mid, while older checkpoints may come
            # from CompactCrossAttentionRouter and include a task classifier
            # head. The feature encoder does not own classifier params, so we
            # keep only the overlapping keys here.
            if not ckpt_expert_names or list(ckpt_expert_names) == list(self.expert_names):
                for key, value in current_state.items():
                    if key in saved_state:
                        loaded[key] = saved_state[key]
                module.load_state_dict(loaded, strict=False)
                skipped = sorted(set(saved_state.keys()) - set(current_state.keys()))
                if skipped:
                    print(
                        f"[INFO] skipped incompatible {which} keys when loading "
                        f"legacy checkpoint: {skipped}"
                    )
                return

            for key, value in current_state.items():
                if key not in saved_state:
                    continue
                src = saved_state[key]
                if key == "classifier.weight":
                    remapped = value.clone()
                    for new_idx, task in enumerate(self.expert_names):
                        if task in ckpt_expert_names:
                            old_idx = ckpt_expert_names.index(task)
                            remapped[new_idx] = src[old_idx]
                    loaded[key] = remapped
                elif key == "classifier.bias":
                    remapped = value.clone()
                    for new_idx, task in enumerate(self.expert_names):
                        if task in ckpt_expert_names:
                            old_idx = ckpt_expert_names.index(task)
                            remapped[new_idx] = src[old_idx]
                    loaded[key] = remapped
                else:
                    loaded[key] = src
            module.load_state_dict(loaded, strict=False)
            print(
                f"[INFO] remapped {which} checkpoint expert heads from "
                f"{ckpt_expert_names} to {self.expert_names}"
            )

        if "pair_first_encoder" in state:
            self.router_first.load_state_dict(state["pair_first_encoder"])
        else:
            _load_router_with_task_remap(self.router_first, state["router_first"], which="router_first")
        if "pair_mid_encoder" in state:
            self.router_mid.load_state_dict(state["pair_mid_encoder"])
        else:
            _load_router_with_task_remap(self.router_mid, state["router_mid"], which="router_mid")
        if "pair_classifier" in state:
            self.pair_classifier.load_state_dict(state["pair_classifier"], strict=False)
        if "bert_encoder" in state:
            self.bert.load_state_dict(state["bert_encoder"], strict=False)

    ####控制哪些 module 要訓練。
    def set_trainable(self, freeze_bert: bool = True, freeze_router_first: bool = False, freeze_router_mid: bool = False):
        """Train router components while keeping the shared BERT encoder fixed."""
        for p in self.router_first.parameters():
            p.requires_grad = not freeze_router_first
        for p in self.router_mid.parameters():
            p.requires_grad = not freeze_router_mid
        for p in self.pair_classifier.parameters():
            p.requires_grad = not (freeze_router_first and freeze_router_mid)
        for p in self.bert.parameters():
            p.requires_grad = False

    ####抽 first_vec/mid_vec 時刻意使用 NULL_EXPERT，也就是只從 base model 抽 prompt feature。
    ####這是合理的 router input contract：feature 不先洩漏候選 expert 的輸出，
    ####真正哪個 expert pair 較好則由後面的 loss_matrix 監督學習。
    ####runtime 也必須用相同的 NULL_EXPERT feature extraction 才能對齊訓練。
    @torch.no_grad()
    def extract_prompt_vectors(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        set_all_experts(self.model, NULL_EXPERT_ID)
        return self.vector_extractor.extract(input_ids=input_ids, attention_mask=attention_mask)

    #####算分數的function, 主要接參數，剩下看分到哪個function
    @torch.no_grad()
    def score_all_route_pairs(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        targets: Sequence[str],
        source_texts: Sequence[str],
        task_names: Sequence[str],
        llm_tokenizer,
        score_mode: str = "official_eval_aligned_generation",
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        score_mode = str(score_mode)
        if score_mode == "official_generation_only":
            return self._score_all_route_pairs_generation_only(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                targets=targets,
                source_texts=source_texts,
                task_names=task_names,
                llm_tokenizer=llm_tokenizer,
            )
        if score_mode == "official_eval_aligned_generation":
            return self._score_all_route_pairs_official_eval_aligned_generation(
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                targets=targets,
                source_texts=source_texts,
                task_names=task_names,
                llm_tokenizer=llm_tokenizer,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
        raise ValueError(f"Unknown score_mode: {score_mode}")

    @torch.no_grad()
    def score_all_single_experts(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        targets: Sequence[str],
        source_texts: Sequence[str],
        task_names: Sequence[str],
        llm_tokenizer,
        score_mode: str = "official_eval_aligned_generation",
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Score T candidates where one LoRA expert is active on every layer."""
        score_mode = str(score_mode)
        if score_mode not in {"official_generation_only", "official_eval_aligned_generation"}:
            raise ValueError(f"Unknown score_mode: {score_mode}")
        batch_size = prompt_input_ids.size(0)
        device = prompt_input_ids.device
        num_experts = len(self.expert_names)
        loss_vector = torch.empty(batch_size, num_experts, dtype=torch.float32, device=device)
        correct_vector = torch.zeros(batch_size, num_experts, dtype=torch.bool, device=device)
        prediction_vectors: List[List[str]] = [["" for _ in range(num_experts)] for _ in range(batch_size)]

        if score_mode == "official_generation_only":
            generation_indices = list(range(batch_size))
            first_token_indices = []
        else:
            generation_indices = [
                idx for idx, task_name in enumerate(task_names) if _task_uses_generation_evaluator(task_name)
            ]
            first_token_indices = [
                idx for idx, task_name in enumerate(task_names) if not _task_uses_generation_evaluator(task_name)
            ]
        generation_index_tensor = (
            torch.tensor(generation_indices, dtype=torch.long, device=device) if generation_indices else None
        )
        first_token_index_tensor = (
            torch.tensor(first_token_indices, dtype=torch.long, device=device) if first_token_indices else None
        )

        for expert_idx, expert_name in enumerate(self.expert_names):
            set_all_experts(self.model, self.task_to_expert_id[expert_name])
            candidate_loss = torch.empty(batch_size, dtype=torch.float32, device=device)
            if first_token_index_tensor is not None:
                ids = prompt_input_ids.index_select(0, first_token_index_tensor)
                mask = prompt_attention_mask.index_select(0, first_token_index_tensor)
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False, return_dict=True).logits
                subset_tasks = [task_names[idx] for idx in first_token_indices]
                subset_targets = [targets[idx] for idx in first_token_indices]
                losses, correct, predictions = compute_first_token_target_scores(
                    logits=logits,
                    prompt_attention_mask=mask,
                    targets=subset_targets,
                    task_names=subset_tasks,
                    tokenizer=llm_tokenizer,
                )
                candidate_loss.index_copy_(0, first_token_index_tensor, losses.to(device=device, dtype=torch.float32))
                correct_vector[:, expert_idx].index_copy_(
                    0, first_token_index_tensor, correct.to(device=device, dtype=torch.bool)
                )
                for local_idx, sample_idx in enumerate(first_token_indices):
                    prediction_vectors[sample_idx][expert_idx] = str(predictions[local_idx])
            if generation_index_tensor is not None:
                subset_tasks = [task_names[idx] for idx in generation_indices]
                predictions = self.generate_under_current_pair(
                    prompt_input_ids=prompt_input_ids.index_select(0, generation_index_tensor),
                    prompt_attention_mask=prompt_attention_mask.index_select(0, generation_index_tensor),
                    tokenizer=llm_tokenizer,
                    task_names=subset_tasks,
                )
                subset_targets = [targets[idx] for idx in generation_indices]
                subset_sources = [source_texts[idx] for idx in generation_indices]
                losses = compute_generated_dataset_scores(
                    predictions=predictions,
                    targets=subset_targets,
                    task_names=subset_tasks,
                    source_texts=subset_sources,
                ).to(device=device, dtype=torch.float32)
                candidate_loss.index_copy_(0, generation_index_tensor, losses)
                correct_vector[:, expert_idx].index_copy_(0, generation_index_tensor, losses <= 1e-6)
                for local_idx, sample_idx in enumerate(generation_indices):
                    prediction_vectors[sample_idx][expert_idx] = str(predictions[local_idx])
            loss_vector[:, expert_idx] = candidate_loss

        set_all_experts(self.model, NULL_EXPERT_ID)
        self.last_route_correct_vector = correct_vector
        self.last_route_prediction_vectors = prediction_vectors
        return loss_vector

    ####先配置每個 sample、每個 (first expert, mid expert) pair 的計分結果：
    ####loss_matrix[B, T, T] 存 cost，correct_matrix[B, T, T] 存是否答對，
    ####prediction_matrices[B][T][T] 存該 pair 的預測文字，稍後逐 pair 填入。
    def _init_score_buffers(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, List[List[List[str]]]]:
        num_tasks = len(self.expert_names)
        loss_matrix = torch.empty(batch_size, num_tasks, num_tasks, dtype=torch.float32, device=device)
        correct_matrix = torch.zeros(batch_size, num_tasks, num_tasks, dtype=torch.bool, device=device)
        prediction_matrices: List[List[List[str]]] = [
            [["" for _ in range(num_tasks)] for _ in range(num_tasks)]
            for _ in range(batch_size)
        ]
        return loss_matrix, correct_matrix, prediction_matrices

    def _set_active_expert_pair(self, first_task: str, mid_task: str) -> None:
        first_eid = self.task_to_expert_id[first_task]
        mid_eid = self.task_to_expert_id[mid_task]
        set_all_experts(self.model, NULL_EXPERT_ID)
        set_layer_range_expert(self.model, self.first_layer_idx, self.middle_layer_idx - 1, first_eid)
        set_layer_range_expert(self.model, self.middle_layer_idx, self.num_layers - 1, mid_eid)

    ####設定args的matrix指到哪一個matrix
    def _finalize_score_state(
        self,
        correct_matrix: torch.Tensor,
        prediction_matrices: List[List[List[str]]],
    ) -> None:
        set_all_experts(self.model, NULL_EXPERT_ID)
        self.last_route_correct_matrix = correct_matrix
        self.last_route_prediction_matrices = prediction_matrices

    #####所有task都call opencompass的算分方法
    def _score_all_route_pairs_generation_only(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        targets: Sequence[str],
        source_texts: Sequence[str],
        task_names: Sequence[str],
        llm_tokenizer,
    ) -> torch.Tensor:
        batch_size = prompt_input_ids.size(0)
        device = prompt_input_ids.device
        generation_indices = list(range(batch_size))
        generation_index_tensor = torch.arange(batch_size, dtype=torch.long, device=device)
        loss_matrix, correct_matrix, prediction_matrices = self._init_score_buffers(
            batch_size=batch_size,
            device=device,
        )

        for first_tid, first_task in enumerate(self.expert_names):
            for mid_tid, mid_task in enumerate(self.expert_names):
                self._set_active_expert_pair(first_task=first_task, mid_task=mid_task)
                combo_loss = torch.empty(batch_size, dtype=torch.float32, device=device)
                generated_texts = self.generate_under_current_pair(
                    prompt_input_ids=prompt_input_ids.index_select(0, generation_index_tensor),
                    prompt_attention_mask=prompt_attention_mask.index_select(0, generation_index_tensor),
                    tokenizer=llm_tokenizer,
                    task_names=task_names,
                )
                generation_loss = compute_generated_dataset_scores(
                    predictions=generated_texts,
                    targets=targets,
                    task_names=task_names,
                    source_texts=source_texts,
                ).to(device=device, dtype=torch.float32)
                combo_loss.index_copy_(0, generation_index_tensor, generation_loss)
                correct_matrix[:, first_tid, mid_tid].index_copy_(
                    0,
                    generation_index_tensor,
                    generation_loss <= 1e-6,
                )
                for local_idx, sample_idx in enumerate(generation_indices):
                    prediction_matrices[sample_idx][first_tid][mid_tid] = str(generated_texts[local_idx])
                loss_matrix[:, first_tid, mid_tid] = combo_loss

        self._finalize_score_state(
            correct_matrix=correct_matrix,
            prediction_matrices=prediction_matrices,
        )
        return loss_matrix


    ###除了iwslt2017跟squad20都用token logits的算法, 要講解一下我看不懂
    def _score_all_route_pairs_official_eval_aligned_generation(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        targets: Sequence[str],
        source_texts: Sequence[str],
        task_names: Sequence[str],
        llm_tokenizer,
        input_ids: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = prompt_input_ids.size(0)
        device = prompt_input_ids.device
        loss_matrix, correct_matrix, prediction_matrices = self._init_score_buffers(
            batch_size=batch_size,
            device=device,
        )
        generation_indices = [
            idx for idx, task_name in enumerate(task_names)
            if _task_uses_generation_evaluator(task_name)
        ]
        first_token_indices = [
            idx for idx, task_name in enumerate(task_names)
            if not _task_uses_generation_evaluator(task_name)
        ]

        first_token_index_tensor = (
            torch.tensor(first_token_indices, dtype=torch.long, device=device)
            if first_token_indices
            else None
        )
        generation_index_tensor = (
            torch.tensor(generation_indices, dtype=torch.long, device=device)
            if generation_indices
            else None
        )

        for first_tid, first_task in enumerate(self.expert_names):
            for mid_tid, mid_task in enumerate(self.expert_names):
                self._set_active_expert_pair(first_task=first_task, mid_task=mid_task)
                combo_loss = torch.empty(batch_size, dtype=torch.float32, device=device)

                if first_token_index_tensor is not None:
                    first_token_prompt_ids = prompt_input_ids.index_select(0, first_token_index_tensor)
                    first_token_prompt_mask = prompt_attention_mask.index_select(0, first_token_index_tensor)
                    first_token_logits = self.model(
                        input_ids=first_token_prompt_ids,
                        attention_mask=first_token_prompt_mask,
                        use_cache=False,
                        return_dict=True,
                    ).logits
                    first_token_tasks = [task_names[idx] for idx in first_token_indices]
                    first_token_targets = [targets[idx] for idx in first_token_indices]
                    first_token_loss, first_token_correct, first_token_predictions = compute_first_token_target_scores(
                        logits=first_token_logits,
                        prompt_attention_mask=first_token_prompt_mask,
                        targets=first_token_targets,
                        task_names=first_token_tasks,
                        tokenizer=llm_tokenizer,
                    )
                    combo_loss.index_copy_(
                        0,
                        first_token_index_tensor,
                        first_token_loss.to(device=device, dtype=torch.float32),
                    )
                    correct_matrix[:, first_tid, mid_tid].index_copy_(
                        0,
                        first_token_index_tensor,
                        first_token_correct.to(device=device, dtype=torch.bool),
                    )
                    for local_idx, sample_idx in enumerate(first_token_indices):
                        prediction_matrices[sample_idx][first_tid][mid_tid] = str(first_token_predictions[local_idx])

                if generation_index_tensor is not None:
                    generation_tasks = [task_names[idx] for idx in generation_indices]
                    generated_texts = self.generate_under_current_pair(
                        prompt_input_ids=prompt_input_ids.index_select(0, generation_index_tensor),
                        prompt_attention_mask=prompt_attention_mask.index_select(0, generation_index_tensor),
                        tokenizer=llm_tokenizer,
                        task_names=generation_tasks,
                    )
                    generation_targets = [targets[idx] for idx in generation_indices]
                    generation_source_texts = [source_texts[idx] for idx in generation_indices]
                    generation_loss = compute_generated_dataset_scores(
                        predictions=generated_texts,
                        targets=generation_targets,
                        task_names=generation_tasks,
                        source_texts=generation_source_texts,
                    ).to(device=device, dtype=torch.float32)
                    combo_loss.index_copy_(0, generation_index_tensor, generation_loss)
                    correct_matrix[:, first_tid, mid_tid].index_copy_(
                        0,
                        generation_index_tensor,
                        generation_loss <= 1e-6,
                    )
                    for local_idx, sample_idx in enumerate(generation_indices):
                        prediction_matrices[sample_idx][first_tid][mid_tid] = str(generated_texts[local_idx])

                loss_matrix[:, first_tid, mid_tid] = combo_loss

        self._finalize_score_state(
            correct_matrix=correct_matrix,
            prediction_matrices=prediction_matrices,
        )
        return loss_matrix

    ####在目前已設定好的 expert pair 下 generate。
    @torch.no_grad()
    def generate_under_current_pair(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        tokenizer,
        task_names: Sequence[str],
    ) -> List[str]:
        max_new_tokens = max(task_max_new_tokens(task) for task in task_names)
        outputs = self.model.generate(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            max_new_tokens=max_new_tokens,
            # `do_sample=False` 且 `num_beams=1` 表示每一步直接取最高機率 token：
            # 這裡的 generation model 輸出是 greedy decoding，不是 sampling 或 beam search。
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        prompt_seq_len = prompt_input_ids.size(1)
        texts = []
        for idx in range(outputs.size(0)):
            new_tokens = outputs[idx, prompt_seq_len:]
            texts.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return texts

    def forward_router(
        self,
        bert_input_ids: torch.Tensor,
        bert_attention_mask: torch.Tensor,
        bert_token_type_ids: Optional[torch.Tensor],
        first_vec: torch.Tensor,
        mid_vec: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ####BERT 回傳形狀為 [B, S, H_bert]：B 是 batch size，S 是 token 長度，
        ####H_bert 是每個 token 的 BERT hidden feature 維度。
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        ####它做的事情是：用 first_vec 當 query, 用 BERT 倒數兩層 hidden states 當 memory, 做一個 cross-attention 風格的 feature extraction, 輸出一個 feature vector
        first_feat = self.router_first(
            llama_vec=first_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        mid_feat = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        ####把兩個 feature 串起來。因為現在不是分開做 first/mid task classification，而是要直接做 pair classification。所以最自然的做法就是：先分別抽 first 與 mid feature, 再把它們拼在一起，讓最後 classifier 看兩邊的聯合資訊
        pair_feat = torch.cat([first_feat, mid_feat], dim=-1)
        ####輸出[B, num_pairs]
        pair_logits = self.pair_classifier(pair_feat)
        ####reshape成[B, T, T], T是expert數量
        ####pair_prob[b, i, j] = 第 b 筆 sample 選 i->j 的機率
        pair_prob = torch.softmax(pair_logits, dim=-1).view(pair_logits.size(0), len(self.expert_names), len(self.expert_names))
        ###P(first = i) = sum_j P(first = i, mid = j), sum(dim=2) 是把 mid expert 維度加總。
        ###pair_prob 對 mid 維度加總後得到 first expert 的 marginal probability；
        ###對它取 log 可當成 first expert 的 logits 提供額外 loss/metrics 使用。
        ###`clamp_min(1e-12)` 只是在取 log 前避免機率數值下溢成 0 而得到 -inf。
        first_logits = torch.log(pair_prob.sum(dim=2).clamp_min(1e-12))
        ####同上
        mid_logits = torch.log(pair_prob.sum(dim=1).clamp_min(1e-12))

        '''
        pair_logits:
        真正 joint pair classifier 的輸出，用來訓練 pair route。

        first_logits:
        從 pair distribution marginal 出來的 first expert logits。

        mid_logits:
        從 pair distribution marginal 出來的 mid expert logits。

        '''
        return pair_logits, first_logits, mid_logits


# 同時建立兩種 LLM 輸入：`prompt_input_ids` 只含 prompt，供抽 feature、first-token
# scoring 與 generation 使用；`input_ids`/`labels` 是 prompt+target 的 teacher-forcing
# 格式，其中 prompt 和 padding 的 label 設為 -100，代表不納入 token loss。
def build_lm_batch(
    tokenizer,
    prompts: Sequence[str],
    targets: Sequence[str],
    max_length: int,
    add_eos_to_target: bool,
) -> Dict[str, torch.Tensor]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("Tokenizer pad_token_id is required.")

    prompt_ids_list = []
    target_ids_list = []
    for prompt, target in zip(prompts, targets):
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(target, add_special_tokens=False)
        if add_eos_to_target and tokenizer.eos_token_id is not None:
            target_ids = target_ids + [tokenizer.eos_token_id]
        prompt_ids_list.append(prompt_ids)
        target_ids_list.append(target_ids)

    input_ids = []
    labels = []
    attention_masks = []
    prompt_input_ids = []
    prompt_attention_masks = []

    max_prompt_len = min(max((len(x) for x in prompt_ids_list), default=1), max_length)
    max_full_len = 1
    full_seqs = []
    full_labels = []
    for prompt_ids, target_ids in zip(prompt_ids_list, target_ids_list):
        # TODO: 若 prompt 超過 max_length，目前會從尾端截掉內容，可能連題目尾端的
        # Answer: 或 target 一起移除；尚未以實際 tokenizer 統計是否超過 768 token。
        full_ids = (prompt_ids + target_ids)[:max_length]
        usable_prompt_len = min(len(prompt_ids), len(full_ids))
        seq_labels = [-100] * usable_prompt_len + full_ids[usable_prompt_len:]
        full_seqs.append(full_ids)
        full_labels.append(seq_labels)
        max_full_len = max(max_full_len, len(full_ids))

    for prompt_ids, full_ids, seq_labels in zip(prompt_ids_list, full_seqs, full_labels):
        prompt_ids = prompt_ids[:max_prompt_len]
        prompt_pad = max_prompt_len - len(prompt_ids)
        prompt_input_ids.append([pad_id] * prompt_pad + prompt_ids)
        prompt_attention_masks.append([0] * prompt_pad + [1] * len(prompt_ids))

        pad_len = max_full_len - len(full_ids)
        input_ids.append(full_ids + [pad_id] * pad_len)
        labels.append(seq_labels + [-100] * pad_len)
        attention_masks.append([1] * len(full_ids) + [0] * pad_len)

    return {
        "prompt_input_ids": torch.tensor(prompt_input_ids, dtype=torch.long),
        "prompt_attention_mask": torch.tensor(prompt_attention_masks, dtype=torch.long),
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }

def _target_text_for_first_token_score(task_name: str, target: str) -> str:
    task_name = str(task_name)
    if task_name == "sst2":
        normalized = normalize_sst2_label(target)
        if normalized == "1":
            return "positive"
        if normalized == "0":
            return "negative"
    return str(target)


def _first_target_token_id_from_text(tokenizer, task_name: str, target: str) -> Optional[int]:
    target_text = _target_text_for_first_token_score(task_name, target)
    ids = tokenizer.encode(target_text, add_special_tokens=False)
    return ids[0] if ids else None


# Legacy implementation kept here for reference only.
# def compute_first_token_target_scores(
#     logits: torch.Tensor,
#     prompt_attention_mask: torch.Tensor,
#     targets: Sequence[str],
#     task_names: Sequence[str],
#     tokenizer,
# ) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
#     positions = torch.arange(prompt_attention_mask.size(1), device=prompt_attention_mask.device)
#     last_prompt_pos = (prompt_attention_mask.to(torch.long) * positions).max(dim=1).values
#     batch_idx = torch.arange(logits.size(0), device=logits.device)
#     answer_logits = logits[batch_idx, last_prompt_pos, :]
#     full_log_probs = torch.log_softmax(answer_logits, dim=-1)
#     pred_token_ids = answer_logits.argmax(dim=-1)
#     pred_tokens = tokenizer.batch_decode(pred_token_ids.unsqueeze(1), skip_special_tokens=True)
#
#     costs: List[torch.Tensor] = []
#     correct_flags: List[torch.Tensor] = []
#     predictions: List[str] = []
#     for idx, (task_name, target) in enumerate(zip(task_names, targets)):
#         target_token_id = _first_target_token_id_from_text(tokenizer, task_name, target)
#         if target_token_id is None:
#             costs.append(torch.zeros((), dtype=torch.float32, device=logits.device))
#             correct_flags.append(torch.zeros((), dtype=torch.bool, device=logits.device))
#             predictions.append(str(pred_tokens[idx]).strip())
#             continue
#         costs.append((-full_log_probs[idx, target_token_id]).to(torch.float32))
#         correct_flags.append((pred_token_ids[idx] == int(target_token_id)).to(torch.bool))
#         predictions.append(str(pred_tokens[idx]).strip())
#
#     return (
#         torch.stack(costs, dim=0).to(torch.float32),
#         torch.stack(correct_flags, dim=0).to(torch.bool),
#         predictions,
#     )


def _single_token_option_candidate_ids(tokenizer, option_text: str) -> List[int]:
    candidate_ids: List[int] = []
    seen = set()
    for variant in (str(option_text), f" {option_text}", f"\n{option_text}"):
        token_ids = tokenizer.encode(variant, add_special_tokens=False)
        if len(token_ids) != 1:
            continue
        token_id = int(token_ids[0])
        if token_id in seen:
            continue
        seen.add(token_id)
        candidate_ids.append(token_id)
    return candidate_ids


def _option_label_log_probs(
    log_probs: torch.Tensor,
    tokenizer,
    task_name: str,
) -> Optional[Dict[str, torch.Tensor]]:
    option_labels = router_option_labels(task_name)
    if not option_labels:
        return None

    option_log_probs: Dict[str, torch.Tensor] = {}
    for option_label in option_labels:
        candidate_ids = _single_token_option_candidate_ids(tokenizer, option_label)
        if not candidate_ids:
            continue
        option_log_probs[str(option_label)] = torch.logsumexp(
            log_probs[torch.tensor(candidate_ids, device=log_probs.device, dtype=torch.long)],
            dim=0,
        )
    return option_log_probs or None


def compute_first_token_target_scores(
    logits: torch.Tensor,
    prompt_attention_mask: torch.Tensor,
    targets: Sequence[str],
    task_names: Sequence[str],
    tokenizer,
) -> tuple[torch.Tensor, torch.Tensor, List[str]]:
    positions = torch.arange(prompt_attention_mask.size(1), device=prompt_attention_mask.device)
    last_prompt_pos = (prompt_attention_mask.to(torch.long) * positions).max(dim=1).values
    batch_idx = torch.arange(logits.size(0), device=logits.device)
    answer_logits = logits[batch_idx, last_prompt_pos, :]
    full_log_probs = torch.log_softmax(answer_logits, dim=-1)
    # 非 generation task 不會展開生成序列；這裡只用下一個 token 的 argmax
    # 作為單一步驟 greedy 預測，並以 gold first token 的 full-vocab NLL 當 cost。
    pred_token_ids = answer_logits.argmax(dim=-1)
    pred_tokens = tokenizer.batch_decode(pred_token_ids.unsqueeze(1), skip_special_tokens=True)

    costs: List[torch.Tensor] = []
    correct_flags: List[torch.Tensor] = []
    predictions: List[str] = []
    for idx, (task_name, target) in enumerate(zip(task_names, targets)):
        option_log_probs = _option_label_log_probs(
            log_probs=full_log_probs[idx],
            tokenizer=tokenizer,
            task_name=str(task_name),
        )
        if option_log_probs is not None:
            gold_label = normalize_router_option_label(task_name, target)
            if gold_label in option_log_probs:
                pred_label = max(
                    option_log_probs.items(),
                    key=lambda item: float(item[1].detach().item()),
                )[0]
                costs.append((-option_log_probs[gold_label]).to(torch.float32))
                correct_flags.append(torch.tensor(pred_label == gold_label, dtype=torch.bool, device=logits.device))
                predictions.append(pred_label)
                continue

        target_token_id = _first_target_token_id_from_text(tokenizer, task_name, target)
        if target_token_id is None:
            costs.append(torch.zeros((), dtype=torch.float32, device=logits.device))
            correct_flags.append(torch.zeros((), dtype=torch.bool, device=logits.device))
            predictions.append(str(pred_tokens[idx]).strip())
            continue
        costs.append((-full_log_probs[idx, target_token_id]).to(torch.float32))
        correct_flags.append((pred_token_ids[idx] == int(target_token_id)).to(torch.bool))
        predictions.append(str(pred_tokens[idx]).strip())

    return (
        torch.stack(costs, dim=0).to(torch.float32),
        torch.stack(correct_flags, dim=0).to(torch.bool),
        predictions,
    )


def _task_uses_generation_evaluator(task_name: str) -> bool:
    task_name = str(task_name)
    return task_name in QA_STYLE_TASKS or task_name in TRANSLATION_STYLE_TASKS


def compute_official_evaluator_sample_score(
    prediction: str,
    target: str,
    task_name: str,
    source_text: Optional[str] = None,
) -> float:
    ###lazy-load：第一次真的需要 evaluator 時才 import/建立，後續 sample 重用同一組實例。
    runtime = _get_opencompass_eval_runtime()
    task = str(task_name)
    #####不同 task 有不同規則，/home/kaia/opencompass/task_eval_specs.py, 例如：race：選項抽取 + accuracy, squad2：SQuAD evaluator, iwslt2017：BLEU
    spec: Optional[TaskEvalSpec] = TASK_EVAL_SPECS.get(task)
    if spec is None:
        pred = normalize_qa_text(prediction)
        gold = normalize_qa_text(target)
        result = runtime["acc"].score([pred], [gold])
        return 1.0 - float(result["accuracy"]) / 100.0
    ####prediction 後處理, 例如選擇題可能只抽出 A/B/C/D
    processed_prediction = apply_postprocessor([prediction],
                                               spec.pred_postprocessor)[0]
    prediction_for_eval = prediction if spec.use_raw_prediction else processed_prediction
    if spec.reference_adapter is not None:
        reference = spec.reference_adapter(str(source_text or ""), target)
    elif task == "sst2":
        reference = normalize_sst2_label(target)
    elif task == "boolq":
        reference = normalize_boolq_label(target)
    else:
        reference = target

    #####依 task 規格組出 evaluator 需要的參數，例如 references、test_set 或
    #####origin_prompt；不同 benchmark 的官方評分介面需要的欄位不完全相同。
    score_kwargs = {}
    if spec.extra_score_kwargs_builder is not None:
        score_kwargs.update(
            spec.extra_score_kwargs_builder(str(source_text or ""), target))
    if "references" not in score_kwargs:
        score_kwargs["references"] = [reference]

    if spec.dataset_postprocessor is not None and "references" in score_kwargs:
        refs = score_kwargs["references"]
        if refs and isinstance(refs[0], list):
            refs = [apply_postprocessor(ref_list, spec.dataset_postprocessor) for ref_list in refs]
        else:
            refs = apply_postprocessor(refs, spec.dataset_postprocessor)
        score_kwargs["references"] = refs

    ###這步就是evaluator打分的部份
    result = runtime[spec.evaluator_key].score(
        predictions=[prediction_for_eval], **score_kwargs)
    metric_name = spec.score_family if spec.score_family in result else None
    if metric_name is None:
        metric_name = "accuracy" if "accuracy" in result else "score"
    
    ###放入loss_matrix的東西，就是cost = 1.0 - float(result[metric_name]) / 100.0
    ###目前這條 generation evaluator contract 假設官方 metric 的範圍是 0~100，
    ###因此用 cost = 1 - score / 100，分數越高 cost 越低。
    ###若未來加入回傳範圍不同的新 evaluator，必須先在 task spec 統一尺度再使用此式。
    return 1.0 - float(result[metric_name]) / 100.0


#####generation 最多新增幾個 token 由這裡依 sample task 決定，不是 OpenCompass
#####dataset config 決定。同一個 batch 混有多種 task 時，呼叫端會取其中最大值。
def task_max_new_tokens(task_name: str) -> int:
    task_name = str(task_name)
    if task_name in MCQ_STYLE_TASKS:
        return 4
    if task_name in BINARY_STYLE_TASKS:
        return 4
    if task_name in QA_STYLE_TASKS:
        return 32
    if task_name in TRANSLATION_STYLE_TASKS:
        return 128
    return 32


def compute_generated_dataset_scores(
    predictions: Sequence[str],
    targets: Sequence[str],
    task_names: Sequence[str],
    source_texts: Optional[Sequence[str]] = None,
) -> torch.Tensor:
    scores = []
    if source_texts is None:
        source_texts = [""] * len(predictions)
    ###對 batch 每一筆 sample, 用它自己的 task evaluator 打分, 回傳一個 cost tensor, 分數的部份可能會有問題, 因為不確定是不是範圍都是(1~100)：sst2跟medmcqa很有可能有問題，已知sst2只有全對跟全錯
    for pred, target, task, source_text in zip(predictions, targets, task_names, source_texts):
        scores.append(
            compute_official_evaluator_sample_score(
                prediction=pred,
                target=target,
                task_name=task,
                source_text=source_text,
            ))
    return torch.tensor(scores, dtype=torch.float32)


def save_runtime_compatible_router_bundle(
    model: JointAnswerSupervisionRouterModel,
    bert_tokenizer,
    output_dir: str,
    state: Dict,
    router_config: Dict,
):
    os.makedirs(output_dir, exist_ok=True)
    torch.save(state, os.path.join(output_dir, "router_heads.pt"))
    save_json(router_config, os.path.join(output_dir, "router_config.json"))

    encoder_dir = os.path.join(output_dir, "encoder")
    model.bert.encoder.save_pretrained(encoder_dir)
    bert_tokenizer.save_pretrained(encoder_dir)


@torch.no_grad()
def evaluate(
    model: JointAnswerSupervisionRouterModel,
    loader: DataLoader,
    llm_tokenizer,
    bert_tokenizer,
    device: torch.device,
    max_llm_len: int,
    max_bert_len: int,
    add_eos_to_target: bool,
    train_mode: str,
    joint_loss: str,
    pseudo_ce_weight: float,
    pseudo_ce_margin: float,
    pair_loss_normalization: str,
    correct_soft_ce_temperature: float,
    self_preserve_weight: float,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}
    pred_first_all: List[int] = []
    pred_mid_all: List[int] = []
    best_first_all: List[int] = []
    best_mid_all: List[int] = []
    task_ids_all: List[int] = []
    task_names_all: List[str] = []
    oracle_debug_acc = init_oracle_debug_accumulator(model.expert_names)

    progress = make_progress(loader, total=len(loader), desc="eval")
    for batch in progress:
        lm_batch = build_lm_batch(
            tokenizer=llm_tokenizer,
            prompts=batch.texts,
            targets=batch.targets,
            max_length=max_llm_len,
            add_eos_to_target=add_eos_to_target,
        )
        prompt_input_ids = lm_batch["prompt_input_ids"].to(device, non_blocking=True)
        prompt_attention_mask = lm_batch["prompt_attention_mask"].to(device, non_blocking=True)
        input_ids = lm_batch["input_ids"].to(device, non_blocking=True)
        attention_mask = lm_batch["attention_mask"].to(device, non_blocking=True)
        labels = lm_batch["labels"].to(device, non_blocking=True)
        task_ids = batch.task_ids.to(device, non_blocking=True)

        bert_enc = bert_tokenizer(
            batch.source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_bert_len,
        )
        bert_input_ids = bert_enc["input_ids"].to(device, non_blocking=True)
        bert_attention_mask = bert_enc["attention_mask"].to(device, non_blocking=True)
        bert_token_type_ids = bert_enc.get("token_type_ids")
        if bert_token_type_ids is not None:
            bert_token_type_ids = bert_token_type_ids.to(device, non_blocking=True)

        ####先抽 prompt vector
        first_vec, mid_vec = model.extract_prompt_vectors(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
        )
        loss_matrix = model.score_all_route_pairs(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            targets=batch.targets,
            source_texts=batch.source_texts,
            task_names=batch.task_names,
            llm_tokenizer=llm_tokenizer,
            # Full-training 固定使用與目前 router supervision 對齊的混合計分定義。
            score_mode="official_eval_aligned_generation",
        )
        correct_matrix = model.last_route_correct_matrix
        pair_logits, logits_first, logits_mid = model.forward_router(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=first_vec.to(torch.float32),
            mid_vec=mid_vec.to(torch.float32),
        )
        loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
            pair_logits=pair_logits,
            logits_first=logits_first,
            logits_mid=logits_mid,
            loss_matrix=loss_matrix,
            correct_matrix=correct_matrix,
            task_ids=task_ids,
            mode=train_mode,
            joint_loss=joint_loss,
            pseudo_ce_weight=pseudo_ce_weight,
            margin=pseudo_ce_margin,
            loss_normalization=pair_loss_normalization,
            correct_soft_ce_temperature=correct_soft_ce_temperature,
            self_preserve_weight=self_preserve_weight,
        )

        pred_pair = pair_logits.argmax(dim=-1)
        pred_first = pred_pair // loss_matrix.size(2)
        pred_mid = pred_pair % loss_matrix.size(2)
        batch_size = pred_first.size(0)
        batch_stats = compute_routing_accuracy_stats(
            pred_first=pred_first,
            pred_mid=pred_mid,
            best_first=best_first,
            best_mid=best_mid,
            task_ids=task_ids,
            expert_names=model.expert_names,
            sample_task_names=batch.task_names,
        )
        score_stats = compute_route_score_stats(
            loss_matrix=loss_matrix,
            pred_pair=pred_pair,
            task_ids=task_ids,
            loss_normalization=pair_loss_normalization,
            expert_names=model.expert_names,
            sample_task_names=batch.task_names,
        )
        update_oracle_debug_accumulator(
            acc=oracle_debug_acc,
            loss_matrix=loss_matrix,
            task_ids=task_ids,
            sample_task_names=batch.task_names,
        )
        pred_first_all.extend(pred_first.cpu().tolist())
        pred_mid_all.extend(pred_mid.cpu().tolist())
        best_first_all.extend(best_first.cpu().tolist())
        best_mid_all.extend(best_mid.cpu().tolist())
        task_ids_all.extend(task_ids.detach().cpu().tolist())
        task_names_all.extend(str(name) for name in batch.task_names)

        total_loss += loss.item() * batch_size
        total_samples += batch_size
        for key, value in batch_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size
        for key, value in score_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * batch_size
        for key in [
            "expected_loss",
            "correct_soft_ce",
            "correct_conf_ce",
            "self_preserving_correct_conf_ce",
            "correct_max_margin",
            "correct_margin_available_ratio",
            "correct_target_available_ratio",
            "avg_correct_pairs",
            "main_pair_ce",
            "pseudo_ce_pair",
            "pseudo_ce_first",
            "pseudo_ce_mid",
            "best_pair_loss",
            "margin_active_ratio",
        ]:
            if key in metrics:
                metric_totals[key] = metric_totals.get(key, 0.0) + metrics[key] * batch_size

        if tqdm is not None:
            progress.set_postfix(
                loss=f"{(total_loss / max(total_samples, 1)):.4f}",
                pair=f"{(metric_totals.get('pair_acc', 0.0) / max(total_samples, 1)):.4f}",
                router=f"{(metric_totals.get('router_argmax_score', 0.0) / max(total_samples, 1)):.2f}",
                self_score=f"{(metric_totals.get('fixed_self_score', 0.0) / max(total_samples, 1)):.2f}",
            )

    denom = max(total_samples, 1)
    result = {"loss": total_loss / denom}
    for key, value in metric_totals.items():
        result[key] = value / denom
    result["routing_summary"] = build_routing_summary(
        pred_first_all=pred_first_all,
        pred_mid_all=pred_mid_all,
        best_first_all=best_first_all,
        best_mid_all=best_mid_all,
        task_ids_all=task_ids_all,
        expert_names=model.expert_names,
        sample_task_names_all=task_names_all,
    )
    result["oracle_debug_summary"] = build_oracle_debug_summary(oracle_debug_acc)
    return result

## CLI 傳進來的逗號分隔字串 轉成 Python list。e.g. --expert_names iwslt2017,medmcqa,race 變成 ["iwslt2017", "medmcqa", "race"]
def parse_csv_arg(raw: Optional[str]) -> Optional[List[str]]:
    if not raw:
        return None
    items = [part.strip() for part in raw.split(",") if part.strip()]
    return items or None


###就是有進度條的東西
def make_progress(iterable, total: int, desc: str):
    if tqdm is None:
        return iterable
    return tqdm(iterable, total=total, desc=desc, dynamic_ncols=True, leave=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="router_train_datasets_0511")
    ####資料來源要用哪些task來訓練router, 不填就用data_root底下所有的資料夾
    parser.add_argument(
        "--task_names",
        type=str,
        default=None,
        help="comma-separated task names. default: discover all subdirs under data_root",
    )
    ####資料來源要用哪個data_root根目錄來eval, 不填就用data_root底下所有的資料夾
    parser.add_argument(
        "--eval_data_root",
        type=str,
        default=None,
        help="optional evaluation dataset root. default: use data_root",
    )
    ####指定 eval 用哪些 task。不填就掃 eval_data_root底下所有的資料夾
    parser.add_argument(
        "--eval_task_names",
        type=str,
        default=None,
        help="comma-separated eval task names. default: use discovered tasks under eval_data_root",
    )
    parser.add_argument(
        "--eval_split",
        type=str,
        default="validation",
        choices=["train", "validation", "test"],
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="skip training and only evaluate router checkpoint on eval datasets",
    )
    #### 訓練每幾個 epoch 額外跑一次 eval dataset。適合看新資料訓練時，有沒有把舊資料能力搞壞。
    parser.add_argument(
        "--eval_data_during_train",
        action="store_true",
        help="also evaluate eval_data_root after training epochs, useful for monitoring old tasks while training on new tasks",
    )
    ####搭配上面那個參數，每幾個 epoch eval 一次。
    parser.add_argument(
        "--eval_data_every_epochs",
        type=int,
        default=1,
        help="run eval_data_root evaluation every N epochs when --eval_data_during_train is set",
    )
    ####router 可以選的 expert 名單。這個會決定 pair 空間大小：
    parser.add_argument(
        "--expert_names",
        type=str,
        default=None,
        help="comma-separated expert names to route among. default: infer from provided LoRA paths",
    )
    parser.add_argument("--base_model_path", type=str, required=True)
    parser.add_argument("--router_bert_init", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    ###從既有 router checkpoint 繼續訓練或 eval。
    parser.add_argument("--load_router_ckpt_dir", type=str, default=None)
    parser.add_argument("--lora_iwslt", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_iwslt')
    parser.add_argument("--lora_medmcqa", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_medmcqa')
    parser.add_argument("--lora_race", type=str, default='./saves/llama2-7b-chat-hf/lora/sft_race')
    parser.add_argument("--lora_squad2", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_squad20")
    parser.add_argument("--lora_sst2", type=str, default="./saves/llama2-7b-chat-hf/lora/sft_sst2")
    #parser.add_argument("--lora_piqa", type=str, default=None)
    #parser.add_argument("--lora_copa", type=str, default=None)
    #parser.add_argument("--lora_hellaswag", type=str, default=None)
    #parser.add_argument("--lora_boolq", type=str, default=None)
    #parser.add_argument("--lora_siqa", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_train_samples", type=int, default=500)
    parser.add_argument("--max_val_samples", type=int, default=250)
    ####LLM prompt+target 最大 token 長度。越大越吃 GPU。
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    ####router_dim 是 router 內部 attention 投影維度：LLM query 與 BERT key/value
    ####會各自投影到這個維度。較小會形成資訊瓶頸但降低參數量，較大則成本更高；
    ####這是可訓練壓縮，不保證無損，需以 validation 結果選擇。
    ####CompactRouterFeatureEncoder 最後串接 `q` 與 attention context `ctx`，
    ####兩者各為 router_dim，所以每一個 router_first/router_mid feature 維度
    ####是 `router_dim * 2`；當 router_dim=512 時就是 1024。
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument(
        "--router_pooling",
        type=str,
        default="mean",
        choices=["last_token", "mean", "lastk_mean"],
    )
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16"])
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument(
        "--freeze_bert",
        action="store_true",
        default=True,
        help="Retained for command compatibility; the BERT encoder is always frozen.",
    )
    ####router head 的 freeze flags 仍採 store_true；BERT 則固定凍結。
    parser.add_argument("--freeze_router_first", action="store_true")
    parser.add_argument("--freeze_router_mid", action="store_true")
    parser.add_argument("--train_mode", type=str, default="joint", choices=["stage1", "stage2", "joint"])
    add_joint_loss_arguments(parser)
    ####`action="store_true"` 表示 CLI 有寫 `--add_eos_to_target` 才會設為 True；
    ####它會在 teacher-forcing label 的 target 後附加 EOS，原意是也監督答案結束。
    ####目前正式 `official_eval_aligned_generation` 計分不使用這份 sequence labels：
    ####classification 走 first-token NLL，generation task 走生成後 evaluator，
    ####因此此 flag 現在不會改變正式 loss_matrix；實驗設定仍會記入輸出 config。
    parser.add_argument("--add_eos_to_target", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save_every_epoch", action="store_true")
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="opencompass")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None, help="comma-separated wandb tags")
    args = parser.parse_args()

    global tqdm
    if args.disable_tqdm:
        tqdm = None

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise ImportError("--wandb was set but wandb is not installed") from e
        wandb_tags = parse_csv_arg(args.wandb_tags)
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=wandb_tags,
            config=vars(args),
            dir=args.output_dir,
        )
        print(f"[INFO] wandb enabled project={args.wandb_project}")

    requested_tasks = parse_csv_arg(args.task_names)
    requested_eval_tasks = parse_csv_arg(args.eval_task_names)
    eval_data_root = args.eval_data_root or args.data_root

    all_lora_paths = {
        "iwslt2017": args.lora_iwslt,
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "squad2": args.lora_squad2,
        "sst2": args.lora_sst2,
        #####這裡只登記實際提供 LoRA 的 expert 是可以的；piqa/copa/hellaswag/boolq/siqa
        #####仍可作為 sample task 訓練或評估，但沒有自己的 self expert，task_id 會是 -1。
        #####若要讓 router 可以選它們，才需要加入 LoRA path 並納入 expert_names。
        #"piqa": args.lora_piqa,
        #"copa": args.lora_copa,
        #"hellaswag": args.lora_hellaswag,
        #"boolq": args.lora_boolq,
        #"siqa": args.lora_siqa,
    }

    ###解析 expert names
    requested_experts = parse_csv_arg(args.expert_names)
    ###決定這次真正要用哪些 experts
    expert_names, expert_name_source = discover_expert_names(requested_experts, all_lora_paths)
    print(f"[INFO] expert_names={expert_names}")
    print(f"[INFO] expert_name_source={expert_name_source}")
    lora_paths = {}
    missing_loras = []
    for task in expert_names:
        path = all_lora_paths.get(task)
        if not path:
            missing_loras.append(task)
        else:
            lora_paths[task] = path
    if missing_loras:
        raise ValueError(
            "Missing LoRA paths for selected tasks. "
            f"Please provide CLI args for: {missing_loras}"
        )

    llm_tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if llm_tokenizer.pad_token_id is None:
        llm_tokenizer.pad_token = llm_tokenizer.eos_token
    llm_tokenizer.padding_side = "left"
    bert_tokenizer = AutoTokenizer.from_pretrained(args.router_bert_init)

    train_task_names: List[str] = []
    train_loader = None
    val_loader = None
    if not args.eval_only:
        train_ds, train_task_names = build_dataset(
            data_root=args.data_root,
            split="train",
            requested_tasks=requested_tasks,
            expert_names=expert_names,
            max_samples=args.max_train_samples,
            seed=args.seed,
        )
        print(f"[INFO] train_task_names={train_task_names}")
        val_ds, val_task_names = build_dataset(
            data_root=args.data_root,
            split="validation",
            requested_tasks=requested_tasks,
            expert_names=expert_names,
            max_samples=args.max_val_samples,
            seed=args.seed,
        )
        print(f"[INFO] val_task_names={val_task_names}")
        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=Collator(),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.eval_batch_size,
            shuffle=False,
            collate_fn=Collator(),
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
    else:
        print("[INFO] eval_only=True, skipping training dataset construction")

    eval_ds, eval_task_names = build_dataset(
        data_root=eval_data_root,
        split=args.eval_split,
        requested_tasks=requested_eval_tasks,
        expert_names=expert_names,
        max_samples=args.max_val_samples,
        seed=args.seed,
    )
    print(f"[INFO] eval_task_names={eval_task_names}")
    eval_loader = DataLoader(
        eval_ds,
        batch_size=args.eval_batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = JointAnswerSupervisionRouterModel(
        base_model_path=args.base_model_path,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        router_dim=args.router_dim,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        expert_names=expert_names,
        router_pooling=args.router_pooling,
        router_pooling_last_k=args.router_pooling_last_k,
    )
    if args.load_router_ckpt_dir:
        model.load_router_weights(args.load_router_ckpt_dir)
        print(f"[INFO] loaded router weights from {args.load_router_ckpt_dir}")
    
    ###`joint` 預設不凍結 router_first 或 router_mid，兩者與 pair_classifier 一起訓練；
    ###`stage1` 會凍結 router_mid，`stage2` 會凍結 router_first。BERT 在本實作固定凍結。
    freeze_router_first = args.freeze_router_first or (args.train_mode == "stage2")
    freeze_router_mid = args.freeze_router_mid or (args.train_mode == "stage1")
    model.set_trainable(
        freeze_bert=args.freeze_bert,
        freeze_router_first=freeze_router_first,
        freeze_router_mid=freeze_router_mid,
    )
    model.to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    trainable_param_count = sum(p.numel() for p in trainable_params)
    print(
        "[INFO] trainable_parts="
        f"bert:{not args.freeze_bert} "
        f"router_first:{not freeze_router_first} "
        f"router_mid:{not freeze_router_mid} "
        f"num_params={trainable_param_count}"
    )
    print(
        f"[INFO] train_mode={args.train_mode} joint_loss={args.joint_loss} "
        f"pseudo_ce_weight={args.pseudo_ce_weight} pseudo_ce_margin={args.pseudo_ce_margin} "
        f"correct_soft_ce_temperature={args.correct_soft_ce_temperature} "
        f"self_preserve_weight={args.self_preserve_weight} "
        f"pair_loss_normalization={args.pair_loss_normalization}"
    )
    config = vars(args).copy()
    config["train_task_names"] = train_task_names
    config["expert_names"] = expert_names
    config["eval_task_names"] = eval_task_names
    save_json(config, os.path.join(args.output_dir, "train_config.json"))

    if args.eval_only:
        if not args.load_router_ckpt_dir:
            raise ValueError("--eval_only requires --load_router_ckpt_dir")
        eval_metrics = evaluate(
            model=model,
            loader=eval_loader,
            llm_tokenizer=llm_tokenizer,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_llm_len=args.max_llm_len,
            max_bert_len=args.max_bert_len,
            add_eos_to_target=args.add_eos_to_target,
            train_mode=args.train_mode,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            pair_loss_normalization=args.pair_loss_normalization,
            correct_soft_ce_temperature=args.correct_soft_ce_temperature,
            self_preserve_weight=args.self_preserve_weight,
        )
        print(
            f"[EVAL] split={args.eval_split} loss={eval_metrics['loss']:.4f} "
            f"router_score={eval_metrics['router_argmax_score']:.2f} "
            f"self_score={optional_float(eval_metrics.get('fixed_self_score'), digits=2)} "
            f"oracle_score={eval_metrics['oracle_best_pair_score']:.2f} "
            f"first_acc={eval_metrics['first_acc']:.4f} "
            f"mid_acc={eval_metrics['mid_acc']:.4f} "
            f"joint_acc={eval_metrics['joint_acc']:.4f} "
            f"pair_acc={eval_metrics['pair_acc']:.4f} "
            f"self_first={optional_float(eval_metrics.get('self_first_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"self_mid={optional_float(eval_metrics.get('self_mid_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"self_joint={optional_float(eval_metrics.get('self_joint_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"self_pair={optional_float(eval_metrics.get('self_pair_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"oracle_first_self={optional_float(eval_metrics.get('oracle_first_self_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"oracle_mid_self={optional_float(eval_metrics.get('oracle_mid_self_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"oracle_self_pair={optional_float(eval_metrics.get('oracle_self_pair_acc') if eval_metrics.get('self_applicable_ratio', 1.0) > 0 else None)}"
        )
        print_routing_summary(tag=f"EVAL-{args.eval_split}", summary=eval_metrics["routing_summary"])
        print_oracle_debug_summary(tag=f"EVAL-{args.eval_split}", summary=eval_metrics["oracle_debug_summary"])
        save_json(
            eval_metrics["routing_summary"],
            os.path.join(args.output_dir, f"routing_summary_{args.eval_split}.json"),
        )
        save_json(
            eval_metrics["oracle_debug_summary"],
            os.path.join(args.output_dir, f"oracle_debug_summary_{args.eval_split}.json"),
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    f"eval/{k}": v
                    for k, v in eval_metrics.items()
                    if isinstance(v, (int, float))
                }
            )
            wandb_run.finish()
        return

    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    best_router_score = float("-inf")
    best_epoch = -1

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        train_pred_first_all: List[int] = []
        train_pred_mid_all: List[int] = []
        train_best_first_all: List[int] = []
        train_best_mid_all: List[int] = []
        train_task_ids_all: List[int] = []
        train_task_names_all: List[str] = []
        train_oracle_debug_acc = init_oracle_debug_accumulator(expert_names)

        train_progress = make_progress(
            train_loader,
            total=len(train_loader),
            desc=f"train epoch {epoch}/{args.epochs}",
        )
        for step, batch in enumerate(train_progress, start=1):
            ###產出：prompt_input_ids, prompt_attention_mask, input_ids, attention_mask, labels
            lm_batch = build_lm_batch(
                tokenizer=llm_tokenizer,
                prompts=batch.texts,
                targets=batch.targets,
                max_length=args.max_llm_len,
                add_eos_to_target=args.add_eos_to_target,
            )
            prompt_input_ids = lm_batch["prompt_input_ids"].to(device, non_blocking=True)
            prompt_attention_mask = lm_batch["prompt_attention_mask"].to(device, non_blocking=True)
            input_ids = lm_batch["input_ids"].to(device, non_blocking=True)
            attention_mask = lm_batch["attention_mask"].to(device, non_blocking=True)
            labels = lm_batch["labels"].to(device, non_blocking=True)
            task_ids = batch.task_ids.to(device, non_blocking=True)

            bert_enc = bert_tokenizer(
                batch.source_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_bert_len,
            )
            bert_input_ids = bert_enc["input_ids"].to(device, non_blocking=True)
            bert_attention_mask = bert_enc["attention_mask"].to(device, non_blocking=True)
            bert_token_type_ids = bert_enc.get("token_type_ids")
            if bert_token_type_ids is not None:
                bert_token_type_ids = bert_token_type_ids.to(device, non_blocking=True)

            ####開torch.no_grad(), 因為我不想對 base LLM/backbone 回傳梯度
            with torch.no_grad():
                first_vec, mid_vec = model.extract_prompt_vectors(
                    input_ids=prompt_input_ids,
                    attention_mask=prompt_attention_mask,
                )
                loss_matrix = model.score_all_route_pairs(
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    targets=batch.targets,
                    source_texts=batch.source_texts,
                    task_names=batch.task_names,
                    llm_tokenizer=llm_tokenizer,
                    # Full-training 固定使用與目前 router supervision 對齊的混合計分定義。
                    score_mode="official_eval_aligned_generation",
                )
                correct_matrix = model.last_route_correct_matrix

            pair_logits, logits_first, logits_mid = model.forward_router(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=first_vec.to(torch.float32),
                mid_vec=mid_vec.to(torch.float32),
            )
            loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
                pair_logits=pair_logits,
                logits_first=logits_first,
                logits_mid=logits_mid,
                loss_matrix=loss_matrix,
                correct_matrix=correct_matrix,
                task_ids=task_ids,
                mode=args.train_mode,
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                margin=args.pseudo_ce_margin,
                loss_normalization=args.pair_loss_normalization,
                correct_soft_ce_temperature=args.correct_soft_ce_temperature,
                self_preserve_weight=args.self_preserve_weight,
            )

            optimizer.zero_grad()
            ####base LLM、LoRA 與 BERT 已凍結，但 backward 仍會沿計算圖計算 router 的梯度；
            ####optimizer 只更新 requires_grad=True 的 router_first/router_mid/pair_classifier。
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            ####下方不再更新模型，而是累積這個 epoch 的 loss、router argmax pair、
            ####oracle best pair 與 self/oracle 對照統計，供進度列、wandb 與 summary JSON 輸出。
            batch_size = len(batch.texts)
            running_loss += loss.item() * batch_size
            running_count += batch_size
            pred_pair = pair_logits.argmax(dim=-1)
            pred_first = pred_pair // loss_matrix.size(2)
            pred_mid = pred_pair % loss_matrix.size(2)
            train_pred_first_all.extend(pred_first.detach().cpu().tolist())
            train_pred_mid_all.extend(pred_mid.detach().cpu().tolist())
            train_best_first_all.extend(best_first.detach().cpu().tolist())
            train_best_mid_all.extend(best_mid.detach().cpu().tolist())
            train_task_ids_all.extend(task_ids.detach().cpu().tolist())
            train_task_names_all.extend(str(name) for name in batch.task_names)
            update_oracle_debug_accumulator(
                acc=train_oracle_debug_acc,
                loss_matrix=loss_matrix,
                task_ids=task_ids,
                sample_task_names=batch.task_names,
            )

            if tqdm is not None:
                batch_stats = compute_routing_accuracy_stats(
                    pred_first=pred_first,
                    pred_mid=pred_mid,
                    best_first=best_first,
                    best_mid=best_mid,
                    task_ids=task_ids,
                    expert_names=expert_names,
                    sample_task_names=batch.task_names,
                )
                train_progress.set_postfix(
                    loss=f"{(running_loss / max(running_count, 1)):.4f}",
                    exp=f"{metrics['expected_loss']:.4f}",
                    pair=f"{metrics['best_pair_loss']:.4f}",
                    f_acc=f"{batch_stats['first_acc']:.4f}",
                    m_acc=f"{batch_stats['mid_acc']:.4f}",
                    self_acc=f"{batch_stats['self_joint_acc']:.4f}",
                )

            if step % 10 == 0 or step == len(train_loader):
                batch_stats = compute_routing_accuracy_stats(
                    pred_first=pred_first,
                    pred_mid=pred_mid,
                    best_first=best_first,
                    best_mid=best_mid,
                    task_ids=task_ids,
                    expert_names=expert_names,
                    sample_task_names=batch.task_names,
                )
                avg_loss = running_loss / max(running_count, 1)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": step + (epoch - 1) * len(train_loader),
                            "train/loss": avg_loss,
                            "train/main_pair_ce": metrics["main_pair_ce"],
                            "train/expected_loss": metrics["expected_loss"],
                            "train/correct_soft_ce": metrics.get("correct_soft_ce", 0.0),
                            "train/correct_conf_ce": metrics.get("correct_conf_ce", 0.0),
                            "train/self_preserving_correct_conf_ce": metrics.get("self_preserving_correct_conf_ce", 0.0),
                            "train/correct_max_margin": metrics.get("correct_max_margin", 0.0),
                            "train/correct_margin_available_ratio": metrics.get("correct_margin_available_ratio", 0.0),
                            "train/correct_target_available_ratio": metrics.get("correct_target_available_ratio", 0.0),
                            "train/avg_correct_pairs": metrics.get("avg_correct_pairs", 0.0),
                            "train/best_pair_loss": metrics["best_pair_loss"],
                            "train/first_acc": batch_stats["first_acc"],
                            "train/mid_acc": batch_stats["mid_acc"],
                            "train/joint_acc": batch_stats["joint_acc"],
                            "train/pair_acc": batch_stats["pair_acc"],
                            "train/self_first_acc": batch_stats["self_first_acc"],
                            "train/self_mid_acc": batch_stats["self_mid_acc"],
                            "train/self_joint_acc": batch_stats["self_joint_acc"],
                            "train/self_pair_acc": batch_stats["self_pair_acc"],
                            "train/oracle_first_self_acc": batch_stats["oracle_first_self_acc"],
                            "train/oracle_mid_self_acc": batch_stats["oracle_mid_self_acc"],
                            "train/oracle_self_joint_acc": batch_stats["oracle_self_joint_acc"],
                            "train/oracle_self_pair_acc": batch_stats["oracle_self_pair_acc"],
                            "train/margin_active_ratio": metrics.get("margin_active_ratio", 0.0),
                            "train/lr": scheduler.get_last_lr()[0],
                        }
                    )
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} pair_ce={metrics['main_pair_ce']:.4f} expected={metrics['expected_loss']:.4f} "
                    f"correct_conf={metrics.get('correct_conf_ce', 0.0):.4f} "
                    f"best_pair={metrics['best_pair_loss']:.4f} "
                    f"first_acc={batch_stats['first_acc']:.4f} mid_acc={batch_stats['mid_acc']:.4f} "
                    f"pair_acc={batch_stats['pair_acc']:.4f} "
                    f"self_first={optional_float(batch_stats.get('self_first_acc') if batch_stats.get('self_applicable_ratio', 1.0) > 0 else None)} "
                    f"self_mid={optional_float(batch_stats.get('self_mid_acc') if batch_stats.get('self_applicable_ratio', 1.0) > 0 else None)} "
                    f"oracle_self_pair={optional_float(batch_stats.get('oracle_self_pair_acc') if batch_stats.get('self_applicable_ratio', 1.0) > 0 else None)}"
                )

        train_routing_summary = build_routing_summary(
            pred_first_all=train_pred_first_all,
            pred_mid_all=train_pred_mid_all,
            best_first_all=train_best_first_all,
            best_mid_all=train_best_mid_all,
            task_ids_all=train_task_ids_all,
            expert_names=expert_names,
            sample_task_names_all=train_task_names_all,
        )
        print_routing_summary(tag=f"TRAIN-EPOCH{epoch}", summary=train_routing_summary)
        save_json(
            train_routing_summary,
            os.path.join(args.output_dir, f"routing_summary_train_epoch{epoch}.json"),
        )
        train_oracle_debug_summary = build_oracle_debug_summary(train_oracle_debug_acc)
        print_oracle_debug_summary(tag=f"TRAIN-EPOCH{epoch}", summary=train_oracle_debug_summary)
        save_json(
            train_oracle_debug_summary,
            os.path.join(args.output_dir, f"oracle_debug_train_epoch{epoch}.json"),
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            llm_tokenizer=llm_tokenizer,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_llm_len=args.max_llm_len,
            max_bert_len=args.max_bert_len,
            add_eos_to_target=args.add_eos_to_target,
            train_mode=args.train_mode,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
            pair_loss_normalization=args.pair_loss_normalization,
            correct_soft_ce_temperature=args.correct_soft_ce_temperature,
            self_preserve_weight=args.self_preserve_weight,
        )
        print(
            f"[VAL] epoch={epoch} loss={val_metrics['loss']:.4f} "
            f"router_score={val_metrics['router_argmax_score']:.2f} "
            f"self_score={optional_float(val_metrics.get('fixed_self_score'), digits=2)} "
            f"oracle_score={val_metrics['oracle_best_pair_score']:.2f} "
            f"first_acc={val_metrics['first_acc']:.4f} "
            f"mid_acc={val_metrics['mid_acc']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.4f} "
            f"self_first={optional_float(val_metrics.get('self_first_acc') if val_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"self_mid={optional_float(val_metrics.get('self_mid_acc') if val_metrics.get('self_applicable_ratio', 1.0) > 0 else None)} "
            f"oracle_self_pair={optional_float(val_metrics.get('oracle_self_pair_acc') if val_metrics.get('self_applicable_ratio', 1.0) > 0 else None)}"
        )
        print_routing_summary(tag=f"VAL-EPOCH{epoch}", summary=val_metrics["routing_summary"])
        save_json(
            val_metrics["routing_summary"],
            os.path.join(args.output_dir, f"routing_summary_val_epoch{epoch}.json"),
        )
        print_oracle_debug_summary(tag=f"VAL-EPOCH{epoch}", summary=val_metrics["oracle_debug_summary"])
        save_json(
            val_metrics["oracle_debug_summary"],
            os.path.join(args.output_dir, f"oracle_debug_val_epoch{epoch}.json"),
        )
        if wandb_run is not None:
            payload = {
                "val/epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/main_pair_ce": val_metrics["main_pair_ce"],
                "val/expected_loss": val_metrics["expected_loss"],
                "val/correct_soft_ce": val_metrics.get("correct_soft_ce", 0.0),
                "val/correct_conf_ce": val_metrics.get("correct_conf_ce", 0.0),
                "val/self_preserving_correct_conf_ce": val_metrics.get("self_preserving_correct_conf_ce", 0.0),
                "val/correct_max_margin": val_metrics.get("correct_max_margin", 0.0),
                "val/correct_margin_available_ratio": val_metrics.get("correct_margin_available_ratio", 0.0),
                "val/correct_target_available_ratio": val_metrics.get("correct_target_available_ratio", 0.0),
                "val/avg_correct_pairs": val_metrics.get("avg_correct_pairs", 0.0),
                "val/router_argmax_score": val_metrics["router_argmax_score"],
                "val/fixed_self_score": val_metrics["fixed_self_score"],
                "val/oracle_best_pair_score": val_metrics["oracle_best_pair_score"],
                "val/oracle_avg_gap": val_metrics["oracle_debug_summary"]["avg_gap"],
                "val/oracle_p50_gap": val_metrics["oracle_debug_summary"]["p50_gap"],
                "val/oracle_avg_self_minus_oracle": val_metrics["oracle_debug_summary"]["avg_self_minus_oracle"],
                "val/first_acc": val_metrics["first_acc"],
                "val/mid_acc": val_metrics["mid_acc"],
                "val/pair_acc": val_metrics["pair_acc"],
                "val/joint_acc": val_metrics["joint_acc"],
                "val/best_router_argmax_score": max(best_router_score, val_metrics["router_argmax_score"]),
            }
            wandb_run.log(payload)

        if args.eval_data_during_train and epoch % max(1, args.eval_data_every_epochs) == 0:
            eval_data_metrics = evaluate(
                model=model,
                loader=eval_loader,
                llm_tokenizer=llm_tokenizer,
                bert_tokenizer=bert_tokenizer,
                device=device,
                max_llm_len=args.max_llm_len,
                max_bert_len=args.max_bert_len,
                add_eos_to_target=args.add_eos_to_target,
                train_mode=args.train_mode,
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                pseudo_ce_margin=args.pseudo_ce_margin,
                pair_loss_normalization=args.pair_loss_normalization,
                correct_soft_ce_temperature=args.correct_soft_ce_temperature,
                self_preserve_weight=args.self_preserve_weight,
            )
            print(
                f"[EVAL-DATA] epoch={epoch} split={args.eval_split} "
                f"loss={eval_data_metrics['loss']:.4f} "
                f"router_score={eval_data_metrics['router_argmax_score']:.2f} "
                f"self_score={optional_float(eval_data_metrics.get('fixed_self_score'), digits=2)} "
                f"oracle_score={eval_data_metrics['oracle_best_pair_score']:.2f} "
                f"first_acc={eval_data_metrics['first_acc']:.4f} "
                f"mid_acc={eval_data_metrics['mid_acc']:.4f} "
                f"pair_acc={eval_data_metrics['pair_acc']:.4f}"
            )
            print_routing_summary(
                tag=f"EVAL-DATA-EPOCH{epoch}",
                summary=eval_data_metrics["routing_summary"],
            )
            save_json(
                eval_data_metrics["routing_summary"],
                os.path.join(args.output_dir, f"routing_summary_eval_data_epoch{epoch}.json"),
            )
            print_oracle_debug_summary(
                tag=f"EVAL-DATA-EPOCH{epoch}",
                summary=eval_data_metrics["oracle_debug_summary"],
            )
            save_json(
                eval_data_metrics["oracle_debug_summary"],
                os.path.join(args.output_dir, f"oracle_debug_eval_data_epoch{epoch}.json"),
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "eval_data/epoch": epoch,
                        "eval_data/loss": eval_data_metrics["loss"],
                        "eval_data/router_argmax_score": eval_data_metrics["router_argmax_score"],
                        "eval_data/fixed_self_score": eval_data_metrics["fixed_self_score"],
                        "eval_data/oracle_best_pair_score": eval_data_metrics["oracle_best_pair_score"],
                        "eval_data/first_acc": eval_data_metrics["first_acc"],
                        "eval_data/mid_acc": eval_data_metrics["mid_acc"],
                        "eval_data/pair_acc": eval_data_metrics["pair_acc"],
                    }
                )

        state = {
            "pair_first_encoder": model.router_first.state_dict(),
            "pair_mid_encoder": model.router_mid.state_dict(),
            "pair_classifier": model.pair_classifier.state_dict(),
            "bert_encoder": model.bert.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }

        if args.save_every_epoch:
            torch.save(state, os.path.join(args.output_dir, f"router_heads_epoch{epoch}.pt"))

        if val_metrics["router_argmax_score"] > best_router_score:
            best_router_score = val_metrics["router_argmax_score"]
            best_epoch = epoch
            save_runtime_compatible_router_bundle(
                model=model,
                bert_tokenizer=bert_tokenizer,
                output_dir=args.output_dir,
                state=state,
                router_config={
                    "task_names": expert_names,
                    "expert_names": expert_names,
                    "train_task_names": train_task_names,
                    "first_layer_idx": args.first_layer_idx,
                    "middle_layer_idx": args.middle_layer_idx,
                    "router_max_len": args.max_bert_len,
                    "router_feature_type": "answer_supervision_prompt_last_valid_token",
                    "train_mode": args.train_mode,
                    "joint_loss": args.joint_loss,
                    "router_pooling": args.router_pooling,
                    "router_pooling_last_k": args.router_pooling_last_k,
                    "llama_hidden_size": get_hidden_size(model.model),
                    "router_dim": int(args.router_dim),
                    "sample_feature_mode": "none",
                    "sample_feature_dim": 0,
                    "feature_contract_version": 1,
                    "freeze_bert": args.freeze_bert,
                    "pseudo_ce_margin": args.pseudo_ce_margin,
                    "pseudo_ce_weight": args.pseudo_ce_weight,
                    "pair_loss_normalization": args.pair_loss_normalization,
                    "best_router_argmax_score": best_router_score,
                    "best_epoch": best_epoch,
                    "best_val_loss": val_metrics["loss"],
                },
            )
            print(
                f"[SAVE] best checkpoint updated at epoch={epoch} "
                f"router_score={best_router_score:.2f}"
            )

    print(f"[DONE] best_router_argmax_score={best_router_score:.2f} best_epoch={best_epoch}")
    if wandb_run is not None:
        wandb_run.summary["best_router_argmax_score"] = best_router_score
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
