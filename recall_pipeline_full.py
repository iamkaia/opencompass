#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Paper-Strict RECALL merging (5 tasks, LoRA, LLaMA-2-7b-chat-hf, bf16)

- 從各個 LoRA checkpoint 抽 representation
- KMeans 在 anchor task 上挑典型樣本
- 對每層、每個 LoRA 用 RBF similarity 算 softmax 權重
- 在 LoRA 參數層面做 layer-wise 加權融合，輸出一個新的 LoRA adapter
"""

import os
import json
import random
import gc
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from safetensors.torch import load_file
from sklearn.cluster import KMeans
from tqdm import tqdm

# =========================================
# 全域設定（請依你機器路徑微調）
# =========================================
CONFIG = {
    "base_model": "meta-llama/Llama-2-7b-chat-hf",

    # LoRA 根目錄（不要給 checkpoints/*）
    "adapter_roots": {
        "sst2":      "../opencompass/saves/llama2-7b-chat-hf/lora/sft_sst2",
        # 你現在的實際路徑看起來是 sft_squad20，如果是 sft_squad2 就改掉這行
        "squad2":    "../opencompass/saves/llama2-7b-chat-hf/lora/sft_squad20",
        "iwslt2017": "../opencompass/saves/llama2-7b-chat-hf/lora/sft_iwslt",
        "race":      "../opencompass/saves/llama2-7b-chat-hf/lora/sft_race",
        "medmcqa":   "../opencompass/saves/llama2-7b-chat-hf/lora/sft_medmcqa",
    },

    # SFT 用的 jsonl（你自己做的 data；路徑視你實際情況調）
    "datasets": {
        "sst2":      "../LLaMA-Factory/data/sst2.jsonl",
        # 同樣這裡路徑如果叫 squad2.jsonl 就改回去
        "squad2":    "../LLaMA-Factory/data/squad20.jsonl",
        "iwslt2017": "../LLaMA-Factory/data/iwslt.jsonl",
        "race":      "../LLaMA-Factory/data/race.jsonl",
        "medmcqa":   "../LLaMA-Factory/data/medmcqa.jsonl",
    },

    # KMeans 抽樣 pool size & 代表樣本數 m
    "cluster_samples_per_task": 2000,
    "samples_per_task": 20,      # 想更敏感一點可以調成 100

    "max_length": 512,
    "batch_size": 4,

    # RBF gamma（None 則自動設為 1/hidden_dim）
    "rbf_gamma": None,

    # 預設 anchor（可以用 --anchor_task 覆蓋）
    "anchor_task": "race",

    # 預設輸出路徑（可以用 --output_dir 覆蓋）
    "output_dir": "./recall_fused_strict",

    "seed": 42,
}

device = "cuda" if torch.cuda.is_available() else "cpu"


# ----------------- CLI 參數 -----------------
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--anchor_task", type=str, default=None)
parser.add_argument("--output_dir", type=str, default=None)
args = parser.parse_args()

if args.anchor_task is not None:
    CONFIG["anchor_task"] = args.anchor_task

if args.output_dir is not None:
    CONFIG["output_dir"] = args.output_dir


# =========================================
# 小工具
# =========================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_jsonl_inputs(path: str, max_n: int = None) -> List[str]:
    """讀 jsonl，優先取 'input'，空的話用 'instruction' / 'prompt'。"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj.get("input", "")
            if not text:
                text = obj.get("instruction", "") or obj.get("prompt", "")

            if not text:
                continue

            data.append(text)
            if max_n is not None and len(data) >= max_n:
                break
    return data


def resolve_adapter_path(root: str) -> str:
    """這裡永遠用最外層 LoRA 目錄（不要去 checkpoints 裡抓）。"""
    print(f"[INFO] Using top-level LoRA at: {root}")
    return root


def build_base_tokenizer():
    tok = AutoTokenizer.from_pretrained(CONFIG["base_model"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


# ============== representation 抽取 ==============

def get_embeddings_last_layer(model, tokenizer, texts: List[str]) -> np.ndarray:
    """最後一層 hidden state + mean pooling，給 KMeans 用。"""
    all_vecs = []
    bs = CONFIG["batch_size"]
    max_len = CONFIG["max_length"]

    model.eval()
    for i in tqdm(range(0, len(texts), bs), desc="Extract feats (last layer)"):
        batch = texts[i:i+bs]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)

        hidden = outputs.hidden_states[-1]              # (B, T, D)
        mask = tokenized["attention_mask"].unsqueeze(-1).float()
        pooled = (hidden * mask).sum(1) / mask.sum(1)   # (B, D)

        all_vecs.append(pooled.cpu())

        del tokenized, outputs, hidden, mask, pooled
        torch.cuda.empty_cache()

    return torch.cat(all_vecs, dim=0).numpy()


def select_typical_samples(anchor_model, tokenizer, texts: List[str]) -> List[str]:
    """在 anchor task 上，用 KMeans 選出 m 個典型樣本。"""
    print(f"[STEP] Selecting typical samples from anchor task '{CONFIG['anchor_task']}'...")
    m = CONFIG["samples_per_task"]

    if len(texts) <= m:
        print(f"[WARN] texts <= samples_per_task ({len(texts)} <= {m}), 全部當典型樣本用。")
        return texts

    pool_n = min(CONFIG["cluster_samples_per_task"], len(texts))
    rng = np.random.default_rng(CONFIG["seed"])
    pool_idx = rng.choice(len(texts), size=pool_n, replace=False)
    pool_texts = [texts[i] for i in pool_idx]

    feats = get_embeddings_last_layer(anchor_model, tokenizer, pool_texts)
    hidden_dim = feats.shape[1]

    if CONFIG["rbf_gamma"] is None:
        CONFIG["rbf_gamma"] = 1.0 / hidden_dim
        print(f"[INFO] Set RBF gamma = 1.0 / hidden_dim = {CONFIG['rbf_gamma']:.6f}")

    print(f"[INFO] KMeans on {pool_n} samples, k = {m} ...")
    kmeans = KMeans(n_clusters=m, random_state=CONFIG["seed"], n_init="auto")
    kmeans.fit(feats)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    selected_pool_idx = []
    for c in range(m):
        idxs = np.where(labels == c)[0]
        if len(idxs) == 0:
            continue
        sub = feats[idxs] - centers[c]
        dist2 = np.sum(sub * sub, axis=1)
        best_local = idxs[np.argmin(dist2)]
        selected_pool_idx.append(best_local)

    global_idx = [pool_idx[i] for i in selected_pool_idx]
    global_idx = sorted(set(global_idx))
    print(f"[INFO] Selected {len(global_idx)} typical samples.")
    return [texts[i] for i in global_idx]


def get_layerwise_reps(model, tokenizer, texts: List[str]) -> List[np.ndarray]:
    """給一組 texts，用 LoRA model 抽每層 mean-pooled hidden state。"""
    print("[STEP] Extracting layer-wise representations...")
    bs = CONFIG["batch_size"]
    max_len = CONFIG["max_length"]

    model.eval()
    all_layers = None

    for i in tqdm(range(0, len(texts), bs), desc="Layer-wise feats"):
        batch = texts[i:i+bs]
        tokenized = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**tokenized, output_hidden_states=True)

        hs = outputs.hidden_states[1:]  # drop embedding layer
        mask = tokenized["attention_mask"].unsqueeze(-1).float()

        if all_layers is None:
            L = len(hs)
            all_layers = [[] for _ in range(L)]

        for li, h in enumerate(hs):
            pooled = (h * mask).sum(1) / mask.sum(1)  # (B, D)
            all_layers[li].append(pooled.cpu())

        del tokenized, outputs, hs, mask, pooled
        torch.cuda.empty_cache()

    reps = [torch.cat(chunks, dim=0).numpy() for chunks in all_layers]
    return reps  # list length = num_layers, each (m, D)


# ============== RBF 相似度 + 權重 ==============

def rbf_similarity(a: np.ndarray, b: np.ndarray, gamma: float) -> float:
    """平均 RBF similarity over m 個典型樣本。"""
    diff = a - b
    dist2 = np.sum(diff * diff, axis=1)
    sim = np.exp(-gamma * dist2)
    return float(sim.mean())


def compute_layer_weights(
    reps_by_task: Dict[str, List[np.ndarray]],
    tasks: List[str],
) -> Dict[int, Dict[str, float]]:
    """reps_by_task[task][layer] -> (m, D)，輸出每層的 softmax 權重。"""
    anchor = CONFIG["anchor_task"]
    gamma = CONFIG["rbf_gamma"]
    assert anchor in reps_by_task, f"anchor task '{anchor}' not in reps_by_task"

    num_layers = len(next(iter(reps_by_task.values())))
    layer_weights: Dict[int, Dict[str, float]] = {}

    print("[STEP] Computing RBF similarities + softmax weights per layer...")
    for li in range(num_layers):
        sims = []
        for t in tasks:
            sim = rbf_similarity(reps_by_task[anchor][li], reps_by_task[t][li], gamma)
            sims.append(sim)

        sims = np.array(sims, dtype=np.float32)
        w = np.exp(sims) / np.exp(sims).sum()
        layer_weights[li] = {t: float(w[idx]) for idx, t in enumerate(tasks)}

    return layer_weights


# ============== LoRA 相關工具 ==============

def get_lora_state_dict(adapter_path: str) -> Dict[str, torch.Tensor]:
    """讀取某個 LoRA adapter 的 safetensors state dict。"""
    st_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(st_path):
        raise FileNotFoundError(f"adapter_model.safetensors not found under {adapter_path}")
    sd = load_file(st_path)
    return sd


def normalize_template_keys(sd_keys):
    """把 template_peft.state_dict() 的 key prefix 剝掉，得到 normalized key 集合。"""
    normalized = set()
    for k in sd_keys:
        k2 = k
        for prefix in [
            "base_model.model.model.",
            "base_model.model.",
            "base_model.",
            "model.model.",
            "model.",
        ]:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]
        # 預防 lora_A.default.weight / lora_B.default.weight 差異
        k2 = k2.replace(".lora_A.default.weight", ".lora_A.weight")
        k2 = k2.replace(".lora_B.default.weight", ".lora_B.weight")
        normalized.add(k2)
    return normalized


def normalize_lora_keys(sd, template_keys=None):
    """把 LoRA state_dict 的 key prefix 移除，並對齊 template_keys。"""
    new_sd = {}
    for k, v in sd.items():
        k2 = k
        for prefix in [
            "base_model.model.model.",
            "base_model.model.",
            "base_model.",
            "model.model.",
            "model.",
        ]:
            if k2.startswith(prefix):
                k2 = k2[len(prefix):]

        k2 = k2.replace(".lora_A.default.weight", ".lora_A.weight")
        k2 = k2.replace(".lora_B.default.weight", ".lora_B.weight")

        if template_keys is None or k2 in template_keys:
            new_sd[k2] = v

    return new_sd


def merge_lora_weights(
    layer_weights: Dict[int, Dict[str, float]],
    lora_sds: Dict[str, Dict[str, torch.Tensor]],
    tasks: List[str],
    template_peft: PeftModel,
) -> PeftModel:
    """
    真正做 RECALL 融合的地方：
    - layer_weights: layer_idx -> {task: weight}
    - lora_sds: task -> normalized LoRA state_dict
    - template_peft: 當作容器的 PeftModel（會被覆寫 LoRA 權重）
    """
    print("[STEP] Merging LoRA weights (layer-wise, RBF-softmax)...")

    # 1. 找出所有 LoRA 的共同 key（在 normalized 空間）
    common_keys = set(lora_sds[tasks[0]].keys())
    for t in tasks[1:]:
        common_keys &= set(lora_sds[t].keys())
    common_keys = sorted(common_keys)

    print(f"[INFO] Number of common LoRA params: {len(common_keys)}")
    if len(common_keys) == 0:
        raise RuntimeError("[FATAL] No common LoRA keys to merge.")

    # 2. 建立 normalized_key -> real_key 映射
    template_sd = template_peft.state_dict()
    norm2real = {}

    for real_k in template_sd.keys():
        k = real_k
        for prefix in [
            "base_model.model.model.",
            "base_model.model.",
            "base_model.",
            "model.model.",
            "model.",
        ]:
            if k.startswith(prefix):
                k = k[len(prefix):]

        k = k.replace(".lora_A.default.weight", ".lora_A.weight")
        k = k.replace(".lora_B.default.weight", ".lora_B.weight")

        norm2real[k] = real_k

    missing = [k for k in common_keys if k not in norm2real]
    if missing:
        raise RuntimeError(
            f"[FATAL] {len(missing)} LoRA keys not found in template model. "
            f"Example: {missing[0]}"
        )

    # 3. 逐 param 做 weighted sum
    for name in tqdm(common_keys, desc="Merging layers"):
        # 解析 layer index（用來查 layer_weights）
        layer_idx = None
        for pat in ["model.layers.", "layers."]:
            if pat in name:
                try:
                    after = name.split(pat)[1]
                    layer_idx = int(after.split(".")[0])
                    break
                except Exception:
                    layer_idx = None

        if layer_idx is not None and layer_idx in layer_weights:
            ws = np.array(
                [layer_weights[layer_idx][t] for t in tasks],
                dtype=np.float32,
            )
        else:
            ws = np.ones(len(tasks), dtype=np.float32) / len(tasks)

        merged = None
        for ti, t in enumerate(tasks):
            w = ws[ti]
            param = lora_sds[t][name].to(torch.float32)
            merged = param * w if merged is None else merged + w * param

        real_key = norm2real[name]
        target_dtype = template_sd[real_key].dtype
        template_sd[real_key] = merged.to(target_dtype)

    template_peft.load_state_dict(template_sd, strict=False)
    return template_peft


# =========================================
# 主流程
# =========================================

def main():
    gc.collect()
    torch.cuda.empty_cache()
    set_seed(CONFIG["seed"])

    # 這裡 tasks 順序就是 RECALL 中的 N 個 task
    tasks = ["sst2", "squad2", "iwslt2017", "race", "medmcqa"]
    anchor = CONFIG["anchor_task"]
    assert anchor in tasks, f"anchor_task '{anchor}' must be in {tasks}"

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    print("========== [STEP 0] Load tokenizer & base config ==========")
    tokenizer = build_base_tokenizer()
    tokenizer.padding_side = "right"
    base_cfg = AutoConfig.from_pretrained(CONFIG["base_model"])
    print(f"[INFO] num_hidden_layers = {base_cfg.num_hidden_layers}")

    print("========== [STEP 1] Load datasets ==========")
    all_texts = {}
    for t in tasks:
        path = CONFIG["datasets"][t]
        texts = load_jsonl_inputs(path)
        all_texts[t] = texts
        print(f"[DATA] {t}: {len(texts)} samples from {path}")

    print("========== [STEP 2] Select typical samples on anchor task ==========")
    anchor_root = CONFIG["adapter_roots"][anchor]
    anchor_adapter = resolve_adapter_path(anchor_root)

    base_anchor = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": device},
    )
    anchor_model = PeftModel.from_pretrained(
        base_anchor,
        anchor_adapter,
        torch_dtype=torch.bfloat16,
    )
    anchor_model.eval()

    typical_texts = select_typical_samples(anchor_model, tokenizer, all_texts[anchor])
    m = len(typical_texts)
    print(f"[INFO] Using m={m} typical samples for RECALL similarities.")

    del anchor_model, base_anchor
    torch.cuda.empty_cache()

    print("========== [STEP 3] Extract layer-wise reps for each task ==========")
    reps_by_task: Dict[str, List[np.ndarray]] = {}

    for t in tasks:
        print(f"\n[Task {t}] ----")
        root = CONFIG["adapter_roots"][t]
        adapter_path = resolve_adapter_path(root)

        base = AutoModelForCausalLM.from_pretrained(
            CONFIG["base_model"],
            torch_dtype=torch.bfloat16,
            device_map={"": device},
        )
        lora_model = PeftModel.from_pretrained(
            base,
            adapter_path,
            torch_dtype=torch.bfloat16,
        )
        lora_model.eval()

        reps = get_layerwise_reps(lora_model, tokenizer, typical_texts)
        reps_by_task[t] = reps

        del lora_model, base
        torch.cuda.empty_cache()

    print("========== [STEP 4] Compute RBF-softmax weights per layer ==========")
    layer_weights = compute_layer_weights(reps_by_task, tasks)
    # 若想 debug 也可以在這邊印每層權重
    # for li, w in layer_weights.items():
    #     print(f"[DEBUG] layer {li}: {w}")

    print("========== [STEP 5] Build template PEFT model ==========")
    # ⭐ 這裡用「當前 anchor 的 LoRA」當 template，跟論文一致
    anchor_adapter_path = resolve_adapter_path(CONFIG["adapter_roots"][anchor])
    base_for_merge = AutoModelForCausalLM.from_pretrained(
        CONFIG["base_model"],
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},  # merge 在 CPU 上做就好
    )
    template_peft = PeftModel.from_pretrained(
        base_for_merge,
        anchor_adapter_path,
        torch_dtype=torch.bfloat16,
        device_map={"": "cpu"},
    )
    template_peft.eval()

    print("[INFO] Normalizing template keys...")
    template_keys_raw = set(template_peft.state_dict().keys())
    template_keys = normalize_template_keys(template_keys_raw)
    print(f"[INFO] Template keys: {len(template_keys)} normalized keys")

    print("========== [STEP 6] Load & normalize LoRA state dicts ==========")
    lora_sds: Dict[str, Dict[str, torch.Tensor]] = {}
    for t in tasks:
        root = CONFIG["adapter_roots"][t]
        adapter_path = resolve_adapter_path(root)

        print(f"[INFO] Loading LoRA for task {t}: {adapter_path}")
        raw_sd = get_lora_state_dict(adapter_path)
        norm_sd = normalize_lora_keys(raw_sd, template_keys)
        print(f"[INFO]   {t}: {len(norm_sd)} / {len(raw_sd)} keys kept after normalization")
        lora_sds[t] = norm_sd

    print("========== [STEP 7] Merge LoRA weights ==========")
    fused_model = merge_lora_weights(layer_weights, lora_sds, tasks, template_peft)

    print("========== [STEP 8] Save fused adapter ==========")
    fused_model = fused_model.to("cpu")

    import shutil
    if os.path.exists(CONFIG["output_dir"]):
        shutil.rmtree(CONFIG["output_dir"])
    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    fused_model.save_pretrained(CONFIG["output_dir"])
    tokenizer.save_pretrained(CONFIG["output_dir"])
    print(f"🎉 Fusion complete. Saved fused LoRA to: {CONFIG['output_dir']}")


if __name__ == "__main__":
    main()

