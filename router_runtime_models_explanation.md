# OpenCompass Runtime Router Model 三個檔案解釋

本文依照目前工作樹中的以下三支檔案整理：

1. [`opencompass/models/router_moe_llama_internal_compact_cached_joint.py`](./opencompass/models/router_moe_llama_internal_compact_cached_joint.py)
2. [`opencompass/models/router_moe_shared.py`](./opencompass/models/router_moe_shared.py)
3. [`opencompass/models/unified_moe_core_internal_router_compact_cached_joint.py`](./opencompass/models/unified_moe_core_internal_router_compact_cached_joint.py)

這三支檔案負責 **OpenCompass evaluation runtime**：讀取已訓練好的 router checkpoint，對每筆 OpenCompass prompt 選一個 `(first expert, mid expert)` pair，掛上對應 LoRA 後生成答案。

它們不是建 cache 時呼叫的 [`router_answer_supervision_core.py`](./router_answer_supervision_core.py)。兩條路徑共享相同的 expert/feature 設計，但執行入口不同。

---

## 0. 三支檔案分工

| 檔案 | 負責內容 | 不負責內容 |
|---|---|---|
| `router_moe_llama_internal_compact_cached_joint.py` | OpenCompass model wrapper、接收 `max_out_len`、整理 prompt、紀錄 routing log | 不直接計算 router feature 或套 LoRA |
| `router_moe_shared.py` | 將 base LLM linear layer 包成 hard-routed LoRA layer、管理 `NULL_EXPERT` 與 LoRA slot | 不決定要選哪個 pair |
| `unified_moe_core_internal_router_compact_cached_joint.py` | 載入 checkpoint、抽 feature、跑 pair classifier、切換 expert、正式 generate | 不處理 OpenCompass inferencer config 的優先順序 |

一句話流程：

```text
OpenCompass config / dataset infer_cfg
    -> wrapper 收到 prompt 與 max_out_len
    -> runtime core 用 NULL_EXPERT 抽 prompt feature
    -> router 預測 first/mid expert pair
    -> shared LoRA mechanism 將指定 layer 切到該 pair
    -> base LLM + selected LoRA 生成答案
```

---

## 1. OpenCompass 真正如何呼叫這個 Model

### 1.1 Model config 建立 wrapper

例如：

```python
# opencompass/configs/models/hf_llama/moe_lora_internal_compact_cachedjoint_0524_all7_3expert.py
models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        path="meta-llama/Llama-2-7b-chat-hf",
        router_ckpt_dir="./router_0524_correct_conf_ce_t1_all7_3expert",
        lora_paths=dict(
            medmcqa="...",
            race="...",
            sst2="...",
        ),
        max_out_len=128,
    )
]
```

這個 config 建立的是 `RouterMoELlamaInternalCompactCachedJoint` wrapper，不是 training 時的 `JointAnswerSupervisionRouterModel`。

### 1.2 `max_out_len` 的 OpenCompass 呼叫鏈

```text
model config 的 max_out_len
或 dataset infer_cfg 明確指定的 max_out_len
        |
        v
opencompass/tasks/openicl_infer.py
        |
        v
GenInferencer.max_out_len
        |
        v
model.generate_from_template(..., max_out_len=...)
        |
        v
RouterMoELlamaInternalCompactCachedJoint.generate(...)
        |
        v
gen_kwargs["max_new_tokens"] = max_out_len
        |
        v
UnifiedMoECoreInternalRouterCompactCachedJoint.generate(...)
        |
        v
Hugging Face self.model.generate(...)
```

優先順序是：

```text
dataset infer_cfg.max_out_len > model config.max_out_len
```

理由是 `opencompass/tasks/openicl_infer.py` 只有在 dataset inferencer 沒寫 `max_out_len` 時，才把 model config 的值填進去。

例如：

| Dataset | Dataset config 是否自帶 `max_out_len` | 若 model config 是 `128`，runtime 有效值 |
|---|---:|---:|
| `race_gen_sft_prompt.py` | 無 | `128` |
| `medmcqa_gen_sft_prompt.py` | 無 | `128` |
| `sst2_gen.py` | 無 | `128` |
| `squad20_gen_sft_prompt.py` | `50` | `50` |

### 1.3 與 `task_max_new_tokens()` 的關係

`router_answer_supervision_core.py` 中的 `task_max_new_tokens()` 只用於：

```text
建 cached loss matrix / heavy router training 時的 generation-based supervision
```

OpenCompass runtime 走的是本文件的 wrapper/core，不會呼叫 `task_max_new_tokens()`。

---

## 2. `router_moe_llama_internal_compact_cached_joint.py`

### 2.1 它是 OpenCompass 的 Model Wrapper

核心 class：

```python
class RouterMoELlamaInternalCompactCachedJoint(HuggingFacewithChatTemplate):
```

它被 `@MODELS.register_module()` 註冊，因此 model config 可以寫：

```python
type=RouterMoELlamaInternalCompactCachedJoint
```

這個 wrapper 的工作是配合 OpenCompass `BaseModel` 介面。它不直接實作 router network，而是在 `__init__()` 裡建立：

```python
self.core = UnifiedMoECoreInternalRouterCompactCachedJoint(...)
```

真正的 LLM、LoRA 與 router forward 都在 `self.core` 中完成。

### 2.2 `__init__()` 接收哪些重要設定

| 參數 | 用途 |
|---|---|
| `path` | base LLM 路徑 |
| `router_ckpt_dir` | 已訓練 router 的 `router_config.json`、`router_heads.pt`、`encoder/` |
| `router_bert_init` | router BERT tokenizer 來源 |
| `lora_paths` | 本次 runtime 可掛載的 LoRA task -> adapter path |
| `max_out_len` | model config 的預設最大生成 token 數 |
| `max_seq_len` | runtime prompt 送進 LLM 前的最大 token 長度 |
| `force_first_task`, `force_mid_task` | debug/ablation 時強制選指定 expert |
| `debug_router_*` | 印出或保存 routing 決策 |

注意：

```python
self.max_out_len = max_out_len
```

只是保存 model 預設設定。實際每個 dataset 的 generation 長度仍可能被 dataset `infer_cfg.max_out_len` 覆蓋。

### 2.3 `_to_prompt_str()`：將 OpenCompass 輸入轉成字串

OpenCompass 可能傳入：

- 純字串；
- 包有 `prompt` / `text` / `input` 的 dict；
- chat message list。

`_to_prompt_str()` 會將這些輸入統一轉成實際送進 router 與 LLM 的 prompt string。

對 chat messages，若 tokenizer 有 `apply_chat_template()`，會使用：

```python
tokenizer.apply_chat_template(
    msgs,
    tokenize=False,
    add_generation_prompt=True,
)
```

因此 model/tokenizer 更換後，runtime prompt 文字格式可能跟舊模型不同；router prompt alignment 應以 runtime verifier 再確認。

### 2.4 `generate()`：接 OpenCompass 輸出長度，再交給 core

關鍵片段：

```python
def generate(self, inputs, max_out_len, ...):
    gen_kwargs = self.generation_kwargs.copy()
    gen_kwargs.update(kwargs)
    if max_out_len is not None:
        gen_kwargs["max_new_tokens"] = int(max_out_len)
    ...
    one_out = self.core.generate([prompt], gen_kwargs=gen_kwargs, dataset_name=dataset_name)
```

這裡完成兩件事：

1. 將 OpenCompass 使用的名稱 `max_out_len` 轉成 Hugging Face 使用的 `max_new_tokens`。
2. 將每筆 prompt 交給 runtime core 做 route-then-generate。

### 2.5 Routing log

wrapper 透過：

```python
self.core.cached_first_eid
self.core.cached_mid_eid
```

記錄每筆 prompt 最後選到的 pair。

`debug_router_record_path` 啟用時，每筆會輸出：

| 欄位 | 意義 |
|---|---|
| `dataset` | 本筆來源 dataset |
| `sample_index` | 該 dataset 內的流水號 |
| `prompt_sha1` | prompt hash，用來檢查同一筆 prompt |
| `pred_pair` | router 選到的 `task -> task` |
| `first_eid`, `mid_eid` | LLM LoRA slot id，`0` 為 null，真 expert 從 `1` 開始 |
| `prompt_preview` | prompt 前段文字 |

---

## 3. `router_moe_shared.py`

這支檔案定義 **LoRA expert 如何實際掛到 base LLM 裡面**。

### 3.1 `NULL_EXPERT_ID = 0`

```python
NULL_EXPERT_ID = 0
```

這是 **LLM LoRA slot id**，不是 router output matrix index。

假設 router 可選三個 experts：

```python
task_names = ["medmcqa", "race", "sst2"]
```

那 LLM 內的 slot 是：

| LLM LoRA slot id | 意義 |
|---:|---|
| `0` | `NULL_EXPERT`，不加 task LoRA |
| `1` | `medmcqa` LoRA |
| `2` | `race` LoRA |
| `3` | `sst2` LoRA |

而 router pair classifier 的 axis 則是：

| Router axis index | 意義 |
|---:|---|
| `0` | `medmcqa` |
| `1` | `race` |
| `2` | `sst2` |

兩者之間的轉換是：

```python
lora_slot_id = router_axis_index + 1
```

### 3.2 `HardRoutedLoRALinear`

這個 class 把原本 LLM 的一個 `nn.Linear` 包成：

```text
原始 linear 輸出 + 目前 active expert 的 LoRA 增量
```

forward 等價於：

```python
y = base_linear(x)
d = (x @ A[active_expert].T) @ B[active_expert].T
return y + (alpha / r) * d
```

重要設計：

- `base_linear` 凍結；
- 每一個 slot 都有獨立的 `A`、`B`；
- 一次只使用一個 `active_expert`，所以是 hard routing，不是 weighted sum；
- slot `0` 的 `A/B` 保持為零，使用它時就等同只使用 base model。

### 3.3 `patch_causal_lm_with_hard_routed_lora()`

它將 decoder 每一層中的以下 linear modules 換成 `HardRoutedLoRALinear`：

```text
MLP:       gate_proj, up_proj, down_proj
Attention: q_proj, k_proj, v_proj, o_proj
```

呼叫時 runtime core 會使用：

```python
num_experts = 1 + len(task_names)
```

多出來的 `1` 就是保留給 `NULL_EXPERT` 的 slot `0`。

### 3.4 `set_all_experts()` 與 `set_layer_range_expert()`

```python
set_all_experts(model, NULL_EXPERT_ID)
```

代表所有 patched LoRA layer 都回到 base model 狀態。runtime 在每筆 prompt 開始時使用它，確保 prepass 抽到的是未套 task LoRA 的 feature。

```python
set_layer_range_expert(model, start_idx, end_idx, eid)
```

只切換特定 layer 範圍使用某個 LoRA slot。

因此當 router 選出：

```text
first expert = race
mid expert = sst2
```

runtime 可以設定：

```text
前段 layers -> race LoRA slot
後段 layers -> sst2 LoRA slot
```

### 3.5 `load_lora_into_expert()`

這個函式讀取單一 adapter 的 `adapter_model.safetensors`，把對應的 `lora_A.weight` / `lora_B.weight` 複製到指定 slot。

它只負責把 LoRA 權重安裝到 slot，不負責決定 inference 時哪個 slot 被啟動；啟動決策由 runtime core 的 router 輸出控制。

---

## 4. `unified_moe_core_internal_router_compact_cached_joint.py`

這支是 runtime 真正的核心：

```python
class UnifiedMoECoreInternalRouterCompactCachedJoint:
```

它的設計是：

```text
每筆 prompt 先用 NULL_EXPERT 做 prepass
    -> 抽 first/mid LLM vector + BERT memory
    -> pair classifier 預測一組 expert pair
    -> 將前/後 layer range 掛上該 pair 的 LoRA
    -> 再做一次正式 generation
```

### 4.1 讀取 router checkpoint contract

runtime 先讀：

```python
router_ckpt_dir/router_config.json
```

重要欄位：

| 欄位 | 用途 |
|---|---|
| `task_names` | router output axis 的 expert 名稱與順序 |
| `first_layer_idx`, `middle_layer_idx` | 前/後段切點以及抽 feature 的 layer |
| `router_max_len` | BERT router prompt 的最大長度 |
| `router_pooling`, `router_pooling_last_k` | LLM hidden vector 如何 pool 成 feature |
| `router_dim` | router feature encoder 維度 |
| `llama_hidden_size` | 建 cache/訓練時 LLM hidden size，供 runtime 檢查 model 相容性 |
| `sample_feature_mode`, `sample_feature_dim` | runtime 僅允許 `none` / `0` |

### 4.2 為什麼 `lora_paths` 必須與 checkpoint 完全對上

程式會檢查：

```python
missing_lora_tasks = [task for task in self.task_names if task not in lora_paths]
extra_lora_tasks = [task for task in lora_paths if task not in self.task_names]
```

原因是 router classifier 輸出順序由 checkpoint 的 `task_names` 固定。

若 checkpoint 定義：

```python
["medmcqa", "race", "sst2"]
```

則 pair classifier 的 `3 x 3` 輸出 axis 就是這三個 expert。runtime 不可以臨時少掛一個、增加一個，或用不同排序解讀相同 logits。

### 4.3 載入 base LLM 與 LoRA slots

```python
self.model = patch_llama_with_hard_routed_lora(
    self.model,
    num_experts=1 + len(self.task_names),
    ...
)
```

然後：

```python
for task in self.task_names:
    task_id = self.task_names.index(task)
    expert_id = task_id + 1
    load_lora_into_expert(self.model, adapter_dir, expert_id)
```

再次強調：

```text
pair classifier index 0 = task_names[0]
LLM LoRA slot id     1 = task_names[0]
LLM LoRA slot id     0 = NULL_EXPERT
```

### 4.4 載入 router heads

runtime 建立：

```python
self.bert_encoder
self.router_first
self.router_mid
self.pair_classifier
```

並從：

```python
router_ckpt_dir/router_heads.pt
```

載入權重。

`router_first` 與 `router_mid` 各自將：

```text
LLM prompt vector + BERT memory
```

轉成 feature，最後：

```python
pair_logits = self.pair_classifier(torch.cat([first_feat, mid_feat], dim=-1))
```

一次輸出所有 `(first, mid)` pair 的 logits。

### 4.5 每筆 prompt 的 runtime generation 流程

`generate()` 對每筆 prompt 做以下事情。

#### Step 1：清空上筆 routing 狀態

```python
self._reset_runtime_cache()
```

其中最重要的是：

```python
set_all_experts(self.model, NULL_EXPERT_ID)
```

確保抽 feature 前沒有沿用前一筆 sample 選到的 LoRA。

#### Step 2：用相同 prompt 建 BERT memory

```python
self._encode_bert_memory(prompt)
```

BERT 會使用 `router_max_len` 截斷 prompt，回傳倒數第二層與最後一層 hidden states，供 router feature encoder 做 cross-attention。

#### Step 3：以 NULL_EXPERT 執行 LLM prepass

```python
prepass = self.model(
    **inp,
    output_hidden_states=True,
)
```

因為 Step 1 已將所有 LoRA slot 設為 null，此處的 hidden states 是 base LLM prompt representation，不是任一 candidate expert 的結果。

#### Step 4：依 training contract pooling prompt vector

```python
first_vec = self._pool_prompt_vector(hidden_states[self.first_layer_idx], inp["attention_mask"])
mid_vec = self._pool_prompt_vector(hidden_states[self.middle_layer_idx], inp["attention_mask"])
```

可用 pooling：

| `router_pooling` | 意義 |
|---|---|
| `last_token` | 取 prompt 最後 token 的 hidden vector |
| `mean` | 對所有有效 prompt token 平均 |
| `lastk_mean` | 對最後 `router_pooling_last_k` 個有效 token 平均 |

這個設定必須和建 cache/訓練 checkpoint 時一致，否則 runtime router 看到的 feature distribution 會改變。

#### Step 5：預測 pair

```python
pred_pair = int(pair_logits.argmax(dim=-1).item())
first_eid = (pred_pair // num_tasks) + 1
mid_eid = (pred_pair % num_tasks) + 1
```

`pair_logits` 的 index 是 `0..E*E-1`；加 `1` 後才成為實際 LLM LoRA slot id，因為 slot `0` 保留給 null。

例如三個 experts：

```python
task_names = ["medmcqa", "race", "sst2"]
```

| `pred_pair` | Router pair | 實際 LLM slots |
|---:|---|---|
| `0` | `medmcqa -> medmcqa` | `1 -> 1` |
| `1` | `medmcqa -> race` | `1 -> 2` |
| `5` | `race -> sst2` | `2 -> 3` |

#### Step 6：掛上選中的 LoRA pair

```python
set_layer_range_expert(self.model, self.first_layer_idx, self.middle_layer_idx - 1, first_eid)
set_layer_range_expert(self.model, self.middle_layer_idx, self.num_layers - 1, mid_eid)
```

前半段 layer 使用 `first_eid`，後半段 layer 使用 `mid_eid`。

#### Step 7：正式 generation

```python
args["do_sample"] = False
args["temperature"] = 0.0
args["top_p"] = 1.0
args["num_beams"] = 1
out = self.model.generate(**inp, **args)
```

這裡 runtime 明確強制：

```text
greedy decoding
```

即使外部傳入 sampling 或 beam search 設定，也會被這四行覆蓋。

### 4.6 `force_first_task` / `force_mid_task`

這兩個欄位用於 debug 或 ablation：

```python
force_first_task="race"
force_mid_task="sst2"
```

若有指定，runtime 會將不符合條件的 pair logits mask 成 `-inf`，迫使 argmax 選指定方向。

它不改 router checkpoint 權重，只限制推論時可被選的 pair。

---

## 5. 與 Cache/Training 路徑的關係

| 行為 | Cache / training scoring | OpenCompass runtime evaluation |
|---|---|---|
| 入口 | `router_answer_supervision_core.py` | 本文件三支 runtime 檔案 |
| Router feature prepass | `NULL_EXPERT` | `NULL_EXPERT` |
| Pair axis | `expert_names x expert_names` | checkpoint `task_names x task_names` |
| LoRA slot `0` | `NULL_EXPERT` | `NULL_EXPERT` |
| 真 expert slot | router index `+1` | router index `+1` |
| Classification/MCQ scoring | first-token full-vocab NLL | OpenCompass generation + evaluator |
| QA/translation scoring | generation evaluator | OpenCompass generation + evaluator |
| Generation 長度來源 | `task_max_new_tokens()` | OpenCompass `max_out_len` -> `max_new_tokens` |

因此兩條路徑要一致的重點是：

1. prompt string；
2. expert 名稱與順序；
3. router pooling / layer index / hidden size feature contract；
4. 對同樣使用 generation supervision 的 task，輸出長度與 decoding 規則。

---

## 6. 相對於 Git `HEAD` 的現行工作樹差異

比較方式：

```bash
git diff HEAD -- \
  opencompass/models/router_moe_llama_internal_compact_cached_joint.py \
  opencompass/models/router_moe_shared.py \
  opencompass/models/unified_moe_core_internal_router_compact_cached_joint.py
```

目前三支檔案都有未提交修改。`git log` 中最近一次觸及這三支檔案的 commit 是：

```text
01826323 0513:[ADD] debug_router_record_path: 輸出每筆 sample 的 routing JSONL
```

### 6.1 `router_moe_llama_internal_compact_cached_joint.py`

現行工作樹新增：

```python
force_first_task: Optional[str] = None
force_mid_task: Optional[str] = None
```

並將它們傳進 runtime core：

```python
force_first_task=force_first_task
force_mid_task=force_mid_task
```

`HEAD` 中原本固定傳：

```python
force_first_task=None
force_mid_task=None
```

影響：

- 現在 model config 可直接強制 first/mid expert，方便做 self-pair 或固定 pair ablation。
- 正常未設定時，行為與 `HEAD` 相同。

### 6.2 `router_moe_shared.py`

現行工作樹只修改中文註解：

- 說明本檔放置 LoRA expert mechanism。
- 說明 `set_all_experts()` 會被用於抽 cache/runtime feature 時清除 task LoRA。

影響：

```text
沒有執行行為差異。
```

### 6.3 `unified_moe_core_internal_router_compact_cached_joint.py`

這支有實際 runtime contract 改動。

#### 新增 checkpoint feature contract 讀取

現行工作樹會從 `router_config.json` 讀取：

```python
router_pooling
router_pooling_last_k
router_dim
sample_feature_mode
sample_feature_dim
```

`HEAD` 版本沒有完整讀取這些欄位。

#### 禁止 runtime 不支援的 sample feature

現行工作樹新增：

```python
if self.sample_feature_mode != "none" or self.sample_feature_dim != 0:
    raise ValueError(...)
```

意義是：如果 checkpoint 是用額外 cached sample feature 訓練的，但 runtime 沒有同樣 feature，直接拒絕執行，避免靜默輸入不一致。

#### 檢查 base LLM hidden size

現行工作樹新增：

```python
expected_llama_hidden = cfg.get("llama_hidden_size")
if expected_llama_hidden != self.model.config.hidden_size:
    raise ValueError(...)
```

意義是：避免拿某個 hidden dimension 訓練的 router checkpoint，錯掛到不同 hidden dimension 的 base model。

#### `router_dim` 改由 checkpoint 決定

`HEAD` 建 router heads 時使用建構參數 `router_dim`；現行工作樹使用：

```python
self.router_dim = int(cfg.get("router_dim", router_dim))
```

並用 `self.router_dim` 建立 module。

意義是：runtime 優先依 checkpoint 原始架構重建 router，而不是依 evaluation config 意外傳入不同維度。

#### Runtime pooling 由固定 last-token 改為依 checkpoint contract

`HEAD` 原本固定：

```python
first_vec = hidden_states[self.first_layer_idx][:, -1, :]
mid_vec = hidden_states[self.middle_layer_idx][:, -1, :]
```

現行工作樹新增 `_pool_prompt_vector()`，支援：

```python
last_token
mean
lastk_mean
```

並使用 checkpoint 的 `router_pooling` / `router_pooling_last_k`。

影響：

- 如果 cache/training 使用 `mean` 或 `lastk_mean`，現行 runtime 可以保持 feature contract 對齊。
- `HEAD` 會一律取 last token，對非 `last_token` checkpoint 會造成 runtime feature mismatch。

---

## 7. 目前最重要的判斷

### `task_max_new_tokens()` 會不會影響這三支 runtime 檔案？

不會。

OpenCompass runtime 使用：

```text
dataset infer_cfg.max_out_len 或 model config.max_out_len
    -> wrapper.generate(max_out_len=...)
    -> gen_kwargs["max_new_tokens"]
    -> runtime core self.model.generate()
```

### `NULL_EXPERT_ID=0` 是否與 router pair index 衝突？

不衝突。

```text
LLM LoRA slot 0      = NULL_EXPERT
router pair axis 0   = task_names[0] 的真 expert
兩者透過 +1 轉換
```

### 現行工作樹相對 `HEAD` 最重要的改善是什麼？

runtime 已開始遵守 training/cache checkpoint 內的 feature contract，特別是：

```text
router_pooling
router_pooling_last_k
router_dim
llama_hidden_size
sample_feature_mode
```

其中 pooling 對齊最直接影響 router 選 pair 的輸入內容。
