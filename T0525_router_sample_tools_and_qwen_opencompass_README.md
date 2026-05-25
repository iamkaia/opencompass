# T0525 Router Sample 檢視與 Qwen OpenCompass 執行指南

## 本輪設定

這份文件對應以下實驗：

```text
sample tasks baseline: medmcqa, race, sst2
sample tasks 4sum:     medmcqa, race, sst2 + one of boolq/rte/siqa/piqa
experts:               medmcqa, race, sst2
router objective:      correct_conf_ce
Qwen base model:       Qwen/Qwen3-4B-Instruct-2507
Qwen dtype:            float16
Qwen routed layers:    first_layer_idx=0, middle_layer_idx=18
```

SST2 的新 scoring contract 是：

```text
target=0 -> negative
target=1 -> positive
```

cache 中的 `loss_matrix` 對 SST2 計算的是 gold word option
`negative` / `positive` 的 aggregated next-token NLL，不再用數字 token `0/1`。
新 cache manifest 會記錄：

```json
"sst2_option_labels": "words"
```

## Tools Table

| Tool | 用途 | 主要輸入/輸出 |
| --- | --- | --- |
| `tools/run_T0525_qwen3_fixopt2.sh` | Qwen cache、router 訓練、逐筆快速檢視、完整 JSON export 的統一入口 | jobs: `cache_*`, `train_*`, `inspect_*`, `export_*` |
| `tools/inspect_T0525_qwen3_router_records.py` | Qwen router record 快速看 top-k target/pred/self | 讀 best epoch 的 `route_records_*_epoch*.json` |
| `tools/export_router_sample_details.py` | 將 cache pair outputs 與 router records 合併成完整 sample JSON | 輸出含 matrices 與 self/selected pair summary |
| `tools/run_T0525_llama_currentcode_cache_compare.sh` | Llama eval record、完整 JSON export，以及 current-code cache 差異比較入口 | jobs: `eval_*`, `inspect_*`, `export_*`, `cache_*`, `compare_*` |
| `tools/inspect_T0525_llama_router_records.py` | Llama router record 快速看 top-k target/pred/self | 讀 `route_records_*_eval_only.json` |
| `tools/compare_T0525_llama_cached_matrices.py` | 比較舊 Llama cache 和 current-code rebuild cache | 列出各 task matrix/target 改變數量 |
| `tools/run_T0525_qwen3_opencompass_sst2words.sh` | Qwen 五個 router 的 OpenCompass 七-task evaluation 入口 | jobs: `mrs`, `boolq`, `rte`, `siqa`, `piqa` |

## Cache 與 Router Record 存什麼

### Cache Chunk

`build_cached_router_pair_dataset.py` 產生的 `chunk_*.pt` 每個 sample 已保存：

| 欄位 | 意義 |
| --- | --- |
| `task` | sample 的 task name |
| `prompt_text` | 送給 LLM/BERT 的 canonical prompt |
| `target` | gold answer |
| `loss_matrix` | 每個 expert pair 對 gold target 的 loss/NLL |
| `correct_matrix` | 每個 expert pair 的輸出是否答對 |
| `prediction_matrix` | 每個 expert pair 實際輸出的答案 |
| `first_vec`, `mid_vec` | cache 給 router 用的 LLM hidden features |

expert 順序固定為：

```text
["medmcqa", "race", "sst2"]
```

因此所有 `3 x 3` matrix 的座標都是：

```text
[
  [medmcqa->medmcqa, medmcqa->race, medmcqa->sst2],
  [race->medmcqa,    race->race,    race->sst2],
  [sst2->medmcqa,    sst2->race,    sst2->sst2],
]
```

### Router Record

使用 `--save_route_records --eval_train_each_epoch` 訓練 router 後，record 會保存：

| 欄位 | 意義 |
| --- | --- |
| `gold_pair` | raw `loss_matrix` 最低的單一 pair |
| `pred_pair` | router argmax 選到的 pair |
| `pred_correct` | router 選到的 pair 是否答對 |
| `any_pair_correct` | 是否至少存在一個答對 pair |
| `router_target_matrix` | `correct_conf_ce` 的 supervision target distribution |
| `pair_prob_matrix` | 訓練後 router 預測的 pair probability distribution |

`router_target_matrix` 是由同一份 cached `loss_matrix` 與
`correct_matrix`，依 `correct_conf_ce` 公式重建的 target。它不是訓練
mini-batch 當時保存下來的 tensor；但在 cache 和 loss 設定不變時，數值
上就是當時 back propagation 使用的 supervision distribution。

### 完整 Sample JSON

`tools/export_router_sample_details.py` 將 cache 與 router record 合併。每筆
sample 會有：

```text
idx
cache_root
task
prompt
target
expert_names
loss_matrix
correct_matrix
prediction_matrix
router_pred_pair
router_gold_lowest_loss_pair
router_pred_matrix
router_target_matrix
router_pred_correct
any_pair_correct
self_expert
selected_expert_pair
```

其中 `self_expert` 和 `selected_expert_pair` 已整理出：

```text
pair
answer
loss
correct
router_pred_prob
router_target_prob
```

`medmcqa/race/sst2` 有 self expert；`boolq/rte/siqa/piqa` 沒有掛自己的
LoRA，因此這四種 sample 的 `self_expert` 是 `null`。

## Qwen Tools 用法

Qwen 新輸出皆使用 `sst2words` 後綴：

```text
0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words
0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words
router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert_sst2words
router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert_sst2words
router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert_sst2words
router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert_sst2words
router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert_sst2words
```

### 快速查看 Router Target 與 Prediction

```bash
bash tools/run_T0525_qwen3_fixopt2.sh inspect_mrs --split train_eval --task sst2 --limit 10 --topk 3
```

```bash
bash tools/run_T0525_qwen3_fixopt2.sh inspect_boolq --split train_eval --task boolq --limit 10 --topk 3
```

把 `inspect_boolq` 換成：

```text
inspect_rte
inspect_siqa
inspect_piqa
```

可查看另外三個 4sum router。`--split` 可用：

```text
train_eval
val
```

只看統計：

```bash
bash tools/run_T0525_qwen3_fixopt2.sh inspect_mrs --split train_eval --summary_only
```

### 匯出完整 JSON

Baseline MRS：

```bash
bash tools/run_T0525_qwen3_fixopt2.sh export_mrs_train
bash tools/run_T0525_qwen3_fixopt2.sh export_mrs_val
```

4sum：

```bash
bash tools/run_T0525_qwen3_fixopt2.sh export_boolq_train
bash tools/run_T0525_qwen3_fixopt2.sh export_boolq_val
bash tools/run_T0525_qwen3_fixopt2.sh export_rte_train
bash tools/run_T0525_qwen3_fixopt2.sh export_rte_val
bash tools/run_T0525_qwen3_fixopt2.sh export_siqa_train
bash tools/run_T0525_qwen3_fixopt2.sh export_siqa_val
bash tools/run_T0525_qwen3_fixopt2.sh export_piqa_train
bash tools/run_T0525_qwen3_fixopt2.sh export_piqa_val
```

輸出檔以例如下：

```text
T0525_qwen3_mrs_train_sample_details_sst2words.json
T0525_qwen3_mrs_val_sample_details_sst2words.json
T0525_qwen3_4sum_boolq_train_sample_details_sst2words.json
T0525_qwen3_4sum_boolq_val_sample_details_sst2words.json
```

## Llama Tools 用法

### 建 Record 與快速看

MRS record 已可直接查看：

```bash
bash tools/run_T0525_llama_currentcode_cache_compare.sh inspect_mrs --split train --task sst2 --limit 10 --topk 3
```

四個 4sum 需要先用 checkpoint 對 cache 做 `eval_only` 以補出新的
`router_target_matrix` / `router_pred_matrix` record：

```bash
bash tools/run_T0525_llama_currentcode_cache_compare.sh eval_boolq
bash tools/run_T0525_llama_currentcode_cache_compare.sh eval_rte
bash tools/run_T0525_llama_currentcode_cache_compare.sh eval_siqa
bash tools/run_T0525_llama_currentcode_cache_compare.sh eval_piqa
```

之後查看：

```bash
bash tools/run_T0525_llama_currentcode_cache_compare.sh inspect_boolq --split train --task boolq --limit 10 --topk 3
```

### 匯出完整 JSON

```bash
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_mrs_train
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_mrs_val
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_boolq_train
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_boolq_val
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_rte_train
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_rte_val
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_siqa_train
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_siqa_val
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_piqa_train
bash tools/run_T0525_llama_currentcode_cache_compare.sh export_piqa_val
```

### 比較舊 Cache 與 Current-Code Cache

在 GPU 可用時重建：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_llama_currentcode_cache_compare.sh cache_mrs
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_llama_currentcode_cache_compare.sh cache_4other
```

比較：

```bash
bash tools/run_T0525_llama_currentcode_cache_compare.sh compare_mrs
bash tools/run_T0525_llama_currentcode_cache_compare.sh compare_4other
```

## Qwen OpenCompass Evaluation

### Model Configs

| Router Run | Model Config |
| --- | --- |
| MRS baseline | `opencompass/configs/models/qwen3/moe_lora_internal_compact_cachedjoint_0525_mrs_3expert_sst2words.py` |
| MRS + BoolQ | `opencompass/configs/models/qwen3/moe_lora_internal_compact_cachedjoint_0525_4sum_boolq_mrs_3expert_sst2words.py` |
| MRS + RTE | `opencompass/configs/models/qwen3/moe_lora_internal_compact_cachedjoint_0525_4sum_rte_mrs_3expert_sst2words.py` |
| MRS + SIQA | `opencompass/configs/models/qwen3/moe_lora_internal_compact_cachedjoint_0525_4sum_siqa_mrs_3expert_sst2words.py` |
| MRS + PIQA | `opencompass/configs/models/qwen3/moe_lora_internal_compact_cachedjoint_0525_4sum_piqa_mrs_3expert_sst2words.py` |

每條 OpenCompass job 均評估同一組七個 official dataset configs：

```text
SuperGLUE_RTE_gen
piqa_gen
race_gen_sft_prompt
sst2_gen
medmcqa_gen_sft_prompt
siqa_gen
SuperGLUE_BoolQ_gen
```

這樣 baseline 與每個 4sum router 的 benchmark score 和 runtime routing
debug record 才能直接比較。

### 執行 Commands

每個 job 都會載入一份 Qwen model，並各自寫入一個獨立 log。使用以下 command 逐個啟動；同一張 GPU 請等上一條完成後再啟動下一條：

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_qwen3_opencompass_sst2words.sh mrs
```

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_qwen3_opencompass_sst2words.sh boolq
```

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_qwen3_opencompass_sst2words.sh rte
```

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_qwen3_opencompass_sst2words.sh siqa
```

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/run_T0525_qwen3_opencompass_sst2words.sh piqa
```

每次建議同一張 GPU 只跑一條。logs：

```text
T0525_opencompass_qwen3_mrs_3expert_sst2words.log
T0525_opencompass_qwen3_4sum_boolq_mrs_3expert_sst2words.log
T0525_opencompass_qwen3_4sum_rte_mrs_3expert_sst2words.log
T0525_opencompass_qwen3_4sum_siqa_mrs_3expert_sst2words.log
T0525_opencompass_qwen3_4sum_piqa_mrs_3expert_sst2words.log
```

runtime router debug records：

```text
T0525_opencompass_router_records_qwen3_mrs_3expert_sst2words.jsonl
T0525_opencompass_router_records_qwen3_4sum_boolq_mrs_3expert_sst2words.jsonl
T0525_opencompass_router_records_qwen3_4sum_rte_mrs_3expert_sst2words.jsonl
T0525_opencompass_router_records_qwen3_4sum_siqa_mrs_3expert_sst2words.jsonl
T0525_opencompass_router_records_qwen3_4sum_piqa_mrs_3expert_sst2words.jsonl
```
