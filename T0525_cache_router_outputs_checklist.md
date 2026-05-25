# T0525 Cache 與 Router 訓練輸出檢查指南

這份文件對應 `0525_4sum_commands_llama_qwen.md` 的實驗設計：

- expert 永遠只有三個：`medmcqa`, `race`, `sst2`
- baseline router 只用 `medmcqa,race,sst2` 樣本訓練
- 4sum router 分別多加入一個 task：`boolq`, `rte`, `siqa`, `piqa`
- loss 使用 `correct_conf_ce`
- 最後要檢查 4sum 後，原本三個 task 在相同 MRS validation set 上的 routing 分布是否仍接近 baseline

## 1. 最重要的結論：最後應該看哪些檔

若只是快速判斷實驗結果，優先順序如下。

| 目的 | 最先看的檔案 | 看什麼 |
| --- | --- | --- |
| 確認 cache 建對了 | `<cache_root>/cache_config.json` 與 `<cache_root>/validation/manifest.json` | model、dtype、layer、task、expert、樣本數、score mode |
| 確認 router 訓練完成且最佳 checkpoint 是哪個 epoch | `<router_out_dir>/router_config.json` 與 `<router_out_dir>/best_metrics.json` | `best_epoch`, `best_route_correct_acc`, `best_val_loss`, validation metrics |
| 確認三個 expert task 是否 routing 回自己 | `<router_out_dir>/best_metrics.json` 內的 `metrics.routing_summary.per_task` | `pred_self_first_rate`, `pred_self_mid_rate`, `top_pred_pairs` |
| 確認 4sum 是否破壞 baseline 的 routing 分布 | `<eval_out_dir>/routing_summary_val_eval_only.json` | 在同一份 MRS validation 上比較 `all_pred_pairs` 與 `per_task` |
| 找出為什麼某些 sample 被 route 到別的 expert | `<router_out_dir>/route_records_val_epoch<best_epoch>.json` 或 eval-only 對應檔 | 每筆 `pred_pair`, `gold_pair`, `pred_correct`, `pred_answer`, `gold_answer`, raw loss |
| 看 cache 原始 oracle/self pair 品質 | `tools/inspect_cached_router_pair_chunk.py` | cache 內 loss/correct/prediction matrix 與 self-vs-oracle 統計 |

注意：你的最終目標是「4sum router 在原本三個 task 上，維持與 baseline 類似的 router 分布」。因此不能只看 4sum 訓練自己的 validation 指標，因為 4sum validation 會含新 task。必須把每個 4sum checkpoint 都用 `eval_only` 跑在同一個 MRS cache 上，再和 baseline 比。

## 2. 本次路徑與目前已存在的產物

### Cache roots

| Model | Cache 類型 | 路徑 | 預期內容 | 目前狀態 |
| --- | --- | --- | --- | --- |
| Llama | 原始三 task | `0525_llama_cache_mrs_3expert_official_eval_aligned` | train `600`、validation `150` | 已存在 |
| Llama | 四個額外候選 task | `0525_llama_cache_4other_3expert_official_eval_aligned` | train `800`、validation `200` | 已存在 |
| Qwen FP16 | 原始三 task | `0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned` | train `600`、validation `150` | 已存在 |
| Qwen FP16 | 四個額外候選 task | `0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned` | train `800`、validation `200` | 已存在 |

### Router output directories

目前 workspace 實際已看到：

```text
router_0525_llama_correct_conf_ce_t1_mrs_3expert/
```

這是 Llama baseline router，已包含最佳 checkpoint 與 epoch summary。

按照 command 文件完成後，還會有以下 training output directories：

```text
router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert/
router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert/
router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert/
router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert/

router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert/
router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert/
router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert/
router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert/
router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert/
```

按照 command 文件完成 4sum 後的 MRS-only 評估，還會有以下 evaluation directories：

```text
eval_0525_llama_4sum_boolq_on_mrs_3expert/
eval_0525_llama_4sum_rte_on_mrs_3expert/
eval_0525_llama_4sum_siqa_on_mrs_3expert/
eval_0525_llama_4sum_piqa_on_mrs_3expert/

eval_0525_qwen3_fp16_4sum_boolq_on_mrs_3expert/
eval_0525_qwen3_fp16_4sum_rte_on_mrs_3expert/
eval_0525_qwen3_fp16_4sum_siqa_on_mrs_3expert/
eval_0525_qwen3_fp16_4sum_piqa_on_mrs_3expert/
```

## 3. 建完 Cache 之後會存什麼

Cache 是昂貴的 LLM + LoRA route-pair scoring 結果。後面的 router 訓練只讀 cache，不會重新跑三個 LoRA expert 做 scoring。

每個 cache root 結構如下：

```text
<cache_root>/
├── cache_config.json
├── train/
│   ├── manifest.json
│   └── chunk_00000.pt
└── validation/
    ├── manifest.json
    └── chunk_00000.pt
```

實作位置：

- `build_cached_router_pair_dataset.py::main()` 寫 `cache_config.json`
- `build_cached_router_pair_dataset.py::process_split()` 寫 split manifest 與 chunk
- `build_cached_router_pair_dataset.py::save_chunk()` 以 `torch.save({"items": items}, ...)` 寫 `.pt`

### 3.1 `cache_config.json`

這是整個 cache 的建立設定。最需要確認的欄位如下。

| 欄位 | 意義 | 本次 Llama | 本次 Qwen FP16 |
| --- | --- | --- | --- |
| `task_names` | 這個 cache 收進來的 sample task | MRS 或 4other | MRS 或 4other |
| `expert_names` | 實際可 route 的 expert | `medmcqa,race,sst2` | `medmcqa,race,sst2` |
| `base_model_path` | 產生特徵與 pair score 的 base LLM | `meta-llama/Llama-2-7b-chat-hf` | `Qwen/Qwen3-4B-Instruct-2507` |
| `dtype` | 建 cache 時 LLM 計算 dtype | `float16` | `float16` |
| `score_mode` | pair correctness/loss 生成模式 | `official_eval_aligned_generation` | `official_eval_aligned_generation` |
| `first_layer_idx` | 第一段 router feature 所取 layer 起點 | `0` | `0` |
| `middle_layer_idx` | 中間 router feature 切點 | `15` | `18` |
| `llama_hidden_size` | LLM hidden dimension；欄位名是 legacy 命名 | `4096` | `2560` |
| `router_pooling` | hidden state pool 方法 | `mean` | `mean` |

第一個檢查：

```bash
cat 0525_llama_cache_mrs_3expert_official_eval_aligned/cache_config.json
cat 0525_llama_cache_4other_3expert_official_eval_aligned/cache_config.json
cat 0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned/cache_config.json
cat 0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned/cache_config.json
```

### 3.2 `train/manifest.json` 與 `validation/manifest.json`

Manifest 是 trainer 讀取 cache 時的入口。它列出 chunk 檔、樣本數、task/expert contract 與 feature contract。

本次應看到的樣本數：

| Cache | train `num_items` | validation `num_items` | `task_names` |
| --- | ---: | ---: | --- |
| MRS cache | 600 | 150 | `medmcqa,race,sst2` |
| 4other cache | 800 | 200 | `boolq,rte,siqa,piqa` |

本次所有 manifest 都應共同滿足：

```json
{
  "expert_names": ["medmcqa", "race", "sst2"],
  "num_tasks": 3,
  "supervision_type": "cached_loss_matrix",
  "has_correct_matrix": true,
  "has_option_prob_matrix": false,
  "has_prediction_matrix": true,
  "score_mode": "official_eval_aligned_generation"
}
```

`has_option_prob_matrix=false` 很重要：這批 cache 有 evaluator-aligned prediction/correctness matrix，但沒有 option probability matrix。因此 `option_prob_available_ratio` 應為 `0.0`，不要拿 weighted option probability 類型的 metric 當主要結果。

第二個檢查：

```bash
cat 0525_llama_cache_mrs_3expert_official_eval_aligned/train/manifest.json
cat 0525_llama_cache_mrs_3expert_official_eval_aligned/validation/manifest.json
cat 0525_llama_cache_4other_3expert_official_eval_aligned/train/manifest.json
cat 0525_llama_cache_4other_3expert_official_eval_aligned/validation/manifest.json
```

Qwen 時將上面兩個 cache root 換成：

```text
0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned
0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned
```

### 3.3 `chunk_00000.pt` 每筆樣本內有什麼

每個 chunk 是一個 `{"items": [...]}` 的 PyTorch object。每筆 sample 保存：

| 欄位 | 意義 | 判讀用途 |
| --- | --- | --- |
| `text` | canonical prompt；供 legacy reader 使用 | 應與 `prompt_text` 一樣 |
| `source_text` | 此 cache 寫入時同樣保存 canonical prompt | 應與 `text`、`prompt_text` 一樣 |
| `prompt_text` | router/BERT 實際看的 prompt | prompt contract 檢查 |
| `original_source_text` | dataset 原始來源文字 | 追查資料來源用 |
| `target` | gold target | 判讀答案 |
| `task` | sample 屬於哪個 dataset | 分 task 統計 |
| `task_id` | 若 task 也是 expert，對應的 expert index；非 expert task 可能是 `-1` | 只有 MRS task 有 self-expert 意義 |
| `first_vec` | LLM prompt feature 的第一段向量，轉成 `float16` 存 CPU | router input |
| `mid_vec` | LLM prompt feature 的中段向量，轉成 `float16` 存 CPU | router input |
| `loss_matrix` | 每個 route pair 的 loss/cost，shape 是 `3 x 3` | oracle pair 與 `correct_conf_ce` weighting |
| `correct_matrix` | 每個 route pair 最終 prediction 是否正確，shape 是 `3 x 3` | `route_correct_acc` 與 training target |
| `prediction_matrix` | 每個 route pair 產生/後處理後的答案 | sample-level debugging |
| `pair_label` | `loss_matrix` 最小值對應的 flatten pair id | raw oracle pair label |
| `first_label` | raw oracle pair 第一段 expert index | raw oracle first expert |
| `mid_label` | raw oracle pair 第二段 expert index | raw oracle mid expert |

三個 expert 的 index 順序固定為：

```text
0 = medmcqa
1 = race
2 = sst2
```

因此 `3 x 3` matrix 的 row 是 first expert，column 是 mid expert，flatten pair id 如下：

| pair id | Route pair |
| ---: | --- |
| 0 | `medmcqa->medmcqa` |
| 1 | `medmcqa->race` |
| 2 | `medmcqa->sst2` |
| 3 | `race->medmcqa` |
| 4 | `race->race` |
| 5 | `race->sst2` |
| 6 | `sst2->medmcqa` |
| 7 | `sst2->race` |
| 8 | `sst2->sst2` |

### 3.4 Cache 建好後實際建議看的 summary

repo 已有 inspection tool：

```bash
/home/u9472191/.conda/envs/opencompass/bin/python tools/inspect_cached_router_pair_chunk.py --cache_root 0525_llama_cache_mrs_3expert_official_eval_aligned --split validation --summary_only
```

需要看某個 task 的 sample 細節與所有 `3 x 3` matrix：

```bash
/home/u9472191/.conda/envs/opencompass/bin/python tools/inspect_cached_router_pair_chunk.py --cache_root 0525_llama_cache_mrs_3expert_official_eval_aligned --split validation --task medmcqa --limit 3 --show_matrices --show_pair_ranking
```

這個 tool 的 `oracle=self summary` 是 cache 本身的 oracle 統計，不是已訓練 router 的 prediction 統計：

| tool 輸出 | 意義 |
| --- | --- |
| `self_pair_acc` | loss 最低的 oracle pair 是否恰好是 task 自己的 expert pair |
| `self_answer_acc` | 固定用 self pair 時答案正確率 |
| `oracle_answer_acc` | 用 loss 最低的 pair 時答案正確率 |
| `avg_gap` | `self_loss - oracle_loss`；越大代表固定 self 比 oracle 差得越多 |
| `top_oracle_pairs` | cache label 最常出現的最小 loss pair |

## 4. `correct_conf_ce` 到底在訓練什麼

這批 router command 使用：

```text
--joint_loss correct_conf_ce
--correct_soft_ce_temperature 1.0
--pair_loss_normalization sample_minmax
--supervision_mode oracle_loss
```

實作在 `router_pair_common.py::compute_pair_losses()`。

它的邏輯不是「強迫 medmcqa sample 永遠 route 到 `medmcqa->medmcqa`」：

1. 對每筆 sample，先從 `correct_matrix` 找到所有答對的 route pairs。
2. 只在這些答對 pairs 中，用 normalized `loss_matrix` 產生 soft target。
3. loss 越低的正確 pair 會獲得越高 target probability。
4. router 用 cross entropy 去擬合這個「正確而且較低 loss」的 pair distribution。
5. 若該 sample 沒有任何正確 pair，該 sample 不會提供 `correct_conf_ce` target。

所以你要分開看以下三件事：

| 問題 | 看哪個指標 |
| --- | --- |
| router 是否選到 cache 中 loss 最小的 pair | `pair_acc`, `pred_pair == gold_pair`, `match` |
| router 選的 pair 是否答對題目 | `route_correct_acc`, route record 的 `pred_correct` |
| router 是否維持三個原始 task route 回自己 | `routing_summary.per_task[].pred_self_first_rate`, `pred_self_mid_rate`, `top_pred_pairs` |

重要提醒：

- `pair_acc` 低不一定代表 `correct_conf_ce` 訓練失敗，因為可能有多個 route pair 都能答對。
- `route_correct_acc` 高也不代表 self-routing 維持住，因為它可能靠別的 expert pair 答對。
- 你的主要目標需要另外檢查 `per_task` 的 predicted pair distribution。

## 5. Router 訓練完之後會存什麼

訓練輸出目錄，例如：

```text
router_0525_llama_correct_conf_ce_t1_mrs_3expert/
├── train_config.json
├── router_config.json
├── best_metrics.json
├── encoder/
│   ├── config.json
│   └── model.safetensors
├── router_heads.pt
├── routing_summary_train_epoch1.json
├── routing_summary_val_epoch1.json
├── routing_summary_train_eval_epoch1.json
├── oracle_debug_train_epoch1.json
├── oracle_debug_val_epoch1.json
├── route_records_val_epoch1.json
└── route_records_train_eval_epoch1.json
```

其中 epoch 檔案會從 `1` 持續到 early stopping 或訓練結束。你這次 command 帶有：

```text
--eval_train_each_epoch
--save_route_records
--early_stop_patience 2
```

因此每個 epoch 都會額外保留 validation 與 train-eval 的 summary/records；但真正可拿去 load 的模型只有最佳 checkpoint 寫入的 `encoder/` 與 `router_heads.pt`。

### 5.1 `train_config.json`

這是實際跑訓練時解析完成的參數與 cache contract。優先確認：

| 欄位 | 要確認什麼 |
| --- | --- |
| `resolved_feature_roots` | baseline 只應含 MRS cache；4sum 應含 MRS 與 4other cache |
| `resolved_sample_task_names` | baseline 是三個 task；4sum 是三個 task 加指定的一個新 task |
| `resolved_expert_names` | 永遠只有 `medmcqa,race,sst2` |
| `joint_loss` | `correct_conf_ce` |
| `best_metric` | `route_correct_acc` |
| `feature_contract` | 與 cache 的 layer/hidden size/pooling 完全一致 |

4sum 訓練雖然提供兩個 cache roots，但 trainer 會根據 `--sample_task_names` 過濾樣本。例如 `4sum_boolq` 只取 `medmcqa,race,sst2,boolq`，不會同時拿到 `rte,siqa,piqa`。

另外，trainer 會檢查合併的 cache 是否有一致的：

```text
expert_names
score_mode
option_prob_normalization
sst2_option_labels
first_layer_idx
middle_layer_idx
router_pooling
router_pooling_last_k
llama_hidden_size
```

所以 Llama cache 不能和 Qwen cache 混在同一個 training command 中。

### 5.2 `router_config.json`

這是最佳 checkpoint 的 runtime contract 與最佳 epoch 摘要。主要欄位：

| 欄位 | 意義 |
| --- | --- |
| `expert_names` | router 能輸出的 expert 集合 |
| `train_task_names` | 這次 router training 用到的 sample tasks |
| `num_pairs` | 這次是 `3 x 3 = 9` |
| `joint_loss` | 使用的 loss |
| `best_epoch` | 寫入 checkpoint 的最佳 epoch |
| `best_metric` | 選 checkpoint 的依據 |
| `best_metric_value` | 最佳 validation metric 值 |
| `best_route_correct_acc` | 最佳 epoch 的 router correctness |
| `best_val_loss` | 最佳 epoch 的 validation objective loss |

### 5.3 `encoder/` 與 `router_heads.pt`

這兩者才是實際模型參數：

| 檔案 | 保存內容 |
| --- | --- |
| `encoder/model.safetensors` | BERT encoder 的 HuggingFace `save_pretrained` 輸出 |
| `encoder/config.json` | BERT encoder config |
| `router_heads.pt` | `bert_encoder`, `router_first`, `router_mid`, `pair_classifier` state dict |

後續 `--load_from <router_out_dir>` 主要就是從該目錄載入 router checkpoint，以便在固定 MRS validation cache 上做 `eval_only`。

### 5.4 `best_metrics.json`

這是最快判斷 router 訓練結果的檔。它只對應最佳 epoch，而不是最後一個 epoch。

要優先讀：

| 欄位 | 意義 | 你的判斷方式 |
| --- | --- | --- |
| `best_epoch` | 哪個 epoch 被保存成 checkpoint | 確認 early stop 下最佳點 |
| `metrics.route_correct_acc` | router 選到的 pair 是否答對 | 主要 correctness 指標 |
| `metrics.fixed_self_correct_acc` | 永遠強制 self pair 的答對率 | router 是否比固定 self 好的參考 |
| `metrics.any_pair_correct_rate` | cache 中至少存在一個正確 pair 的比例 | correctness 上限背景 |
| `metrics.pair_acc` | router pair 是否完全等於最小 loss oracle pair | 不等同答對率 |
| `metrics.self_pair_acc` | router prediction 是否為 self pair，僅原始三 task 適用 | self-routing 全局概況 |
| `metrics.oracle_self_pair_acc` | cache 最小 loss oracle 是否為 self pair | 用來理解 label，不是 router 預測 |
| `metrics.routing_summary.per_task` | 每個 task 的 routing distribution | 你的核心檢查 |

檢查 command：

```bash
jq '{best_epoch, route_correct_acc: .metrics.route_correct_acc, fixed_self_correct_acc: .metrics.fixed_self_correct_acc, pair_acc: .metrics.pair_acc, self_pair_acc: .metrics.self_pair_acc, oracle_self_pair_acc: .metrics.oracle_self_pair_acc, per_task: [.metrics.routing_summary.per_task[] | {task, pred_self_first_rate, pred_self_mid_rate, gold_self_pair_rate, top_pred_pairs}]}' router_0525_llama_correct_conf_ce_t1_mrs_3expert/best_metrics.json
```

### 5.5 `routing_summary_*_epochN.json`

這些檔案專門描述 routing distribution：

| 檔名 | 含義 |
| --- | --- |
| `routing_summary_train_epochN.json` | 當 epoch 訓練過程中的 predicted distribution |
| `routing_summary_val_epochN.json` | 當 epoch 在 validation 上的 distribution |
| `routing_summary_train_eval_epochN.json` | epoch 結束後重新 eval train set 的 distribution |

檔案中的核心欄位：

| 欄位 | 意義 |
| --- | --- |
| `all_pred_pairs` | router 實際預測的完整 9-pair 分布 |
| `all_gold_pairs` | cache loss 最小 pair 的完整 9-pair 分布 |
| `per_task[].pred_self_first_rate` | 該 task 的 first router 選到自身 expert 比例 |
| `per_task[].pred_self_mid_rate` | 該 task 的 mid router 選到自身 expert 比例 |
| `per_task[].gold_self_pair_rate` | cache oracle pair 是自身 pair 的比例 |
| `per_task[].top_pred_pairs` | 該 task 上 router 最常預測的 pair |
| `per_task[].all_pred_pairs` | 該 task 上完整 predicted pair distribution |

對你關心的 `medmcqa`, `race`, `sst2`：

- `top_pred_pairs[0].name` 最理想分別是 `medmcqa->medmcqa`、`race->race`、`sst2->sst2`
- `pred_self_first_rate` 與 `pred_self_mid_rate` 越接近 baseline 越符合 4sum 目標
- 更完整的比較應看 `all_pred_pairs` 分布，而不只看 top-1 pair

### 5.6 `oracle_debug_*_epochN.json`

這些檔只看 cache oracle 結構與 self-vs-oracle 差距，不是 router predicted distribution。

| 欄位 | 意義 |
| --- | --- |
| `avg_gap`, `p50_gap`, `p90_gap` | 最佳 pair 與次佳 pair 的 loss 差距；gap 小代表 label 容易搖擺 |
| `avg_self_minus_oracle` | 固定 self pair loss 減掉 oracle pair loss；越大代表 oracle 明顯偏離 self |
| `top_gold_pairs` | 最小 loss oracle pair 分布 |
| `per_task` | 各 task 的 oracle 結構 |

它能回答「為什麼 oracle 本身不是 self」；不能單獨回答「訓練後 router 是否仍回自己」。

### 5.7 `route_records_*_epochN.json`

你用了 `--save_route_records`，所以每筆 evaluated sample 都會寫出細節。

| 欄位 | 意義 |
| --- | --- |
| `task` | sample dataset |
| `pred_pair` | router 選出的 pair |
| `gold_pair` | cache 最小 loss pair |
| `match` | `pred_pair == gold_pair` |
| `pred_correct` | router 選的 pair 是否答對 |
| `gold_correct` | oracle pair 是否答對 |
| `any_pair_correct` | 九個 pairs 內是否至少有一個答對 |
| `pred_raw_loss` | router pair 的 raw cached loss |
| `gold_raw_loss` | oracle pair 的 raw cached loss |
| `pred_answer` | router pair 的最終答案 |
| `gold_answer` | oracle pair 的最終答案 |
| `prompt_sha1` | prompt hash，用於追 sample |
| `prompt_text`, `target` | 實際 prompt 與 gold target |

判讀時請分開：

| 現象 | 解讀 |
| --- | --- |
| `match=true` | router 完整模仿最小 loss pair |
| `match=false` 且 `pred_correct=true` | router 沒模仿 oracle，但仍選到正確 pair；對 `correct_conf_ce` 不一定是問題 |
| `pred_pair` 不是 self，但 `pred_correct=true` | 答對了，但不符合你想維持 self-routing 的額外目標 |
| `any_pair_correct=false` | 該 sample 在這三個 experts 的所有 route pairs 下都無法答對；router 無法靠選 pair 解決 |

## 6. `eval_only` 完成後會存什麼

`eval_only` 用來把一個已訓練好的 4sum router，重新在相同的 MRS cache 上跑 evaluation。這是比較 4sum 是否改變原三 task routing 分布的正確方式。

每個 eval output directory 會有：

```text
<eval_out_dir>/
├── train_config.json
├── routing_summary_val_eval_only.json
├── oracle_debug_val_eval_only.json
├── route_records_val_eval_only.json
├── routing_summary_train_eval_only.json
├── oracle_debug_train_eval_only.json
└── route_records_train_eval_only.json
```

因為 `eval_only` 不訓練新 router，它不會產生新的：

```text
encoder/
router_heads.pt
router_config.json
best_metrics.json
```

你要看的主要是：

```text
routing_summary_val_eval_only.json
route_records_val_eval_only.json
```

## 7. 你的最終 4sum 比較流程

### 7.1 Baseline 對照

Llama baseline 目前已存在：

```text
router_0525_llama_correct_conf_ce_t1_mrs_3expert/best_metrics.json
```

它的 best epoch 是 `3`，在 MRS validation `150` 筆樣本上，已看到：

| 指標 | 值 | 解讀 |
| --- | ---: | --- |
| `route_correct_acc` | `0.5733` | router pair 答對率 |
| `fixed_self_correct_acc` | `0.5800` | 固定 self pair 答對率 |
| `pair_acc` | `0.5200` | router 完整等於最低 loss oracle pair 的比例 |
| `self_pair_acc` | `0.9533` | router 在三個原 task 上預測 self pair 的總比例 |
| `oracle_self_pair_acc` | `0.5067` | cache oracle pair 本身是 self 的比例 |
| `router_argmax_score` | `85.2090` | normalized cost 導出的 router score |
| `fixed_self_score` | `85.3892` | normalized cost 導出的固定 self score |

各 task 的 baseline routing 分布：

| Task | `pred_self_first_rate` | `pred_self_mid_rate` | Top predicted pair | Top pair rate | `gold_self_pair_rate` |
| --- | ---: | ---: | --- | ---: | ---: |
| `medmcqa` | `0.98` | `0.86` | `medmcqa->medmcqa` | `0.86` | `0.10` |
| `race` | `1.00` | `1.00` | `race->race` | `1.00` | `0.66` |
| `sst2` | `1.00` | `1.00` | `sst2->sst2` | `1.00` | `0.76` |

這組數字直接說明：

- baseline router 的 predicted distribution 幾乎是回自己的 task，尤其 `race` 與 `sst2`。
- `medmcqa` 仍有一部分 middle route 到 `race`。
- oracle distribution 並不全是 self，尤其 `medmcqa` 的 `gold_self_pair_rate=0.10`；因此你的 self-routing 目標不能只拿 `gold_pair`/`pair_acc` 代表。

### 7.2 每個 4sum router 都要在同一個 MRS validation 上比較

Llama 比較檔：

```text
baseline:
router_0525_llama_correct_conf_ce_t1_mrs_3expert/best_metrics.json

4sum boolq:
eval_0525_llama_4sum_boolq_on_mrs_3expert/routing_summary_val_eval_only.json

4sum rte:
eval_0525_llama_4sum_rte_on_mrs_3expert/routing_summary_val_eval_only.json

4sum siqa:
eval_0525_llama_4sum_siqa_on_mrs_3expert/routing_summary_val_eval_only.json

4sum piqa:
eval_0525_llama_4sum_piqa_on_mrs_3expert/routing_summary_val_eval_only.json
```

Qwen FP16 的比較檔同理：

```text
baseline:
router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert/best_metrics.json

4sum:
eval_0525_qwen3_fp16_4sum_<boolq|rte|siqa|piqa>_on_mrs_3expert/routing_summary_val_eval_only.json
```

### 7.3 比較時建議記錄的表格

每個 model 各做一張表，三個 task 分開填。

| Model | Training set | Eval task | Self first rate | Self mid rate | Self pair/top-pair rate | Top pair | Route correct acc |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: |
| Llama | MRS baseline | medmcqa | 0.98 | 0.86 | 0.86 | medmcqa->medmcqa | baseline overall 0.5733 |
| Llama | MRS + boolq | medmcqa | 待填 | 待填 | 待填 | 待填 | 待填 |
| Llama | MRS + rte | medmcqa | 待填 | 待填 | 待填 | 待填 | 待填 |
| Llama | MRS + siqa | medmcqa | 待填 | 待填 | 待填 | 待填 | 待填 |
| Llama | MRS + piqa | medmcqa | 待填 | 待填 | 待填 | 待填 | 待填 |

`race` 與 `sst2` 也各填同樣表格。Qwen FP16 等 baseline 跑完後照同一方式填。

你的目標成立時，4sum eval-only 的 MRS distribution 應接近 baseline：

- `race` 與 `sst2` 最理想仍幾乎都是各自的 self pair。
- `medmcqa` 不必強迫變成 `1.0`，應優先看是否維持 baseline 約 `medmcqa->medmcqa` 為主的分布，而非被新 task 訓練明顯推向其他 pair。
- 同時檢查 `route_correct_acc`，避免 routing distribution 看似維持，但答題 correctness 大幅下降。

## 8. 常用檢查命令

### 8.1 看 cache 是否完整

```bash
find 0525_llama_cache_mrs_3expert_official_eval_aligned 0525_llama_cache_4other_3expert_official_eval_aligned 0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned 0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned -maxdepth 2 -type f | sort
```

### 8.2 看 baseline 最佳 checkpoint 摘要

```bash
jq '{best_epoch, route_correct_acc: .metrics.route_correct_acc, fixed_self_correct_acc: .metrics.fixed_self_correct_acc, pair_acc: .metrics.pair_acc, self_pair_acc: .metrics.self_pair_acc, oracle_self_pair_acc: .metrics.oracle_self_pair_acc, per_task: [.metrics.routing_summary.per_task[] | {task, pred_self_first_rate, pred_self_mid_rate, gold_self_pair_rate, top_pred_pairs}]}' router_0525_llama_correct_conf_ce_t1_mrs_3expert/best_metrics.json
```

### 8.3 看某個 4sum eval-only 在 MRS validation 的 routing summary

```bash
jq '{pair_acc, self_pair_acc, oracle_self_pair_acc, all_pred_pairs, per_task: [.per_task[] | {task, pred_self_first_rate, pred_self_mid_rate, gold_self_pair_rate, top_pred_pairs, all_pred_pairs}]}' eval_0525_llama_4sum_boolq_on_mrs_3expert/routing_summary_val_eval_only.json
```

### 8.4 看某個 4sum 的 sample-level 失敗案例

例如列出 routing 沒有回自己或答案錯的 `medmcqa` samples：

```bash
jq '.records[] | select(.task == "medmcqa" and (.pred_pair != "medmcqa->medmcqa" or .pred_correct == false)) | {task, pred_pair, gold_pair, match, pred_correct, gold_correct, pred_raw_loss, gold_raw_loss, pred_answer, gold_answer, target, prompt_sha1}' eval_0525_llama_4sum_boolq_on_mrs_3expert/route_records_val_eval_only.json
```

### 8.5 看 log 是否正常完成

Cache log 完成時末尾應有：

```text
[CACHE] split=validation ...
[DONE] cached dataset build finished
```

```bash
tail -40 T0525_llama_build_cache_mrs_3expert.log
tail -40 T0525_llama_build_cache_4other_3expert.log
tail -40 T0525_qwen3_fp16_build_cache_mrs_3expert.log
tail -40 T0525_qwen3_fp16_build_cache_4other_3expert.log
```

Router log 完成後請看：

```bash
tail -100 T0525_llama_router_mrs_3expert.log
tail -100 T0525_llama_router_4sum_boolq_mrs_3expert.log
tail -100 T0525_llama_router_4sum_rte_mrs_3expert.log
tail -100 T0525_llama_router_4sum_siqa_mrs_3expert.log
tail -100 T0525_llama_router_4sum_piqa_mrs_3expert.log
```

log 會列每個 epoch 的 `[VAL]`、`[ROUTE]`、`[ORACLE]`、`[SAVE]` 與 `[EARLY_STOP]`。真正保存的 checkpoint 以 `router_config.json` / `best_metrics.json` 的 `best_epoch` 為準，不要只看最後印出的 epoch。

## 9. 實作來源對照

| 功能 | 實際程式位置 |
| --- | --- |
| 寫 cache config、manifest、chunk item | `build_cached_router_pair_dataset.py::main`, `process_split`, `save_chunk` |
| 讀取一個或多個 cache roots、檢查 contract、篩 sample tasks | `train_internal_two_router_compact_cached_joint.py::CachedLossMatrixDataset` |
| `correct_conf_ce` loss 如何從 `correct_matrix` 與 `loss_matrix` 形成 target | `router_pair_common.py::compute_pair_losses` |
| routing distribution summary 欄位來源 | `router_pair_common.py::build_routing_summary` |
| oracle/self gap summary 欄位來源 | `router_pair_common.py::build_oracle_debug_summary` |
| 每筆 route record 與 metrics 產生 | `train_internal_two_router_compact_cached_joint.py::evaluate` |
| 最佳 router checkpoint 寫檔 | `train_internal_two_router_compact_cached_joint.py::save_ckpt` |
| Cache 逐筆 matrix 檢查工具 | `tools/inspect_cached_router_pair_chunk.py` |
