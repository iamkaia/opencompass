# T0525 Qwen3 fixopt2 Router Records Check

## 目的

這份文件用來檢查 Qwen3 `fixopt2` router 訓練完成後，對同一筆 sample：

- 訓練 supervision 給 router 的 pair 分布：`router_target_matrix`
- router 實際預測的 pair 分布：`pair_prob_matrix`
- router 最後 argmax 選的 pair：`pred_pair`
- cache 中 gold target loss 最低的 pair：`gold_pair`
- sample 有自己的 expert 時，router 是否選回 `task->task`

查看工具：

```bash
tools/inspect_T0525_qwen3_router_records.py
```

此工具會自動讀每個 router output 的 `best_metrics.json`，使用 `best_epoch` 對應的 records 檔案，不需要手動指定 epoch。

## 現有 Router Outputs

| Run Key | 訓練內容 | Best Epoch | Validation `route_correct_acc` | `correct_target_available_ratio` |
| --- | --- | ---: | ---: | ---: |
| `mrs` | `medmcqa,race,sst2` | 1 | 0.5867 | 0.7333 |
| `boolq` | `medmcqa,race,sst2,boolq` | 1 | 0.6550 | 0.7850 |
| `rte` | `medmcqa,race,sst2,rte` | 2 | 0.6250 | 0.7550 |
| `siqa` | `medmcqa,race,sst2,siqa` | 1 | 0.6350 | 0.7600 |
| `piqa` | `medmcqa,race,sst2,piqa` | 2 | 0.6500 | 0.7750 |

三個 expert 的矩陣順序固定為：

```text
medmcqa, race, sst2
```

因此 `3 x 3` 矩陣座標對應如下：

```text
[
  [medmcqa->medmcqa, medmcqa->race, medmcqa->sst2],
  [race->medmcqa,    race->race,    race->sst2],
  [sst2->medmcqa,    sst2->race,    sst2->sst2],
]
```

## Records 檔案

### Train Split

訓練資料的逐筆 sample records 使用：

```text
route_records_train_eval_epoch*.json
```

這不是 gradient update 當下的 mini-batch dump，而是每個 epoch 訓練完成後，用當下 router 對完整 train split 重新 evaluation 的逐筆結果。

例如 baseline 的 best epoch 是 1，因此工具會讀：

```text
router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert_fixopt2/route_records_train_eval_epoch1.json
```

### Validation Split

validation 資料的逐筆 sample records 使用：

```text
route_records_val_epoch*.json
```

例如 `rte` 4sum router 的 best epoch 是 2，因此工具會讀：

```text
router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert_fixopt2/route_records_val_epoch2.json
```

## 每筆 Sample 的欄位

| 欄位 | 意義 |
| --- | --- |
| `task` | sample 原始 task |
| `target` | gold answer label |
| `gold_pair` | cache 的 `loss_matrix` 中 gold target loss 最低的 pair |
| `gold_raw_loss` | `gold_pair` 對 gold target 的 cost |
| `pred_pair` | router `pair_prob_matrix` argmax 選出的 pair |
| `pred_pair_prob` | router 分給 `pred_pair` 的機率 |
| `pred_raw_loss` | `pred_pair` 對 gold target 的 cost |
| `pred_correct` | `pred_pair` 的 greedy answer 是否答對 |
| `any_pair_correct` | 該 sample 是否至少存在一個答對的 pair |
| `router_target_matrix` | `correct_conf_ce` 建出的 pair supervision distribution |
| `pair_prob_matrix` | router 訓練後輸出的 pair probability distribution |

### `gold_pair` 與 `router_target_matrix` 不一定相同

`gold_pair` 只看 `loss_matrix` 最低的單一 pair。

你現在用的是 `correct_conf_ce`，所以 `router_target_matrix` 的邏輯是：

1. 用 `correct_matrix` 保留答對的 pairs。
2. 答錯的 pairs target probability 設為 0。
3. 答對的 pairs 之間，依 gold target loss 高低分配 probability。

因此，檢查訓練成功時，主要比較的是：

```text
router_pred_topk vs router_target_topk
```

不是只看 `pred_pair == gold_pair`。

## Train Split 查看 Commands

以下 commands 都可直接貼入 tmux。

### Baseline MRS 全部 Task

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --split train_eval --limit 10 --topk 3
```

### Baseline MRS 分 Task

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --split train_eval --task medmcqa --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --split train_eval --task race --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --split train_eval --task sst2 --limit 10 --topk 3
```

### 4sum 新增 Task

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --split train_eval --task boolq --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py rte --split train_eval --task rte --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py siqa --split train_eval --task siqa --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py piqa --split train_eval --task piqa --limit 10 --topk 3
```

### 4sum 中原本三個 Expert Task 是否維持原分布

例如檢查加入 `boolq` 後，原來的 `medmcqa/race/sst2`：

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --split train_eval --task medmcqa --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --split train_eval --task race --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --split train_eval --task sst2 --limit 10 --topk 3
```

把第一個參數 `boolq` 換成 `rte`、`siqa`、`piqa`，即可檢查另外三個 4sum router。

## Validation Split 查看 Commands

### Baseline MRS

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --limit 10 --topk 3
```

### 4sum 新增 Task

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --task boolq --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py rte --task rte --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py siqa --task siqa --limit 10 --topk 3
```

```bash
python tools/inspect_T0525_qwen3_router_records.py piqa --task piqa --limit 10 --topk 3
```

## 只看 Aggregate Summary

如果不需要逐筆 sample，只想先看正確率與 self expert 被選中的比例：

```bash
python tools/inspect_T0525_qwen3_router_records.py mrs --split train_eval --summary_only
```

```bash
python tools/inspect_T0525_qwen3_router_records.py boolq --split train_eval --summary_only
```

```bash
python tools/inspect_T0525_qwen3_router_records.py rte --split train_eval --summary_only
```

```bash
python tools/inspect_T0525_qwen3_router_records.py siqa --split train_eval --summary_only
```

```bash
python tools/inspect_T0525_qwen3_router_records.py piqa --split train_eval --summary_only
```

## 怎麼判斷 Self Expert

只有 `medmcqa`、`race`、`sst2` 有自己的 expert：

| Sample Task | Self Pair |
| --- | --- |
| `medmcqa` | `medmcqa->medmcqa` |
| `race` | `race->race` |
| `sst2` | `sst2->sst2` |

工具每筆會印：

```text
self=medmcqa->medmcqa chosen=False pred_rank=3 pred_prob=0.1306 target_rank=1 target_prob=0.1442
```

解讀：

- `chosen=False`：router argmax 沒有選 self pair。
- `pred_rank=3`：在 router 輸出的所有 pair 機率中，self pair 排第 3。
- `target_rank=1`：訓練 supervision target 希望 self pair 排第 1。
- `pred_prob` 與 `target_prob`：可直接比較 router 是否接近 supervision distribution。

`boolq`、`rte`、`siqa`、`piqa` 沒有自己的 expert，因此顯示：

```text
self=N/A
```

## 實際輸出怎麼判斷

每筆 sample 建議依序看：

1. `any_pair_correct=True`  
   表示這筆資料至少存在答對的 pair，`correct_conf_ce` 有可用 supervision。

2. `router_target_top3`  
   這是訓練希望 router 分配機率的位置。

3. `router_pred_top3`  
   這是 router 實際學出的機率位置。

4. `pred_correct=True`  
   即使 `pred_pair` 不等於最低 loss 的 `gold_pair`，只要它是答對的 pair，對 `correct_conf_ce` 來說不一定是失敗。

5. `self=... chosen=...`  
   對 `medmcqa/race/sst2` 判斷 router 是否選回自己的 expert。

