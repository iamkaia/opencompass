# 兩層實驗說明

這份說明把目前的 router 實驗拆成兩層：

1. 第一層：`loss_matrix` / supervision 是怎麼來的
2. 第二層：router 拿到同一份 `loss_matrix` 之後，怎麼學

這兩層一定要分開，不然很容易把「分數算歪」和「trainer 學歪」混在一起。

---

## 這次固定的 3-task 組合

這一版統一改成：

- `sst2`
- `race`
- `squad20`

注意：

- 在 **OpenCompass dataset 名稱** 裡是 `squad20`
- 在 **router training / expert key / LoRA key** 裡仍然用 `squad2`

也就是說，訓練指令會寫：

- `--task_names sst2,race,squad2`
- `--expert_names sst2,race,squad2`

但 OpenCompass eval 會寫：

- `glue/sst2_gen`
- `race/race_gen_sft_prompt`
- `squad20/squad20_gen_sft_prompt`

這不是衝突，是目前 repo 裡 task key 和 eval dataset 名稱本來就不同。

---

## 你現在先跑哪個？

先跑這組最乾淨：

1. 先建一份共用的小樣本 feature cache
2. 用同一份 cache 分別跑：
   - `ce_pair`
   - `expected_loss`
   - `ce_pair_plus_expected`
3. 三個 checkpoint 再各自跑 OpenCompass

這組實驗在測的是：

- 同一份 supervision 固定
- 只改第二層 `joint_loss`

所以它回答的是：

- 是 `joint_loss` 的問題比較大？
- 還是同一份 supervision 下，不管怎麼學都差不多？

---

## 第一步：先建共用 feature cache

```bash
nohup python build_cached_router_pair_dataset.py \
  --data_root ./router_train_datasets_5org \
  --feature_root ./router_cached_features_3task_sst2_race_squad20_taskaware_small \
  --task_names sst2,race,squad2 \
  --expert_names sst2,race,squad2 \
  --base_model_path meta-llama/Llama-2-7b-chat-hf \
  --router_bert_init ./task_classifier_ckpt \
  --batch_size 8 \
  --max_train_samples 300 \
  --max_val_samples 150 \
  --max_llm_len 768 \
  --first_layer_idx 0 \
  --middle_layer_idx 15 \
  --router_dim 512 \
  --router_pooling mean \
  --dtype float16 \
  --r 8 \
  --alpha 32 \
  --lora_sst2 ./saves/llama2-7b-chat-hf/lora/sft_sst2 \
  --lora_race ./saves/llama2-7b-chat-hf/lora/sft_race \
  --lora_squad2 ./saves/llama2-7b-chat-hf/lora/sft_squad20 \
  > "./nohup_build_cached_3task_sst2_race_squad20_taskaware_small_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

看進度：

```bash
tail -f nohup_build_cached_3task_sst2_race_squad20_taskaware_small_*.log
```

等這步完成，再往下跑三個 trainer。

---

## 第一層：`loss_matrix` / supervision

第一層在問的是：

- 每個 sample、每個 pair 的 cost 是怎麼算出來的？
- oracle route label 會不會太 noisy？
- 分數會不會太集中？
- 某些 pair 會不會永遠特別容易贏？

### 相關程式

- [train_joint_answer_supervision_router.py](./train_joint_answer_supervision_router.py)
  - `extract_prompt_vectors()`
  - `score_all_route_pairs()`
  - task-aware 分流
  - `_task_uses_generation_evaluator()`
  - `compute_option_nll_proxy_scores()`
- [build_cached_router_pair_dataset.py](./build_cached_router_pair_dataset.py)
  - 抽 cached vectors
  - 建 cached `loss_matrix`

### 這一層會影響什麼

- `first_vec`
- `mid_vec`
- `loss_matrix`
- `pair_label`
- `first_label`
- `mid_label`

也就是說，這一步不只是「抽 feature」，也是在「決定 supervision」。

### 這一層常見變因

- `router_pooling`
  - `mean`
  - `lastk_mean`
  - `last`
- `score_mode` / route scoring 邏輯
  - `dataset_auto`
  - task-aware
  - official evaluator 路線
- task set / expert set

---

## 第二層：trainer 怎麼學同一份 `loss_matrix`

第二層在問的是：

- 同一份 cached `loss_matrix` 固定不變時
- router 用哪種目標學比較穩？

### 相關程式

- [train_internal_two_router_compact_cached_joint.py](./train_internal_two_router_compact_cached_joint.py)
  - `compute_pair_losses()`
  - `--joint_loss` CLI
  - train loop 使用 `joint_loss`
  - eval 使用 `joint_loss`
- [train_joint_answer_supervision_router.py](./train_joint_answer_supervision_router.py)
  - online `compute_pair_losses()`
  - online `--joint_loss` CLI

### 現在支援的 `joint_loss`

- `ce_pair`
- `expected_loss`
- `ce_pair_plus_expected`

### 這三個意思是什麼

#### `ce_pair`

- 先從 `loss_matrix` 找出 cost 最小的 pair
- 把那個 pair 當唯一正解
- 用 hard classification 去學

#### `expected_loss`

- 不把 supervision 壓成唯一 pair label
- 直接讓 router 最小化 pair 分布下的期望 cost

#### `ce_pair_plus_expected`

- 同時用 hard pair label 和 soft expected cost
- `expected_loss` 會當 regularization

---

## 最重要的區分

如果你跑：

- `ce_pair`
- `expected_loss`
- `ce_pair_plus_expected`

而且它們都吃 **同一份 `feature_root`**，那代表：

- `loss_matrix` 是一樣的
- supervision 是一樣的
- 差別只在 trainer objective

所以這組實驗是在測：

- 第二層：router 怎麼吃 supervision

不是在測：

- 第一層：supervision 怎麼算

如果你改的是：

- `feature_root`
- pooling
- score path
- cache build 邏輯

那你在測的是第一層。

---

## 目前第一層的 task-aware 路徑

現在 [train_joint_answer_supervision_router.py](./train_joint_answer_supervision_router.py) 裡的
`score_all_route_pairs()` 是 task-aware 的：

- 分類型任務：
  - 走 proxy
  - 用 `compute_option_nll_proxy_scores()`
- 生成型任務：
  - 走 `generate`
  - 再跑 evaluator

這次三個 task 的分流是：

- `sst2` 走分類 proxy
- `race` 走分類 proxy
- `squad2` / `squad20` 走 generation evaluator

---

## 3-task 小樣本實驗

這組實驗的目的是：

- 固定同一份小樣本 task-aware cache
- 只改 `joint_loss`
- 先把第二層看清楚

任務：

- `sst2`
- `race`
- `squad20`

---

## 第 2 步：訓練 `ce_pair`

```bash
nohup python train_internal_two_router_compact_cached_joint.py \
  --feature_root ./router_cached_features_3task_sst2_race_squad20_taskaware_small \
  --bert_init ./task_classifier_ckpt \
  --out_dir ./router_ckpt_cachedjoint_3task_sst2_race_squad20_taskaware_small_cepair \
  --task_names sst2,race,squad2 \
  --batch_size 128 \
  --epochs 10 \
  --lr 2e-4 \
  --max_bert_len 512 \
  --router_dim 512 \
  --freeze_bert \
  --joint_loss ce_pair \
  --pseudo_ce_weight 0.1 \
  --pseudo_ce_margin 0.1 \
  --wandb \
  --wandb_project router_answer_supervision \
  --wandb_name 3task_sst2_race_squad20_taskaware_small_cepair \
  --wandb_tags 3task,sst2,race,squad20,cached-joint,offline,taskaware,small,ce-pair \
  > "./nohup_train_3task_sst2_race_squad20_taskaware_small_cepair_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## 第 3 步：訓練 `expected_loss`

```bash
nohup python train_internal_two_router_compact_cached_joint.py \
  --feature_root ./router_cached_features_3task_sst2_race_squad20_taskaware_small \
  --bert_init ./task_classifier_ckpt \
  --out_dir ./router_ckpt_cachedjoint_3task_sst2_race_squad20_taskaware_small_expected \
  --task_names sst2,race,squad2 \
  --batch_size 128 \
  --epochs 10 \
  --lr 2e-4 \
  --max_bert_len 512 \
  --router_dim 512 \
  --freeze_bert \
  --joint_loss expected_loss \
  --pseudo_ce_weight 0.1 \
  --pseudo_ce_margin 0.1 \
  --wandb \
  --wandb_project router_answer_supervision \
  --wandb_name 3task_sst2_race_squad20_taskaware_small_expected \
  --wandb_tags 3task,sst2,race,squad20,cached-joint,offline,taskaware,small,expected-loss \
  > "./nohup_train_3task_sst2_race_squad20_taskaware_small_expected_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

## 第 4 步：訓練 `ce_pair_plus_expected`

```bash
nohup python train_internal_two_router_compact_cached_joint.py \
  --feature_root ./router_cached_features_3task_sst2_race_squad20_taskaware_small \
  --bert_init ./task_classifier_ckpt \
  --out_dir ./router_ckpt_cachedjoint_3task_sst2_race_squad20_taskaware_small_cepair_plusexp \
  --task_names sst2,race,squad2 \
  --batch_size 128 \
  --epochs 10 \
  --lr 2e-4 \
  --max_bert_len 512 \
  --router_dim 512 \
  --freeze_bert \
  --joint_loss ce_pair_plus_expected \
  --pseudo_ce_weight 0.1 \
  --pseudo_ce_margin 0.1 \
  --wandb \
  --wandb_project router_answer_supervision \
  --wandb_name 3task_sst2_race_squad20_taskaware_small_cepair_plusexp \
  --wandb_tags 3task,sst2,race,squad20,cached-joint,offline,taskaware,small,ce-pair-plus-expected \
  > "./nohup_train_3task_sst2_race_squad20_taskaware_small_cepair_plusexp_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

---

## 已經補好的 OpenCompass config

### Model config

- [moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_cepair.py](./opencompass/configs/models/hf_llama/moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_cepair.py)
- [moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_expected.py](./opencompass/configs/models/hf_llama/moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_expected.py)
- [moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_cepair_plusexp.py](./opencompass/configs/models/hf_llama/moe_lora_internal_compact_cachedjoint_3task_sst2_race_squad20_cepair_plusexp.py)

### Eval config

- [router_3task_sst2_race_squad20_cepair_direct.py](./opencompass/configs/eval/router_3task_sst2_race_squad20_cepair_direct.py)
- [router_3task_sst2_race_squad20_expected_direct.py](./opencompass/configs/eval/router_3task_sst2_race_squad20_expected_direct.py)
- [router_3task_sst2_race_squad20_cepair_plusexp_direct.py](./opencompass/configs/eval/router_3task_sst2_race_squad20_cepair_plusexp_direct.py)

這三份 eval config 的 dataset 組成是：

- `sst2` full
- `race` full
- `squad20` full

---

## OpenCompass 命令

### `ce_pair`

```bash
nohup python run.py opencompass/configs/eval/router_3task_sst2_race_squad20_cepair_direct.py \
  -w outputs/router_3task_sst2_race_squad20_cepair_direct \
  > "./nohup_oc_3task_sst2_race_squad20_cepair_direct_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

### `expected_loss`

```bash
nohup python run.py opencompass/configs/eval/router_3task_sst2_race_squad20_expected_direct.py \
  -w outputs/router_3task_sst2_race_squad20_expected_direct \
  > "./nohup_oc_3task_sst2_race_squad20_expected_direct_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

### `ce_pair_plus_expected`

```bash
nohup python run.py opencompass/configs/eval/router_3task_sst2_race_squad20_cepair_plusexp_direct.py \
  -w outputs/router_3task_sst2_race_squad20_cepair_plusexp_direct \
  > "./nohup_oc_3task_sst2_race_squad20_cepair_plusexp_direct_$(date +%Y%m%d_%H%M%S).log" 2>&1 &
```

---

## 你之後要看什麼

先看 trainer log：

- `val/router_score`
- `val/pair_acc`
- `top_pred_pairs`

再看 OpenCompass 最終表。

如果：

- 三個 `joint_loss` 差很多

代表第二層影響很大。

如果：

- 三個 `joint_loss` 差不多都差

代表更像第一層 supervision 本身有問題。
