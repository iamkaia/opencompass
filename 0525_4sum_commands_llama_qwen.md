# 0525 Router 4sum Commands: Llama and Qwen3

所有 command 都在 `/home/u9472191/opencompass` 下執行。

本輪使用的 dataset tasks：

```text
baseline / 4sum retained tasks: medmcqa,race,sst2
new 4sum tasks: boolq,rte,siqa,piqa
experts: medmcqa,race,sst2
```

`0525_router_train_datasets` 中的 `iwslt2017`、`squad2` 不加入這組 4sum 實驗。

注意：

- 使用的 objective 參數名稱是 `correct_conf_ce`。
- 四個 4sum router 都從新初始化開始，在 `MRS + 一個新 task` 的四-task 混合資料上訓練；這樣比較的是 4sum mixture 本身是否保留舊 task routing 分布。
- 只有 `eval_only` command 需要 `--load_from`，因為它要載入已訓練完成的 4sum router 來比較 MRS validation 分布。
- 同一張 GPU 時，cache command 請逐條跑；training command 也請逐條跑。

---

## A. Llama-2-7B-Chat

### A1. 建 Cache：MRS 三個 task

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u build_cached_router_pair_dataset.py --data_root "0525_router_train_datasets" --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --base_model_path "meta-llama/Llama-2-7b-chat-hf" --router_bert_init "task_classifier_ckpt" --batch_size 8 --max_train_samples 200 --max_val_samples 50 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 > T0525_llama_build_cache_mrs_3expert.log 2>&1 &
```

### A2. 建 Cache：四個額外 task

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u build_cached_router_pair_dataset.py --data_root "0525_router_train_datasets" --feature_root "0525_llama_cache_4other_3expert_official_eval_aligned" --task_names boolq,rte,siqa,piqa --expert_names medmcqa,race,sst2 --base_model_path "meta-llama/Llama-2-7b-chat-hf" --router_bert_init "task_classifier_ckpt" --batch_size 8 --max_train_samples 200 --max_val_samples 50 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 > T0525_llama_build_cache_4other_3expert.log 2>&1 &
```

### A3. 訓練 Baseline：只有 MRS

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_llama_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_llama_router_mrs_3expert.log 2>&1 &
```

以下四條 4sum training 不載入 baseline checkpoint；baseline 是對照組，4sum router 是各自獨立訓練的實驗組。

### A4. 4sum：MRS + BoolQ

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_llama_cache_mrs_3expert_official_eval_aligned,0525_llama_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,boolq --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_llama_router_4sum_boolq_mrs_3expert.log 2>&1 &
```

### A5. 4sum：MRS + RTE

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_llama_cache_mrs_3expert_official_eval_aligned,0525_llama_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,rte --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_llama_router_4sum_rte_mrs_3expert.log 2>&1 &
```

### A6. 4sum：MRS + SIQA

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_llama_cache_mrs_3expert_official_eval_aligned,0525_llama_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,siqa --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_llama_router_4sum_siqa_mrs_3expert.log 2>&1 &
```

### A7. 4sum：MRS + PIQA

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_llama_cache_mrs_3expert_official_eval_aligned,0525_llama_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,piqa --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_llama_router_4sum_piqa_mrs_3expert.log 2>&1 &
```

### A8. 4sum 後只評估 MRS Validation：BoolQ Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_llama_4sum_boolq_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_llama_4sum_boolq_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_llama_eval_4sum_boolq_on_mrs.log 2>&1 &
```

### A9. 4sum 後只評估 MRS Validation：RTE Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_llama_4sum_rte_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_llama_4sum_rte_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_llama_eval_4sum_rte_on_mrs.log 2>&1 &
```

### A10. 4sum 後只評估 MRS Validation：SIQA Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_llama_4sum_siqa_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_llama_4sum_siqa_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_llama_eval_4sum_siqa_on_mrs.log 2>&1 &
```

### A11. 4sum 後只評估 MRS Validation：PIQA Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_llama_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_llama_4sum_piqa_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_llama_4sum_piqa_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_llama_eval_4sum_piqa_on_mrs.log 2>&1 &
```

---

## B. Qwen3-4B-Instruct-2507 FP16

Qwen 這組明確使用 `--dtype float16`。由於先前已存在一份 `bfloat16` 的 Qwen MRS cache，這組輸出路徑加上 `_fp16_`，避免混用不同 cache scoring precision。

### B1. 建 Cache：MRS 三個 task

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u build_cached_router_pair_dataset.py --data_root "0525_router_train_datasets" --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --base_model_path "Qwen/Qwen3-4B-Instruct-2507" --router_bert_init "task_classifier_ckpt" --lora_medmcqa "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa" --lora_race "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race" --lora_sst2 "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2" --batch_size 8 --max_train_samples 200 --max_val_samples 50 --middle_layer_idx 18 --dtype float16 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 > T0525_qwen3_fp16_build_cache_mrs_3expert.log 2>&1 &
```

### B2. 建 Cache：四個額外 task

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u build_cached_router_pair_dataset.py --data_root "0525_router_train_datasets" --feature_root "0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned" --task_names boolq,rte,siqa,piqa --expert_names medmcqa,race,sst2 --base_model_path "Qwen/Qwen3-4B-Instruct-2507" --router_bert_init "task_classifier_ckpt" --lora_medmcqa "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa" --lora_race "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race" --lora_sst2 "saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2" --batch_size 8 --max_train_samples 200 --max_val_samples 50 --middle_layer_idx 18 --dtype float16 --score_mode official_eval_aligned_generation --chunk_size 2048 --seed 42 > T0525_qwen3_fp16_build_cache_4other_3expert.log 2>&1 &
```

### B3. 訓練 Baseline：只有 MRS

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_qwen3_fp16_router_mrs_3expert.log 2>&1 &
```

以下四條 4sum training 不載入 baseline checkpoint；baseline 是對照組，4sum router 是各自獨立訓練的實驗組。

### B4. 4sum：MRS + BoolQ

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned,0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,boolq --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_qwen3_fp16_router_4sum_boolq_mrs_3expert.log 2>&1 &
```

### B5. 4sum：MRS + RTE

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned,0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,rte --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_qwen3_fp16_router_4sum_rte_mrs_3expert.log 2>&1 &
```

### B6. 4sum：MRS + SIQA

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned,0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,siqa --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_qwen3_fp16_router_4sum_siqa_mrs_3expert.log 2>&1 &
```

### B7. 4sum：MRS + PIQA

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_roots "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned,0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --out_dir "router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert" --sample_task_names medmcqa,race,sst2,piqa --expert_names medmcqa,race,sst2 --batch_size 16 --epochs 10 --lr 2e-4 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --best_metric route_correct_acc --early_stop_patience 2 --eval_train_each_epoch --save_route_records > T0525_qwen3_fp16_router_4sum_piqa_mrs_3expert.log 2>&1 &
```

### B8. 4sum 後只評估 MRS Validation：BoolQ Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_qwen3_fp16_4sum_boolq_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_qwen3_fp16_eval_4sum_boolq_on_mrs.log 2>&1 &
```

### B9. 4sum 後只評估 MRS Validation：RTE Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_qwen3_fp16_4sum_rte_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_qwen3_fp16_eval_4sum_rte_on_mrs.log 2>&1 &
```

### B10. 4sum 後只評估 MRS Validation：SIQA Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_qwen3_fp16_4sum_siqa_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_qwen3_fp16_eval_4sum_siqa_on_mrs.log 2>&1 &
```

### B11. 4sum 後只評估 MRS Validation：PIQA Router

```bash
CUDA_VISIBLE_DEVICES=0 nohup /home/u9472191/.conda/envs/opencompass/bin/python -u train_internal_two_router_compact_cached_joint.py --feature_root "0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned" --bert_init "task_classifier_ckpt" --load_from "router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert" --out_dir "eval_0525_qwen3_fp16_4sum_piqa_on_mrs_3expert" --sample_task_names medmcqa,race,sst2 --expert_names medmcqa,race,sst2 --batch_size 16 --joint_loss correct_conf_ce --correct_soft_ce_temperature 1.0 --pair_loss_normalization sample_minmax --supervision_mode oracle_loss --save_route_records --eval_only > T0525_qwen3_fp16_eval_4sum_piqa_on_mrs.log 2>&1 &
```

---

## C. 完成後看哪裡

### Cache 是否正確

Llama：

```bash
cat 0525_llama_cache_mrs_3expert_official_eval_aligned/train/manifest.json
```

```bash
cat 0525_llama_cache_4other_3expert_official_eval_aligned/train/manifest.json
```

Qwen3：

```bash
cat 0525_qwen3_fp16_cache_mrs_3expert_official_eval_aligned/train/manifest.json
```

```bash
cat 0525_qwen3_fp16_cache_4other_3expert_official_eval_aligned/train/manifest.json
```

每一份的 `expert_names` 都應為：

```json
["medmcqa", "race", "sst2"]
```

### Baseline 分布

```text
router_0525_llama_correct_conf_ce_t1_mrs_3expert/best_metrics.json
router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert/best_metrics.json
```

看：

```text
metrics.routing_summary.per_task[].top_pred_pairs
metrics.routing_summary.per_task[].all_pred_pairs
metrics.routing_summary.per_task[].pred_self_first_rate
metrics.routing_summary.per_task[].pred_self_mid_rate
metrics.routing_summary.per_task[].gold_self_pair_rate
```

### 4sum 是否破壞舊三個 task 的 routing 分布

Llama：

```text
eval_0525_llama_4sum_boolq_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_llama_4sum_rte_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_llama_4sum_siqa_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_llama_4sum_piqa_on_mrs_3expert/routing_summary_val_eval_only.json
```

Qwen3：

```text
eval_0525_qwen3_fp16_4sum_boolq_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_qwen3_fp16_4sum_rte_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_qwen3_fp16_4sum_siqa_on_mrs_3expert/routing_summary_val_eval_only.json
eval_0525_qwen3_fp16_4sum_piqa_on_mrs_3expert/routing_summary_val_eval_only.json
```

這些 4sum evaluation 都是在同一份 MRS validation cache 上跑，所以可以直接和各自模型的 baseline `best_metrics.json` 比：

```text
per_task[].all_pred_pairs
per_task[].top_pred_pairs
per_task[].pred_self_first_rate
per_task[].pred_self_mid_rate
```

### 重要限制

`correct_conf_ce` 的 target 是「所有答對 pairs 中偏向低 cost 的 pair」，不是硬性要求 `task -> task`。若 baseline cache 的 `gold_self_pair_rate` 低，router 不回 self 是 objective 與 cache supervision 一致的結果，不應直接解讀為 4sum 失敗。
