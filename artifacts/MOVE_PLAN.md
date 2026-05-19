# 搬遷計畫

這份文件先記錄建議搬遷順序。真正搬檔案前要先確認是否有 config、script 或歷史指令依賴原本的 root 相對路徑。

## 第一批：低風險，可先搬

這些通常不會被程式當成輸入讀取，搬走後比較不會影響訓練或評估。

目標：`artifacts/logs/`

- `*.log`
- `nohup.out`
- `nohup_*.log`
- `run_opencompass_*.log`
- `run_*_*.log`
- `train_*.log`
- `051*_*.log`
- `cl_*.log`
- `eval_old5_*.log`

目標：`artifacts/analysis/`

- `oracle_analysis_*.json`
- `compare_*.json`
- `debug_router_pair_generation_*.json`
- `debug_router_pair_generation_*.jsonl`
- `*_matrix_check*.json`
- `*_records*.jsonl`，但只有在確認 config 沒有直接引用時才搬

目標：`artifacts/tmp/`

- `.swp`
- `*.swp`
- `__pycache__/`
- `tmp/`
- `tmp_eval_*`
- `tmp_cache_probe*`
- `icl_inference_output/`，如果只是空的或短期輸出

## 第二批：可搬，但要同步改路徑

這些經常被 config 或 script 以 root 相對路徑引用。搬之前需要更新引用，或保留相容 symlink。

目標：`artifacts/checkpoints/`

- `router_ckpt_*`
- `router_debug_*`
- `all_router_ckpt/`
- `task_classifier_ckpt/`
- `saves/`
- `router_qwen3_*/`

常見引用位置：

- `opencompass/configs/models/hf_llama/*.py`
- `opencompass/configs/models/qwen3/*.py`
- `train_joint_answer_supervision_router.py`
- `train_task_classifier.py`
- `extract_llama_vectors_chunked.py`

目標：`artifacts/datasets/`

- `router_train_datasets*`
- `datasets_classifier*`
- `datasets_opencompass_eval_*`
- `expert_sft_datasets_*`
- `hf_external_sources_*`
- `data/`，需先確認是否為專案必要資料

常見引用位置：

- `train_task_classifier.py`
- `sample_router_train_datasets.py`
- wandb run config snapshots
- 手動訓練指令

目標：`artifacts/caches/`

- `cached_router_pair_*`
- `router_cached_features_*`
- `router_cache_*`
- `router_features_*`

常見引用位置：

- `train_internal_two_router_compact_cached_joint.py`
- `train_joint_answer_supervision_router.py`
- wandb run config snapshots
- 近期 `0513` / `0518` 訓練指令

## 第三批：先歸檔，不急著改路徑

目標：`artifacts/archive/`

- `before_0406/`
- `before_0408/`
- `run_log_0420_before/`
- `eval_only_*`
- `router_eval_old5_*`
- `router_eval_only_*`
- 其他不確定是否還會用、但也不應該留在 root 的舊輸出

## 暫時不要搬

以下如果直接搬，現有 config 很容易壞掉：

- `saves/`
- `task_classifier_ckpt/`
- 目前 active config 指到的 `router_ckpt_*`
- 目前 active config 指到的 `router_debug_*`
- `0513_opencompass_router_records_w05.jsonl`
- `0513_opencompass_router_records_qwen3_3sample_5expert_w05.jsonl`

這些要搬的話，建議先做其中一種相容策略：

- 更新所有 config/script 的路徑到 `artifacts/...`
- 或保留 root symlink 指回 `artifacts/...`
- 或先只搬不會被 active config 使用的舊 checkpoint

## 建議順序

1. 先搬 logs、swap、明顯 tmp。
2. 跑一次 `git status --short` 和 `rg` 檢查 root 是否乾淨很多。
3. 盤點 active config 目前使用哪些 checkpoint/dataset/cache。
4. 對第二批產物選擇「改路徑」或「symlink」策略。
5. 最後再處理大型 checkpoint、dataset、cache。
