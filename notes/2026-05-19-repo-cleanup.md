# 2026-05-19 repo 整理紀錄

## 今天做了什麼

- 建立 `artifacts/` 作為實驗產物目錄。
- 建立 `artifacts/README.md`，定義 logs、analysis、checkpoints、datasets、caches、archive、tmp 的用途。
- 建立 `artifacts/MOVE_PLAN.md`，記錄搬遷順序與路徑風險。
- 清掉前一版留下的空目錄 `artifacts/cache/` 和 `artifacts/eval_records/`。
- 搬 root 第一層的低風險產物：
  - 300 個 log 到 `artifacts/logs/`
  - 11 個分析 JSON/JSONL 到 `artifacts/analysis/`
  - 13 個 tmp/swp/scratch 項目到 `artifacts/tmp/`
- 刪掉 `opencompass/configs/models/hf_llama/.moe_lora_internal_compact_end2end_bs128_lr2e4_freezebert.py.swp`。
- 把舊的第三批 archive 候選搬到 `artifacts/archive/`，但保留 `0514` 的 router eval 結果在 root。
- 補上 `.gitignore` 規則，避免 `artifacts/` 內的大型產物、router checkpoint/cache/debug output、generated dataset 被誤提交。
- 建立 `tools/README.md`，說明現有 tools 的用途分類。
- 把三個舊 router training scripts 移到 `tools/legacy/router_training/`，並補上 legacy README。

## 刻意沒有搬的東西

以下路徑常被 config 或 script 用 root 相對路徑引用，暫時不直接搬：

- `router_ckpt_*`
- `router_debug_*`
- `saves/`
- `task_classifier_ckpt/`
- `router_train_datasets*`
- `cached_router_pair_*`
- `router_cached_features_*`
- `0513_opencompass_router_records_w05.jsonl`
- `0513_opencompass_router_records_qwen3_3sample_5expert_w05.jsonl`

## 仍留在 root 的 0514 結果

- `router_eval_only_cl_boolq_on_orig_0514`
- `router_eval_only_cl_commonsenseqa_on_orig_0514`
- `router_eval_only_cl_hellaswag_on_orig_0514`
- `router_eval_only_cl_piqa_on_orig_0514`
- `router_eval_only_cl_rte_on_orig_0514`
- `router_eval_only_cl_siqa_on_orig_0514`
- `router_eval_only_w05_orig_0514`

## 下一步建議

1. 先盤點目前 active config 實際指向哪些 checkpoint、dataset、cache。
2. 對 checkpoint/dataset/cache 決定使用「改路徑」或「root symlink」策略。
3. 再搬第二批大型產物，避免打壞現有 eval / train config。
4. commit 前先用 `git status --short --untracked-files=all` 確認 staging 範圍，不要把實驗產物或未確認的新 config 一起提交。
