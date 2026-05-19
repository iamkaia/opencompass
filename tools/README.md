# Tools

這個目錄放可重複使用的輔助腳本。原則上，仍可能被拿來處理資料、檢查設定、分析輸出或輔助 OpenCompass 流程的腳本放在這裡；大型執行產物則放到 `artifacts/`。

## 目前工具分類

### Router / MoE 分析與輸出檢查

- `analyze_opencompass_routing_logs.py`：分析 OpenCompass routing log。
- `export_router_sample_outputs.py`：匯出 router sample output，方便人工檢查。
- `sample_router_train_datasets.py`：從 router train dataset 抽樣。
- `rebuild_balanced_replay_dataset.py`：重建 balanced replay dataset。

### Dataset 建置與轉換

- `build_aligned_training_dirs_0511.py`：建立對齊後的訓練資料目錄。
- `build_hf_external_sources_0511.py`：建立 HuggingFace external source 資料。
- `build_method1_hard_datasets.py`：建立 method1 hard dataset。
- `build_opencompass_siqa_data.py`：建立 OpenCompass SIQA 資料。
- `build_piqa_custom_dataset.py`：建立 PIQA custom dataset。
- `build_router_train_dataset_piqa_copa.py`：建立 PIQA/COPA router training dataset。
- `build_small_eval_dataset.py`：建立小型 eval dataset。
- `convert_alignmentbench.py`：轉換 AlignmentBench 資料。
- `update_dataset_suffix.py`：更新 dataset suffix。

### OpenCompass 輔助工具

- `case_analyzer.py`：分析個案輸出。
- `chatml_format_test.py`：測試 ChatML 格式。
- `collect_code_preds.py`：收集 code prediction。
- `compare_configs.py`：比較 config。
- `list_configs.py`：列出 config。
- `prediction_merger.py`：合併 prediction。
- `prompt_viewer.py`：檢視 prompt。
- `test_api_model.py`：測試 API model。
- `viz_multi_model.py`：多模型結果視覺化。

## Legacy

`tools/legacy/` 放目前不屬於 active workflow、但仍想保留作為參考的舊腳本。這些腳本不應該被視為主要入口；如果再次使用，建議先檢查路徑、依賴和 config 是否仍有效。
