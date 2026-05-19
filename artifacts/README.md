# 產物目錄

這個目錄用來放執行後產生的輸出，以及不適合留在 repo 根目錄的大型實驗產物。

建議結構：

- `artifacts/logs/`：訓練、評估、nohup、debug 的 log
- `artifacts/analysis/`：summary、jsonl diff、csv 報表等分析輸出
- `artifacts/checkpoints/`：模型 checkpoint
- `artifacts/datasets/`：生成後的資料集快照
- `artifacts/caches/`：router feature cache、pair dataset cache 等
- `artifacts/archive/`：先保留但平常不會直接用的舊輸出
- `artifacts/tmp/`：短期 scratch、probe、臨時 debug 輸出

這個目錄可以保留少量說明文件，但不要把大型生成產物提交進 git。
