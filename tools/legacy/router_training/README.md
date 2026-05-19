# Legacy Router Training Scripts

這個目錄放舊版 router training scripts。目前它們不在 active workflow 裡，但先保留作為歷史參考。

目前移入的腳本：

- `train_internal_two_router_compact.py`
- `train_internal_two_router_compact_cached_answer_supervision.py`
- `train_internal_two_router_compact_end2end.py`

使用前請先檢查：

- hard-coded dataset / checkpoint path 是否仍存在。
- 是否依賴 root 底下舊的 `router_*` 產物。
- 是否已被 newer training script 取代。
