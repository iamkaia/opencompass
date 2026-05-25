# `router_pair_common.py` 函式解釋

本文依照目前工作樹中的 [`router_pair_common.py`](./router_pair_common.py) 內容整理，重點解釋你在檔案中加問號的三個 function：

1. `compute_pair_losses()`
2. `compute_routing_accuracy_stats()`
3. `update_oracle_debug_accumulator()`

這支檔案不是產生 cache 的地方，而是「使用 cache 裡已經算好的每個 expert pair 成本，訓練 router 並產生統計報表」的共用函式。

---

## 0. 先理解資料代表什麼

假設：

| 符號 | 意思 |
|---|---|
| `B` | batch size |
| `E` | expert 數量 |
| `loss_matrix.shape = [B, E, E]` | 每筆 sample 在每種 `(first expert, mid expert)` 路徑下的成本 |
| `pair_logits.shape = [B, E*E]` | router 對所有 expert pair 的輸出分數 |
| `correct_matrix.shape = [B, E, E]` | 每個 pair 的最終回答是否正確 |
| `task_ids.shape = [B]` | 每筆 sample 自己所屬 task 對應的 expert id |

### `loss_matrix`

```python
loss_matrix[b, first_id, mid_id]
```

表示第 `b` 筆 sample 經過：

```text
first expert = first_id
mid expert   = mid_id
```

之後得到的成本。**成本越小越好**。

如果有 3 個 experts，單筆 sample 的 matrix 可能長這樣：

```text
             mid=0   mid=1   mid=2
first=0       0.80    0.25    0.71
first=1       0.50    0.42    0.30
first=2       0.61    0.10    0.39
```

這筆 sample 的 oracle best pair 就是：

```text
first=2, mid=1，因為 cost=0.10 最低
```

### `pair_logits`

router 不直接輸出一個二維 matrix，而是輸出展平後的 pair 分數：

```python
pair_logits.shape = [B, E * E]
```

pair id 的排列方式是：

```python
pair_id = first_id * E + mid_id
```

若 `E=3`：

| `pair_id` | pair |
|---|---|
| `0` | `0 -> 0` |
| `1` | `0 -> 1` |
| `2` | `0 -> 2` |
| `3` | `1 -> 0` |
| `4` | `1 -> 1` |
| `5` | `1 -> 2` |
| `6` | `2 -> 0` |
| `7` | `2 -> 1` |
| `8` | `2 -> 2` |

`pair_logits` 越大，表示 router 越想選那個 pair。之後會用：

```python
pair_prob = torch.softmax(pair_logits, dim=-1)
pred_pair = pair_logits.argmax(dim=-1)
```

分別取得 router 的機率分布與最終選擇。

### `correct_matrix`

`correct_matrix[b, i, j]` 代表 pair `i -> j` 最後是否答對。

它和 `loss_matrix` 不完全相同：

- `loss_matrix`：某個 scoring contract 下的 cost，例如 NLL 類成本，越低越好。
- `correct_matrix`：答案最後是否判定正確，是布林訊號。

因此可能存在：

```text
兩個 pair 都答對，但其中一個 loss 比另一個低。
```

### `self pair`

若一筆 `piqa` sample 對應到 `piqa` expert，則：

```text
piqa -> piqa
```

就是這筆 sample 的 `self pair`。

它代表固定送回自己的 task expert，而不是讓 router 在所有 experts 間自由挑選。

---

## 1. 整支檔案的 Function 地圖

| Function | 作用 | 是否影響 gradient |
|---|---|---|
| `add_joint_loss_arguments()` | 加入 loss 相關 CLI 參數 | 間接影響 |
| `normalize_pair_loss_matrix()` | 每筆 sample 的 pair cost 正規化 | 會，被訓練 loss 使用 |
| `compute_pair_losses()` | 計算真正訓練 router 的 loss 與 oracle pair | **會** |
| `score_from_cost()` | 把 cost 轉成較易讀的分數 | 不會 |
| `resolve_self_expert_ids()` | 找每筆 sample 對應的 self expert | 不會 |
| `task_names_from_ids()` | task id 轉 task name | 不會 |
| `optional_float()` / `optional_rate()` | log 顯示格式 | 不會 |
| `masked_mean()` | 只在有效 sample 上平均 | 不會 |
| `compute_routing_accuracy_stats()` | 算 router 選 oracle / self 的比例 | **不會，純統計** |
| `build_routing_summary()` | 整理整個 epoch 的 routing 分布 | 不會 |
| `print_routing_summary()` | 印 routing 報表 | 不會 |
| `compute_route_score_stats()` | 比較 router、oracle、固定 self 的 cost/score | 不會 |
| `init_oracle_debug_accumulator()` | 建立 oracle debug 累積容器 | 不會 |
| `update_oracle_debug_accumulator()` | 將 batch 的 oracle 結構累積進容器 | **不會，純統計** |
| `build_oracle_debug_summary()` | 將累積資料整理成摘要 | 不會 |
| `print_oracle_debug_summary()` | 印 oracle debug 摘要 | 不會 |
| `flatten_routing_summary()` | 轉成 wandb 容易 log 的 dict | 不會 |

在 cached trainer 中實際流程是：

```text
model forward
  -> pair_logits
  -> compute_pair_losses() 算出 loss
  -> loss.backward() 更新 router
  -> accuracy / route score / oracle debug functions 只做統計
```

對應程式位置：

- router forward：[`train_internal_two_router_compact_cached_joint.py:1162`](./train_internal_two_router_compact_cached_joint.py#L1162)
- 計算訓練 loss：[`train_internal_two_router_compact_cached_joint.py:1173`](./train_internal_two_router_compact_cached_joint.py#L1173)
- 真正反向傳播：[`train_internal_two_router_compact_cached_joint.py:1203`](./train_internal_two_router_compact_cached_joint.py#L1203)
- 算 routing 統計：[`train_internal_two_router_compact_cached_joint.py:1212`](./train_internal_two_router_compact_cached_joint.py#L1212)
- 累積 oracle debug：[`train_internal_two_router_compact_cached_joint.py:1245`](./train_internal_two_router_compact_cached_joint.py#L1245)

---

## 2. `add_joint_loss_arguments()`

位置：[`router_pair_common.py:21`](./router_pair_common.py#L21)

這個 function 只是把可選的 loss 參數加入 command-line parser。

### `--joint_loss`

可選：

| 選項 | 主要概念 |
|---|---|
| `ce_pair` | 把最低 cost pair 當成唯一分類答案 |
| `expected_loss` | 最小化 router 選擇後的期望 cost |
| `ce_pair_plus_expected` | hard CE 加 expected cost |
| `correct_soft_ce` | 所有答對 pair 平均分配 target probability |
| `correct_conf_ce` | 所有答對 pair 中，偏好低 cost pair |
| `self_preserving_correct_conf_ce` | 正確 self pair 可保留指定權重 |
| `correct_max_margin` | 最高正確 pair logit 要高於最高錯誤 pair logit |
| `correct_conf_ce_plus_margin` | `correct_conf_ce` 加 margin loss |

### 其他參數

| 參數 | 用途 |
|---|---|
| `--pseudo_ce_weight` | 組合 objective 中第二項的係數 |
| `--pseudo_ce_margin` | 最佳與第二佳 pair 的 gap 門檻，或 correct/wrong margin |
| `--correct_soft_ce_temperature` | `correct_conf_ce` 的 target 分布溫度 |
| `--self_preserve_weight` | 正確 self pair 要保留多少 target mass |
| `--pair_loss_normalization` | pair cost 是否做逐 sample min-max |

---

## 3. `normalize_pair_loss_matrix()`

位置：[`router_pair_common.py:62`](./router_pair_common.py#L62)

```python
def normalize_pair_loss_matrix(loss_matrix: torch.Tensor, method: str) -> torch.Tensor:
    method = str(method)
    loss_matrix = loss_matrix.to(torch.float32)
    if method == "none":
        return loss_matrix
    if method != "sample_minmax":
        raise ValueError(f"Unknown pair loss normalization: {method}")

    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    min_values = flat_loss.min(dim=-1, keepdim=True).values
    max_values = flat_loss.max(dim=-1, keepdim=True).values
    denom = (max_values - min_values).clamp_min(1e-6)
    normalized_flat = (flat_loss - min_values) / denom
    return normalized_flat.view_as(loss_matrix)
```

### 逐行解釋

```python
method = str(method)
loss_matrix = loss_matrix.to(torch.float32)
```

將參數與 tensor dtype 整理成穩定格式。

```python
if method == "none":
    return loss_matrix
```

如果不想正規化，就直接使用 cache 中原始成本。

```python
if method != "sample_minmax":
    raise ValueError(...)
```

目前只支援：

- `none`
- `sample_minmax`

```python
flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
```

把每筆 sample 的 `[E, E]` matrix 展平成 `[E*E]`。

```python
min_values = flat_loss.min(dim=-1, keepdim=True).values
max_values = flat_loss.max(dim=-1, keepdim=True).values
```

對每筆 sample 個別找最低與最高成本。

```python
denom = (max_values - min_values).clamp_min(1e-6)
```

避免所有 pair cost 一樣時除以零。

```python
normalized_flat = (flat_loss - min_values) / denom
```

做逐 sample min-max normalization：

```text
normalized_cost = (cost - sample_min) / (sample_max - sample_min)
```

因此每筆 sample：

- 最佳 pair cost 變 `0`
- 最差 pair cost 變 `1`
- 其他 pairs 落在 `0~1`

這不會改變 pair 排名，只會改變 loss 尺度。

---

## 4. 問號 Function：`compute_pair_losses()`

位置：[`router_pair_common.py:79`](./router_pair_common.py#L79)

這個 function 是 router 訓練的核心。它做兩件事：

1. 根據 cache 找到 oracle pair。
2. 根據你選的 `joint_loss` 算出可以 `backward()` 的 router 訓練 loss。

### 4.1 函式參數

```python
def compute_pair_losses(
    pair_logits: torch.Tensor,
    logits_first: Optional[torch.Tensor] = None,
    logits_mid: Optional[torch.Tensor] = None,
    loss_matrix: Optional[torch.Tensor] = None,
    correct_matrix: Optional[torch.Tensor] = None,
    task_ids: Optional[torch.Tensor] = None,
    mode: str = "joint",
    joint_loss: str = "expected_loss",
    pseudo_ce_weight: float = 0.0,
    margin: float = 0.0,
    loss_normalization: str = "sample_minmax",
    correct_soft_ce_temperature: float = 1.0,
    self_preserve_weight: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
```

| 參數 | 內容 |
|---|---|
| `pair_logits` | router 對 `E*E` 個 pair 的輸出 `[B, E*E]` |
| `logits_first` | stage1 router 輸出；joint cached trainer 通常不傳 |
| `logits_mid` | stage2 router 輸出；joint cached trainer 通常不傳 |
| `loss_matrix` | cache 中所有 pair 的 cost `[B, E, E]` |
| `correct_matrix` | 每個 pair 是否答對 `[B, E, E]` |
| `task_ids` | sample 自己 task 的 expert id，用於 self pair |
| `mode` | `stage1` / `stage2` / `joint` |
| `joint_loss` | joint mode 下真正使用哪種 loss |
| `pseudo_ce_weight` | 組合 loss 的附加項係數 |
| `margin` | hard pair gap 門檻或 correct/wrong margin |
| `loss_normalization` | pair cost 如何正規化 |
| `correct_soft_ce_temperature` | 正確 pairs 間 target 權重的溫度 |
| `self_preserve_weight` | 正確 self pair 的保留權重 |

### 4.2 第 94-102 行：一定要有 `loss_matrix`

```python
if loss_matrix is None:
    raise ValueError("loss_matrix is required")

mode = str(mode)
joint_loss = str(joint_loss)
normalized_loss_matrix = normalize_pair_loss_matrix(
    loss_matrix=loss_matrix,
    method=loss_normalization,
)
```

逐行意思：

| 程式 | 解釋 |
|---|---|
| `if loss_matrix is None` | 沒有 cache cost，就不知道哪個 pair 好 |
| `mode = str(mode)` | 將模式固定成字串 |
| `joint_loss = str(joint_loss)` | 將 objective 名稱固定成字串 |
| `normalize_pair_loss_matrix(...)` | 產生訓練用的 normalized cost matrix |

### 4.3 第 103-108 行：從 `loss_matrix` 找 oracle pair

```python
flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
normalized_flat_loss = normalized_loss_matrix.view(normalized_loss_matrix.size(0), -1)
flat_best = flat_loss.argmin(dim=-1)
num_tasks = loss_matrix.size(2)
best_first = flat_best // num_tasks
best_mid = flat_best % num_tasks
```

逐行意思：

```python
flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
```

將：

```text
[B, E, E]
```

轉為：

```text
[B, E*E]
```

每個欄位就是一種 pair。

```python
normalized_flat_loss = normalized_loss_matrix.view(...)
```

同樣將 normalized cost 攤平，後面計算 expected loss 使用。

```python
flat_best = flat_loss.argmin(dim=-1)
```

對每筆 sample 找出 cost 最低的 pair id，也就是 oracle pair。

```python
num_tasks = loss_matrix.size(2)
best_first = flat_best // num_tasks
best_mid = flat_best % num_tasks
```

將 flatten pair id 拆回：

```text
oracle first expert
oracle mid expert
```

例如 `E=3`，若 `flat_best=7`：

```text
best_first = 7 // 3 = 2
best_mid   = 7 % 3  = 1
```

代表 oracle 是：

```text
2 -> 1
```

### 4.4 第 109-111 行：`expected_loss`

```python
pair_prob = torch.softmax(pair_logits, dim=-1)
expected_loss = (pair_prob * normalized_flat_loss).sum(dim=-1).mean()
log_pair_prob = torch.log_softmax(pair_logits, dim=-1)
```

```python
pair_prob = torch.softmax(pair_logits, dim=-1)
```

將 router 的 raw logits 轉成每個 pair 的選擇機率。

```python
expected_loss = (pair_prob * normalized_flat_loss).sum(dim=-1).mean()
```

對每筆 sample 計算：

```text
Σ router 選該 pair 的機率 * 該 pair 的 normalized cost
```

例如：

```text
pair cost = [0.0, 0.2, 1.0]
router prob = [0.1, 0.8, 0.1]

expected cost = 0.1*0.0 + 0.8*0.2 + 0.1*1.0
              = 0.26
```

訓練會推動 router 把機率移到 cost 較低的 pair。

```python
log_pair_prob = torch.log_softmax(pair_logits, dim=-1)
```

這是後面 soft-label cross entropy 使用的 `log p(pair)`。

### 4.5 第 113-120 行：先建立可安全使用的零 loss

```python
zero_loss = pair_logits.sum() * 0.0
correct_soft_ce = zero_loss
correct_conf_ce = zero_loss
self_preserving_correct_conf_ce = zero_loss
correct_max_margin = zero_loss
correct_target_available = torch.zeros(loss_matrix.size(0), dtype=torch.bool, device=pair_logits.device)
correct_margin_available = torch.zeros(loss_matrix.size(0), dtype=torch.bool, device=pair_logits.device)
avg_correct_pairs = torch.tensor(0.0, device=pair_logits.device)
```

重點是：

```python
zero_loss = pair_logits.sum() * 0.0
```

它的數值為 `0`，但仍連在 router 的 computation graph 上。

用途是：若某個 batch 沒有可以計算的正確 pair target，函式仍能安全回傳可 backward 的零值，而不會因為回傳純 Python `0` 或無關 tensor 破壞流程。

### 4.6 第 121-140 行：`correct_soft_ce`

```python
if correct_matrix is not None:
    flat_correct = correct_matrix.to(device=pair_logits.device, dtype=torch.float32).view(loss_matrix.size(0), -1)
    correct_counts = flat_correct.sum(dim=-1, keepdim=True)
    correct_target_available = correct_counts.squeeze(-1) > 0
    avg_correct_pairs = (
        correct_counts[correct_target_available].mean()
        if correct_target_available.any()
        else torch.tensor(0.0, device=pair_logits.device)
    )
    target_prob = torch.where(
        correct_counts > 0,
        flat_correct / correct_counts.clamp_min(1.0),
        torch.zeros_like(flat_correct),
    )
    correct_soft_ce_all = -(target_prob * log_pair_prob).sum(dim=-1)
    correct_soft_ce = (
        correct_soft_ce_all[correct_target_available].mean()
        if correct_target_available.any()
        else zero_loss
    )
```

逐行拆開：

```python
flat_correct = correct_matrix...view(loss_matrix.size(0), -1)
```

把布林 correct matrix 攤平成每個 pair 的 `0/1`：

```text
[B, E, E] -> [B, E*E]
```

```python
correct_counts = flat_correct.sum(dim=-1, keepdim=True)
```

計算每筆 sample 有幾個 pairs 答對。

例如：

```text
flat_correct = [1, 0, 1, 0]
correct_counts = 2
```

```python
correct_target_available = correct_counts.squeeze(-1) > 0
```

標記哪些 samples 至少存在一個正確 pair。沒有任何 pair 答對的 sample，不能拿來做「選正確 pair」的 CE。

```python
avg_correct_pairs = correct_counts[correct_target_available].mean()
```

只是一項報表：有正確 pair 的 samples 中，平均一筆有多少個正確 pairs。

```python
target_prob = flat_correct / correct_counts
```

在所有正確 pairs 上平均分配 target probability。

例如：

```text
flat_correct = [1, 0, 1, 0]
target_prob  = [0.5, 0.0, 0.5, 0.0]
```

```python
correct_soft_ce_all = -(target_prob * log_pair_prob).sum(dim=-1)
```

用 soft target 與 router probability 做 cross entropy。

這個 objective 的語意是：

```text
只要答對的 pair 都可以，正確 pair 之間不區分誰成本比較低。
```

### 4.7 第 141-155 行：`correct_conf_ce`

```python
temperature = max(float(correct_soft_ce_temperature), 1e-6)
correct_conf_logits = -normalized_flat_loss / temperature
correct_conf_logits = correct_conf_logits.masked_fill(flat_correct <= 0, -1e9)
correct_conf_target = torch.softmax(correct_conf_logits, dim=-1)
correct_conf_target = torch.where(
    correct_counts > 0,
    correct_conf_target,
    torch.zeros_like(correct_conf_target),
)
correct_conf_ce_all = -(correct_conf_target * log_pair_prob).sum(dim=-1)
correct_conf_ce = (
    correct_conf_ce_all[correct_target_available].mean()
    if correct_target_available.any()
    else zero_loss
)
```

這段和 `correct_soft_ce` 的關鍵差異是：  
**正確 pairs 不再平均分配權重，而是偏好成本較低的正確 pair。**

```python
temperature = max(float(correct_soft_ce_temperature), 1e-6)
```

避免 temperature 為零。

```python
correct_conf_logits = -normalized_flat_loss / temperature
```

因為成本越低越好，因此加負號：

```text
cost 越低 -> -cost 越大 -> softmax target probability 越高
```

```python
correct_conf_logits = correct_conf_logits.masked_fill(flat_correct <= 0, -1e9)
```

所有答錯的 pairs 都遮掉，不允許它們獲得 target probability。

```python
correct_conf_target = torch.softmax(correct_conf_logits, dim=-1)
```

只在正確 pairs 之間分配概率，而且低 cost pair 分到比較多。

舉例：

```text
pair       是否正確   normalized cost
A          正確       0.10
B          正確       0.80
C          錯誤       0.00
```

雖然 C cost 可能低，但它答錯，所以被 mask 掉。target 只在 A、B 間分配，並偏好 A。

#### temperature 的效果

| Temperature | 效果 |
|---|---|
| 小 | 幾乎只偏向最低 cost 的正確 pair |
| 大 | 正確 pairs 之間權重較平均 |

所以 `correct_conf_ce` 的語意是：

```text
router 要選答對的 pair；若有多個答對 pair，偏好其中成本較低者。
```

### 4.8 第 156-181 行：`self_preserving_correct_conf_ce`

```python
if task_ids is not None:
    preserve_weight = min(max(float(self_preserve_weight), 0.0), 1.0)
    task_ids_device = task_ids.to(device=pair_logits.device, dtype=torch.long)
    valid_self_task = (task_ids_device >= 0) & (task_ids_device < num_tasks)
    self_pair_ids = task_ids_device.clamp(min=0, max=max(num_tasks - 1, 0)) * num_tasks + task_ids_device.clamp(
        min=0, max=max(num_tasks - 1, 0)
    )
```

```python
preserve_weight = min(max(float(self_preserve_weight), 0.0), 1.0)
```

確保 self 保留權重位於 `0~1`。

```python
valid_self_task = (task_ids_device >= 0) & (task_ids_device < num_tasks)
```

不是每筆 sample 都一定有對應的 self expert；這裡判定哪些 sample 可以談 self pair。

```python
self_pair_ids = task_id * num_tasks + task_id
```

取得 flatten 後的 self pair id。

例如 task id `1` 且 `E=3`：

```text
self pair = 1 -> 1
self_pair_id = 1 * 3 + 1 = 4
```

接著：

```python
self_correct = torch.zeros(loss_matrix.size(0), dtype=torch.bool, device=pair_logits.device)
self_correct[valid_self_task] = flat_correct.bool().gather(
    1, self_pair_ids.unsqueeze(1)
).squeeze(1)[valid_self_task]
```

這是在判斷：

```text
每筆 sample 的 self pair 本身是否答對？
```

這非常重要：只有 self pair 答對時才會刻意保留它。self pair 答錯時，不會因為它是自己的 expert 就強行監督 router 選錯路線。

```python
self_target = torch.zeros_like(correct_conf_target)
self_target.scatter_(1, self_pair_ids.unsqueeze(1), 1.0)
```

產生 one-hot 的 self pair target。

```python
soft_self_target = preserve_weight * self_target + (1.0 - preserve_weight) * correct_conf_target
```

將 self one-hot target 與普通 `correct_conf_target` 混合。

若：

```python
self_preserve_weight = 0.5
```

表示：

```text
50% target mass 指定給正確的 self pair
50% target mass 依 correct_conf_ce 分配給所有正確 pairs
```

若：

```python
self_preserve_weight = 1.0
```

表示只要 self pair 正確，就完全要求 router 選 self pair。

```python
self_preserving_target = torch.where(
    (valid_self_task & self_correct).unsqueeze(1),
    soft_self_target,
    correct_conf_target,
)
```

只有同時滿足：

```text
存在 self expert 且 self pair 答對
```

才使用保留 self 的 target；否則直接回到普通的 `correct_conf_target`。

```python
self_preserving_correct_conf_ce_all = -(self_preserving_target * log_pair_prob).sum(dim=-1)
```

最後一樣以 soft-label cross entropy 訓練 router。

### 4.9 第 182-194 行：`correct_max_margin`

```python
flat_correct_bool = flat_correct > 0
flat_wrong_bool = ~flat_correct_bool
correct_margin_available = flat_correct_bool.any(dim=-1) & flat_wrong_bool.any(dim=-1)
masked_correct_logits = pair_logits.masked_fill(~flat_correct_bool, -1e9)
masked_wrong_logits = pair_logits.masked_fill(~flat_wrong_bool, -1e9)
best_correct_logit = masked_correct_logits.max(dim=-1).values
best_wrong_logit = masked_wrong_logits.max(dim=-1).values
correct_max_margin_all = nn.functional.softplus(best_wrong_logit - best_correct_logit + float(margin))
correct_max_margin = (
    correct_max_margin_all[correct_margin_available].mean()
    if correct_margin_available.any()
    else zero_loss
)
```

這個 objective 不管正確 pairs 內部誰最好，只比較：

```text
router 給分最高的正確 pair
vs.
router 給分最高的錯誤 pair
```

它想達成：

```text
best_correct_logit >= best_wrong_logit + margin
```

如果錯誤 pair 的 logit 過高，loss 變大。

```python
softplus(best_wrong_logit - best_correct_logit + margin)
```

是平滑版 margin loss；即使接近邊界仍有平滑 gradient。

### 4.10 第 196-203 行：`ce_pair`

```python
sorted_loss, _ = normalized_flat_loss.sort(dim=-1)
if normalized_flat_loss.size(1) > 1:
    margin_mask = (sorted_loss[:, 1] - sorted_loss[:, 0]) >= float(margin)
else:
    margin_mask = torch.ones_like(flat_best, dtype=torch.bool)

ce_pair_all = nn.functional.cross_entropy(pair_logits, flat_best, reduction="none")
ce_pair = ce_pair_all[margin_mask].mean() if margin_mask.any() else zero_loss
```

`ce_pair` 把 oracle best pair 當成唯一分類標籤：

```text
router 預測 class = loss_matrix 最小的 pair id
```

但有一個額外篩選：

```python
margin_mask = (second_best_cost - best_cost) >= margin
```

意思是，如果最佳 pair 和第二佳 pair 差距太小，就不要強迫 router 把其中一個視為唯一答案。

例如：

```text
best cost       = 0.10
second best cost = 0.11
margin           = 0.05
```

因為：

```text
0.11 - 0.10 = 0.01 < 0.05
```

這筆 sample 不會加入 hard `ce_pair` loss。

### 4.11 第 205-212 行：舊式 stage1 / stage2 loss

```python
ce_first = zero_loss
ce_mid = zero_loss
if logits_first is not None:
    ce_first_all = nn.functional.cross_entropy(logits_first, best_first, reduction="none")
    ce_first = ce_first_all[margin_mask].mean() if margin_mask.any() else zero_loss
if logits_mid is not None:
    ce_mid_all = nn.functional.cross_entropy(logits_mid, best_mid, reduction="none")
    ce_mid = ce_mid_all[margin_mask].mean() if margin_mask.any() else zero_loss
```

這是為了支援舊式拆開訓練的 router：

- stage1 只預測 `first expert`
- stage2 只預測 `mid expert`

目前 cached joint trainer 的 model 直接產生 `pair_logits`，沒有傳 `logits_first` / `logits_mid`，主要走 joint pair loss。

### 4.12 第 214-256 行：最後到底用哪個 loss 訓練

```python
if mode == "stage1":
    ...
elif mode == "stage2":
    ...
elif mode == "joint":
    ...
```

目前 cached trainer 呼叫時未指定 `mode`，因此預設：

```python
mode = "joint"
```

#### `joint_loss == "ce_pair"`

```python
total_loss = ce_pair
```

把最低 cost pair 當作唯一正解分類。

#### `joint_loss == "expected_loss"`

```python
total_loss = expected_loss
```

最小化 router 的期望成本，會利用所有 pairs 的成本差距。

#### `joint_loss == "correct_soft_ce"`

```python
total_loss = correct_soft_ce
```

只要求 router 選到任何答對 pair，所有答對 pair 權重相同。

#### `joint_loss == "correct_conf_ce"`

```python
total_loss = correct_conf_ce
```

只要求 router 選答對 pair，但在答對 pairs 中偏好成本較低者。

#### `joint_loss == "self_preserving_correct_conf_ce"`

```python
total_loss = self_preserving_correct_conf_ce
```

若 self pair 答對，將部分或全部 target mass 特別保留給 self pair。

#### `joint_loss == "correct_max_margin"`

```python
total_loss = correct_max_margin
```

要求 router 對正確 pair 的最高 logit 高於錯誤 pair 的最高 logit。

#### `joint_loss == "correct_conf_ce_plus_margin"`

```python
total_loss = correct_conf_ce + float(pseudo_ce_weight) * correct_max_margin
```

同時要求：

- 選答對且低成本的 pair
- 正確 pair 分數壓過錯誤 pair

#### `joint_loss == "ce_pair_plus_expected"`

```python
total_loss = ce_pair
if pseudo_ce_weight > 0:
    total_loss = total_loss + float(pseudo_ce_weight) * expected_loss
```

同時：

- 把 oracle pair 當硬分類標籤
- 也利用完整 cost matrix 的連續差距

### 4.13 第 258-280 行：metrics 與回傳值

```python
metrics = {
    "expected_loss": ...,
    "correct_soft_ce": ...,
    "correct_conf_ce": ...,
    "self_preserving_correct_conf_ce": ...,
    "correct_max_margin": ...,
    ...
}
return total_loss, metrics, best_first, best_mid, flat_best
```

回傳值：

| 回傳值 | 用途 |
|---|---|
| `total_loss` | 呼叫端真正拿去 `loss.backward()` 的 tensor |
| `metrics` | log / wandb 用的統計數字 |
| `best_first` | 每筆 sample oracle pair 的 first expert |
| `best_mid` | 每筆 sample oracle pair 的 mid expert |
| `flat_best` | 每筆 sample oracle pair 的 flatten id |

### 4.14 這個 function 在 trainer 中真正如何使用

呼叫位置：[`train_internal_two_router_compact_cached_joint.py:1173`](./train_internal_two_router_compact_cached_joint.py#L1173)

```python
oracle_loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
    pair_logits=pair_logits,
    loss_matrix=batch.loss_matrix.to(device),
    correct_matrix=batch.correct_matrix.to(device),
    task_ids=batch.task_ids.to(device),
    joint_loss=args.joint_loss,
    pseudo_ce_weight=args.pseudo_ce_weight,
    margin=args.pseudo_ce_margin,
    loss_normalization=args.pair_loss_normalization,
    correct_soft_ce_temperature=args.correct_soft_ce_temperature,
    self_preserve_weight=args.self_preserve_weight,
)
```

如果：

```python
args.supervision_mode == "oracle_loss"
```

則：

```python
loss = oracle_loss
```

也就是上面 `joint_loss` 選定的 objective 真正用來更新 router。

如果：

```python
args.supervision_mode == "self_pair_ce"
```

則這個函式仍被呼叫來計算 oracle 統計，但真正訓練 loss 被替換成：

```python
loss = compute_self_pair_ce(...)
```

也就是強迫每筆 sample 走自己的 `task -> task`。

---

## 5. `score_from_cost()`

位置：[`router_pair_common.py:283`](./router_pair_common.py#L283)

```python
def score_from_cost(cost: torch.Tensor) -> torch.Tensor:
    return (1.0 - cost) * 100.0
```

如果輸入 cost 已是 `0~1` 的 normalized cost：

| Cost | Score |
|---|---|
| `0.0` | `100` |
| `0.2` | `80` |
| `0.5` | `50` |
| `1.0` | `0` |

這個 score 不是 OpenCompass evaluator accuracy，也不是答案正確率，只是把 normalized cost 轉成容易閱讀的分數方向：

```text
cost 越低，score 越高
```

---

## 6. `resolve_self_expert_ids()`

位置：[`router_pair_common.py:287`](./router_pair_common.py#L287)

這個 function 回答：

```text
這筆 sample 如果要走 self pair，它自己的 expert id 是多少？
```

```python
if sample_task_names is None:
    ids = task_ids.to(device=device, dtype=torch.long)
    valid = (ids >= 0) & (ids < len(expert_names))
    return ids, valid
```

如果沒有額外傳 task name，直接使用 `task_ids`。

```python
expert2id = {name: idx for idx, name in enumerate(expert_names)}
ids = torch.tensor(
    [expert2id.get(str(task_name), -1) for task_name in sample_task_names],
    ...
)
return ids, ids >= 0
```

若有實際 task name，則以名稱查 expert id。找不到時給 `-1`，表示這個 task 沒有可對應的 self expert。

---

## 7. 小工具 Functions

### `task_names_from_ids()`

位置：[`router_pair_common.py:307`](./router_pair_common.py#L307)

將數字 task id 轉成 expert 名稱。若 id 不合法，就顯示：

```text
task_id:<id>
```

### `optional_float()`

位置：[`router_pair_common.py:318`](./router_pair_common.py#L318)

用於 log 顯示：

- 有值：印固定小數位
- `None` 或非法值：印 `n/a`

### `optional_rate()`

位置：[`router_pair_common.py:324`](./router_pair_common.py#L324)

將比例轉成百分比顯示，例如：

```text
0.421 -> 42.10%
```

### `masked_mean()`

位置：[`router_pair_common.py:330`](./router_pair_common.py#L330)

```python
def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if bool(mask.any().item()):
        return float(values[mask].float().mean().item())
    return 0.0
```

只針對 `mask=True` 的 samples 計算平均。

在此檔案中主要用於：

```text
只有真的存在 self expert 的 samples 才計算 self routing 比例。
```

---

## 8. 問號 Function：`compute_routing_accuracy_stats()`

位置：[`router_pair_common.py:337`](./router_pair_common.py#L337)

這個 function **純粹是統計**，不參與 loss，也不會造成任何 gradient 或模型更新。

它要回答兩類問題：

1. router 選到 oracle pair 的比例是多少？
2. router / oracle 各自多常選 self pair？

### 8.1 輸入

```python
def compute_routing_accuracy_stats(
    pred_first: torch.Tensor,
    pred_mid: torch.Tensor,
    best_first: torch.Tensor,
    best_mid: torch.Tensor,
    task_ids: torch.Tensor,
    expert_names: Optional[Sequence[str]] = None,
    sample_task_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
```

| 輸入 | 意思 |
|---|---|
| `pred_first` | router 實際選的 first expert |
| `pred_mid` | router 實際選的 mid expert |
| `best_first` | `loss_matrix` oracle 的 first expert |
| `best_mid` | `loss_matrix` oracle 的 mid expert |
| `task_ids` | sample 自己 task 的 id |
| `expert_names` | expert id 對應名稱 |
| `sample_task_names` | sample 的實際 task 名稱；可用於找 self expert |

### 8.2 第 346-350 行：不需要 gradient

```python
pred_first = pred_first.detach()
pred_mid = pred_mid.detach()
best_first = best_first.detach()
best_mid = best_mid.detach()
task_ids = task_ids.to(device=pred_first.device)
```

`detach()` 明確切斷計算圖，因為這些資料只用來做比較與印報表。

### 8.3 第 351-358 行：找 self expert

```python
if expert_names is None:
    expert_names = [str(idx) for idx in range(int(max(task_ids.max().item() + 1, best_first.max().item() + 1, best_mid.max().item() + 1)))]
self_ids, self_valid = resolve_self_expert_ids(
    task_ids=task_ids,
    expert_names=expert_names,
    sample_task_names=sample_task_names,
    device=pred_first.device,
)
```

如果沒有 expert 名稱，就暫時以 `"0"`, `"1"` 等數字名稱處理。

`self_ids`：每筆 sample 自己 expert 的 id。  
`self_valid`：這筆 sample 是否真的有 self expert 可比較。

### 8.4 第 360-368 行：各種 accuracy 的真正意思

```python
first_correct = (pred_first == best_first).float().mean().item()
mid_correct = (pred_mid == best_mid).float().mean().item()
pair_acc = ((pred_first == best_first) & (pred_mid == best_mid)).float().mean().item()
```

這些指標問的是：

```text
router 是否選到 loss_matrix 中成本最低的 oracle pair？
```

| 指標 | 意思 |
|---|---|
| `first_acc` | router first expert 等於 oracle first expert 的比例 |
| `mid_acc` | router mid expert 等於 oracle mid expert 的比例 |
| `pair_acc` | router 的兩段 expert 都完全等於 oracle pair 的比例 |

這裡的 `pair_acc` **不是答案正確率**。  
若 router 選到另一個也答對、但 cost 稍高的 pair，`pair_acc` 仍會算成錯。

```python
self_first_acc = masked_mean(pred_first == self_ids, self_valid)
self_mid_acc = masked_mean(pred_mid == self_ids, self_valid)
pred_self_pair_acc = masked_mean((pred_first == self_ids) & (pred_mid == self_ids), self_valid)
```

這些指標問的是：

```text
router 自己實際有多常選回 sample 自己的 expert？
```

| 指標 | 意思 |
|---|---|
| `self_first_acc` | router 第一段選 self expert 的比例 |
| `self_mid_acc` | router 第二段選 self expert 的比例 |
| `self_pair_acc` | router 完整選 `self -> self` 的比例 |

```python
oracle_first_self_acc = masked_mean(best_first == self_ids, self_valid)
oracle_mid_self_acc = masked_mean(best_mid == self_ids, self_valid)
oracle_self_pair_acc = masked_mean((best_first == self_ids) & (best_mid == self_ids), self_valid)
```

這些指標問的是：

```text
cache 中的 oracle best pair 本來有多常就是 self pair？
```

| 指標 | 意思 |
|---|---|
| `oracle_first_self_acc` | oracle 第一段為 self expert 的比例 |
| `oracle_mid_self_acc` | oracle 第二段為 self expert 的比例 |
| `oracle_self_pair_acc` | oracle 完整為 `self -> self` 的比例 |

### 8.5 回傳 dict 中容易混淆的欄位

```python
return {
    "first_acc": first_correct,
    "mid_acc": mid_correct,
    "joint_acc": 0.5 * (first_correct + mid_correct),
    "pair_acc": pair_acc,
    ...
}
```

| 欄位 | 注意事項 |
|---|---|
| `joint_acc` | 只是 `first_acc` 與 `mid_acc` 的平均，不代表兩段同時正確 |
| `pair_acc` | 才是完整 pair 是否和 oracle 完全一致 |
| `self_pair_acc` | router 選 self pair 的比例 |
| `oracle_self_pair_acc` | cache oracle 本身是 self pair 的比例 |
| `self_applicable_ratio` | 有 self expert 可供比較的 samples 比例 |

### 8.6 實際讀 log 時怎麼判斷

例如：

```text
pair_acc=0.20
self_pair_acc=0.90
oracle_self_pair_acc=0.25
```

代表：

- router 幾乎都選 `self -> self`
- 但 cache oracle 只有 25% 真的是 self pair
- 所以 router 還沒學到 cache 中的跨 expert 最佳路線

反之：

```text
self_pair_acc=0.80
oracle_self_pair_acc=0.82
```

則表示大量走 self 可能本來就是合理的，而不是 router collapse。

---

## 9. `build_routing_summary()` 與 `print_routing_summary()`

位置：

- [`build_routing_summary()`](./router_pair_common.py#L387)
- [`print_routing_summary()`](./router_pair_common.py#L519)

`compute_routing_accuracy_stats()` 算的是一組 tensors 的簡單比例；`build_routing_summary()` 則是把整個 epoch 收集起來的 prediction / oracle 清單整理成報表。

### `build_routing_summary()` 內容

它會統計：

- 整體 `first_acc` / `mid_acc` / `pair_acc`
- 整體 router 最常選哪些 pairs
- 整體 oracle 最常是哪幾個 pairs
- 每個 task 最常選的 first expert
- 每個 task 最常選的 mid expert
- 每個 task 最常選的 pair
- 每個 task 的 self routing 比例
- 每個 task 所有 pairs 的完整比例

### `print_routing_summary()`

只負責將 summary 印成 log，例如：

```text
[ROUTE][VAL-EPOCH1] pair_acc=... self_pair=... oracle_self_pair=...
```

這份報表分析的是：

```text
router 實際選了什麼，以及它與 oracle 的一致程度。
```

---

## 10. `compute_route_score_stats()`

位置：[`router_pair_common.py:555`](./router_pair_common.py#L555)

這個 function 和 `pair_acc` 的觀點不同。

`pair_acc` 問：

```text
router 選的 pair id 是否完全等於 oracle pair id？
```

`compute_route_score_stats()` 問：

```text
router 選的 pair 實際 cost 表現距離 oracle 有多遠？
```

### 主要邏輯

```python
score_loss_matrix = normalize_pair_loss_matrix(...)
flat_loss = score_loss_matrix.view(score_loss_matrix.size(0), -1)
best_pair = flat_loss.argmin(dim=-1)
```

先取得 normalized costs 與 oracle。

```python
pred_cost = flat_loss[batch_idx, pred_pair]
oracle_cost = flat_loss[batch_idx, best_pair]
```

比較 router 選的 cost 與 oracle cost。

```python
self_cost = score_loss_matrix[batch_idx, self_ids.clamp_min(0), self_ids.clamp_min(0)]
```

另外計算固定走 self pair 的 cost。

### 回傳結果

| 欄位 | 意思 |
|---|---|
| `router_argmax_score` | router 實際選擇的 pair 分數 |
| `oracle_best_pair_score` | oracle 最佳 pair 分數 |
| `fixed_self_score` | 固定 `task -> task` 的分數 |
| `router_argmax_cost` | router 實際選擇的 normalized cost |
| `oracle_best_pair_cost` | oracle normalized cost |
| `fixed_self_cost` | 固定 self normalized cost |

因此你可以用它判斷：

```text
router 是否比固定 self routing 好？
router 距離 oracle 上限還有多遠？
```

---

## 11. `init_oracle_debug_accumulator()`

位置：[`router_pair_common.py:599`](./router_pair_common.py#L599)

這個 function 建立一個空的累積器：

```python
{
    "num_samples": 0,
    "expert_names": list(expert_names),
    "global_gold_pair_counter": Counter(),
    "global_gap_values": [],
    "global_self_minus_oracle_values": [],
    "per_task": defaultdict(...)
}
```

它要存的是 cache oracle 本身的結構：

- oracle 最常是哪個 pair
- oracle 第一名與第二名的 cost 差距
- self pair 比 oracle 差多少
- 各 task 分開後的上述統計

這個 accumulator 不保存 router 預測，也不產生 gradient。

---

## 12. 問號 Function：`update_oracle_debug_accumulator()`

位置：[`router_pair_common.py:619`](./router_pair_common.py#L619)

這個 function **不是訓練，也不是計算 router accuracy**。  
它只是在每個 batch 進來時，分析 cache 中 oracle pair 的分布，然後把資訊累積到 `acc` 中。

### 12.1 輸入

```python
def update_oracle_debug_accumulator(
    acc: Dict,
    loss_matrix: torch.Tensor,
    task_ids: torch.Tensor,
    sample_task_names: Optional[Sequence[str]] = None,
):
```

| 參數 | 意思 |
|---|---|
| `acc` | `init_oracle_debug_accumulator()` 建立的累積器 |
| `loss_matrix` | 目前 batch 的原始 pair cost matrix |
| `task_ids` | 目前 batch samples 的 task id |
| `sample_task_names` | 可選，直接提供 sample 的 task 名稱 |

注意：它沒有收 `pair_logits` 或 `pred_pair`，所以不可能分析 router 預測；它分析的完全是 cache oracle。

### 12.2 第 625-627 行：展平 pair matrix

```python
expert_names = acc["expert_names"]
num_pairs = loss_matrix.size(1) * loss_matrix.size(2)
flat_loss = loss_matrix.view(loss_matrix.size(0), num_pairs)
```

逐行意思：

```python
expert_names = acc["expert_names"]
```

取得 expert id 對應名稱，之後用來把 pair id 顯示為：

```text
piqa->rte
sst2->sst2
```

```python
num_pairs = loss_matrix.size(1) * loss_matrix.size(2)
```

如果有 `E` 個 experts，總共存在 `E*E` 種 pair。

```python
flat_loss = loss_matrix.view(loss_matrix.size(0), num_pairs)
```

把每筆 sample 的 `[E, E]` matrix 攤平成 pair list。

### 12.3 第 628-632 行：找 oracle 第一名、第二名與 gap

```python
sorted_loss, sorted_idx = flat_loss.sort(dim=-1)
best_pair = sorted_idx[:, 0]
best_cost = sorted_loss[:, 0]
second_cost = sorted_loss[:, 1] if num_pairs > 1 else sorted_loss[:, 0]
gap = second_cost - best_cost
```

逐行意思：

```python
sorted_loss, sorted_idx = flat_loss.sort(dim=-1)
```

對每筆 sample 的所有 pair costs 從低排到高。

```python
best_pair = sorted_idx[:, 0]
best_cost = sorted_loss[:, 0]
```

第一個就是 oracle pair 及其成本。

```python
second_cost = sorted_loss[:, 1] if num_pairs > 1 else sorted_loss[:, 0]
```

取得第二佳 pair 成本。若只有一種 pair，就讓第二佳等於最佳。

```python
gap = second_cost - best_cost
```

計算 oracle 優勢大小：

```text
gap = 第二佳成本 - 最佳成本
```

解讀方式：

| `gap` | 意思 |
|---|---|
| 很大 | oracle pair 很明確；選其他 pair 代價大 |
| 很小 | 前兩名幾乎平手；硬要求只選第一名未必合理 |
| `0` | 最佳與第二佳相同或幾乎完全相同 |

### 12.4 第 633-641 行：計算固定 self 比 oracle 差多少

```python
batch_idx = torch.arange(loss_matrix.size(0), device=loss_matrix.device)
self_ids, self_valid = resolve_self_expert_ids(
    task_ids=task_ids,
    expert_names=expert_names,
    sample_task_names=sample_task_names,
    device=loss_matrix.device,
)
self_cost = loss_matrix[batch_idx, self_ids.clamp_min(0), self_ids.clamp_min(0)]
self_minus_oracle = self_cost - best_cost
```

```python
batch_idx = torch.arange(...)
```

建立每筆 sample 的 index，用於一次取出每筆自己的 self pair 成本。

```python
self_ids, self_valid = resolve_self_expert_ids(...)
```

找每筆 sample 的 self expert id，並標記它是否有效。

```python
self_cost = loss_matrix[batch_idx, self_ids.clamp_min(0), self_ids.clamp_min(0)]
```

取得每筆 sample 的：

```text
self -> self 成本
```

`clamp_min(0)` 只是避免無效 id `-1` 造成 indexing error；後續只有 `self_valid=True` 的 sample 會真的被記錄 self 比較。

```python
self_minus_oracle = self_cost - best_cost
```

表示：

```text
固定走 self pair 比 oracle best pair 多出的 cost
```

解讀方式：

| `self_minus_oracle` | 意思 |
|---|---|
| `0` | self pair 本身就是最佳，或與最佳相同 |
| 小正數 | self 稍差，但差異不大 |
| 大正數 | 固定走 self 明顯吃虧，跨 expert pair 重要 |

由於 `best_cost` 已是最低成本，正常情況下這個值不應為負。

### 12.5 第 643-648 行：從 GPU tensor 轉成 Python 可累積資料

```python
best_pair_cpu = best_pair.detach().cpu().tolist()
gap_cpu = gap.detach().cpu().tolist()
self_minus_oracle_cpu = self_minus_oracle.detach().cpu().tolist()
self_valid_cpu = self_valid.detach().cpu().tolist()
if sample_task_names is None:
    sample_task_names = task_names_from_ids(task_ids.detach().cpu().tolist(), expert_names)
```

這裡做 `.detach().cpu().tolist()` 是因為接下來要放進：

- Python `list`
- Python `Counter`
- JSON summary

不需要 GPU tensor，也不需要 gradient。

若 caller 沒提供 sample task names，就從 `task_ids` 映射回名稱。

### 12.6 第 650-669 行：將統計加入 accumulator

```python
acc["num_samples"] += len(best_pair_cpu)
acc["global_gap_values"].extend(float(x) for x in gap_cpu)
```

先記錄：

- 總 sample 數量
- 所有 samples 的 oracle gap

接著逐 sample：

```python
for pair_id, gap_value, delta_value, has_self_expert, task_name in zip(
    best_pair_cpu, gap_cpu, self_minus_oracle_cpu, self_valid_cpu, sample_task_names
):
```

其中：

| 變數 | 意思 |
|---|---|
| `pair_id` | 該 sample 的 oracle pair flatten id |
| `gap_value` | 第二佳比最佳多多少 cost |
| `delta_value` | self pair 比 oracle 多多少 cost |
| `has_self_expert` | 此 sample 是否存在 self expert |
| `task_name` | sample 所屬 task |

```python
first_idx = int(pair_id) // loss_matrix.size(2)
mid_idx = int(pair_id) % loss_matrix.size(2)
pair_name = f"{expert_names[first_idx]}->{expert_names[mid_idx]}"
```

將 flatten pair id 轉成名稱。

例如 `pair_id=7`、`E=3`：

```text
first_idx = 2
mid_idx   = 1
pair_name = expert_names[2] -> expert_names[1]
```

```python
acc["global_gold_pair_counter"][pair_name] += 1
task_bucket = acc["per_task"][task_name]
task_bucket["count"] += 1
task_bucket["gold_pair_counter"][pair_name] += 1
task_bucket["gap_values"].append(float(gap_value))
```

累積：

- 所有 samples 中 oracle pairs 的全域分布
- 每個 task 各自的 oracle pair 分布
- 每個 task 的 oracle gap 分布

```python
if bool(has_self_expert):
    task_bucket["has_self_expert"] = True
    task_bucket["self_minus_oracle_values"].append(float(delta_value))
    acc["global_self_minus_oracle_values"].append(float(delta_value))
```

只有存在 self expert 的 samples 才記錄：

```text
self pair 與 oracle 的差距
```

### 12.7 這個 function 實際回答什麼問題

它最後讓你可以回答：

| 問題 | 對應資料 |
|---|---|
| cache 的 oracle 最常選哪些 expert pair？ | `gold_pair_counter` |
| oracle 選擇是否很明確？ | `gap_values` |
| 固定走自己的 expert 是否接近 oracle？ | `self_minus_oracle_values` |
| 哪些 task 特別需要跨 expert routing？ | 每 task 的 `avg_self_minus_oracle` |

它**不回答**：

| 不回答的問題 | 應看哪裡 |
|---|---|
| router 實際選了哪個 pair？ | `build_routing_summary()` |
| router 是否選到 oracle？ | `compute_routing_accuracy_stats()` |
| router 選的答案是否正確？ | trainer 中使用 `correct_matrix` 的 correctness stats |

---

## 13. `build_oracle_debug_summary()` 與 `print_oracle_debug_summary()`

位置：

- [`build_oracle_debug_summary()`](./router_pair_common.py#L672)
- [`print_oracle_debug_summary()`](./router_pair_common.py#L726)

`update_oracle_debug_accumulator()` 每個 batch 只負責累積；epoch 結束後由 `build_oracle_debug_summary()` 整理成 summary。

### summary 欄位

| 欄位 | 意思 |
|---|---|
| `avg_gap` | oracle 第一名與第二名平均差多少原始 cost |
| `p50_gap` | oracle gap 的中位附近值 |
| `p90_gap` | 90th percentile 附近的 gap |
| `avg_self_minus_oracle` | 固定 self 平均比 oracle 多多少原始 cost |
| `top_gold_pairs` | 最常成為 oracle 的 pairs |
| `per_task` | 每個 task 分開看的相同統計 |

### `routing_summary` 與 `oracle_debug_summary` 差異

| Summary | 看的對象 | 用途 |
|---|---|---|
| `routing_summary` | router 實際預測 | 看 router 學成什麼樣 |
| `oracle_debug_summary` | cache 中的 oracle cost 結構 | 看 supervision 本身偏好什麼 routing |

---

## 14. `flatten_routing_summary()`

位置：[`router_pair_common.py:751`](./router_pair_common.py#L751)

這個 function 將 nested summary 攤平成 wandb 好記錄的 key/value，例如：

```python
{
    "val_route/pair_acc": 0.42,
    "val_route/self_pair_acc": 0.67,
    "val_route_task/piqa_self_first_rate": 0.81,
}
```

只影響 logging，不影響訓練。

---

## 15. 三個問號 Function 的最短結論

### `compute_pair_losses()`

這是**訓練核心**：

```text
根據 cache 的 loss_matrix / correct_matrix，
決定 router 應偏好哪些 pair，
並產生拿去 backward 的 loss。
```

若 `supervision_mode=oracle_loss`，它回傳的 `total_loss` 就是真正訓練 router 的方向。

### `compute_routing_accuracy_stats()`

這是**router prediction 統計**：

```text
router 有多常選到 cache oracle pair？
router 有多常走 self pair？
oracle 本身有多常是 self pair？
```

它不影響模型更新。

### `update_oracle_debug_accumulator()`

這是**cache supervision 結構統計**：

```text
oracle pair 分布是什麼？
oracle 第一名比第二名明確多少？
self pair 比 oracle 差多少？
```

它完全不看 router prediction，也不影響模型更新。

---

## 16. 最容易混淆的地方

### 16.1 `loss_matrix` 最佳不一定等於 answer correctness 唯一答案

`loss_matrix` 表達的是 cost。若有多個 pair 答對，它們仍可能有不同 cost。

因此：

- `ce_pair`：強迫選 cost 最低的唯一 pair
- `correct_soft_ce`：所有答對 pair 都一樣好
- `correct_conf_ce`：所有答對 pair 都可接受，但偏好低 cost

### 16.2 `pair_acc` 不是 benchmark answer accuracy

`pair_acc` 是：

```text
router pair 是否完全等於 loss_matrix argmin pair
```

router 可能選了另一條也答對的路徑，但在 `pair_acc` 仍被視為沒有 match oracle。

### 16.3 `update_oracle_debug_accumulator()` 不是在檢查 router

這個函式輸入根本沒有 `pair_logits` 或 `pred_pair`，它只看 `loss_matrix`。  
所以它是在分析 cache 提供的 supervision 是否偏 self、是否有明確最佳路徑，而不是看 router 是否訓練成功。

