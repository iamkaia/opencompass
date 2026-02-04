'''
import json

path = "/home/kaia/opencompass/outputs/default/20251126_224626/results/llama-2-7b-chat-hf/sst2.json"  # 如果檔名不同就改這裡

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0
total = 0

for k, v in details.items():
    if k == "type":
        continue  # 跳過 "GEN"

    pred = v["predictions"].strip().lower()
    ref = v["references"].strip()   # "1" or "0"

    if pred.startswith("positive"):
        pred_label = "1"
    elif pred.startswith("negative"):
        pred_label = "0"
    else:
        # 其他怪輸出可以先印出來 debug
        print(f"[WARN] unexpected pred: {pred!r}")
        continue

    total += 1
    if pred_label == ref:
        correct += 1

print(f"Correct: {correct}/{total}")
print(f"Accuracy: {correct / total:.4f}")
'''

'''
import json

path = "/home/kaia/opencompass/outputs/default/20251211_011939/results/iwslt_lora/sst2.json"

#path = "/home/kaia/opencompass/outputs/default/20251125_043200/results/llama-2-7b-chat-hf/sst2_gen.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0
total = 0

for k, v in details.items():
    if k == "type":
        continue

    pred = v["predictions"].strip().lower()
    ref = v["references"].strip()

    # 比較寬鬆的規則：只要有出現 positive / negative 就判斷
    has_pos = "positive" in pred
    has_neg = "negative" in pred

    if has_pos and not has_neg:
        pred_label = "1"
    elif has_neg and not has_pos:
        pred_label = "0"
    else:
        print(f"[WARN] ambiguous pred: {pred[:80]!r}")

    total += 1
    if pred_label == ref:
        correct += 1

print(f"Used samples: {total}")
if total > 0:
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total:.4f}")
else:
    print("No valid samples parsed.")
'''

'''
import json

path = "/home/kaia/opencompass/outputs/default/20251211_011939/results/iwslt_lora/sst2.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0
total = 0   # 這裡代表「有成功判成 0/1 的樣本數」

for k, v in details.items():
    if k == "type":
        continue

    pred = v["predictions"].strip().lower()
    ref = v["references"].strip()

    # 比較寬鬆的規則：只要有出現 positive / negative 就判斷
    has_pos = "positive" in pred
    has_neg = "negative" in pred

    if has_pos and not has_neg:
        pred_label = "1"
    elif has_neg and not has_pos:
        pred_label = "0"
    else:
        # 這種情況沒有 pred_label，就不要往下算，直接跳過
        print(f"[WARN] ambiguous pred: {pred[:80]!r}")
        continue

    total += 1
    if pred_label == ref:
        correct += 1

print(f"Used samples: {total}")
if total > 0:
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy: {correct / total:.4f}")
else:
    print("No valid samples parsed.")
'''
'''
import json

path = "/home/kaia/opencompass/outputs/default/20251211_011939/results/iwslt_lora/sst2.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0          # 預測正確的筆數
total = 0            # 總筆數（所有樣本都算進去）
ambiguous = 0        # 無法判斷正負的筆數（可選，看你要不要）

for k, v in details.items():
    if k == "type":
        continue

    pred = v["predictions"].strip().lower()
    ref = v["references"].strip()

    has_pos = "positive" in pred
    has_neg = "negative" in pred

    # 不管怎樣，這一筆都算進分母
    total += 1

    if has_pos and not has_neg:
        pred_label = "1"
    elif has_neg and not has_pos:
        pred_label = "0"
    else:
        # ambiguous：沒有明確 positive / negative
        ambiguous += 1
        print(f"[WARN] ambiguous pred: {pred[:80]!r}")
        # 這一筆直接當作預測錯（不增加 correct），所以直接 continue
        continue

    # 只有在有 pred_label 的情況下才會跑到這裡
    if pred_label == ref:
        correct += 1

print(f"Total samples: {total}")
print(f"Ambiguous preds (counted as wrong): {ambiguous}")
if total > 0:
    print(f"Correct: {correct}/{total}")
    print(f"Accuracy (all samples): {correct / total:.4f}")
else:
    print("No samples found.")
'''
import json
import re

# ====== 1. utility: normalize ======
def normalize(text: str):
    text = text.strip().lower()
    # 移除標點符號
    text = re.sub(r"[^\w\s]", "", text)
    return text


# ====== 2. sentiment lexicon ======
positive_words = ["positive", "good", "great", "excellent", "amazing", "wonderful", "nice"]
negative_words = ["negative", "bad", "terrible", "awful", "horrible", "poor"]


# ====== 3. classification function ======
def classify(pred: str):
    """
    輸入一段模型的 predictions 文字
    輸出:
        "1" → positive
        "0" → negative
        None → ambiguous（算錯）
    """
    p = normalize(pred)

    # --- exact match (最準確) ---
    if p == "positive":
        return "1"
    if p == "negative":
        return "0"

    # --- negation-aware rules ---
    if "not positive" in p or "not good" in p:
        return "0"
    if "not negative" in p or "not bad" in p:
        return "1"

    # --- lexicon fallback ---
    pos = any(w in p for w in positive_words)
    neg = any(w in p for w in negative_words)

    if pos and not neg:
        return "1"
    if neg and not pos:
        return "0"

    # ambiguous
    return None


# ====== 4. load file ======
path = "/home/kaia/opencompass/outputs/default/20251216_202502/results/race_fused_lora/sst2.json"

with open(path, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0
total = 0
ambiguous = 0


# ====== 5. main evaluation loop ======
for k, v in details.items():
    if k == "type":
        continue

    pred = v["predictions"]
    ref = v["references"].strip()  # "1" 或 "0"

    total += 1

    pred_label = classify(pred)

    if pred_label is None:
        ambiguous += 1
        print(f"[WARN] ambiguous: {pred[:80]!r}")
        continue

    if pred_label == ref:
        correct += 1


# ====== 6. final report ======
print("========================================")
print(f"Total samples      : {total}")
print(f"Ambiguous counted  : {ambiguous}")
print(f"Correct predictions: {correct}")
print(f"Accuracy           : {correct / (total):.4f}")
print("========================================")
'''

path_v1 = "/home/kaia/opencompass/outputs/default/20251210_145822/results/race_lora/sst2.json"
path_v2 = "/home/kaia/opencompass/outputs/default/20251210_160448/results/medmcqa_lora/sst2.json"
path_v3 = "/home/kaia/opencompass/outputs/default/20251210_134849/results/squad2_lora/sst2.json"
path_v4 = "/home/kaia/opencompass/outputs/default/20251210_100457/results/sst2_lora/sst2.json"
path_v5 = "/home/kaia/opencompass/outputs/default/20251211_011939/results/iwslt_lora/sst2.json"
path_base = "/home/kaia/opencompass/outputs/default/20251208_000433/results/llama-2-7b-chat-hf/sst2.json"

import json
import re

# ==========================================================
# strict SST-2 classifier
# ==========================================================
def strict_classify(pred: str):
    """
    嚴格判斷 sentiment label

    只接受：
    - negative / Negative / NEGATIVE / negative. / "negative"
    - positive / Positive / POSITIVE / positive. / "positive"

    不接受：
    - "The answer is negative."
    - "negative overall"
    - "not positive"
    - "This is positive!"
    - 任何多餘字詞
    """

    # 去掉前後空白
    p = pred.strip()

    # 去掉前後引號或簡單標點： " ' . , ! ?
    p = p.strip('"\'.!?')
    p = p.lower()

    if p == "negative":
        return "0"
    if p == "positive":
        return "1"

    return None   # ambiguous 或錯誤輸出


# ==========================================================
# Main evaluation
# ==========================================================


with open(path_v5, "r", encoding="utf-8") as f:
    data = json.load(f)

details = data["details"]

correct = 0
total = 0
ambiguous = 0

for k, v in details.items():
    if k == "type":
        continue

    pred = v["predictions"]
    ref = v["references"].strip()    # "0" or "1"

    total += 1

    pred_label = strict_classify(pred)

    if pred_label is None:
        ambiguous += 1
        print(f"[WARN] ambiguous or invalid answer → {pred!r}")
        continue

    if pred_label == ref:
        correct += 1

# ==========================================================
# Final report
# ==========================================================
print("========================================")
print(f"Total samples      : {total}")
print(f"Ambiguous (wrong)  : {ambiguous}")
print(f"Correct            : {correct}")
print(f"Accuracy           : {correct / total:.4f}")
print("========================================")
'''