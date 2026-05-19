#!/usr/bin/env python3
import json
import os

import numpy as np
import transformers
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

DATA_DIR = "datasets_classifier"
MODEL_NAME = "prajjwal1/bert-tiny"
OUT_DIR = "task_classifier_ckpt"

# ------------------------------------------------
# Load jsonl files
# ------------------------------------------------


def load_split(split):
    texts, labels = [], []
    label_map = {}

    for task_id, task in enumerate(sorted(os.listdir(DATA_DIR))):
        label_map[task] = task_id
        path = os.path.join(DATA_DIR, task, f"{split}.jsonl")
        if not os.path.exists(path):
            continue

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                ex = json.loads(line)
                texts.append(ex["text"])
                labels.append(task_id)

    return texts, labels, label_map


train_texts, train_labels, label_map = load_split("train")
val_texts, val_labels, _ = load_split("validation")

id2label = {v: k for k, v in label_map.items()}
num_labels = len(label_map)

print("Label map:", label_map)

# ------------------------------------------------
# Build HF Dataset
# ------------------------------------------------
dataset = DatasetDict({
    "train": Dataset.from_dict({"text": train_texts, "label": train_labels}),  # 原本是text
    "validation": Dataset.from_dict({"text": val_texts, "label": val_labels}),
})

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def tokenize(batch):
    return tokenizer(
        batch["text"],  # 原本是text
        truncation=True,
        padding="max_length",
        max_length=512,
    )


dataset = dataset.map(tokenize, batched=True)
dataset.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# ------------------------------------------------
# Model
# ------------------------------------------------
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label_map,
)

# ------------------------------------------------
# Metrics
# ------------------------------------------------

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


print("transformers version:", transformers.__version__)

common_kwargs = dict(
    output_dir=OUT_DIR,
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    report_to="none",
)


try:
    # 舊版/常見版
    args = TrainingArguments(
        evaluation_strategy="epoch",
        **common_kwargs,
    )
except TypeError:
    # 新版（把 evaluation_strategy 改名成 eval_strategy）
    args = TrainingArguments(
        eval_strategy="epoch",
        **common_kwargs,
    )

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(OUT_DIR)
tokenizer.save_pretrained(OUT_DIR)

print("✅ Task classifier training finished.")
