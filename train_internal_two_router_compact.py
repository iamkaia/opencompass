# train_internal_two_router_compact.py
import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from opencompass.models.router_moe_components import BertExternalEncoder, CompactCrossAttentionRouter


TASK_NAMES = ["iwslt2017", "medmcqa", "race", "squad2", "sst2"]


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


class ChunkedCompactFeatureDataset(Dataset):
    def __init__(self, feature_root: str, split: str):
        split_dir = os.path.join(feature_root, split)
        with open(os.path.join(split_dir, "manifest.json"), "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.items = []
        for fn in manifest["files"]:
            payload = torch.load(os.path.join(split_dir, fn), map_location="cpu")
            self.items.extend(payload["items"])

        if len(self.items) == 0:
            raise ValueError(f"No items loaded from {split_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "text": item["text"],
            "label": int(item["label"]),
            "task": item["task"],
            "first_vec": item["first_vec"],   # [H]
            "mid_vec": item["mid_vec"],       # [H]
        }


@dataclass
class Batch:
    texts: List[str]
    labels: torch.Tensor
    first_vec: torch.Tensor
    mid_vec: torch.Tensor


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        texts = [x["text"] for x in batch]
        labels = torch.tensor([x["label"] for x in batch], dtype=torch.long)

        first_vec = torch.stack([x["first_vec"] for x in batch], dim=0).to(torch.float32)
        mid_vec = torch.stack([x["mid_vec"] for x in batch], dim=0).to(torch.float32)

        return Batch(
            texts=texts,
            labels=labels,
            first_vec=first_vec,
            mid_vec=mid_vec,
        )


class InternalTwoRouterCompactModel(nn.Module):
    def __init__(self, bert_init: str, llama_hidden_size: int, router_dim: int, num_tasks: int):
        super().__init__()
        self.bert = BertExternalEncoder(bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size

        self.router_first = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=num_tasks,
        )
        self.router_mid = CompactCrossAttentionRouter(
            llama_hidden_size=llama_hidden_size,
            bert_hidden_size=bert_hidden_size,
            router_dim=router_dim,
            num_tasks=num_tasks,
        )

    def forward(
        self,
        bert_input_ids,
        bert_attention_mask,
        bert_token_type_ids,
        first_vec,
        mid_vec,
    ):
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )

        logits_first = self.router_first(
            llama_vec=first_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        logits_mid = self.router_mid(
            llama_vec=mid_vec,
            bert_prev=bert_prev,
            bert_last=bert_last,
            bert_attention_mask=bert_attention_mask,
        )
        return logits_first, logits_mid


def set_trainable(model, mode: str, freeze_bert: bool = False):
    for p in model.parameters():
        p.requires_grad = False

    if mode == "stage1":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_first.parameters():
            p.requires_grad = True

    elif mode == "stage2":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_mid.parameters():
            p.requires_grad = True

    elif mode == "joint":
        if not freeze_bert:
            for p in model.bert.parameters():
                p.requires_grad = True
        for p in model.router_first.parameters():
            p.requires_grad = True
        for p in model.router_mid.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown mode: {mode}")


def compute_loss(logits_first, logits_mid, labels, mode: str):
    ce = nn.CrossEntropyLoss()
    if mode == "stage1":
        return ce(logits_first, labels)
    if mode == "stage2":
        return ce(logits_mid, labels)
    if mode == "joint":
        return 0.5 * (ce(logits_first, labels) + ce(logits_mid, labels))
    raise ValueError(f"Unknown mode: {mode}")


@torch.no_grad()
def evaluate(model, loader, bert_tokenizer, device, max_bert_len, mode):
    model.eval()
    ce = nn.CrossEntropyLoss()

    total = 0
    correct_first = 0
    correct_mid = 0
    total_loss_first = 0.0
    total_loss_mid = 0.0

    all_labels = []
    all_preds_primary = []

    for batch in loader:
        labels = batch.labels.to(device)
        first_vec = batch.first_vec.to(device=device, dtype=torch.float32)
        mid_vec = batch.mid_vec.to(device=device, dtype=torch.float32)

        bert_enc = bert_tokenizer(
            batch.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_bert_len,
        )
        bert_input_ids = bert_enc["input_ids"].to(device)
        bert_attention_mask = bert_enc["attention_mask"].to(device)
        bert_token_type_ids = bert_enc.get("token_type_ids", None)
        if bert_token_type_ids is not None:
            bert_token_type_ids = bert_token_type_ids.to(device)

        logits_first, logits_mid = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=first_vec,
            mid_vec=mid_vec,
        )

        loss_first = ce(logits_first, labels)
        loss_mid = ce(logits_mid, labels)

        pred_first = logits_first.argmax(dim=-1)
        pred_mid = logits_mid.argmax(dim=-1)

        bs = labels.size(0)
        total += bs
        total_loss_first += loss_first.item() * bs
        total_loss_mid += loss_mid.item() * bs
        correct_first += (pred_first == labels).sum().item()
        correct_mid += (pred_mid == labels).sum().item()

        all_labels.extend(labels.cpu().tolist())
        if mode == "stage1":
            all_preds_primary.extend(pred_first.cpu().tolist())
        elif mode == "stage2":
            all_preds_primary.extend(pred_mid.cpu().tolist())
        else:
            all_preds_primary.extend(pred_mid.cpu().tolist())

    acc_first = correct_first / max(total, 1)
    acc_mid = correct_mid / max(total, 1)

    cm = confusion_matrix(all_labels, all_preds_primary, labels=list(range(len(TASK_NAMES))))
    report = classification_report(
        all_labels,
        all_preds_primary,
        labels=list(range(len(TASK_NAMES))),
        target_names=TASK_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )

    if mode == "stage1":
        score = acc_first
    elif mode == "stage2":
        score = acc_mid
    else:
        score = 0.5 * (acc_first + acc_mid)

    return {
        "val_loss_first": total_loss_first / max(total, 1),
        "val_loss_mid": total_loss_mid / max(total, 1),
        "acc_first": acc_first,
        "acc_mid": acc_mid,
        "score": score,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }


def save_ckpt(model, out_dir, first_layer_idx, middle_layer_idx, max_bert_len, metrics, epoch):
    os.makedirs(out_dir, exist_ok=True)
    model.bert.encoder.save_pretrained(os.path.join(out_dir, "encoder"))

    torch.save(
        {
            "bert_encoder": model.bert.state_dict(),
            "router_first": model.router_first.state_dict(),
            "router_mid": model.router_mid.state_dict(),
        },
        os.path.join(out_dir, "router_heads.pt"),
    )

    cfg = {
        "task_names": TASK_NAMES,
        "first_layer_idx": first_layer_idx,
        "middle_layer_idx": middle_layer_idx,
        "router_max_len": max_bert_len,
        "router_feature_type": "compact_last_valid_token_vector",
    }
    save_json(cfg, os.path.join(out_dir, "router_config.json"))
    save_json({"best_epoch": epoch, "metrics": metrics}, os.path.join(out_dir, "best_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--bert_init", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--mode", choices=["stage1", "stage2", "joint"], required=True)
    parser.add_argument("--load_from", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)

    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    train_ds = ChunkedCompactFeatureDataset(args.feature_root, "train")
    val_ds = ChunkedCompactFeatureDataset(args.feature_root, "validation")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device == "cuda"),
    )

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
    llama_hidden_size = int(train_ds[0]["first_vec"].numel())

    model = InternalTwoRouterCompactModel(
        bert_init=args.bert_init,
        llama_hidden_size=llama_hidden_size,
        router_dim=args.router_dim,
        num_tasks=len(TASK_NAMES),
    ).to(device)

    if args.load_from is not None:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[LOAD] loaded from {args.load_from}")

    set_trainable(model, args.mode, freeze_bert=args.freeze_bert)

    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    print(f"[INFO] train_size = {len(train_ds)}")
    print(f"[INFO] val_size   = {len(val_ds)}")
    print(f"[INFO] mode = {args.mode}")
    print(f"[INFO] freeze_bert = {args.freeze_bert}")
    print(f"[INFO] steps_per_epoch = {steps_per_epoch}")
    print(f"[INFO] total_steps = {total_steps}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0

        for step, batch in enumerate(train_loader):
            labels = batch.labels.to(device)
            first_vec = batch.first_vec.to(device=device, dtype=torch.float32)
            mid_vec = batch.mid_vec.to(device=device, dtype=torch.float32)

            bert_enc = bert_tokenizer(
                batch.texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_bert_len,
            )
            bert_input_ids = bert_enc["input_ids"].to(device)
            bert_attention_mask = bert_enc["attention_mask"].to(device)
            bert_token_type_ids = bert_enc.get("token_type_ids", None)
            if bert_token_type_ids is not None:
                bert_token_type_ids = bert_token_type_ids.to(device)

            logits_first, logits_mid = model(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=first_vec,
                mid_vec=mid_vec,
            )

            loss = compute_loss(logits_first, logits_mid, labels, args.mode)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"[TRAIN] epoch={epoch} step={step} loss={loss.item():.8f} lr={lr_now:.10f}")

        avg_train_loss = running_loss / max(len(train_loader), 1)

        metrics = evaluate(
            model=model,
            loader=val_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            mode=args.mode,
        )

        print(
            f"[EVAL] epoch={epoch} "
            f"train_loss={avg_train_loss:.8f} "
            f"val_loss_first={metrics['val_loss_first']:.8f} "
            f"val_loss_mid={metrics['val_loss_mid']:.8f} "
            f"acc_first={metrics['acc_first']:.6f} "
            f"acc_mid={metrics['acc_mid']:.6f} "
            f"score={metrics['score']:.6f}"
        )

        improved = metrics["score"] > (best_score + args.early_stop_min_delta)
        if improved:
            best_score = metrics["score"]
            best_epoch = epoch
            no_improve_epochs = 0
            save_ckpt(
                model=model,
                out_dir=args.out_dir,
                first_layer_idx=args.first_layer_idx,
                middle_layer_idx=args.middle_layer_idx,
                max_bert_len=args.max_bert_len,
                metrics={"epoch": epoch, "avg_train_loss": avg_train_loss, **metrics},
                epoch=epoch,
            )
        else:
            no_improve_epochs += 1
            print(
                f"[EARLY_STOP] no improvement for {no_improve_epochs} epoch(s). "
                f"best_score={best_score:.6f} at epoch={best_epoch}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(f"[EARLY_STOP] stop training because patience={args.early_stop_patience} is reached.")
                break

    print(f"[DONE] best_score={best_score:.6f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
