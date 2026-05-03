import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from opencompass.models.router_moe_components import BertExternalEncoder, CompactCrossAttentionRouter


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def parse_task_names(raw: Optional[str], fallback: Optional[Sequence[str]] = None) -> List[str]:
    if raw:
        tasks = [part.strip() for part in raw.split(",") if part.strip()]
        if tasks:
            return tasks
    if fallback:
        return list(fallback)
    raise ValueError("Failed to resolve task names")


class CachedLossMatrixDataset(Dataset):
    def __init__(self, feature_root: str, split: str, selected_task_names: Optional[Sequence[str]] = None):
        split_dir = os.path.join(feature_root, split)
        manifest_path = os.path.join(split_dir, "manifest.json")
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        source_task_names = list(manifest.get("task_names") or manifest.get("expert_names") or [])
        if not source_task_names:
            raise ValueError(f"Missing task_names/expert_names in {manifest_path}")

        if selected_task_names:
            resolved_task_names = [str(name) for name in selected_task_names]
        else:
            resolved_task_names = list(source_task_names)

        missing = [name for name in resolved_task_names if name not in source_task_names]
        if missing:
            raise ValueError(
                f"Requested task_names={missing} are not present in cached feature manifest {manifest_path}. "
                f"Available={source_task_names}"
            )

        selected_indices = [source_task_names.index(name) for name in resolved_task_names]
        old_to_new_task_id = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_indices)}
        selected_index_tensor = torch.tensor(selected_indices, dtype=torch.long)

        self.items = []
        for fn in manifest["files"]:
            payload = torch.load(os.path.join(split_dir, fn), map_location="cpu")
            for item in payload["items"]:
                task_name = str(item["task"])
                if task_name not in resolved_task_names:
                    continue

                loss_matrix = item["loss_matrix"]
                sliced_loss_matrix = loss_matrix.index_select(0, selected_index_tensor).index_select(1, selected_index_tensor)
                flat_loss = sliced_loss_matrix.view(-1)
                best_pair = int(flat_loss.argmin().item())
                num_tasks = len(resolved_task_names)
                best_first = best_pair // num_tasks
                best_mid = best_pair % num_tasks

                remapped = dict(item)
                remapped["task_id"] = int(old_to_new_task_id[int(item["task_id"])])
                remapped["loss_matrix"] = sliced_loss_matrix
                remapped["pair_label"] = best_pair
                remapped["first_label"] = best_first
                remapped["mid_label"] = best_mid
                self.items.append(remapped)

        if not self.items:
            raise ValueError(f"No items loaded from {split_dir}")
        self.task_names = resolved_task_names
        self.source_task_names = source_task_names

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        return {
            "text": item["text"],
            "task": item["task"],
            "task_id": int(item["task_id"]),
            "first_vec": item["first_vec"],
            "mid_vec": item["mid_vec"],
            "loss_matrix": item["loss_matrix"],
            "pair_label": int(item["pair_label"]),
            "first_label": int(item["first_label"]),
            "mid_label": int(item["mid_label"]),
        }


@dataclass
class Batch:
    texts: List[str]
    task_ids: torch.Tensor
    first_vec: torch.Tensor
    mid_vec: torch.Tensor
    loss_matrix: torch.Tensor


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            first_vec=torch.stack([x["first_vec"] for x in batch], dim=0).to(torch.float32),
            mid_vec=torch.stack([x["mid_vec"] for x in batch], dim=0).to(torch.float32),
            loss_matrix=torch.stack([x["loss_matrix"] for x in batch], dim=0).to(torch.float32),
        )


class InternalTwoRouterCachedAnswerSupervisionModel(nn.Module):
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

    def forward(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, first_vec, mid_vec):
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


def set_trainable(model, freeze_bert: bool = False, mode: str = "joint"):
    for p in model.parameters():
        p.requires_grad = False
    if not freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad = True
    if mode in {"stage1", "joint"}:
        for p in model.router_first.parameters():
            p.requires_grad = True
    if mode in {"stage2", "joint"}:
        for p in model.router_mid.parameters():
            p.requires_grad = True


def compute_router_loss(
    logits_first: torch.Tensor,
    logits_mid: torch.Tensor,
    loss_matrix: torch.Tensor,
    mode: str,
    pseudo_ce_weight: float,
):
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    flat_best = flat_loss.argmin(dim=-1)
    num_tasks = loss_matrix.size(2)
    best_first = flat_best // num_tasks
    best_mid = flat_best % num_tasks

    prob_first = torch.softmax(logits_first, dim=-1)
    prob_mid = torch.softmax(logits_mid, dim=-1)
    joint_prob = prob_first.unsqueeze(2) * prob_mid.unsqueeze(1)
    expected_loss = (joint_prob * loss_matrix).sum(dim=(1, 2)).mean()

    ce_first = nn.functional.cross_entropy(logits_first, best_first)
    ce_mid = nn.functional.cross_entropy(logits_mid, best_mid)

    if mode == "stage1":
        total_loss = ce_first
    elif mode == "stage2":
        total_loss = ce_mid
    elif mode == "joint":
        total_loss = expected_loss
        if pseudo_ce_weight > 0:
            total_loss = total_loss + float(pseudo_ce_weight) * 0.5 * (ce_first + ce_mid)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    metrics = {
        "expected_loss": float(expected_loss.detach().item()),
        "pseudo_ce_first": float(ce_first.detach().item()),
        "pseudo_ce_mid": float(ce_mid.detach().item()),
        "best_pair_loss": float(flat_loss.min(dim=-1).values.mean().item()),
    }
    return total_loss, metrics, best_first, best_mid


def score_from_cost(cost: torch.Tensor) -> torch.Tensor:
    return (1.0 - cost) * 100.0


def compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, task_ids):
    pred_first = pred_first.detach()
    pred_mid = pred_mid.detach()
    best_first = best_first.detach()
    best_mid = best_mid.detach()
    task_ids = task_ids.to(device=pred_first.device)
    return {
        "first_acc": float((pred_first == best_first).float().mean().item()),
        "mid_acc": float((pred_mid == best_mid).float().mean().item()),
        "joint_acc": float(0.5 * (((pred_first == best_first).float().mean().item()) + ((pred_mid == best_mid).float().mean().item()))),
        "pair_acc": float(((pred_first == best_first) & (pred_mid == best_mid)).float().mean().item()),
        "self_first_acc": float((pred_first == task_ids).float().mean().item()),
        "self_mid_acc": float((pred_mid == task_ids).float().mean().item()),
        "self_joint_acc": float(0.5 * (((pred_first == task_ids).float().mean().item()) + ((pred_mid == task_ids).float().mean().item()))),
        "self_pair_acc": float(((pred_first == task_ids) & (pred_mid == task_ids)).float().mean().item()),
        "oracle_first_self_acc": float((best_first == task_ids).float().mean().item()),
        "oracle_mid_self_acc": float((best_mid == task_ids).float().mean().item()),
        "oracle_self_joint_acc": float(0.5 * (((best_first == task_ids).float().mean().item()) + ((best_mid == task_ids).float().mean().item()))),
        "oracle_self_pair_acc": float(((best_first == task_ids) & (best_mid == task_ids)).float().mean().item()),
    }


def compute_route_score_stats(loss_matrix: torch.Tensor, pred_first: torch.Tensor, pred_mid: torch.Tensor, task_ids: torch.Tensor):
    batch_idx = torch.arange(loss_matrix.size(0), device=loss_matrix.device)
    pred_cost = loss_matrix[batch_idx, pred_first, pred_mid]
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    best_pair = flat_loss.argmin(dim=-1)
    best_first = best_pair // loss_matrix.size(2)
    best_mid = best_pair % loss_matrix.size(2)
    oracle_cost = loss_matrix[batch_idx, best_first, best_mid]
    self_ids = task_ids.to(loss_matrix.device)
    self_cost = loss_matrix[batch_idx, self_ids, self_ids]
    return {
        "router_argmax_score": float(score_from_cost(pred_cost).mean().item()),
        "oracle_best_pair_score": float(score_from_cost(oracle_cost).mean().item()),
        "fixed_self_score": float(score_from_cost(self_cost).mean().item()),
        "router_argmax_cost": float(pred_cost.mean().item()),
        "oracle_best_pair_cost": float(oracle_cost.mean().item()),
        "fixed_self_cost": float(self_cost.mean().item()),
    }


@torch.no_grad()
def evaluate(model, loader, bert_tokenizer, device, max_bert_len, mode, pseudo_ce_weight):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}

    for batch in loader:
        bert_enc = bert_tokenizer(
            batch.texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_bert_len,
        )
        bert_input_ids = bert_enc["input_ids"].to(device)
        bert_attention_mask = bert_enc["attention_mask"].to(device)
        bert_token_type_ids = bert_enc.get("token_type_ids")
        if bert_token_type_ids is not None:
            bert_token_type_ids = bert_token_type_ids.to(device)

        logits_first, logits_mid = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=batch.first_vec.to(device),
            mid_vec=batch.mid_vec.to(device),
        )
        loss, metrics, best_first, best_mid = compute_router_loss(
            logits_first=logits_first,
            logits_mid=logits_mid,
            loss_matrix=batch.loss_matrix.to(device),
            mode=mode,
            pseudo_ce_weight=pseudo_ce_weight,
        )
        pred_first = logits_first.argmax(dim=-1)
        pred_mid = logits_mid.argmax(dim=-1)
        batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
        score_stats = compute_route_score_stats(batch.loss_matrix.to(device), pred_first, pred_mid, batch.task_ids)

        bs = batch.task_ids.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for source in (metrics, batch_stats, score_stats):
            for key, value in source.items():
                metric_totals[key] = metric_totals.get(key, 0.0) + value * bs

    denom = max(total_samples, 1)
    result = {"loss": total_loss / denom}
    for key, value in metric_totals.items():
        result[key] = value / denom
    return result


def save_ckpt(model, out_dir, task_names, first_layer_idx, middle_layer_idx, max_bert_len, metrics, epoch, mode, pseudo_ce_weight):
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
    save_json(
        {
            "task_names": list(task_names),
            "expert_names": list(task_names),
            "first_layer_idx": first_layer_idx,
            "middle_layer_idx": middle_layer_idx,
            "router_max_len": max_bert_len,
            "router_feature_type": "cached_prompt_vectors_with_loss_matrix",
            "supervision_type": "cached_answer_supervision_expected_plus_pseudo_ce",
            "mode": str(mode),
            "pseudo_ce_weight": float(pseudo_ce_weight),
        },
        os.path.join(out_dir, "router_config.json"),
    )
    save_json({"best_epoch": epoch, "metrics": metrics}, os.path.join(out_dir, "best_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--bert_init", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--task_names", type=str, default=None)
    parser.add_argument("--load_from", type=str, default=None)
    parser.add_argument("--mode", choices=["stage1", "stage2", "joint"], default="joint")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=15)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument("--pseudo_ce_weight", type=float, default=0.5)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] device = {device}")

    selected_task_names = parse_task_names(args.task_names) if args.task_names else None
    train_ds = CachedLossMatrixDataset(args.feature_root, "train", selected_task_names)
    val_ds = CachedLossMatrixDataset(args.feature_root, "validation", train_ds.task_names)
    task_names = list(train_ds.task_names)

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

    model = InternalTwoRouterCachedAnswerSupervisionModel(
        bert_init=args.bert_init,
        llama_hidden_size=llama_hidden_size,
        router_dim=args.router_dim,
        num_tasks=len(task_names),
    ).to(device)

    if args.load_from is not None:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        model.load_state_dict(state, strict=False)
        print(f"[LOAD] loaded from {args.load_from}")

    set_trainable(model, freeze_bert=args.freeze_bert, mode=args.mode)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)

    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * 0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_score = -1e18
    best_epoch = -1
    no_improve_epochs = 0

    print(f"[INFO] train_size = {len(train_ds)}")
    print(f"[INFO] val_size   = {len(val_ds)}")
    print(f"[INFO] task_names = {task_names}")
    print(f"[INFO] mode = {args.mode}")
    print(f"[INFO] freeze_bert = {args.freeze_bert}")
    print(f"[INFO] pseudo_ce_weight = {args.pseudo_ce_weight}")

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        metric_totals: Dict[str, float] = {}
        total_samples = 0

        for step, batch in enumerate(train_loader):
            bert_enc = bert_tokenizer(
                batch.texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_bert_len,
            )
            bert_input_ids = bert_enc["input_ids"].to(device)
            bert_attention_mask = bert_enc["attention_mask"].to(device)
            bert_token_type_ids = bert_enc.get("token_type_ids")
            if bert_token_type_ids is not None:
                bert_token_type_ids = bert_token_type_ids.to(device)

            logits_first, logits_mid = model(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=batch.first_vec.to(device),
                mid_vec=batch.mid_vec.to(device),
            )
            loss, metrics, best_first, best_mid = compute_router_loss(
                logits_first=logits_first,
                logits_mid=logits_mid,
                loss_matrix=batch.loss_matrix.to(device),
                mode=args.mode,
                pseudo_ce_weight=args.pseudo_ce_weight,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_first = logits_first.argmax(dim=-1)
            pred_mid = logits_mid.argmax(dim=-1)
            batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
            score_stats = compute_route_score_stats(batch.loss_matrix.to(device), pred_first, pred_mid, batch.task_ids)

            bs = batch.task_ids.size(0)
            running_loss += loss.item() * bs
            total_samples += bs
            for source in (metrics, batch_stats, score_stats):
                for key, value in source.items():
                    metric_totals[key] = metric_totals.get(key, 0.0) + value * bs

            if step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                avg_loss = running_loss / max(total_samples, 1)
                print(
                    f"[TRAIN] epoch={epoch} step={step} loss={avg_loss:.6f} "
                    f"expected={metric_totals.get('expected_loss', 0.0)/max(total_samples,1):.6f} "
                    f"first_acc={metric_totals.get('first_acc', 0.0)/max(total_samples,1):.4f} "
                    f"mid_acc={metric_totals.get('mid_acc', 0.0)/max(total_samples,1):.4f} "
                    f"pair_acc={metric_totals.get('pair_acc', 0.0)/max(total_samples,1):.4f} "
                    f"lr={lr_now:.10f}"
                )

        avg_train_loss = running_loss / max(total_samples, 1)
        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            mode=args.mode,
            pseudo_ce_weight=args.pseudo_ce_weight,
        )

        print(
            f"[EVAL] epoch={epoch} train_loss={avg_train_loss:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"expected={val_metrics.get('expected_loss', 0.0):.6f} "
            f"first_acc={val_metrics['first_acc']:.4f} mid_acc={val_metrics['mid_acc']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.4f} "
            f"router_score={val_metrics['router_argmax_score']:.2f} "
            f"oracle_score={val_metrics['oracle_best_pair_score']:.2f}"
        )

        current_score = val_metrics["router_argmax_score"]
        improved = current_score > (best_score + args.early_stop_min_delta)
        if improved:
            best_score = current_score
            best_epoch = epoch
            no_improve_epochs = 0
            save_ckpt(
                model=model,
                out_dir=args.out_dir,
                task_names=task_names,
                first_layer_idx=args.first_layer_idx,
                middle_layer_idx=args.middle_layer_idx,
                max_bert_len=args.max_bert_len,
                metrics={"epoch": epoch, "avg_train_loss": avg_train_loss, **val_metrics},
                epoch=epoch,
                mode=args.mode,
                pseudo_ce_weight=args.pseudo_ce_weight,
            )
        else:
            no_improve_epochs += 1
            print(
                f"[EARLY_STOP] no improvement for {no_improve_epochs} epoch(s). "
                f"best_score={best_score:.4f} at epoch={best_epoch}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(f"[EARLY_STOP] stop training because patience={args.early_stop_patience} is reached.")
                break

    print(f"[DONE] best_score={best_score:.4f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()
