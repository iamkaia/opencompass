import argparse
import json
import math
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from opencompass.models.router_moe_components import BertExternalEncoder, CompactRouterFeatureEncoder


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

#####它會讀 feature_root/<split>/manifest.json，再把 chunk 檔載進來。重要的是它在 66-81 行 (line 66) 做了兩件事：
#####如果你只想訓練部分 task，它會先把 loss_matrix slice 成較小的子矩陣
####再從 slice 後的 loss_matrix 重新算一次 pair_label / first_label / mid_label
####所以 cache 可以先建大，再在 trainer 端選 task 子集
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

        self.items = []
        for fn in manifest["files"]:
            payload = torch.load(os.path.join(split_dir, fn), map_location="cpu")
            for item in payload["items"]:
                task_name = str(item["task"])
                if task_name not in resolved_task_names:
                    continue

                loss_matrix = item["loss_matrix"]
                sliced_loss_matrix = loss_matrix.index_select(0, torch.tensor(selected_indices)).index_select(
                    1, torch.tensor(selected_indices)
                )
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
    pair_labels: torch.Tensor
    first_labels: torch.Tensor
    mid_labels: torch.Tensor


class Collator:
    def __call__(self, batch: List[Dict]) -> Batch:
        return Batch(
            texts=[x["text"] for x in batch],
            task_ids=torch.tensor([x["task_id"] for x in batch], dtype=torch.long),
            first_vec=torch.stack([x["first_vec"] for x in batch], dim=0).to(torch.float32),
            mid_vec=torch.stack([x["mid_vec"] for x in batch], dim=0).to(torch.float32),
            loss_matrix=torch.stack([x["loss_matrix"] for x in batch], dim=0).to(torch.float32),
            pair_labels=torch.tensor([x["pair_label"] for x in batch], dtype=torch.long),
            first_labels=torch.tensor([x["first_label"] for x in batch], dtype=torch.long),
            mid_labels=torch.tensor([x["mid_label"] for x in batch], dtype=torch.long),
        )

#### router本人, 沒有base llm
class InternalTwoRouterCachedJointModel(nn.Module):
    def __init__(self, bert_init: str, llama_hidden_size: int, router_dim: int, num_pairs: int):
        super().__init__()
        self.bert = BertExternalEncoder(bert_init)
        bert_hidden_size = self.bert.encoder.config.hidden_size
        self.router_first = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.router_mid = CompactRouterFeatureEncoder(llama_hidden_size, bert_hidden_size, router_dim)
        self.pair_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 4),
            nn.Linear(router_dim * 4, router_dim * 2),
            nn.GELU(),
            nn.Linear(router_dim * 2, num_pairs),
        )

    def forward(self, bert_input_ids, bert_attention_mask, bert_token_type_ids, first_vec, mid_vec):
        bert_prev, bert_last = self.bert(
            input_ids=bert_input_ids,
            attention_mask=bert_attention_mask,
            token_type_ids=bert_token_type_ids,
        )
        first_feat = self.router_first(first_vec, bert_prev, bert_last, bert_attention_mask)
        mid_feat = self.router_mid(mid_vec, bert_prev, bert_last, bert_attention_mask)
        pair_logits = self.pair_classifier(torch.cat([first_feat, mid_feat], dim=-1))
        return pair_logits


def set_trainable(model, freeze_bert: bool = False):
    for p in model.parameters():
        p.requires_grad = False
    if not freeze_bert:
        for p in model.bert.parameters():
            p.requires_grad = True
    for p in model.router_first.parameters():
        p.requires_grad = True
    for p in model.router_mid.parameters():
        p.requires_grad = True
    for p in model.pair_classifier.parameters():
        p.requires_grad = True


def compute_pair_losses(
    pair_logits: torch.Tensor,
    loss_matrix: torch.Tensor,
    joint_loss: str,
    pseudo_ce_weight: float,
    margin: float,
):
    ###flat_best = argmin(loss_matrix) 找 oracle 最佳 pair
    ###expected_loss = sum P(pair) * oracle_cost(pair)
    ###ce_pair = CE(pair_logits, flat_best)
    joint_loss = str(joint_loss)
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    flat_best = flat_loss.argmin(dim=-1)
    num_tasks = loss_matrix.size(2)
    best_first = flat_best // num_tasks
    best_mid = flat_best % num_tasks
    pair_prob = torch.softmax(pair_logits, dim=-1)
    expected_loss = (pair_prob * flat_loss).sum(dim=-1).mean()

    sorted_loss, _ = flat_loss.sort(dim=-1)
    if flat_loss.size(1) > 1:
        margin_mask = (sorted_loss[:, 1] - sorted_loss[:, 0]) >= float(margin)
    else:
        margin_mask = torch.ones_like(flat_best, dtype=torch.bool)
    ce_pair_all = nn.functional.cross_entropy(pair_logits, flat_best, reduction="none")
    ce_pair = ce_pair_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=pair_logits.device)

    ###ce_pair
    ###expected_loss
    ###ce_pair_plus_expected
    if joint_loss == "ce_pair":
        total_loss = ce_pair
    elif joint_loss == "expected_loss":
        total_loss = expected_loss
    elif joint_loss == "ce_pair_plus_expected":
        total_loss = ce_pair
        if pseudo_ce_weight > 0:
            total_loss = total_loss + float(pseudo_ce_weight) * expected_loss
    else:
        raise ValueError(f"Unknown joint_loss: {joint_loss}")
    metrics = {
        "expected_loss": float(expected_loss.detach().item()),
        "main_pair_ce": float(ce_pair.detach().item()),
        "pseudo_ce_pair": float(ce_pair.detach().item()),
        "best_pair_loss": float(sorted_loss[:, 0].mean().item()),
        "margin_active_ratio": float(margin_mask.float().mean().item()),
    }
    return total_loss, metrics, best_first, best_mid, flat_best


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
        "oracle_self_pair_acc": float(((best_first == task_ids) & (best_mid == task_ids)).float().mean().item()),
    }


def compute_route_score_stats(loss_matrix: torch.Tensor, pred_pair: torch.Tensor, task_ids: torch.Tensor):
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    best_pair = flat_loss.argmin(dim=-1)
    batch_idx = torch.arange(loss_matrix.size(0), device=loss_matrix.device)
    task_ids = task_ids.to(loss_matrix.device)

    pred_cost = flat_loss[batch_idx, pred_pair]
    oracle_cost = flat_loss[batch_idx, best_pair]
    self_cost = loss_matrix[batch_idx, task_ids, task_ids]

    return {
        "router_argmax_score": float(score_from_cost(pred_cost).mean().item()),
        "oracle_best_pair_score": float(score_from_cost(oracle_cost).mean().item()),
        "fixed_self_score": float(score_from_cost(self_cost).mean().item()),
        "router_argmax_cost": float(pred_cost.mean().item()),
        "oracle_best_pair_cost": float(oracle_cost.mean().item()),
        "fixed_self_cost": float(self_cost.mean().item()),
    }


def build_routing_summary(
    pred_first_all: Sequence[int],
    pred_mid_all: Sequence[int],
    best_first_all: Sequence[int],
    best_mid_all: Sequence[int],
    task_ids_all: Sequence[int],
    task_names: Sequence[str],
):
    if not pred_first_all:
        return {"num_samples": 0, "top_pred_pairs": [], "per_task": []}

    num_samples = len(pred_first_all)
    stats = compute_routing_accuracy_stats(
        torch.tensor(pred_first_all, dtype=torch.long),
        torch.tensor(pred_mid_all, dtype=torch.long),
        torch.tensor(best_first_all, dtype=torch.long),
        torch.tensor(best_mid_all, dtype=torch.long),
        torch.tensor(task_ids_all, dtype=torch.long),
    )
    pred_pair_counter = Counter()
    gold_pair_counter = Counter()
    task_bucket = defaultdict(lambda: {"pred_first": Counter(), "pred_mid": Counter(), "pred_pair": Counter()})
    for pf, pm, bf, bm, tid in zip(pred_first_all, pred_mid_all, best_first_all, best_mid_all, task_ids_all):
        pred_pair = f"{task_names[pf]}->{task_names[pm]}"
        gold_pair = f"{task_names[bf]}->{task_names[bm]}"
        pred_pair_counter[pred_pair] += 1
        gold_pair_counter[gold_pair] += 1
        task_bucket[int(tid)]["pred_first"][task_names[pf]] += 1
        task_bucket[int(tid)]["pred_mid"][task_names[pm]] += 1
        task_bucket[int(tid)]["pred_pair"][pred_pair] += 1

    def counter_rows(counter: Counter, denom: int, top_k: int = 3):
        return [
            {"name": name, "count": int(count), "rate": float(count / max(denom, 1))}
            for name, count in counter.most_common(top_k)
        ]

    per_task = []
    for task_id, task_name in enumerate(task_names):
        n = sum(1 for x in task_ids_all if x == task_id)
        if n == 0:
            continue
        pred_self_first = sum(1 for pf, tid in zip(pred_first_all, task_ids_all) if tid == task_id and pf == task_id)
        pred_self_mid = sum(1 for pm, tid in zip(pred_mid_all, task_ids_all) if tid == task_id and pm == task_id)
        gold_self_pair = sum(
            1 for bf, bm, tid in zip(best_first_all, best_mid_all, task_ids_all) if tid == task_id and bf == task_id and bm == task_id
        )
        per_task.append(
            {
                "task": task_name,
                "count": int(n),
                "pred_self_first_rate": float(pred_self_first / n),
                "pred_self_mid_rate": float(pred_self_mid / n),
                "gold_self_pair_rate": float(gold_self_pair / n),
                "top_pred_first": counter_rows(task_bucket[task_id]["pred_first"], n),
                "top_pred_mid": counter_rows(task_bucket[task_id]["pred_mid"], n),
                "top_pred_pairs": counter_rows(task_bucket[task_id]["pred_pair"], n),
            }
        )

    return {
        "num_samples": int(num_samples),
        "first_acc": float(stats["first_acc"]),
        "mid_acc": float(stats["mid_acc"]),
        "pair_acc": float(stats["pair_acc"]),
        "self_first_acc": float(stats["self_first_acc"]),
        "self_mid_acc": float(stats["self_mid_acc"]),
        "self_pair_acc": float(stats["self_pair_acc"]),
        "oracle_self_pair_acc": float(stats["oracle_self_pair_acc"]),
        "top_pred_pairs": counter_rows(pred_pair_counter, num_samples, top_k=5),
        "top_gold_pairs": counter_rows(gold_pair_counter, num_samples, top_k=5),
        "per_task": per_task,
    }


def print_routing_summary(tag: str, summary: Dict):
    if int(summary.get("num_samples", 0)) <= 0:
        print(f"[ROUTE][{tag}] no samples")
        return
    top_pairs = ", ".join(f"{row['name']}:{row['rate']:.2%}" for row in summary.get("top_pred_pairs", [])[:3])
    print(
        f"[ROUTE][{tag}] pair_acc={summary.get('pair_acc', 0.0):.4f} "
        f"self_pair={summary.get('self_pair_acc', 0.0):.4f} "
        f"oracle_self_pair={summary.get('oracle_self_pair_acc', 0.0):.4f} "
        f"top_pred_pairs={top_pairs}"
    )
    for row in summary.get("per_task", []):
        top_first = row["top_pred_first"][0]["name"] if row["top_pred_first"] else "-"
        top_mid = row["top_pred_mid"][0]["name"] if row["top_pred_mid"] else "-"
        top_pair = row["top_pred_pairs"][0]["name"] if row["top_pred_pairs"] else "-"
        print(
            f"[ROUTE][{tag}][{row['task']}] n={row['count']} "
            f"self_first={row['pred_self_first_rate']:.2%} "
            f"self_mid={row['pred_self_mid_rate']:.2%} "
            f"oracle_self_pair={row['gold_self_pair_rate']:.2%} "
            f"top_first={top_first} top_mid={top_mid} top_pair={top_pair}"
        )


def flatten_routing_summary(summary: Dict, prefix: str) -> Dict[str, float]:
    payload: Dict[str, float] = {}
    for key in [
        "first_acc",
        "mid_acc",
        "pair_acc",
        "self_first_acc",
        "self_mid_acc",
        "self_pair_acc",
        "oracle_self_pair_acc",
    ]:
        if key in summary:
            payload[f"{prefix}/{key}"] = float(summary[key])

    top_pred_pairs = summary.get("top_pred_pairs", [])
    if top_pred_pairs:
        payload[f"{prefix}/top_pred_pair_rate"] = float(top_pred_pairs[0]["rate"])

    for row in summary.get("per_task", []):
        task = str(row["task"])
        payload[f"{prefix}_task/{task}_self_first_rate"] = float(row["pred_self_first_rate"])
        payload[f"{prefix}_task/{task}_self_mid_rate"] = float(row["pred_self_mid_rate"])
        payload[f"{prefix}_task/{task}_oracle_self_pair_rate"] = float(row["gold_self_pair_rate"])
        if row.get("top_pred_first"):
            payload[f"{prefix}_task/{task}_top_first_rate"] = float(row["top_pred_first"][0]["rate"])
        if row.get("top_pred_mid"):
            payload[f"{prefix}_task/{task}_top_mid_rate"] = float(row["top_pred_mid"][0]["rate"])
        if row.get("top_pred_pairs"):
            payload[f"{prefix}_task/{task}_top_pair_rate"] = float(row["top_pred_pairs"][0]["rate"])
    return payload


@torch.no_grad()
def evaluate(
    model,
    loader,
    bert_tokenizer,
    device,
    max_bert_len,
    task_names,
    joint_loss,
    pseudo_ce_weight,
    pseudo_ce_margin,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    metric_totals: Dict[str, float] = {}
    pred_first_all: List[int] = []
    pred_mid_all: List[int] = []
    best_first_all: List[int] = []
    best_mid_all: List[int] = []
    task_ids_all: List[int] = []

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

        pair_logits = model(
            bert_input_ids=bert_input_ids,
            bert_attention_mask=bert_attention_mask,
            bert_token_type_ids=bert_token_type_ids,
            first_vec=batch.first_vec.to(device),
            mid_vec=batch.mid_vec.to(device),
        )
        loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
            pair_logits=pair_logits,
            loss_matrix=batch.loss_matrix.to(device),
            joint_loss=joint_loss,
            pseudo_ce_weight=pseudo_ce_weight,
            margin=pseudo_ce_margin,
        )
        pred_pair = pair_logits.argmax(dim=-1)
        num_tasks = batch.loss_matrix.size(2)
        pred_first = pred_pair // num_tasks
        pred_mid = pred_pair % num_tasks
        batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
        score_stats = compute_route_score_stats(batch.loss_matrix.to(device), pred_pair, batch.task_ids)

        bs = batch.task_ids.size(0)
        total_loss += loss.item() * bs
        total_samples += bs
        for key, value in metrics.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        for key, value in batch_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        for key, value in score_stats.items():
            metric_totals[key] = metric_totals.get(key, 0.0) + value * bs
        pred_first_all.extend(pred_first.cpu().tolist())
        pred_mid_all.extend(pred_mid.cpu().tolist())
        best_first_all.extend(best_first.cpu().tolist())
        best_mid_all.extend(best_mid.cpu().tolist())
        task_ids_all.extend(batch.task_ids.cpu().tolist())

    denom = max(total_samples, 1)
    result = {"loss": total_loss / denom}
    for key, value in metric_totals.items():
        result[key] = value / denom
    result["routing_summary"] = build_routing_summary(
        pred_first_all, pred_mid_all, best_first_all, best_mid_all, task_ids_all, task_names
    )
    return result


def save_ckpt(model, out_dir, task_names, max_bert_len, metrics, epoch, joint_loss, pseudo_ce_weight):
    os.makedirs(out_dir, exist_ok=True)
    model.bert.encoder.save_pretrained(os.path.join(out_dir, "encoder"))
    torch.save(
        {
            "bert_encoder": model.bert.state_dict(),
            "router_first": model.router_first.state_dict(),
            "router_mid": model.router_mid.state_dict(),
            "pair_classifier": model.pair_classifier.state_dict(),
        },
        os.path.join(out_dir, "router_heads.pt"),
    )
    save_json(
        {
            "task_names": list(task_names),
            "expert_names": list(task_names),
            "num_pairs": len(task_names) * len(task_names),
            "router_max_len": max_bert_len,
            "router_feature_type": "cached_prompt_vectors_with_loss_matrix",
            "supervision_type": "cached_pair_ce_main",
            "joint_loss": str(joint_loss),
            "pseudo_ce_weight": float(pseudo_ce_weight),
            "best_epoch": epoch,
            "best_router_argmax_score": metrics.get("router_argmax_score"),
            "best_val_loss": metrics.get("loss"),
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
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--freeze_bert", action="store_true")
    parser.add_argument(
        "--joint_loss",
        type=str,
        default="ce_pair_plus_expected",
        choices=["ce_pair", "expected_loss", "ce_pair_plus_expected"],
    )
    parser.add_argument("--pseudo_ce_weight", type=float, default=0.0)
    parser.add_argument("--pseudo_ce_margin", type=float, default=0.0)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="router_answer_supervision")
    parser.add_argument("--wandb_name", type=str, default=None)
    parser.add_argument("--wandb_entity", type=str, default=None)
    parser.add_argument("--wandb_group", type=str, default=None)
    parser.add_argument("--wandb_tags", type=str, default=None)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] device={device}")
    wandb_run = None
    if args.wandb:
        try:
            import wandb
        except Exception as e:
            raise ImportError("--wandb was set but wandb is not installed") from e
        wandb_tags = [part.strip() for part in str(args.wandb_tags).split(",") if part.strip()] if args.wandb_tags else None
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_name,
            entity=args.wandb_entity,
            group=args.wandb_group,
            tags=wandb_tags,
            config=vars(args),
            dir=args.out_dir,
        )
        print(f"[INFO] wandb enabled project={args.wandb_project}")

    probe_train_ds = CachedLossMatrixDataset(args.feature_root, "train")
    probe_val_ds = CachedLossMatrixDataset(args.feature_root, "validation")
    task_names = parse_task_names(args.task_names, fallback=probe_train_ds.task_names or probe_val_ds.task_names)
    train_ds = CachedLossMatrixDataset(args.feature_root, "train", selected_task_names=task_names)
    val_ds = CachedLossMatrixDataset(args.feature_root, "validation", selected_task_names=task_names)
    print(f"[INFO] task_names={task_names}")
    print(f"[INFO] num_pairs={len(task_names) * len(task_names)}")
    print(
        f"[INFO] joint_loss={args.joint_loss} "
        f"pseudo_ce_weight={args.pseudo_ce_weight} pseudo_ce_margin={args.pseudo_ce_margin}"
    )
    train_cfg = vars(args).copy()
    train_cfg["resolved_task_names"] = task_names
    save_json(train_cfg, os.path.join(args.out_dir, "train_config.json"))

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=Collator(),
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
    llama_hidden_size = int(train_ds[0]["first_vec"].numel())
    model = InternalTwoRouterCachedJointModel(
        bert_init=args.bert_init,
        llama_hidden_size=llama_hidden_size,
        router_dim=args.router_dim,
        num_pairs=len(task_names) * len(task_names),
    ).to(device)

    if args.load_from is not None:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        if "router_first" in state:
            model.router_first.load_state_dict(state["router_first"], strict=False)
        if "router_mid" in state:
            model.router_mid.load_state_dict(state["router_mid"], strict=False)
        if "pair_classifier" in state:
            model.pair_classifier.load_state_dict(state["pair_classifier"], strict=False)
        if "bert_encoder" in state:
            model.bert.load_state_dict(state["bert_encoder"], strict=False)
        print(f"[LOAD] loaded from {args.load_from}")

    set_trainable(model, freeze_bert=args.freeze_bert)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = math.ceil(len(train_ds) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_router_score = float("-inf")
    best_epoch = -1
    no_improve_epochs = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_count = 0
        running_metric_totals: Dict[str, float] = {}
        train_pred_first_all: List[int] = []
        train_pred_mid_all: List[int] = []
        train_best_first_all: List[int] = []
        train_best_mid_all: List[int] = []
        train_task_ids_all: List[int] = []

        for step, batch in enumerate(train_loader, start=1):
            global_step += 1
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

            pair_logits = model(
                bert_input_ids=bert_input_ids,
                bert_attention_mask=bert_attention_mask,
                bert_token_type_ids=bert_token_type_ids,
                first_vec=batch.first_vec.to(device),
                mid_vec=batch.mid_vec.to(device),
            )
            loss, metrics, best_first, best_mid, flat_best = compute_pair_losses(
                pair_logits=pair_logits,
                loss_matrix=batch.loss_matrix.to(device),
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
                margin=args.pseudo_ce_margin,
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            pred_pair = pair_logits.argmax(dim=-1)
            num_tasks = batch.loss_matrix.size(2)
            pred_first = pred_pair // num_tasks
            pred_mid = pred_pair % num_tasks
            batch_stats = compute_routing_accuracy_stats(pred_first, pred_mid, best_first, best_mid, batch.task_ids)
            score_stats = compute_route_score_stats(batch.loss_matrix.to(device), pred_pair, batch.task_ids)

            bs = batch.task_ids.size(0)
            running_loss += loss.item() * bs
            running_count += bs
            for key, value in metrics.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            for key, value in batch_stats.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            for key, value in score_stats.items():
                running_metric_totals[key] = running_metric_totals.get(key, 0.0) + float(value) * bs
            train_pred_first_all.extend(pred_first.detach().cpu().tolist())
            train_pred_mid_all.extend(pred_mid.detach().cpu().tolist())
            train_best_first_all.extend(best_first.detach().cpu().tolist())
            train_best_mid_all.extend(best_mid.detach().cpu().tolist())
            train_task_ids_all.extend(batch.task_ids.cpu().tolist())

            if step % args.log_every == 0 or step == len(train_loader):
                avg_loss = running_loss / max(running_count, 1)
                avg_metrics = {
                    key: value / max(running_count, 1) for key, value in running_metric_totals.items()
                }
                print(
                    f"[TRAIN] epoch={epoch} step={step}/{len(train_loader)} "
                    f"loss={avg_loss:.4f} pair_ce={avg_metrics['main_pair_ce']:.4f} expected={avg_metrics['expected_loss']:.4f} "
                    f"best_pair={avg_metrics['best_pair_loss']:.4f} "
                    f"router_score={avg_metrics['router_argmax_score']:.2f} "
                    f"self_score={avg_metrics['fixed_self_score']:.2f} "
                    f"first_acc={avg_metrics['first_acc']:.4f} mid_acc={avg_metrics['mid_acc']:.4f} "
                    f"pair_acc={avg_metrics['pair_acc']:.4f} "
                    f"self_first={avg_metrics['self_first_acc']:.4f} self_mid={avg_metrics['self_mid_acc']:.4f} "
                    f"oracle_self_pair={avg_metrics['oracle_self_pair_acc']:.4f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/epoch": epoch,
                            "train/step": global_step,
                            "train/loss": avg_loss,
                            "train/main_pair_ce": avg_metrics["main_pair_ce"],
                            "train/expected_loss": avg_metrics["expected_loss"],
                            "train/best_pair_loss": avg_metrics["best_pair_loss"],
                            "train/pseudo_ce_pair": avg_metrics["pseudo_ce_pair"],
                            "train/margin_active_ratio": avg_metrics["margin_active_ratio"],
                            "train/router_argmax_score": avg_metrics["router_argmax_score"],
                            "train/fixed_self_score": avg_metrics["fixed_self_score"],
                            "train/oracle_best_pair_score": avg_metrics["oracle_best_pair_score"],
                            "train/first_acc": avg_metrics["first_acc"],
                            "train/mid_acc": avg_metrics["mid_acc"],
                            "train/joint_acc": avg_metrics["joint_acc"],
                            "train/pair_acc": avg_metrics["pair_acc"],
                            "train/self_first_acc": avg_metrics["self_first_acc"],
                            "train/self_mid_acc": avg_metrics["self_mid_acc"],
                            "train/self_joint_acc": avg_metrics["self_joint_acc"],
                            "train/self_pair_acc": avg_metrics["self_pair_acc"],
                            "train/oracle_self_pair_acc": avg_metrics["oracle_self_pair_acc"],
                            "train/lr": scheduler.get_last_lr()[0],
                        },
                        step=global_step,
                    )

        train_summary = build_routing_summary(
            train_pred_first_all, train_pred_mid_all, train_best_first_all, train_best_mid_all, train_task_ids_all, task_names
        )
        print_routing_summary(f"TRAIN-EPOCH{epoch}", train_summary)
        save_json(train_summary, os.path.join(args.out_dir, f"routing_summary_train_epoch{epoch}.json"))
        if wandb_run is not None:
            wandb_payload = {
                "train_epoch/epoch": epoch,
                "train_epoch/loss": running_loss / max(running_count, 1),
            }
            wandb_payload.update(flatten_routing_summary(train_summary, prefix="train_epoch_route"))
            wandb_run.log(wandb_payload, step=global_step)

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            bert_tokenizer=bert_tokenizer,
            device=device,
            max_bert_len=args.max_bert_len,
            task_names=task_names,
            joint_loss=args.joint_loss,
            pseudo_ce_weight=args.pseudo_ce_weight,
            pseudo_ce_margin=args.pseudo_ce_margin,
        )
        print(
            f"[VAL] epoch={epoch} loss={val_metrics['loss']:.4f} "
            f"router_score={val_metrics['router_argmax_score']:.2f} "
            f"self_score={val_metrics['fixed_self_score']:.2f} "
            f"oracle_score={val_metrics['oracle_best_pair_score']:.2f} "
            f"first_acc={val_metrics['first_acc']:.4f} "
            f"mid_acc={val_metrics['mid_acc']:.4f} "
            f"pair_acc={val_metrics['pair_acc']:.4f} "
            f"self_first={val_metrics['self_first_acc']:.4f} "
            f"self_mid={val_metrics['self_mid_acc']:.4f} "
            f"oracle_self_pair={val_metrics['oracle_self_pair_acc']:.4f}"
        )
        print_routing_summary(f"VAL-EPOCH{epoch}", val_metrics["routing_summary"])
        save_json(val_metrics["routing_summary"], os.path.join(args.out_dir, f"routing_summary_val_epoch{epoch}.json"))
        if wandb_run is not None:
            wandb_payload = {
                "val/epoch": epoch,
                "val/loss": val_metrics["loss"],
                "val/main_pair_ce": val_metrics.get("main_pair_ce", 0.0),
                "val/expected_loss": val_metrics.get("expected_loss", 0.0),
                "val/best_pair_loss": val_metrics.get("best_pair_loss", 0.0),
                "val/pseudo_ce_pair": val_metrics.get("pseudo_ce_pair", 0.0),
                "val/router_argmax_score": val_metrics["router_argmax_score"],
                "val/fixed_self_score": val_metrics["fixed_self_score"],
                "val/oracle_best_pair_score": val_metrics["oracle_best_pair_score"],
                "val/first_acc": val_metrics["first_acc"],
                "val/mid_acc": val_metrics["mid_acc"],
                "val/joint_acc": val_metrics["joint_acc"],
                "val/pair_acc": val_metrics["pair_acc"],
                "val/self_first_acc": val_metrics["self_first_acc"],
                "val/self_mid_acc": val_metrics["self_mid_acc"],
                "val/self_joint_acc": val_metrics["self_joint_acc"],
                "val/self_pair_acc": val_metrics["self_pair_acc"],
                "val/oracle_self_pair_acc": val_metrics["oracle_self_pair_acc"],
                "val/best_router_argmax_score": max(best_router_score, val_metrics["router_argmax_score"]),
            }
            wandb_payload.update(flatten_routing_summary(val_metrics["routing_summary"], prefix="val_route"))
            wandb_run.log(wandb_payload, step=global_step)

        improved = val_metrics["router_argmax_score"] > (best_router_score + args.early_stop_min_delta)
        if improved:
            best_router_score = val_metrics["router_argmax_score"]
            best_epoch = epoch
            no_improve_epochs = 0
            save_ckpt(
                model=model,
                out_dir=args.out_dir,
                task_names=task_names,
                max_bert_len=args.max_bert_len,
                metrics={"epoch": epoch, **val_metrics},
                epoch=epoch,
                joint_loss=args.joint_loss,
                pseudo_ce_weight=args.pseudo_ce_weight,
            )
            print(f"[SAVE] best checkpoint updated at epoch={epoch} router_score={best_router_score:.2f}")
        else:
            no_improve_epochs += 1
            print(
                f"[EARLY_STOP] no improvement for {no_improve_epochs} epoch(s). "
                f"best_router_score={best_router_score:.2f} at epoch={best_epoch}"
            )
            if no_improve_epochs >= args.early_stop_patience:
                print(f"[EARLY_STOP] stop training because patience={args.early_stop_patience} is reached.")
                break

    if wandb_run is not None:
        wandb_run.summary["best_router_argmax_score"] = best_router_score
        wandb_run.summary["best_epoch"] = best_epoch
        wandb_run.finish()
    print(f"[DONE] best_router_argmax_score={best_router_score:.2f} best_epoch={best_epoch}")


if __name__ == "__main__":
    main()