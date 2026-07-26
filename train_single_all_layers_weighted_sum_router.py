import argparse
import json
import math
import os
import random
from typing import List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from opencompass.models.router_moe_components import BertExternalEncoder, CompactRouterFeatureEncoder


def save_json(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def csv(raw: str) -> List[str]:
    return [part.strip() for part in str(raw).split(",") if part.strip()]


class SingleAllLayersCache(Dataset):
    def __init__(self, roots: Sequence[str], split: str, sample_tasks: Sequence[str], expert_names: Sequence[str]):
        self.items = []
        self.expert_names = list(expert_names)
        self.sample_tasks = list(sample_tasks)
        self.feature_contract = None
        for root in roots:
            manifest_path = os.path.join(root, split, "manifest.json")
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            if manifest.get("route_space") != "single_all_layers":
                raise ValueError(f"Expected route_space=single_all_layers in {manifest_path}")
            source_experts = list(manifest["expert_names"])
            missing = [name for name in self.expert_names if name not in source_experts]
            if missing:
                raise ValueError(f"Missing experts {missing} in {manifest_path}")
            expert_ids = torch.tensor([source_experts.index(name) for name in self.expert_names])
            contract = {
                key: manifest[key]
                for key in (
                    "first_layer_idx",
                    "middle_layer_idx",
                    "router_pooling",
                    "router_pooling_last_k",
                    "llama_hidden_size",
                )
            }
            if self.feature_contract is None:
                self.feature_contract = contract
            elif self.feature_contract != contract:
                raise ValueError(f"Feature contract mismatch in {manifest_path}")
            for filename in manifest["files"]:
                payload = torch.load(os.path.join(root, split, filename), map_location="cpu")
                for item in payload["items"]:
                    if str(item["task"]) not in self.sample_tasks:
                        continue
                    copied = dict(item)
                    copied["loss_vector"] = item["loss_vector"].index_select(0, expert_ids).float()
                    copied["correct_vector"] = item["correct_vector"].index_select(0, expert_ids).bool()
                    copied["item_id"] = f"{os.path.basename(root)}:{split}:{len(self.items):08d}"
                    self.items.append(copied)
        if not self.items:
            raise ValueError(f"No {split} samples for tasks={self.sample_tasks} in roots={list(roots)}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def collate(items):
    return {
        "prompt_texts": [str(item.get("prompt_text", item["text"])) for item in items],
        "tasks": [str(item["task"]) for item in items],
        "item_ids": [str(item["item_id"]) for item in items],
        "first_vec": torch.stack([item["first_vec"] for item in items]).float(),
        "loss_vector": torch.stack([item["loss_vector"] for item in items]).float(),
        "correct_vector": torch.stack([item["correct_vector"] for item in items]).bool(),
    }


class SingleAllLayersRouter(nn.Module):
    def __init__(self, bert_init: str, llm_hidden: int, router_dim: int, num_experts: int):
        super().__init__()
        self.bert = BertExternalEncoder(bert_init)
        bert_hidden = self.bert.encoder.config.hidden_size
        self.router_first = CompactRouterFeatureEncoder(llm_hidden, bert_hidden, router_dim)
        self.first_classifier = nn.Sequential(
            nn.LayerNorm(router_dim * 2),
            nn.Linear(router_dim * 2, router_dim),
            nn.GELU(),
            nn.Linear(router_dim, num_experts),
        )

    def forward(self, input_ids, attention_mask, token_type_ids, first_vec):
        bert_prev, bert_last = self.bert(input_ids, attention_mask, token_type_ids)
        feature = self.router_first(first_vec, bert_prev, bert_last, attention_mask)
        return self.first_classifier(feature)


def build_target(loss_vector, correct_vector, temperature, normalization, empty_fallback):
    loss = loss_vector.float()
    if normalization == "sample_minmax":
        low = loss.min(dim=-1, keepdim=True).values
        high = loss.max(dim=-1, keepdim=True).values
        loss = (loss - low) / (high - low).clamp_min(1e-12)
    available = correct_vector.any(dim=-1)
    logits = -loss / max(float(temperature), 1e-6)
    masked_logits = logits.masked_fill(~correct_vector, -1e9)
    target = torch.softmax(masked_logits, dim=-1)
    if empty_fallback == "uniform":
        empty = torch.full_like(target, 1.0 / target.size(-1))
    elif empty_fallback == "loss_softmax":
        empty = torch.softmax(logits, dim=-1)
    elif empty_fallback == "zero":
        empty = torch.zeros_like(target)
    else:
        raise ValueError(empty_fallback)
    return torch.where(available.unsqueeze(-1), target, empty), available


def forward_batch(model, tokenizer, batch, device, args):
    encoded = tokenizer(
        batch["prompt_texts"],
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.max_bert_len,
    )
    token_type_ids = encoded.get("token_type_ids")
    if token_type_ids is not None:
        token_type_ids = token_type_ids.to(device)
    logits = model(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
        token_type_ids,
        batch["first_vec"].to(device),
    )
    target, available = build_target(
        batch["loss_vector"].to(device),
        batch["correct_vector"].to(device),
        args.target_temperature,
        args.loss_normalization,
        args.target_empty_fallback,
    )
    weights = torch.softmax(logits, dim=-1)
    target_mass = target.sum(dim=-1) > 0
    if bool(target_mass.any().item()):
        weight_mse = nn.functional.mse_loss(weights[target_mass], target[target_mass])
        expert_ce = -(
            target[target_mass] * nn.functional.log_softmax(logits[target_mass], dim=-1)
        ).sum(dim=-1).mean()
        loss = (
            float(args.weighted_sum_mse_weight) * weight_mse
            + float(args.expert_ce_weight) * expert_ce
        )
    else:
        loss = logits.sum() * 0.0
        weight_mse = loss
        expert_ce = loss
    return logits, weights, target, available, target_mass, weight_mse, expert_ce, loss


@torch.no_grad()
def evaluate(model, loader, tokenizer, device, args, records=False):
    model.eval()
    total_loss = 0.0
    total_mse = 0.0
    total_ce = 0.0
    total = 0
    correct = 0
    entropy_sum = 0.0
    output_records = []
    for batch in loader:
        logits, weights, target, available, target_mass, weight_mse, expert_ce, loss = forward_batch(
            model, tokenizer, batch, device, args
        )
        n = int(target_mass.sum().item())
        total_loss += float(loss.item()) * n
        total_mse += float(weight_mse.item()) * n
        total_ce += float(expert_ce.item()) * n
        total += n
        pred = weights.argmax(dim=-1)
        gold = target.argmax(dim=-1)
        correct += int(((pred == gold) & target_mass).sum().item())
        entropy = -(weights.clamp_min(1e-12) * weights.clamp_min(1e-12).log()).sum(dim=-1)
        entropy_sum += float(entropy[target_mass].sum().item())
        if records:
            for idx in range(len(batch["tasks"])):
                output_records.append(
                    {
                        "item_id": batch["item_ids"][idx],
                        "task": batch["tasks"][idx],
                        "pred_expert": args.expert_list[int(pred[idx].item())],
                        "target_expert": args.expert_list[int(gold[idx].item())],
                        "weights": weights[idx].cpu().tolist(),
                        "target_weights": target[idx].cpu().tolist(),
                        "correct_target_available": bool(available[idx].item()),
                    }
                )
    denom = max(total, 1)
    return {
        "weighted_sum_mse": total_mse / denom,
        "expert_ce": total_ce / denom,
        "loss": total_loss / denom,
        "argmax_target_acc": correct / denom,
        "weight_entropy": entropy_sum / denom,
        "target_count": total,
        "records": output_records,
    }


def save_checkpoint(model, args, dataset, llm_hidden, epoch, metrics):
    model.bert.encoder.save_pretrained(os.path.join(args.out_dir, "encoder"))
    torch.save(
        {
            "bert_encoder": model.bert.state_dict(),
            "router_first": model.router_first.state_dict(),
            "first_classifier": model.first_classifier.state_dict(),
        },
        os.path.join(args.out_dir, "router_heads.pt"),
    )
    contract = dataset.feature_contract
    save_json(
        {
            "router_architecture": "single_all_layers",
            "share_first_weights_all_layers": True,
            "task_names": args.expert_list,
            "expert_names": args.expert_list,
            "train_task_names": args.sample_task_list,
            "router_max_len": args.max_bert_len,
            "router_feature_type": "cached_single_all_layers_first_vector",
            "first_layer_idx": int(contract["first_layer_idx"]),
            "middle_layer_idx": int(contract["middle_layer_idx"]),
            "router_pooling": str(contract["router_pooling"]),
            "router_pooling_last_k": int(contract["router_pooling_last_k"]),
            "llama_hidden_size": int(llm_hidden),
            "router_dim": args.router_dim,
            "training_loss": (
                "expert_ce_plus_weighted_sum_mse"
                if float(args.expert_ce_weight) > 0
                else "weighted_sum_mse"
            ),
            "expert_ce_weight": float(args.expert_ce_weight),
            "weighted_sum_mse_weight": float(args.weighted_sum_mse_weight),
            "target_temperature": args.target_temperature,
            "loss_normalization": args.loss_normalization,
            "target_empty_fallback": args.target_empty_fallback,
            "best_epoch": epoch,
            "best_metric": args.best_metric,
            "best_metric_value": metrics[args.best_metric],
        },
        os.path.join(args.out_dir, "router_config.json"),
    )
    clean_metrics = {key: value for key, value in metrics.items() if key != "records"}
    save_json({"best_epoch": epoch, "metrics": clean_metrics}, os.path.join(args.out_dir, "best_metrics.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_roots", required=True)
    parser.add_argument("--bert_init", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--sample_task_names", required=True)
    parser.add_argument("--expert_names", required=True)
    parser.add_argument("--load_from")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--max_bert_len", type=int, default=512)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--target_temperature", type=float, default=0.25)
    parser.add_argument("--loss_normalization", choices=["none", "sample_minmax"], default="none")
    parser.add_argument("--target_empty_fallback", choices=["zero", "uniform", "loss_softmax"], default="uniform")
    parser.add_argument("--expert_ce_weight", type=float, default=0.0)
    parser.add_argument("--weighted_sum_mse_weight", type=float, default=1.0)
    parser.add_argument("--best_metric", choices=["loss", "weighted_sum_mse", "expert_ce"], default="weighted_sum_mse")
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.expert_list = csv(args.expert_names)
    args.sample_task_list = csv(args.sample_task_names)
    roots = csv(args.feature_roots)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=False)
    save_json(vars(args), os.path.join(args.out_dir, "train_config.json"))

    train_ds = SingleAllLayersCache(roots, "train", args.sample_task_list, args.expert_list)
    val_ds = SingleAllLayersCache(roots, "validation", args.sample_task_list, args.expert_list)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    train_eval_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_init)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    llm_hidden = int(train_ds[0]["first_vec"].numel())
    model = SingleAllLayersRouter(args.bert_init, llm_hidden, args.router_dim, len(args.expert_list)).to(device)
    if args.load_from:
        state = torch.load(os.path.join(args.load_from, "router_heads.pt"), map_location="cpu")
        model.bert.load_state_dict(state["bert_encoder"], strict=False)
        model.router_first.load_state_dict(state["router_first"])
        model.first_classifier.load_state_dict(state["first_classifier"])
        print(f"[LOAD] {args.load_from}", flush=True)
    for parameter in model.bert.parameters():
        parameter.requires_grad = False
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    total_steps = math.ceil(len(train_ds) / args.batch_size) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(total_steps * args.warmup_ratio), total_steps
    )
    best = float("inf")
    best_epoch = -1
    stale = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        model.bert.eval()
        for batch in train_loader:
            *_, loss = forward_batch(model, tokenizer, batch, device, args)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
        train_metrics = evaluate(model, train_eval_loader, tokenizer, device, args)
        val_metrics = evaluate(model, val_loader, tokenizer, device, args, records=True)
        print(
            f"[EPOCH {epoch}] train_loss={train_metrics['loss']:.6f} "
            f"train_ce={train_metrics['expert_ce']:.6f} "
            f"train_mse={train_metrics['weighted_sum_mse']:.6f} "
            f"val_loss={val_metrics['loss']:.6f} "
            f"val_ce={val_metrics['expert_ce']:.6f} "
            f"val_mse={val_metrics['weighted_sum_mse']:.6f} "
            f"val_argmax={val_metrics['argmax_target_acc']:.4f} "
            f"val_entropy={val_metrics['weight_entropy']:.4f}",
            flush=True,
        )
        score = val_metrics[args.best_metric]
        if score < best - 1e-4:
            best = score
            best_epoch = epoch
            stale = 0
            save_checkpoint(model, args, train_ds, llm_hidden, epoch, val_metrics)
            save_json(
                {"epoch": epoch, "records": val_metrics["records"]},
                os.path.join(args.out_dir, "route_records_val_best.json"),
            )
        else:
            stale += 1
            if stale >= args.early_stop_patience:
                break
    print(
        f"[DONE] out={args.out_dir} best_epoch={best_epoch} "
        f"best_{args.best_metric}={best:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
