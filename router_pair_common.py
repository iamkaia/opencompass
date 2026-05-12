import math
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn


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


def compute_pair_losses(
    pair_logits: torch.Tensor,
    logits_first: Optional[torch.Tensor] = None,
    logits_mid: Optional[torch.Tensor] = None,
    loss_matrix: Optional[torch.Tensor] = None,
    correct_matrix: Optional[torch.Tensor] = None,
    mode: str = "joint",
    joint_loss: str = "expected_loss",
    pseudo_ce_weight: float = 0.0,
    margin: float = 0.0,
    loss_normalization: str = "sample_minmax",
    correct_soft_ce_temperature: float = 1.0,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor, torch.Tensor, torch.Tensor]:
    if loss_matrix is None:
        raise ValueError("loss_matrix is required")

    mode = str(mode)
    joint_loss = str(joint_loss)
    normalized_loss_matrix = normalize_pair_loss_matrix(
        loss_matrix=loss_matrix,
        method=loss_normalization,
    )
    flat_loss = loss_matrix.view(loss_matrix.size(0), -1)
    normalized_flat_loss = normalized_loss_matrix.view(normalized_loss_matrix.size(0), -1)
    flat_best = flat_loss.argmin(dim=-1)
    num_tasks = loss_matrix.size(2)
    best_first = flat_best // num_tasks
    best_mid = flat_best % num_tasks
    pair_prob = torch.softmax(pair_logits, dim=-1)
    expected_loss = (pair_prob * normalized_flat_loss).sum(dim=-1).mean()
    log_pair_prob = torch.log_softmax(pair_logits, dim=-1)

    correct_soft_ce = torch.tensor(0.0, device=pair_logits.device)
    correct_conf_ce = torch.tensor(0.0, device=pair_logits.device)
    correct_target_available = torch.zeros(loss_matrix.size(0), dtype=torch.bool, device=pair_logits.device)
    avg_correct_pairs = torch.tensor(0.0, device=pair_logits.device)
    if correct_matrix is not None:
        flat_correct = correct_matrix.to(device=pair_logits.device, dtype=torch.float32).view(loss_matrix.size(0), -1)
        correct_counts = flat_correct.sum(dim=-1, keepdim=True)
        correct_target_available = correct_counts.squeeze(-1) > 0
        avg_correct_pairs = correct_counts[correct_target_available].mean() if correct_target_available.any() else torch.tensor(
            0.0, device=pair_logits.device
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
            else torch.tensor(0.0, device=pair_logits.device)
        )
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
            else torch.tensor(0.0, device=pair_logits.device)
        )

    sorted_loss, _ = normalized_flat_loss.sort(dim=-1)
    if normalized_flat_loss.size(1) > 1:
        margin_mask = (sorted_loss[:, 1] - sorted_loss[:, 0]) >= float(margin)
    else:
        margin_mask = torch.ones_like(flat_best, dtype=torch.bool)

    ce_pair_all = nn.functional.cross_entropy(pair_logits, flat_best, reduction="none")
    ce_pair = ce_pair_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=pair_logits.device)

    ce_first = torch.tensor(0.0, device=pair_logits.device)
    ce_mid = torch.tensor(0.0, device=pair_logits.device)
    if logits_first is not None:
        ce_first_all = nn.functional.cross_entropy(logits_first, best_first, reduction="none")
        ce_first = ce_first_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=logits_first.device)
    if logits_mid is not None:
        ce_mid_all = nn.functional.cross_entropy(logits_mid, best_mid, reduction="none")
        ce_mid = ce_mid_all[margin_mask].mean() if margin_mask.any() else torch.tensor(0.0, device=logits_mid.device)

    if mode == "stage1":
        if logits_first is None:
            raise ValueError("stage1 requires logits_first")
        total_loss = ce_first
    elif mode == "stage2":
        if logits_mid is None:
            raise ValueError("stage2 requires logits_mid")
        total_loss = ce_mid
    elif mode == "joint":
        if joint_loss == "ce_pair":
            total_loss = ce_pair
        elif joint_loss == "expected_loss":
            total_loss = expected_loss
        elif joint_loss == "correct_soft_ce":
            if correct_matrix is None:
                raise ValueError("correct_soft_ce requires correct_matrix")
            total_loss = correct_soft_ce
        elif joint_loss == "correct_conf_ce":
            if correct_matrix is None:
                raise ValueError("correct_conf_ce requires correct_matrix")
            total_loss = correct_conf_ce
        elif joint_loss == "ce_pair_plus_expected":
            total_loss = ce_pair
            if pseudo_ce_weight > 0:
                total_loss = total_loss + float(pseudo_ce_weight) * expected_loss
        else:
            raise ValueError(f"Unknown joint_loss: {joint_loss}")
    else:
        raise ValueError(f"Unknown training mode: {mode}")

    metrics = {
        "expected_loss": float(expected_loss.detach().item()),
        "correct_soft_ce": float(correct_soft_ce.detach().item()),
        "correct_conf_ce": float(correct_conf_ce.detach().item()),
        "correct_soft_ce_temperature": float(correct_soft_ce_temperature),
        "correct_target_available_ratio": float(correct_target_available.float().mean().item()),
        "avg_correct_pairs": float(avg_correct_pairs.detach().item()),
        "main_pair_ce": float(ce_pair.detach().item()),
        "pseudo_ce_pair": float(ce_pair.detach().item()),
        "best_pair_loss": float(sorted_loss[:, 0].mean().item()),
        "raw_best_pair_loss": float(flat_loss.gather(1, flat_best.unsqueeze(1)).mean().item()),
        "margin_active_ratio": float(margin_mask.float().mean().item()),
    }
    if logits_first is not None:
        metrics["pseudo_ce_first"] = float(ce_first.detach().item())
    if logits_mid is not None:
        metrics["pseudo_ce_mid"] = float(ce_mid.detach().item())

    return total_loss, metrics, best_first, best_mid, flat_best


def score_from_cost(cost: torch.Tensor) -> torch.Tensor:
    return (1.0 - cost) * 100.0


def resolve_self_expert_ids(
    task_ids: torch.Tensor,
    expert_names: Sequence[str],
    sample_task_names: Optional[Sequence[str]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if sample_task_names is None:
        ids = task_ids.to(device=device, dtype=torch.long)
        valid = (ids >= 0) & (ids < len(expert_names))
        return ids, valid

    expert2id = {name: idx for idx, name in enumerate(expert_names)}
    ids = torch.tensor(
        [expert2id.get(str(task_name), -1) for task_name in sample_task_names],
        dtype=torch.long,
        device=device,
    )
    return ids, ids >= 0


def task_names_from_ids(task_ids: Sequence[int], expert_names: Sequence[str]) -> List[str]:
    names = []
    for task_id in task_ids:
        idx = int(task_id)
        if 0 <= idx < len(expert_names):
            names.append(str(expert_names[idx]))
        else:
            names.append(f"task_id:{idx}")
    return names


def optional_float(value: Optional[float], digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.{digits}f}"


def optional_rate(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "n/a"
    return f"{float(value):.2%}"


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    if bool(mask.any().item()):
        return float(values[mask].float().mean().item())
    return 0.0


def compute_routing_accuracy_stats(
    pred_first: torch.Tensor,
    pred_mid: torch.Tensor,
    best_first: torch.Tensor,
    best_mid: torch.Tensor,
    task_ids: torch.Tensor,
    expert_names: Optional[Sequence[str]] = None,
    sample_task_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    pred_first = pred_first.detach()
    pred_mid = pred_mid.detach()
    best_first = best_first.detach()
    best_mid = best_mid.detach()
    task_ids = task_ids.to(device=pred_first.device)
    if expert_names is None:
        expert_names = [str(idx) for idx in range(int(max(task_ids.max().item() + 1, best_first.max().item() + 1, best_mid.max().item() + 1)))]
    self_ids, self_valid = resolve_self_expert_ids(
        task_ids=task_ids,
        expert_names=expert_names,
        sample_task_names=sample_task_names,
        device=pred_first.device,
    )

    first_correct = (pred_first == best_first).float().mean().item()
    mid_correct = (pred_mid == best_mid).float().mean().item()
    self_first_acc = masked_mean(pred_first == self_ids, self_valid)
    self_mid_acc = masked_mean(pred_mid == self_ids, self_valid)
    oracle_first_self_acc = masked_mean(best_first == self_ids, self_valid)
    oracle_mid_self_acc = masked_mean(best_mid == self_ids, self_valid)
    pred_self_pair_acc = masked_mean((pred_first == self_ids) & (pred_mid == self_ids), self_valid)
    oracle_self_pair_acc = masked_mean((best_first == self_ids) & (best_mid == self_ids), self_valid)
    pair_acc = ((pred_first == best_first) & (pred_mid == best_mid)).float().mean().item()

    return {
        "first_acc": first_correct,
        "mid_acc": mid_correct,
        "joint_acc": 0.5 * (first_correct + mid_correct),
        "pair_acc": pair_acc,
        "self_first_acc": self_first_acc,
        "self_mid_acc": self_mid_acc,
        "self_joint_acc": 0.5 * (self_first_acc + self_mid_acc),
        "self_pair_acc": pred_self_pair_acc,
        "oracle_first_self_acc": oracle_first_self_acc,
        "oracle_mid_self_acc": oracle_mid_self_acc,
        "oracle_self_joint_acc": 0.5 * (oracle_first_self_acc + oracle_mid_self_acc),
        "oracle_self_pair_acc": oracle_self_pair_acc,
        "self_applicable_ratio": float(self_valid.float().mean().item()),
    }


def build_routing_summary(
    pred_first_all: Sequence[int],
    pred_mid_all: Sequence[int],
    best_first_all: Sequence[int],
    best_mid_all: Sequence[int],
    task_ids_all: Sequence[int],
    expert_names: Sequence[str],
    sample_task_names_all: Optional[Sequence[str]] = None,
    top_k: int = 5,
) -> Dict:
    if not pred_first_all:
        return {"num_samples": 0, "top_pred_pairs": [], "per_task": []}
    if sample_task_names_all is None:
        sample_task_names_all = task_names_from_ids(task_ids_all, expert_names)

    stats = compute_routing_accuracy_stats(
        pred_first=torch.tensor(pred_first_all, dtype=torch.long),
        pred_mid=torch.tensor(pred_mid_all, dtype=torch.long),
        best_first=torch.tensor(best_first_all, dtype=torch.long),
        best_mid=torch.tensor(best_mid_all, dtype=torch.long),
        task_ids=torch.tensor(task_ids_all, dtype=torch.long),
        expert_names=expert_names,
        sample_task_names=sample_task_names_all,
    )

    num_samples = len(pred_first_all)
    expert2id = {name: idx for idx, name in enumerate(expert_names)}
    pred_pair_counter = Counter()
    gold_pair_counter = Counter()
    task_bucket: Dict[str, Dict[str, Counter]] = defaultdict(
        lambda: {
            "pred_first": Counter(),
            "pred_mid": Counter(),
            "pred_pair": Counter(),
            "gold_pair": Counter(),
        }
    )

    for pred_first, pred_mid, best_first, best_mid, task_name in zip(
        pred_first_all, pred_mid_all, best_first_all, best_mid_all, sample_task_names_all
    ):
        pred_pair_name = f"{expert_names[pred_first]}->{expert_names[pred_mid]}"
        gold_pair_name = f"{expert_names[best_first]}->{expert_names[best_mid]}"
        pred_pair_counter[pred_pair_name] += 1
        gold_pair_counter[gold_pair_name] += 1

        bucket = task_bucket[str(task_name)]
        bucket["pred_first"][expert_names[pred_first]] += 1
        bucket["pred_mid"][expert_names[pred_mid]] += 1
        bucket["pred_pair"][pred_pair_name] += 1
        bucket["gold_pair"][gold_pair_name] += 1

    def _counter_rows(counter: Counter, denom: int, limit: int) -> List[Dict]:
        rows = []
        for name, count in counter.most_common(limit):
            rows.append({"name": name, "count": int(count), "rate": float(count / max(denom, 1))})
        return rows

    def _all_pair_rows(counter: Counter, denom: int) -> List[Dict]:
        rows = []
        for first_name in expert_names:
            for mid_name in expert_names:
                pair_name = f"{first_name}->{mid_name}"
                count = int(counter.get(pair_name, 0))
                rows.append({"name": pair_name, "count": count, "rate": float(count / max(denom, 1))})
        return rows

    per_task = []
    for task_name in sorted(set(str(name) for name in sample_task_names_all)):
        mask_count = sum(1 for name in sample_task_names_all if str(name) == task_name)
        if mask_count == 0:
            continue

        self_expert_id = expert2id.get(task_name)
        if self_expert_id is None:
            pred_self_first_rate = None
            pred_self_mid_rate = None
            gold_self_pair_rate = None
        else:
            pred_self_first = sum(
                1 for pf, name in zip(pred_first_all, sample_task_names_all)
                if str(name) == task_name and pf == self_expert_id
            )
            pred_self_mid = sum(
                1 for pm, name in zip(pred_mid_all, sample_task_names_all)
                if str(name) == task_name and pm == self_expert_id
            )
            gold_self_pair = sum(
                1
                for bf, bm, name in zip(best_first_all, best_mid_all, sample_task_names_all)
                if str(name) == task_name and bf == self_expert_id and bm == self_expert_id
            )
            pred_self_first_rate = float(pred_self_first / mask_count)
            pred_self_mid_rate = float(pred_self_mid / mask_count)
            gold_self_pair_rate = float(gold_self_pair / mask_count)

        bucket = task_bucket[task_name]
        per_task.append(
            {
                "task": task_name,
                "count": int(mask_count),
                "has_self_expert": self_expert_id is not None,
                "pred_self_first_rate": pred_self_first_rate,
                "pred_self_mid_rate": pred_self_mid_rate,
                "gold_self_pair_rate": gold_self_pair_rate,
                "top_pred_first": _counter_rows(bucket["pred_first"], mask_count, limit=3),
                "top_pred_mid": _counter_rows(bucket["pred_mid"], mask_count, limit=3),
                "top_pred_pairs": _counter_rows(bucket["pred_pair"], mask_count, limit=3),
                "top_gold_pairs": _counter_rows(bucket["gold_pair"], mask_count, limit=3),
                "all_pred_pairs": _all_pair_rows(bucket["pred_pair"], mask_count),
                "all_gold_pairs": _all_pair_rows(bucket["gold_pair"], mask_count),
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
        "self_applicable_ratio": float(stats["self_applicable_ratio"]),
        "top_pred_pairs": _counter_rows(pred_pair_counter, num_samples, limit=top_k),
        "top_gold_pairs": _counter_rows(gold_pair_counter, num_samples, limit=top_k),
        "all_pred_pairs": _all_pair_rows(pred_pair_counter, num_samples),
        "all_gold_pairs": _all_pair_rows(gold_pair_counter, num_samples),
        "per_task": per_task,
    }


def print_routing_summary(tag: str, summary: Dict):
    if int(summary.get("num_samples", 0)) <= 0:
        print(f"[ROUTE][{tag}] no samples")
        return

    top_pairs = ", ".join(f"{row['name']}:{row['rate']:.2%}" for row in summary.get("top_pred_pairs", [])[:3])
    print(
        f"[ROUTE][{tag}] pair_acc={summary.get('pair_acc', 0.0):.4f} "
        f"self_pair={optional_float(summary.get('self_pair_acc') if summary.get('self_applicable_ratio', 1.0) > 0 else None)} "
        f"oracle_self_pair={optional_float(summary.get('oracle_self_pair_acc') if summary.get('self_applicable_ratio', 1.0) > 0 else None)} "
        f"top_pred_pairs={top_pairs}"
    )
    all_pairs = summary.get("all_pred_pairs", [])
    if all_pairs:
        pair_dist = ", ".join(f"{row['name']}:{row['rate']:.2%}" for row in all_pairs)
        print(f"[ROUTE][{tag}][PAIR_DIST] {pair_dist}")
    for row in summary.get("per_task", []):
        top_first = row.get("top_pred_first", [])
        top_mid = row.get("top_pred_mid", [])
        top_pair = row.get("top_pred_pairs", [])
        first_name = top_first[0]["name"] if top_first else "-"
        mid_name = top_mid[0]["name"] if top_mid else "-"
        pair_name = top_pair[0]["name"] if top_pair else "-"
        print(
            f"[ROUTE][{tag}][{row['task']}] n={row['count']} "
            f"self_first={optional_rate(row.get('pred_self_first_rate'))} "
            f"self_mid={optional_rate(row.get('pred_self_mid_rate'))} "
            f"oracle_self_pair={optional_rate(row.get('gold_self_pair_rate'))} "
            f"top_first={first_name} top_mid={mid_name} top_pair={pair_name}"
        )
        task_all_pairs = row.get("all_pred_pairs", [])
        if task_all_pairs:
            task_pair_dist = ", ".join(f"{pair_row['name']}:{pair_row['rate']:.2%}" for pair_row in task_all_pairs)
            print(f"[ROUTE][{tag}][{row['task']}][PAIR_DIST] {task_pair_dist}")


def compute_route_score_stats(
    loss_matrix: torch.Tensor,
    pred_pair: torch.Tensor,
    task_ids: torch.Tensor,
    loss_normalization: str = "sample_minmax",
    expert_names: Optional[Sequence[str]] = None,
    sample_task_names: Optional[Sequence[str]] = None,
) -> Dict[str, float]:
    score_loss_matrix = normalize_pair_loss_matrix(
        loss_matrix=loss_matrix,
        method=loss_normalization,
    )
    flat_loss = score_loss_matrix.view(score_loss_matrix.size(0), -1)
    best_pair = flat_loss.argmin(dim=-1)
    batch_idx = torch.arange(score_loss_matrix.size(0), device=score_loss_matrix.device)

    pred_cost = flat_loss[batch_idx, pred_pair]
    oracle_cost = flat_loss[batch_idx, best_pair]
    if expert_names is None:
        expert_names = [str(idx) for idx in range(score_loss_matrix.size(1))]
    self_ids, self_valid = resolve_self_expert_ids(
        task_ids=task_ids,
        expert_names=expert_names,
        sample_task_names=sample_task_names,
        device=score_loss_matrix.device,
    )
    self_cost = score_loss_matrix[batch_idx, self_ids.clamp_min(0), self_ids.clamp_min(0)]
    if bool(self_valid.any().item()):
        fixed_self_score = float(score_from_cost(self_cost[self_valid]).mean().item())
        fixed_self_cost = float(self_cost[self_valid].mean().item())
    else:
        fixed_self_score = float("nan")
        fixed_self_cost = float("nan")

    return {
        "router_argmax_score": float(score_from_cost(pred_cost).mean().item()),
        "oracle_best_pair_score": float(score_from_cost(oracle_cost).mean().item()),
        "fixed_self_score": fixed_self_score,
        "router_argmax_cost": float(pred_cost.mean().item()),
        "oracle_best_pair_cost": float(oracle_cost.mean().item()),
        "fixed_self_cost": fixed_self_cost,
    }


def init_oracle_debug_accumulator(expert_names: Sequence[str]) -> Dict:
    return {
        "num_samples": 0,
        "expert_names": list(expert_names),
        "global_gold_pair_counter": Counter(),
        "global_gap_values": [],
        "global_self_minus_oracle_values": [],
        "per_task": defaultdict(
            lambda: {
                "count": 0,
                "gold_pair_counter": Counter(),
                "gap_values": [],
                "self_minus_oracle_values": [],
                "has_self_expert": False,
            }
        ),
    }


def update_oracle_debug_accumulator(
    acc: Dict,
    loss_matrix: torch.Tensor,
    task_ids: torch.Tensor,
    sample_task_names: Optional[Sequence[str]] = None,
):
    expert_names = acc["expert_names"]
    num_pairs = loss_matrix.size(1) * loss_matrix.size(2)
    flat_loss = loss_matrix.view(loss_matrix.size(0), num_pairs)
    sorted_loss, sorted_idx = flat_loss.sort(dim=-1)
    best_pair = sorted_idx[:, 0]
    best_cost = sorted_loss[:, 0]
    second_cost = sorted_loss[:, 1] if num_pairs > 1 else sorted_loss[:, 0]
    gap = second_cost - best_cost
    batch_idx = torch.arange(loss_matrix.size(0), device=loss_matrix.device)
    self_ids, self_valid = resolve_self_expert_ids(
        task_ids=task_ids,
        expert_names=expert_names,
        sample_task_names=sample_task_names,
        device=loss_matrix.device,
    )
    self_cost = loss_matrix[batch_idx, self_ids.clamp_min(0), self_ids.clamp_min(0)]
    self_minus_oracle = self_cost - best_cost

    best_pair_cpu = best_pair.detach().cpu().tolist()
    gap_cpu = gap.detach().cpu().tolist()
    self_minus_oracle_cpu = self_minus_oracle.detach().cpu().tolist()
    self_valid_cpu = self_valid.detach().cpu().tolist()
    if sample_task_names is None:
        sample_task_names = task_names_from_ids(task_ids.detach().cpu().tolist(), expert_names)

    acc["num_samples"] += len(best_pair_cpu)
    acc["global_gap_values"].extend(float(x) for x in gap_cpu)

    for pair_id, gap_value, delta_value, has_self_expert, task_name in zip(
        best_pair_cpu, gap_cpu, self_minus_oracle_cpu, self_valid_cpu, sample_task_names
    ):
        first_idx = int(pair_id) // loss_matrix.size(2)
        mid_idx = int(pair_id) % loss_matrix.size(2)
        pair_name = f"{expert_names[first_idx]}->{expert_names[mid_idx]}"
        task_name = str(task_name)

        acc["global_gold_pair_counter"][pair_name] += 1
        task_bucket = acc["per_task"][task_name]
        task_bucket["count"] += 1
        task_bucket["gold_pair_counter"][pair_name] += 1
        task_bucket["gap_values"].append(float(gap_value))
        if bool(has_self_expert):
            task_bucket["has_self_expert"] = True
            task_bucket["self_minus_oracle_values"].append(float(delta_value))
            acc["global_self_minus_oracle_values"].append(float(delta_value))


def build_oracle_debug_summary(acc: Dict, top_k: int = 5) -> Dict:
    num_samples = int(acc.get("num_samples", 0))
    if num_samples <= 0:
        return {"num_samples": 0, "top_gold_pairs": [], "per_task": []}

    def _counter_rows(counter: Counter, denom: int, limit: int) -> List[Dict]:
        rows = []
        for name, count in counter.most_common(limit):
            rows.append({"name": name, "count": int(count), "rate": float(count / max(denom, 1))})
        return rows

    def _mean(values: Sequence[float]) -> float:
        return float(sum(values) / max(len(values), 1))

    def _quantile(values: Sequence[float], q: float) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(float(v) for v in values)
        idx = int(round((len(sorted_values) - 1) * q))
        idx = max(0, min(idx, len(sorted_values) - 1))
        return float(sorted_values[idx])

    global_gaps = acc["global_gap_values"]
    global_deltas = acc["global_self_minus_oracle_values"]
    per_task = []
    for task_name in sorted(acc["per_task"].keys()):
        bucket = acc["per_task"][task_name]
        if bucket["count"] <= 0:
            continue
        delta_values = bucket["self_minus_oracle_values"]
        per_task.append(
            {
                "task": task_name,
                "count": int(bucket["count"]),
                "avg_gap": _mean(bucket["gap_values"]),
                "p50_gap": _quantile(bucket["gap_values"], 0.5),
                "p90_gap": _quantile(bucket["gap_values"], 0.9),
                "has_self_expert": bool(bucket.get("has_self_expert", False)),
                "avg_self_minus_oracle": _mean(delta_values) if delta_values else None,
                "top_gold_pairs": _counter_rows(bucket["gold_pair_counter"], bucket["count"], top_k),
            }
        )

    return {
        "num_samples": num_samples,
        "avg_gap": _mean(global_gaps),
        "p50_gap": _quantile(global_gaps, 0.5),
        "p90_gap": _quantile(global_gaps, 0.9),
        "avg_self_minus_oracle": _mean(global_deltas) if global_deltas else None,
        "top_gold_pairs": _counter_rows(acc["global_gold_pair_counter"], num_samples, top_k),
        "per_task": per_task,
    }


def print_oracle_debug_summary(tag: str, summary: Dict):
    if int(summary.get("num_samples", 0)) <= 0:
        print(f"[ORACLE][{tag}] no samples")
        return

    top_gold = ", ".join(f"{row['name']}:{row['rate']:.2%}" for row in summary.get("top_gold_pairs", [])[:3])
    print(
        f"[ORACLE][{tag}] avg_gap={summary.get('avg_gap', 0.0):.4f} "
        f"p50_gap={summary.get('p50_gap', 0.0):.4f} "
        f"p90_gap={summary.get('p90_gap', 0.0):.4f} "
        f"avg_self_minus_oracle={optional_float(summary.get('avg_self_minus_oracle'))} "
        f"top_gold_pairs={top_gold}"
    )
    for row in summary.get("per_task", []):
        top_gold_rows = row.get("top_gold_pairs", [])
        top_gold_name = top_gold_rows[0]["name"] if top_gold_rows else "-"
        print(
            f"[ORACLE][{tag}][{row['task']}] n={row['count']} "
            f"avg_gap={row['avg_gap']:.4f} "
            f"p50_gap={row['p50_gap']:.4f} "
            f"avg_self_minus_oracle={optional_float(row.get('avg_self_minus_oracle'))} "
            f"top_gold_pair={top_gold_name}"
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
        if key in summary and summary[key] is not None:
            payload[f"{prefix}/{key}"] = float(summary[key])

    top_pred_pairs = summary.get("top_pred_pairs", [])
    if top_pred_pairs:
        payload[f"{prefix}/top_pred_pair_rate"] = float(top_pred_pairs[0]["rate"])

    for row in summary.get("per_task", []):
        task = str(row["task"])
        for source_key, out_key in [
            ("pred_self_first_rate", "self_first_rate"),
            ("pred_self_mid_rate", "self_mid_rate"),
            ("gold_self_pair_rate", "oracle_self_pair_rate"),
        ]:
            value = row.get(source_key)
            if value is not None:
                payload[f"{prefix}_task/{task}_{out_key}"] = float(value)
        if row.get("top_pred_first"):
            payload[f"{prefix}_task/{task}_top_first_rate"] = float(row["top_pred_first"][0]["rate"])
        if row.get("top_pred_mid"):
            payload[f"{prefix}_task/{task}_top_mid_rate"] = float(row["top_pred_mid"][0]["rate"])
        if row.get("top_pred_pairs"):
            payload[f"{prefix}_task/{task}_top_pair_rate"] = float(row["top_pred_pairs"][0]["rate"])
    return payload
