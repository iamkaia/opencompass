import argparse
import hashlib
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch


def parse_csv(raw: Optional[str]) -> Optional[List[str]]:
    if raw is None:
        return None
    items = [part.strip() for part in str(raw).split(",") if part.strip()]
    return items or None


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def save_jsonl(rows: Iterable[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def prompt_sha1(task: str, prompt_text: str, target: str) -> str:
    raw = f"{task}\n{prompt_text}\n{target}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def load_manifest(feature_root: str, split: str) -> Dict:
    path = os.path.join(feature_root, split, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def resolve_names(
    manifest: Dict,
    requested_tasks: Optional[Sequence[str]],
    requested_experts: Optional[Sequence[str]],
) -> Tuple[List[str], List[str], List[str], List[str]]:
    source_tasks = list(manifest.get("task_names") or [])
    source_experts = list(manifest.get("expert_names") or source_tasks)
    tasks = list(requested_tasks or source_tasks)
    experts = list(requested_experts or source_experts)
    missing_tasks = [name for name in tasks if name not in source_tasks]
    if missing_tasks:
        raise ValueError(f"Missing sample tasks in cache: {missing_tasks}. Available={source_tasks}")
    missing_experts = [name for name in experts if name not in source_experts]
    if missing_experts:
        raise ValueError(f"Missing experts in cache: {missing_experts}. Available={source_experts}")
    return source_tasks, source_experts, tasks, experts


def option_labels_for_task(task: str) -> Optional[List[str]]:
    task = str(task)
    if task in {"race", "medmcqa", "hellaswag"}:
        return ["A", "B", "C", "D"]
    if task in {"piqa", "copa", "boolq"}:
        return ["A", "B"]
    if task == "siqa":
        return ["A", "B", "C"]
    if task == "sst2":
        return ["0", "1"]
    return None


def pair_name(experts: Sequence[str], pair_id: int) -> str:
    num_experts = len(experts)
    return f"{experts[pair_id // num_experts]}->{experts[pair_id % num_experts]}"


def collect_items(
    feature_root: str,
    split: str,
    tasks: Sequence[str],
    source_experts: Sequence[str],
    experts: Sequence[str],
) -> Dict[str, Dict]:
    manifest = load_manifest(feature_root, split)
    split_dir = os.path.join(feature_root, split)
    expert_indices = torch.tensor([source_experts.index(name) for name in experts], dtype=torch.long)
    items = {}
    for filename in manifest["files"]:
        payload = torch.load(os.path.join(split_dir, filename), map_location="cpu")
        for item in payload["items"]:
            task = str(item["task"])
            if task not in tasks:
                continue
            prompt_text = str(item.get("prompt_text", item["text"]))
            target = str(item.get("target", ""))
            key = prompt_sha1(task, prompt_text, target)
            loss_matrix = item["loss_matrix"].index_select(0, expert_indices).index_select(1, expert_indices).to(torch.float32)
            correct_matrix = item.get("correct_matrix")
            if correct_matrix is None:
                correct_matrix = torch.zeros_like(loss_matrix, dtype=torch.bool)
            else:
                correct_matrix = correct_matrix.index_select(0, expert_indices).index_select(1, expert_indices).to(torch.bool)
            option_prob_matrix = item.get("option_prob_matrix")
            if option_prob_matrix is not None:
                option_prob_matrix = option_prob_matrix.index_select(0, expert_indices).index_select(1, expert_indices).to(torch.float32)
            prediction_matrix = item.get("prediction_matrix")
            items[key] = {
                "key": key,
                "task": task,
                "text": str(item["text"]),
                "prompt_text": prompt_text,
                "target": target,
                "loss_matrix": loss_matrix,
                "correct_matrix": correct_matrix,
                "option_prob_matrix": option_prob_matrix,
                "prediction_matrix": prediction_matrix,
            }
    return items


def summarize_item(item: Dict, experts: Sequence[str]) -> Dict:
    loss_matrix = item["loss_matrix"]
    correct_matrix = item["correct_matrix"]
    flat_loss = loss_matrix.view(-1)
    flat_correct = correct_matrix.view(-1)
    sorted_ids = torch.argsort(flat_loss)
    topk = min(25, flat_loss.numel())
    task = str(item["task"])
    self_idx = experts.index(task) if task in experts else -1
    option_labels = option_labels_for_task(task)
    option_prob_matrix = item.get("option_prob_matrix")
    prediction_matrix = item.get("prediction_matrix")

    all_pairs = []
    for rank in range(topk):
        pair_id = int(sorted_ids[rank].item())
        first_idx = pair_id // loss_matrix.size(1)
        mid_idx = pair_id % loss_matrix.size(1)
        row = {
            "rank_by_loss": rank + 1,
            "pair_id": pair_id,
            "pair": f"{experts[first_idx]}->{experts[mid_idx]}",
            "loss": float(flat_loss[pair_id].item()),
            "correct": bool(flat_correct[pair_id].item()),
        }
        if prediction_matrix is not None:
            row["prediction"] = str(prediction_matrix[first_idx][mid_idx])
        if option_prob_matrix is not None:
            probs = option_prob_matrix[first_idx, mid_idx].view(-1)
            pred_idx = int(probs.argmax().item())
            row["option_probs"] = [float(x) for x in probs.tolist()]
            row["pred_option_idx"] = pred_idx
            row["pred_option_prob"] = float(probs[pred_idx].item())
            if option_labels and pred_idx < len(option_labels):
                row["pred_option_label"] = option_labels[pred_idx]
        all_pairs.append(row)

    best_pair_id = int(sorted_ids[0].item())
    best_pair = pair_name(experts, best_pair_id)
    summary = {
        "task": task,
        "target": item["target"],
        "self_pair": f"{task}->{task}" if self_idx >= 0 else None,
        "self_pair_id": int(self_idx * loss_matrix.size(1) + self_idx) if self_idx >= 0 else None,
        "self_loss": float(loss_matrix[self_idx, self_idx].item()) if self_idx >= 0 else None,
        "self_correct": bool(correct_matrix[self_idx, self_idx].item()) if self_idx >= 0 else None,
        "oracle_pair": best_pair,
        "oracle_pair_id": best_pair_id,
        "oracle_loss": float(flat_loss[best_pair_id].item()),
        "oracle_correct": bool(flat_correct[best_pair_id].item()),
        "num_correct_pairs": int(flat_correct.sum().item()),
        "any_pair_correct": bool(flat_correct.any().item()),
        "all_pairs": all_pairs,
    }
    return summary


def load_route_records(path: Optional[str]) -> Dict[str, Dict]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])
    by_key = {}
    for record in records:
        key = record.get("prompt_sha1")
        if not key:
            task = str(record.get("task", ""))
            prompt_text = str(record.get("prompt_text", ""))
            target = str(record.get("target", ""))
            key = prompt_sha1(task, prompt_text, target)
        by_key[key] = record
    return by_key


def build_summary_table(
    label: str,
    items: Dict[str, Dict],
    tasks: Sequence[str],
    experts: Sequence[str],
) -> Dict:
    task_rows = []
    total_n = 0
    total_self = 0
    total_any = 0
    for task in tasks:
        task_items = [item for item in items.values() if str(item["task"]) == str(task)]
        n = len(task_items)
        if n <= 0:
            continue
        self_idx = experts.index(task) if task in experts else -1
        self_correct = 0
        any_correct = 0
        for item in task_items:
            correct_matrix = item["correct_matrix"]
            if self_idx >= 0:
                self_correct += int(correct_matrix[self_idx, self_idx].item())
            any_correct += int(correct_matrix.view(-1).any().item())
        task_rows.append(
            {
                "task": task,
                "self_acc": float(self_correct / n) if self_idx >= 0 else None,
                "any_pair_upper_bound": float(any_correct / n),
                "n": n,
            }
        )
        total_n += n
        total_self += self_correct
        total_any += any_correct
    return {
        "label": label,
        "per_task": task_rows,
        "overall_self_acc": float(total_self / total_n) if total_n else 0.0,
        "any_pair_upper_bound": float(total_any / total_n) if total_n else 0.0,
        "n": total_n,
    }


def render_markdown_table(old_summary: Dict, new_summary: Dict, notes: Dict[str, str]) -> str:
    task_order = [row["task"] for row in old_summary["per_task"]]
    old_map = {row["task"]: row for row in old_summary["per_task"]}
    new_map = {row["task"]: row for row in new_summary["per_task"]}
    lines = [
        "| Cache / scoring 方法 | " + " | ".join(f"{task} self" for task in task_order) + " | overall self | any-pair upper bound | 重點 |",
        "| --- | " + " | ".join(["---"] * len(task_order)) + " | --- | --- | --- |",
    ]
    for label, summary, row_map in (
        (old_summary["label"], old_summary, old_map),
        (new_summary["label"], new_summary, new_map),
    ):
        values = []
        for task in task_order:
            acc = row_map.get(task, {}).get("self_acc")
            values.append(f"{acc:.3f}" if acc is not None else "n/a")
        values.append(f"{summary['overall_self_acc']:.3f}")
        values.append(f"{summary['any_pair_upper_bound']:.3f}")
        values.append(notes.get(label, ""))
        lines.append("| " + " | ".join([label] + values) + " |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old_feature_root", required=True)
    parser.add_argument("--new_feature_root", required=True)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--sample_task_names", default=None)
    parser.add_argument("--expert_names", default=None)
    parser.add_argument("--old_route_records", default=None)
    parser.add_argument("--new_route_records", default=None)
    parser.add_argument("--out_prefix", required=True)
    args = parser.parse_args()

    requested_tasks = parse_csv(args.sample_task_names)
    requested_experts = parse_csv(args.expert_names)
    old_manifest = load_manifest(args.old_feature_root, args.split)
    _, old_source_experts, tasks, experts = resolve_names(old_manifest, requested_tasks, requested_experts)
    new_manifest = load_manifest(args.new_feature_root, args.split)
    _, new_source_experts, _, _ = resolve_names(new_manifest, tasks, experts)

    old_items = collect_items(args.old_feature_root, args.split, tasks, old_source_experts, experts)
    new_items = collect_items(args.new_feature_root, args.split, tasks, new_source_experts, experts)
    shared_keys = sorted(set(old_items) & set(new_items))
    if not shared_keys:
        raise ValueError("No shared samples found between old/new feature roots")

    old_records = load_route_records(args.old_route_records)
    new_records = load_route_records(args.new_route_records)
    sample_rows = []
    for key in shared_keys:
        old_summary = summarize_item(old_items[key], experts)
        new_summary = summarize_item(new_items[key], experts)
        row = {
            "key": key,
            "task": old_summary["task"],
            "target": old_summary["target"],
            "text": old_items[key]["text"],
            "prompt_text": old_items[key]["prompt_text"],
            "old_method": old_summary,
            "new_method": new_summary,
        }
        if key in old_records:
            row["old_router_prediction"] = old_records[key]
        if key in new_records:
            row["new_router_prediction"] = new_records[key]
        sample_rows.append(row)

    old_summary = build_summary_table(os.path.basename(args.old_feature_root), old_items, tasks, experts)
    new_summary = build_summary_table(os.path.basename(args.new_feature_root), new_items, tasks, experts)
    notes = {
        os.path.basename(args.old_feature_root): "SST2 若明顯偏低，通常代表舊 scoring 對二分類選項的 cost / correct 判定沒有對齊 OpenCompass。",
        os.path.basename(args.new_feature_root): "若 SST2 回到接近 OpenCompass，代表新的 scoring 比較貼近實際推論評估。",
    }
    table_md = render_markdown_table(old_summary, new_summary, notes)
    summary = {
        "split": args.split,
        "sample_task_names": tasks,
        "expert_names": experts,
        "shared_samples": len(shared_keys),
        "old_feature_root": args.old_feature_root,
        "new_feature_root": args.new_feature_root,
        "old_summary": old_summary,
        "new_summary": new_summary,
        "table_markdown": table_md,
    }

    save_json(summary, f"{args.out_prefix}_summary.json")
    with open(f"{args.out_prefix}_summary.md", "w", encoding="utf-8") as f:
        f.write(table_md)
    save_jsonl(sample_rows, f"{args.out_prefix}_samples.jsonl")
    print(table_md.strip())
    print(f"[DONE] wrote {args.out_prefix}_summary.json")
    print(f"[DONE] wrote {args.out_prefix}_summary.md")
    print(f"[DONE] wrote {args.out_prefix}_samples.jsonl")


if __name__ == "__main__":
    main()
