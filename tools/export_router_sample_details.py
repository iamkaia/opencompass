import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch


def load_cache_items(cache_roots: Sequence[Path], split: str, sample_tasks: set[str] | None) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for root in cache_roots:
        manifest = json.loads((root / split / "manifest.json").read_text())
        expert_names = list(manifest["expert_names"])
        for chunk_file in manifest["files"]:
            payload = torch.load(root / split / chunk_file, map_location="cpu")
            for item in payload["items"]:
                if sample_tasks and str(item["task"]) not in sample_tasks:
                    continue
                row = dict(item)
                row["_cache_root"] = str(root)
                row["_expert_names"] = expert_names
                rows.append(row)
    return rows


def to_json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json_value(item) for item in value]
    return value


def pair_value(matrix: Any, pair_id: int, num_experts: int) -> Any:
    if matrix is None:
        return None
    return matrix[pair_id // num_experts][pair_id % num_experts]


def pair_id_from_name(pair_name: str, expert_names: Sequence[str]) -> int | None:
    if "->" not in str(pair_name):
        return None
    first, mid = str(pair_name).split("->", 1)
    if first not in expert_names or mid not in expert_names:
        return None
    return expert_names.index(first) * len(expert_names) + expert_names.index(mid)


def pair_summary(item: Dict[str, Any], record: Dict[str, Any], pair_id: int | None) -> Dict[str, Any] | None:
    if pair_id is None:
        return None
    expert_names = item["_expert_names"]
    num_experts = len(expert_names)
    pair_name = f"{expert_names[pair_id // num_experts]}->{expert_names[pair_id % num_experts]}"
    return {
        "pair": pair_name,
        "answer": pair_value(item.get("prediction_matrix"), pair_id, num_experts),
        "loss": pair_value(item.get("loss_matrix"), pair_id, num_experts),
        "correct": pair_value(item.get("correct_matrix"), pair_id, num_experts),
        "router_pred_prob": pair_value(record.get("pair_prob_matrix"), pair_id, num_experts),
        "router_target_prob": pair_value(record.get("router_target_matrix"), pair_id, num_experts),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_roots", required=True, help="Comma-separated cache roots in training load order.")
    parser.add_argument("--records_path", required=True, type=Path)
    parser.add_argument("--split", choices=["train", "validation"], required=True)
    parser.add_argument("--sample_tasks", default=None, help="Optional comma-separated selected tasks.")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()

    cache_roots = [Path(part.strip()) for part in args.cache_roots.split(",") if part.strip()]
    sample_tasks = {part.strip() for part in args.sample_tasks.split(",") if part.strip()} if args.sample_tasks else None
    items = load_cache_items(cache_roots, args.split, sample_tasks)
    record_obj = json.loads(args.records_path.read_text())
    records = record_obj["records"]
    if len(items) != len(records):
        raise ValueError(f"Cache item count {len(items)} != route record count {len(records)}")

    exported = []
    for idx, (item, record) in enumerate(zip(items, records)):
        if str(item["task"]) != str(record["task"]) or str(item["target"]) != str(record["target"]):
            raise ValueError(f"Order mismatch at idx={idx}: cache={item['task']}/{item['target']} record={record['task']}/{record['target']}")
        expert_names = item["_expert_names"]
        self_id = pair_id_from_name(f"{item['task']}->{item['task']}", expert_names)
        selected_id = pair_id_from_name(str(record["pred_pair"]), expert_names)
        exported.append(
            {
                "idx": idx,
                "cache_root": item["_cache_root"],
                "task": item["task"],
                "prompt": item.get("prompt_text", item.get("text", "")),
                "target": item["target"],
                "expert_names": expert_names,
                "loss_matrix": item["loss_matrix"],
                "correct_matrix": item["correct_matrix"],
                "prediction_matrix": item.get("prediction_matrix"),
                "router_pred_pair": record["pred_pair"],
                "router_gold_lowest_loss_pair": record["gold_pair"],
                "router_pred_matrix": record.get("pair_prob_matrix"),
                "router_target_matrix": record.get("router_target_matrix"),
                "router_pred_correct": record.get("pred_correct"),
                "any_pair_correct": record.get("any_pair_correct"),
                "self_expert": pair_summary(item, record, self_id),
                "selected_expert_pair": pair_summary(item, record, selected_id),
            }
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(to_json_value({"records": exported}), ensure_ascii=False, indent=2))
    print(f"saved {len(exported)} records to {args.out}")


if __name__ == "__main__":
    main()
