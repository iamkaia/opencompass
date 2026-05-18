#!/usr/bin/env python3
import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch


def tensor_to_python(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_route_records(path: Optional[str], out_dir: Optional[str], split: str) -> Dict[str, Dict]:
    if not path and out_dir:
        patterns = [
            os.path.join(out_dir, f"route_records_{split}_epoch*.json"),
            os.path.join(out_dir, f"route_records_{split}_eval_only.json"),
        ]
        matches: List[str] = []
        for pattern in patterns:
            matches.extend(glob.glob(pattern))
        if matches:
            path = sorted(matches)[-1]

    if not path:
        return {}

    payload = load_json(path)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not isinstance(records, list):
        raise ValueError(f"Unsupported route record format: {path}")
    return {str(record["item_id"]): record for record in records if "item_id" in record}


def pair_name(experts: List[str], pair_id: int) -> str:
    num_experts = len(experts)
    first_idx = pair_id // num_experts
    mid_idx = pair_id % num_experts
    return f"{experts[first_idx]}->{experts[mid_idx]}"


def matrix_value(matrix: Any, first_idx: int, mid_idx: int) -> Any:
    if matrix is None:
        return None
    value = matrix[first_idx][mid_idx]
    return tensor_to_python(value)


def build_pair_rows(item: Dict, experts: List[str], include_option_probs: bool) -> List[Dict]:
    loss_matrix = item["loss_matrix"].to(torch.float32)
    correct_matrix = item.get("correct_matrix")
    prediction_matrix = item.get("prediction_matrix")
    option_prob_matrix = item.get("option_prob_matrix")

    rows = []
    for first_idx, first in enumerate(experts):
        for mid_idx, mid in enumerate(experts):
            pair_id = first_idx * len(experts) + mid_idx
            row = {
                "pair_id": pair_id,
                "pair": f"{first}->{mid}",
                "first_expert": first,
                "mid_expert": mid,
                "loss": float(loss_matrix[first_idx, mid_idx].item()),
                "correct": (
                    None
                    if correct_matrix is None
                    else bool(correct_matrix[first_idx, mid_idx].item())
                ),
                "prediction": matrix_value(prediction_matrix, first_idx, mid_idx),
            }
            if include_option_probs and option_prob_matrix is not None:
                probs = option_prob_matrix[first_idx, mid_idx].to(torch.float32).view(-1)
                row["option_probs"] = [float(x) for x in probs.tolist()]
            rows.append(row)
    return rows


def export_samples(
    cache_root: str,
    split: str,
    route_records_by_item: Dict[str, Dict],
    include_all_pairs: bool,
    include_option_probs: bool,
) -> Dict:
    split_dir = Path(cache_root) / split
    manifest = load_json(str(split_dir / "manifest.json"))
    experts = list(manifest["expert_names"])
    records = []
    sample_idx = 0

    for filename in manifest["files"]:
        payload = torch.load(split_dir / filename, map_location="cpu")
        for item in payload["items"]:
            item_id = f"{split}:{sample_idx:08d}"
            pairs = build_pair_rows(item, experts, include_option_probs=include_option_probs)
            loss_matrix = item["loss_matrix"].to(torch.float32)
            oracle_pair_id = int(loss_matrix.view(-1).argmin().item())
            route_record = route_records_by_item.get(item_id)

            record = {
                "item_id": item_id,
                "task": item["task"],
                "target": item.get("target"),
                "prompt_text": item.get("prompt_text"),
                "source_text": item.get("text"),
                "oracle_best_pair_id": oracle_pair_id,
                "oracle_best_pair": pair_name(experts, oracle_pair_id),
                "oracle_best_output": pairs[oracle_pair_id].get("prediction"),
                "oracle_best_loss": pairs[oracle_pair_id].get("loss"),
                "oracle_best_correct": pairs[oracle_pair_id].get("correct"),
            }

            task_name = str(item["task"])
            if task_name in experts:
                self_idx = experts.index(task_name)
                self_pair_id = self_idx * len(experts) + self_idx
                self_row = pairs[self_pair_id]
                record.update(
                    {
                        "self_expert_available": True,
                        "self_pair_id": self_pair_id,
                        "self_pair": self_row["pair"],
                        "self_output": self_row.get("prediction"),
                        "self_loss": self_row.get("loss"),
                        "self_correct": self_row.get("correct"),
                        "self_is_oracle_best": bool(self_pair_id == oracle_pair_id),
                    }
                )
            else:
                record.update(
                    {
                        "self_expert_available": False,
                        "self_pair_id": None,
                        "self_pair": None,
                        "self_output": None,
                        "self_loss": None,
                        "self_correct": None,
                        "self_is_oracle_best": False,
                    }
                )

            if route_record is not None:
                pred_pair_id = int(route_record["pred_pair_id"])
                pred_first = pred_pair_id // len(experts)
                pred_mid = pred_pair_id % len(experts)
                pred_row = pairs[pred_pair_id]
                record.update(
                    {
                        "router_pred_pair_id": pred_pair_id,
                        "router_pred_pair": route_record.get("pred_pair") or pair_name(experts, pred_pair_id),
                        "router_selected_output": pred_row.get("prediction"),
                        "router_selected_loss": float(loss_matrix[pred_first, pred_mid].item()),
                        "router_selected_correct": pred_row.get("correct"),
                        "router_matches_oracle_pair": bool(route_record.get("match", pred_pair_id == oracle_pair_id)),
                        "router_selected_self_pair": bool(
                            record["self_expert_available"] and pred_pair_id == record["self_pair_id"]
                        ),
                        "top_router_pairs": route_record.get("top_router_pairs", []),
                        "pair_probs": route_record.get("pair_probs", []),
                    }
                )

            if include_all_pairs:
                record["pairs"] = pairs
            records.append(record)
            sample_idx += 1

    return {
        "cache_root": cache_root,
        "split": split,
        "num_samples": len(records),
        "expert_names": experts,
        "has_route_records": bool(route_records_by_item),
        "samples": records,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Export cached router per-sample outputs into one JSON file."
    )
    parser.add_argument("--cache_root", required=True)
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--output_json", required=True)
    parser.add_argument(
        "--route_records",
        default=None,
        help="Optional route_records_*.json from train_internal_two_router_compact_cached_joint.py.",
    )
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Optional router output dir. If --route_records is omitted, the newest route_records file is used.",
    )
    parser.add_argument("--no_all_pairs", action="store_true", help="Only export oracle/router selected outputs.")
    parser.add_argument("--include_option_probs", action="store_true")
    args = parser.parse_args()

    route_records = load_route_records(args.route_records, args.out_dir, args.split)
    payload = export_samples(
        cache_root=args.cache_root,
        split=args.split,
        route_records_by_item=route_records,
        include_all_pairs=not args.no_all_pairs,
        include_option_probs=args.include_option_probs,
    )

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"[SAVE] {output_path}")


if __name__ == "__main__":
    main()
