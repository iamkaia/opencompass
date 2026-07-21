#!/usr/bin/env python
import argparse
import json
import os
import shutil
from pathlib import Path

import torch


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path):
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def diagonal_predictions(prediction_matrix, num_experts: int):
    if prediction_matrix is None:
        return None
    out = []
    for expert_idx in range(num_experts):
        try:
            out.append(prediction_matrix[expert_idx][expert_idx])
        except Exception:
            out.append("")
    return out


def convert_item(item: dict, num_experts: int) -> dict:
    loss_matrix = item["loss_matrix"].float()
    if tuple(loss_matrix.shape) != (num_experts, num_experts):
        raise ValueError(f"Unexpected loss_matrix shape={tuple(loss_matrix.shape)} num_experts={num_experts}")

    correct_matrix = item.get("correct_matrix")
    if correct_matrix is None:
        correct_matrix = torch.zeros_like(loss_matrix, dtype=torch.bool)
    else:
        correct_matrix = correct_matrix.bool()

    loss_vector = loss_matrix.diagonal().clone()
    correct_vector = correct_matrix.diagonal().clone()
    expert_label = int(loss_vector.argmin().item())

    out = dict(item)
    out["loss_vector"] = loss_vector
    out["correct_vector"] = correct_vector
    out["prediction_vector"] = diagonal_predictions(item.get("prediction_matrix"), num_experts)
    out["expert_label"] = expert_label
    out["diag_source_pair_label"] = int(expert_label * num_experts + expert_label)

    for key in (
        "loss_matrix",
        "correct_matrix",
        "prediction_matrix",
        "pair_label",
        "first_label",
        "mid_label",
        "option_prob_matrix",
        "has_correct_matrix",
    ):
        out.pop(key, None)
    return out


def convert_split(src_root: Path, dst_root: Path, split: str):
    src_split = src_root / split
    dst_split = dst_root / split
    manifest_path = src_split / "manifest.json"
    manifest = load_json(manifest_path)

    if manifest.get("route_space") != "pair":
        raise ValueError(f"Expected route_space=pair in {manifest_path}")

    expert_names = list(manifest["expert_names"])
    num_experts = len(expert_names)
    dst_split.mkdir(parents=True, exist_ok=True)

    total = 0
    files = []
    for filename in manifest["files"]:
        payload = torch.load(src_split / filename, map_location="cpu")
        converted = []
        for item in payload["items"]:
            converted.append(convert_item(item, num_experts))
        out_payload = dict(payload)
        out_payload["items"] = converted
        torch.save(out_payload, dst_split / filename)
        files.append(filename)
        total += len(converted)

    out_manifest = dict(manifest)
    out_manifest.update(
        {
            "num_items": total,
            "files": files,
            "supervision_type": "cached_diag_single_loss_vector_from_pair_matrix",
            "route_space": "single_all_layers",
            "diag_source_route_space": "pair",
            "diag_source_cache_root": str(src_root),
            "has_correct_matrix": False,
            "has_correct_vector": True,
            "has_prediction_matrix": False,
            "has_prediction_vector": True,
        }
    )
    save_json(out_manifest, dst_split / "manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_cache", required=True)
    parser.add_argument("--output_cache", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    src_root = Path(args.source_cache)
    dst_root = Path(args.output_cache)
    if not (src_root / "train" / "manifest.json").exists() or not (src_root / "validation" / "manifest.json").exists():
        raise FileNotFoundError(f"Missing train/validation manifests under {src_root}")

    if dst_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output cache exists: {dst_root}")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "validation"):
        convert_split(src_root, dst_root, split)

    cache_config = src_root / "cache_config.json"
    if cache_config.exists():
        copied = load_json(cache_config)
        copied["route_space"] = "single_all_layers"
        copied["supervision_type"] = "cached_diag_single_loss_vector_from_pair_matrix"
        copied["diag_source_cache_root"] = str(src_root)
        save_json(copied, dst_root / "cache_config.json")

    print(f"[DONE] diag cache: {dst_root}")


if __name__ == "__main__":
    main()
