import argparse
import json
import os
from typing import Dict, List

import torch


def save_json(obj: Dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def normalize_loss_matrix(loss_matrix: torch.Tensor, method: str) -> torch.Tensor:
    loss_matrix = loss_matrix.to(torch.float32)
    if method == "none":
        return loss_matrix
    if method != "sample_minmax":
        raise ValueError(f"Unsupported normalization method: {method}")

    flat = loss_matrix.view(-1)
    min_value = flat.min()
    max_value = flat.max()
    denom = (max_value - min_value).clamp_min(1e-6)
    return (loss_matrix - min_value) / denom


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_split(
    src_root: str,
    dst_root: str,
    split: str,
    method: str,
) -> int:
    src_split_dir = os.path.join(src_root, split)
    dst_split_dir = os.path.join(dst_root, split)
    os.makedirs(dst_split_dir, exist_ok=True)

    manifest = load_json(os.path.join(src_split_dir, "manifest.json"))
    normalized_items = 0
    for filename in manifest["files"]:
        src_path = os.path.join(src_split_dir, filename)
        dst_path = os.path.join(dst_split_dir, filename)
        payload = torch.load(src_path, map_location="cpu")
        items: List[Dict] = payload["items"]
        converted_items: List[Dict] = []
        for item in items:
            converted = dict(item)
            converted["loss_matrix"] = normalize_loss_matrix(
                item["loss_matrix"],
                method=method,
            )
            converted_items.append(converted)
            normalized_items += 1
        torch.save({"items": converted_items}, dst_path)

    updated_manifest = dict(manifest)
    updated_manifest["loss_normalization"] = method
    updated_manifest["loss_normalization_scope"] = "per_sample"
    save_json(updated_manifest, os.path.join(dst_split_dir, "manifest.json"))
    return normalized_items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_root", type=str, required=True)
    parser.add_argument("--dst_root", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="sample_minmax",
        choices=["none", "sample_minmax"],
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="train,validation",
        help="comma-separated splits to normalize",
    )
    args = parser.parse_args()

    os.makedirs(args.dst_root, exist_ok=True)
    split_names = [part.strip() for part in str(args.splits).split(",") if part.strip()]

    cache_config_path = os.path.join(args.src_root, "cache_config.json")
    if os.path.exists(cache_config_path):
        cache_config = load_json(cache_config_path)
        cache_config["loss_normalization"] = args.method
        cache_config["loss_normalization_scope"] = "per_sample"
        cache_config["source_feature_root"] = args.src_root
        save_json(cache_config, os.path.join(args.dst_root, "cache_config.json"))

    total_items = 0
    for split in split_names:
        split_items = normalize_split(
            src_root=args.src_root,
            dst_root=args.dst_root,
            split=split,
            method=args.method,
        )
        total_items += split_items
        print(
            f"[NORMALIZE] split={split} items={split_items} "
            f"method={args.method} out={os.path.join(args.dst_root, split)}",
            flush=True,
        )

    print(
        f"[DONE] src_root={args.src_root} dst_root={args.dst_root} "
        f"method={args.method} total_items={total_items}",
        flush=True,
    )


if __name__ == "__main__":
    main()
