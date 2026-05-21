import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from mmengine.config import Config

from opencompass.models.base import LMTemplateParser
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.utils import build_dataset_from_cfg
from opencompass.utils.datasets import get_data_path


os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


HF_DATASETS_ROOT = Path("/home/u9472191/.cache/huggingface/datasets")
HF_MODULES_ROOT = Path("/home/u9472191/.cache/huggingface/modules")
WORKSPACE_ROOT = Path("/home/u9472191/opencompass")


DATASET_CONFIGS = [
    ("boolq", "opencompass/configs/datasets/SuperGLUE_BoolQ/SuperGLUE_BoolQ_gen.py"),
    ("commonsenseqa", "opencompass/configs/datasets/commonsenseqa/commonsenseqa_gen.py"),
    ("hellaswag", "opencompass/configs/datasets/hellaswag/hellaswag_gen.py"),
    ("iwslt2017", "opencompass/configs/datasets/iwslt2017/iwslt2017_gen_sft_prompt.py"),
    ("medmcqa", "opencompass/configs/datasets/medmcqa/medmcqa_gen_sft_prompt.py"),
    ("piqa", "opencompass/configs/datasets/piqa/piqa_gen.py"),
    ("race", "opencompass/configs/datasets/race/race_gen_sft_prompt.py"),
    ("rte", "opencompass/configs/datasets/SuperGLUE_RTE/SuperGLUE_RTE_gen.py"),
    ("siqa", "opencompass/configs/datasets/siqa/siqa_gen.py"),
    ("squad2", "opencompass/configs/datasets/squad20/squad20_gen_sft_prompt.py"),
    ("sst2", "opencompass/configs/datasets/glue/sst2_gen.py"),
]


def newest_matching_dir(pattern: str) -> str:
    matches = sorted(HF_DATASETS_ROOT.glob(pattern), key=lambda p: p.stat().st_mtime, reverse=True)
    return str(matches[0]) if matches else ""


def abs_path_str(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((WORKSPACE_ROOT / path).resolve())


def resolve_dataset_source_paths(task: str, dataset_cfg: Dict) -> Dict:
    raw_path = dataset_cfg.get("path", "")
    dataset_type = dataset_cfg.get("type")
    type_name = getattr(dataset_type, "__name__", str(dataset_type))
    reader_cfg = dataset_cfg.get("reader_cfg", {})
    source = {
        "dataset_type": type_name,
        "config_path_value": raw_path,
        "reader_train_split": reader_cfg.get("train_split", "train"),
        "reader_test_split": reader_cfg.get("test_split", "test"),
        "resolved_paths": [],
        "effective_sampling_source": "",
        "notes": [],
    }

    if raw_path.startswith("/"):
        source["resolved_paths"] = [raw_path]
        source["effective_sampling_source"] = "single dataset file mapped to dataset.train"
        return source

    if task == "boolq":
        resolved = get_data_path(raw_path)
        source["resolved_paths"] = [resolved]
        source["effective_sampling_source"] = "single dataset file mapped to dataset.train"
        source["notes"].append("BoolQDatasetV2 loads one jsonl file; BaseDataset maps it to both train/test.")
        return source

    if task == "piqa":
        resolved = get_data_path(raw_path)
        source["resolved_paths"] = [
            abs_path_str(str(Path(resolved) / "train.jsonl")),
            abs_path_str(str(Path(resolved) / "train-labels.lst")),
            abs_path_str(str(Path(resolved) / "dev.jsonl")),
            abs_path_str(str(Path(resolved) / "dev-labels.lst")),
        ]
        source["effective_sampling_source"] = "dataset.train -> local train split"
        return source

    if task == "siqa":
        resolved = get_data_path(raw_path)
        source["resolved_paths"] = [
            abs_path_str(str(Path(resolved) / "train.jsonl")),
            abs_path_str(str(Path(resolved) / "train-labels.lst")),
            abs_path_str(str(Path(resolved) / "dev.jsonl")),
            abs_path_str(str(Path(resolved) / "dev-labels.lst")),
        ]
        source["effective_sampling_source"] = "dataset.train -> local train split"
        return source

    if task == "race":
        resolved = get_data_path(raw_path)
        dataset_name = dataset_cfg.get("name", "")
        source["resolved_paths"] = [
            str(Path(resolved) / "validation" / f"{dataset_name}.jsonl"),
            str(Path(resolved) / "test" / f"{dataset_name}.jsonl"),
        ]
        source["effective_sampling_source"] = "dataset.train -> reader_cfg.train_split=validation"
        return source

    if task == "squad2":
        resolved = get_data_path(raw_path, local_mode=True)
        source["resolved_paths"] = [resolved]
        source["effective_sampling_source"] = "single dataset file mapped to dataset.train"
        return source

    if task == "iwslt2017":
        dataset_name = dataset_cfg.get("name", "")
        cache_dir = newest_matching_dir(f"{raw_path}/{dataset_name}/*/*")
        module_dir = HF_MODULES_ROOT / "datasets_modules" / "datasets" / raw_path
        source["resolved_paths"] = [cache_dir, str(module_dir)]
        source["effective_sampling_source"] = "dataset.train -> reader_cfg.train_split=validation"
        source["notes"].append("HF arrow cache dir plus cached dataset module dir.")
        return source

    if task == "medmcqa":
        cache_dir = newest_matching_dir("openlifescienceai___medmcqa/default/*/*")
        source["resolved_paths"] = [cache_dir]
        source["effective_sampling_source"] = "single dataset loaded from HF validation split, then mapped to dataset.train"
        source["notes"].append("MedmcqaDataset.load hardcodes split='validation'.")
        return source

    if task == "sst2":
        dataset_name = dataset_cfg.get("name", "")
        cache_dir = newest_matching_dir(f"{raw_path}/{dataset_name}/*/*")
        source["resolved_paths"] = [cache_dir]
        source["effective_sampling_source"] = "dataset.train -> HF train split"
        return source

    source["notes"].append("No special resolver matched.")
    return source


def summarize_source_details(source_details: Dict[str, Dict]) -> str:
    parts = []
    for abbr, detail in source_details.items():
        split = detail.get("effective_sampling_source", "")
        paths = detail.get("resolved_paths", [])
        preview = paths[0] if paths else detail.get("config_path_value", "")
        parts.append(f"{abbr}:{split}:{preview}")
    return " | ".join(parts)


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def write_jsonl(path: str, rows: Iterable[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def dataset_cfgs_from_file(config_path: str) -> List[Dict]:
    cfg = Config.fromfile(config_path)
    output = []
    for key, value in cfg.items():
        if key.endswith("_datasets") and isinstance(value, list):
            output.extend(value)
    if not output and "datasets" in cfg:
        output.extend(cfg.datasets)
    if not output:
        raise ValueError(f"No *_datasets list found in {config_path}")
    return output


def normalize_target(task: str, row: Dict, output_column: str):
    if task == "siqa":
        label = row.get("label")
        if label is not None:
            return str(label).strip()
        ref = row.get(output_column)
        if isinstance(ref, dict) and "label" in ref:
            return "ABC"[int(ref["label"])]
    if task == "medmcqa":
        value = row.get(output_column)
        if isinstance(value, int):
            return "ABCD"[value]
        return str(value).strip().upper()[:1]
    if task == "sst2":
        return str(row.get(output_column)).strip()
    value = row.get(output_column)
    if value is None and "label" in row:
        value = row["label"]
    return str(value).strip()


def build_prompt_generator(dataset_cfg: Dict, selected_indices: Sequence[int]):
    cfg = copy.deepcopy(dataset_cfg)
    dataset = build_dataset_from_cfg(cfg)
    train_ds = dataset.train
    selected_ds = train_ds.select(list(selected_indices))
    dataset.reader.dataset["test"] = selected_ds

    infer_cfg = cfg["infer_cfg"]
    ice_template = None
    if "ice_template" in infer_cfg:
        ice_template = ICL_PROMPT_TEMPLATES.build(infer_cfg["ice_template"])
    prompt_template = None
    if "prompt_template" in infer_cfg:
        prompt_template = ICL_PROMPT_TEMPLATES.build(infer_cfg["prompt_template"])

    retriever_cfg = copy.deepcopy(infer_cfg["retriever"])
    retriever_cfg["dataset"] = dataset
    retriever = ICL_RETRIEVERS.build(retriever_cfg)
    ice_idx_list = retriever.retrieve()
    parser = LMTemplateParser(meta_template=None)

    prompts = []
    rows = []
    for local_idx, ice_idx in enumerate(ice_idx_list):
        ice = retriever.generate_ice(ice_idx, ice_template=ice_template)
        prompt_obj = retriever.generate_prompt_for_generate_task(
            local_idx,
            ice,
            gen_field_replace_token=infer_cfg["inferencer"].get("gen_field_replace_token", ""),
            ice_template=ice_template,
            prompt_template=prompt_template,
        )
        prompts.append(parser.parse_template(prompt_obj, mode="gen"))
        rows.append(selected_ds[local_idx])
    return prompts, rows, len(train_ds), cfg.get("abbr", "dataset"), dataset.reader.output_column


def sample_candidates(
    candidates: Sequence[Tuple[int, int]],
    count: int,
    rng: random.Random,
    excluded: set,
) -> List[Tuple[int, int]]:
    available = [item for item in candidates if item not in excluded]
    shuffled = list(available)
    rng.shuffle(shuffled)
    return shuffled[: min(count, len(shuffled))]


def fixed_retriever_exclusions(dataset_cfg: Dict) -> set:
    retriever = dataset_cfg.get("infer_cfg", {}).get("retriever", {})
    fixed = retriever.get("fix_id_list", [])
    return set(int(idx) for idx in fixed)


def build_task(task: str, config_path: str, train_count: int, val_count: int, seed: int):
    dataset_cfgs = dataset_cfgs_from_file(config_path)
    train_lengths = []
    exclusions_by_cfg = []
    candidates = []
    source_details = {}
    for cfg_idx, dataset_cfg in enumerate(dataset_cfgs):
        dataset = build_dataset_from_cfg(copy.deepcopy(dataset_cfg))
        n = len(dataset.train)
        train_lengths.append(n)
        excluded_indices = fixed_retriever_exclusions(dataset_cfg)
        exclusions_by_cfg.append(excluded_indices)
        abbr = dataset_cfg.get("abbr", f"dataset_{cfg_idx}")
        source_details[abbr] = {
            "cfg_idx": cfg_idx,
            **resolve_dataset_source_paths(task, dataset_cfg),
        }
        for row_idx in range(n):
            if row_idx in excluded_indices:
                continue
            candidates.append((cfg_idx, row_idx))

    rng_train = random.Random(seed + 1009)
    rng_val = random.Random(seed + 2003)
    train_pairs = sample_candidates(candidates, train_count, rng_train, excluded=set())
    val_pairs = sample_candidates(candidates, val_count, rng_val, excluded=set(train_pairs))

    def materialize(split: str, pairs: Sequence[Tuple[int, int]]) -> List[Dict]:
        grouped: Dict[int, List[int]] = defaultdict(list)
        for cfg_idx, row_idx in pairs:
            grouped[cfg_idx].append(row_idx)

        generated = {}
        for cfg_idx, indices in grouped.items():
            prompts, source_rows, _n, abbr, output_column = build_prompt_generator(dataset_cfgs[cfg_idx], indices)
            for row_idx, prompt, source_row in zip(indices, prompts, source_rows):
                target = normalize_target(task, source_row, output_column)
                generated[(cfg_idx, row_idx)] = {
                    "_sample_id": f"{task}:train:{abbr}:{row_idx}:{sha1_text(prompt + str(target))[:16]}",
                    "_source_config": config_path,
                    "_opencompass_abbr": abbr,
                    "_source_index": int(row_idx),
                    "target": target,
                    "text": prompt,
                    "source_text": prompt,
                    "label": task,
                }
        return [generated[pair] for pair in pairs]

    return {
        "train_rows": materialize("train", train_pairs),
        "validation_rows": materialize("validation", val_pairs),
        "metadata": {
            "config_path": config_path,
            "dataset_abbrs": [cfg.get("abbr", "dataset") for cfg in dataset_cfgs],
            "source_train_rows_by_abbr": {
                cfg.get("abbr", f"dataset_{idx}"): train_lengths[idx]
                for idx, cfg in enumerate(dataset_cfgs)
            },
            "source_details": source_details,
            "fixed_retriever_excluded_indices_by_abbr": {
                cfg.get("abbr", f"dataset_{idx}"): sorted(exclusions_by_cfg[idx])
                for idx, cfg in enumerate(dataset_cfgs)
                if exclusions_by_cfg[idx]
            },
            "written_train_rows": len(train_pairs),
            "written_validation_rows": len(val_pairs),
            "train_sample_ids": [],
            "validation_sample_ids": [],
            "effective_sampling_source": "dataset.train from OpenCompass config",
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Build router_train_datasets from final OpenCompass eval dataset configs."
    )
    parser.add_argument("--output_root", default="router_train_datasets_opencompass_eval_0520")
    parser.add_argument("--train_samples", type=int, default=500)
    parser.add_argument("--val_samples", type=int, default=250)
    parser.add_argument("--seed", type=int, default=520)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if os.path.exists(args.output_root):
        if not args.overwrite:
            raise FileExistsError(f"Output root exists: {args.output_root}. Use --overwrite.")
        shutil.rmtree(args.output_root)
    os.makedirs(args.output_root, exist_ok=True)

    summary = {
        "output_root": args.output_root,
        "train_samples_per_task": args.train_samples,
        "val_samples_per_task": args.val_samples,
        "seed": args.seed,
        "note": (
            "Rows are sampled from dataset.train exposed by the OpenCompass config path."
        ),
        "tasks": {},
    }

    for task_idx, (task, config_path) in enumerate(DATASET_CONFIGS):
        result = build_task(
            task=task,
            config_path=config_path,
            train_count=args.train_samples,
            val_count=args.val_samples,
            seed=args.seed + task_idx * 7919,
        )
        train_rows = result["train_rows"]
        val_rows = result["validation_rows"]
        write_jsonl(os.path.join(args.output_root, task, "train.jsonl"), train_rows)
        write_jsonl(os.path.join(args.output_root, task, "validation.jsonl"), val_rows)

        meta = result["metadata"]
        meta["train_sample_ids"] = [row["_sample_id"] for row in train_rows[:5]]
        meta["validation_sample_ids"] = [row["_sample_id"] for row in val_rows[:5]]
        summary["tasks"][task] = meta
        source_trace = summarize_source_details(meta.get("source_details", {}))
        print(
            f"[OK] task={task} train={len(train_rows)} validation={len(val_rows)} "
            f"source={source_trace}",
            flush=True,
        )

    with open(os.path.join(args.output_root, "alignment_meta.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
