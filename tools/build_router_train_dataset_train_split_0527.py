import argparse
import copy
import hashlib
import json
import random
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Iterable, List, Sequence, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from mmengine.config import Config

from opencompass.models.base import LMTemplateParser
from opencompass.openicl import DatasetReader
from opencompass.registry import ICL_PROMPT_TEMPLATES, ICL_RETRIEVERS
from opencompass.utils.datasets import get_data_path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
HF_DATASETS_ROOT = Path.home() / ".cache" / "huggingface" / "datasets"

DATASET_CONFIGS = {
    "boolq": "opencompass/configs/datasets/SuperGLUE_BoolQ/SuperGLUE_BoolQ_gen.py",
    "iwslt2017": "opencompass/configs/datasets/iwslt2017/iwslt2017_gen_sft_prompt.py",
    "medmcqa": "opencompass/configs/datasets/medmcqa/medmcqa_gen_sft_prompt.py",
    "openbookqa": "opencompass/configs/datasets/obqa/obqa_gen.py",
    "arc_c": "opencompass/configs/datasets/ARC_c/ARC_c_gen.py",
    "piqa": "opencompass/configs/datasets/piqa/piqa_gen.py",
    "race": "opencompass/configs/datasets/race/race_gen_sft_prompt.py",
    "rte": "opencompass/configs/datasets/SuperGLUE_RTE/SuperGLUE_RTE_gen.py",
    "siqa": "opencompass/configs/datasets/siqa/siqa_gen.py",
    "squad2": "opencompass/configs/datasets/squad20/squad20_gen_sft_prompt.py",
    "sst2": "opencompass/configs/datasets/glue/sst2_gen.py",
}
DEFAULT_TASKS = list(DATASET_CONFIGS)


def sha1_text(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()


def read_jsonl(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def arrow_path(pattern: str, split_name: str) -> Path | None:
    matches = sorted(HF_DATASETS_ROOT.glob(pattern))
    if not matches:
        return None
    path = matches[-1] / split_name
    return path if path.exists() else None


def dataset_paths(dataset: Dataset) -> List[str]:
    return sorted({
        item["filename"] for item in getattr(dataset, "cache_files", [])
        if item.get("filename")
    })


def load_cached_or_hf(
    task: str,
    split: str,
    cached_path: Path | None,
    candidates: Sequence[Tuple[str, str | None]],
    allow_download: bool,
) -> Tuple[Dataset, Dict]:
    if cached_path is not None:
        dataset = Dataset.from_file(str(cached_path))
        return dataset, {
            "raw_split": split,
            "resolved_paths": [str(cached_path)],
            "row_count": len(dataset),
            "loader": "cached_arrow",
        }
    if not allow_download:
        raise FileNotFoundError(
            f"{task}: missing cached raw {split} split. Rerun with --allow_download "
            "or provide the corresponding local train path option."
        )
    errors = []
    for path, name in candidates:
        try:
            dataset = load_dataset(path, name, split=split, trust_remote_code=True)
            return dataset, {
                "raw_split": split,
                "resolved_paths": dataset_paths(dataset) or [f"hf://{path}/{name or ''}:{split}"],
                "row_count": len(dataset),
                "loader": f"huggingface:{path}/{name or ''}",
            }
        except Exception as exc:  # pragma: no cover - depends on installed dataset access.
            errors.append(f"{path}/{name or ''}: {exc}")
    raise RuntimeError(f"{task}: failed to load raw {split} split: {' | '.join(errors)}")


def local_jsonl_dataset(path: Path, split: str, loader: str) -> Tuple[Dataset, Dict]:
    if not path.exists():
        raise FileNotFoundError(str(path))
    dataset = Dataset.from_list(read_jsonl(path))
    return dataset, {
        "raw_split": split,
        "resolved_paths": [str(path.resolve())],
        "row_count": len(dataset),
        "loader": loader,
    }


def load_squad_json(path: Path, split: str) -> Tuple[Dataset, Dict]:
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)["data"]
    rows = []
    for article in raw:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                answers = [item["text"] for item in qa.get("answers", [])]
                if qa.get("is_impossible", False) or not answers:
                    answers = ["impossible to answer"]
                rows.append({
                    "context": paragraph["context"],
                    "question": qa["question"],
                    "answers": list(dict.fromkeys(answers)),
                })
    dataset = Dataset.from_list(rows)
    return dataset, {
        "raw_split": split,
        "resolved_paths": [str(path.resolve())],
        "row_count": len(dataset),
        "loader": "local_squad_json",
    }


def map_boolq(dataset: Dataset) -> Dataset:
    def convert(row):
        value = row["label"]
        if isinstance(value, str):
            value = value.lower() in {"true", "1", "yes"}
        return {**row, "label": "A" if bool(value) else "B"}

    return Dataset.from_list([convert(row) for row in dataset])


def map_rte(dataset: Dataset) -> Dataset:
    def convert(row):
        value = row["label"]
        if isinstance(value, str):
            is_entailment = value == "entailment"
        else:
            is_entailment = int(value) == 0
        return {**row, "label": "A" if is_entailment else "B"}

    return Dataset.from_list([convert(row) for row in dataset])


def map_medmcqa(dataset: Dataset) -> Dataset:
    def convert(row):
        question = re.sub(r"^[ \t]*[A-D][\.\:][^\n]*\n?", "", row["question"],
                          flags=re.MULTILINE)
        question = re.sub(r"^Question:\s*", "", question, flags=re.IGNORECASE).strip()
        return {**row, "question": question}

    return Dataset.from_list([convert(row) for row in dataset])


def map_iwslt(dataset: Dataset) -> Dataset:
    return Dataset.from_list([row["translation"] for row in dataset])


def map_openbookqa(dataset: Dataset) -> Dataset:
    rows = []
    for row in dataset:
        if "question" in row:
            question = row["question"]
            choices = question["choices"]
            stem = question["stem"]
        else:
            choices = row["choices"]
            stem = row["question_stem"]
        labels = choices["label"]
        texts = choices["text"]
        if len(texts) != 4:
            continue
        by_label = dict(zip(labels, texts))
        rows.append({
            "question_stem": stem,
            "A": by_label.get("A", texts[0]),
            "B": by_label.get("B", texts[1]),
            "C": by_label.get("C", texts[2]),
            "D": by_label.get("D", texts[3]),
            "answerKey": row["answerKey"],
        })
    return Dataset.from_list(rows)


def map_arc(dataset: Dataset) -> Dataset:
    rows = []
    for row in dataset:
        question = row["question"]
        if isinstance(question, dict):
            stem = question["stem"]
            choices = question["choices"]
        else:
            stem = question
            choices = row["choices"]
        labels = choices["label"]
        texts = choices["text"]
        if len(texts) != 4:
            continue
        answer_key = "ABCD"[labels.index(row["answerKey"])]
        rows.append({
            "question": stem,
            "textA": texts[0],
            "textB": texts[1],
            "textC": texts[2],
            "textD": texts[3],
            "answerKey": answer_key,
        })
    return Dataset.from_list(rows)


def load_local_piqa(root: Path, split: str) -> Tuple[Dataset, Dict]:
    stem = "train" if split == "train" else "dev"
    data_path = root / f"{stem}.jsonl"
    label_path = root / f"{stem}-labels.lst"
    rows = []
    with data_path.open("r", encoding="utf-8") as data_f, label_path.open("r", encoding="utf-8") as label_f:
        for raw, label in zip(data_f, label_f):
            row = json.loads(raw)
            row.pop("id", None)
            row["answer"] = "AB"[int(label.strip())]
            rows.append(row)
    dataset = Dataset.from_list(rows)
    return dataset, {
        "raw_split": split,
        "resolved_paths": [str(data_path.resolve()), str(label_path.resolve())],
        "row_count": len(dataset),
        "loader": "local_piqa",
    }


def load_local_siqa(root: Path, split: str) -> Tuple[Dataset, Dict]:
    stem = "train" if split == "train" else "dev"
    data_path = root / f"{stem}.jsonl"
    label_path = root / f"{stem}-labels.lst"
    rows = []
    with data_path.open("r", encoding="utf-8") as data_f, label_path.open("r", encoding="utf-8") as label_f:
        for raw, label in zip(data_f, label_f):
            row = json.loads(raw)
            answer = int(label.strip())
            row["label"] = " ABC"[answer]
            row["all_labels"] = {
                "candidates": [
                    [f"A. {row['answerA']}", "A", row["answerA"]],
                    [f"B. {row['answerB']}", "B", row["answerB"]],
                    [f"C. {row['answerC']}", "C", row["answerC"]],
                ],
                "label": answer - 1,
            }
            rows.append(row)
    dataset = Dataset.from_list(rows)
    return dataset, {
        "raw_split": split,
        "resolved_paths": [str(data_path.resolve()), str(label_path.resolve())],
        "row_count": len(dataset),
        "loader": "local_siqa",
    }


def load_local_race(root: Path, split: str, name: str) -> Tuple[Dataset, Dict]:
    path = root / split / f"{name}.jsonl"
    rows = []
    for item in read_jsonl(path):
        rows.append({
            "article": item["article"],
            "question": item["question"],
            "A": item["options"][0],
            "B": item["options"][1],
            "C": item["options"][2],
            "D": item["options"][3],
            "answer": item["answer"],
        })
    dataset = Dataset.from_list(rows)
    return dataset, {
        "raw_split": split,
        "resolved_paths": [str(path.resolve())],
        "row_count": len(dataset),
        "loader": "local_race",
    }


def load_task_sources(task: str, args) -> Tuple[Dict[str, Tuple[Dataset, Dict]], Dict[str, Dict]]:
    if task == "boolq":
        eval_ds, eval_info = local_jsonl_dataset(
            Path(get_data_path("opencompass/boolq")), "validation", "opencompass_eval_local_jsonl"
        )
        train_path = Path(args.boolq_train_path) if args.boolq_train_path else None
        if train_path and train_path.exists():
            train_ds, train_info = local_jsonl_dataset(train_path, "train", "local_boolq")
        else:
            train_ds, train_info = load_cached_or_hf(
                task, "train",
                arrow_path("super_glue/boolq/*/*", "super_glue-train.arrow"),
                [("aps/super_glue", "boolq"), ("super_glue", "boolq")],
                args.allow_download,
            )
        return {"BoolQ": (map_boolq(train_ds), train_info)}, {"BoolQ": {**eval_info, "row_count": len(eval_ds)}}

    if task == "rte":
        eval_path = Path(args.rte_eval_path)
        eval_ds, eval_info = local_jsonl_dataset(eval_path, "validation", "opencompass_eval_local_jsonl")
        train_ds, train_info = load_cached_or_hf(
            task, "train",
            arrow_path("super_glue/rte/*/*", "super_glue-train.arrow"),
            [("aps/super_glue", "rte"), ("super_glue", "rte")],
            args.allow_download,
        )
        return {"RTE": (map_rte(train_ds), train_info)}, {"RTE": eval_info}

    if task == "piqa":
        root = Path(get_data_path("opencompass/piqa"))
        train = load_local_piqa(root, "train")
        evaluation = load_local_piqa(root, "validation")
        return {"piqa": train}, {"piqa": evaluation[1]}

    if task == "siqa":
        root = Path(get_data_path("opencompass/siqa"))
        train = load_local_siqa(root, "train")
        evaluation = load_local_siqa(root, "validation")
        return {"siqa": train}, {"siqa": evaluation[1]}

    if task == "sst2":
        train_ds, train_info = load_cached_or_hf(
            task, "train", arrow_path("glue/sst2/*/*", "glue-train.arrow"),
            [("glue", "sst2")], args.allow_download
        )
        eval_ds, eval_info = load_cached_or_hf(
            task, "validation", arrow_path("glue/sst2/*/*", "glue-validation.arrow"),
            [("glue", "sst2")], args.allow_download
        )
        convert = lambda row: {**row, "label": "negative" if int(row["label"]) == 0 else "positive"}
        return {"sst2": (Dataset.from_list([convert(row) for row in train_ds]), train_info)}, {"sst2": {**eval_info, "row_count": len(eval_ds)}}

    if task == "medmcqa":
        train_ds, train_info = load_cached_or_hf(
            task, "train",
            arrow_path("openlifescienceai___medmcqa/default/*/*", "medmcqa-train.arrow"),
            [("openlifescienceai/medmcqa", None)], args.allow_download
        )
        eval_ds, eval_info = load_cached_or_hf(
            task, "validation",
            arrow_path("openlifescienceai___medmcqa/default/*/*", "medmcqa-validation.arrow"),
            [("openlifescienceai/medmcqa", None)], args.allow_download
        )
        return {"medmcqa": (map_medmcqa(train_ds), train_info)}, {"medmcqa": {**eval_info, "row_count": len(eval_ds)}}

    if task == "openbookqa":
        eval_ds, eval_info = local_jsonl_dataset(
            Path(get_data_path("./data/openbookqa/Main/test.jsonl", local_mode=True)),
            "test",
            "opencompass_eval_local_jsonl",
        )
        train_ds, train_info = load_cached_or_hf(
            task, "train",
            arrow_path("allenai___openbookqa/main/*/*", "openbookqa-train.arrow"),
            [("allenai/openbookqa", "main")], args.allow_download,
        )
        return {"openbookqa": (map_openbookqa(train_ds), train_info)}, {"openbookqa": {**eval_info, "row_count": len(eval_ds)}}

    if task == "arc_c":
        eval_ds, eval_info = local_jsonl_dataset(
            Path(get_data_path("opencompass/ai2_arc-dev")),
            "dev",
            "opencompass_eval_local_jsonl",
        )
        train_path = Path(args.arc_c_train_path)
        if train_path.exists():
            train_ds, train_info = local_jsonl_dataset(train_path, "train", "local_arc_c_jsonl")
        else:
            train_ds, train_info = load_cached_or_hf(
                task, "train",
                arrow_path("allenai___ai2_arc/ARC-Challenge/*/*", "ai2_arc-train.arrow"),
                [("allenai/ai2_arc", "ARC-Challenge")], args.allow_download,
            )
        return {"ARC-c": (map_arc(train_ds), train_info)}, {"ARC-c": {**eval_info, "row_count": len(eval_ds)}}

    if task == "iwslt2017":
        train_ds, train_info = load_cached_or_hf(
            task, "train",
            arrow_path("iwslt2017/iwslt2017-en-fr/*/*", "iwslt2017-train.arrow"),
            [("iwslt2017", "iwslt2017-en-fr")], args.allow_download
        )
        eval_ds, eval_info = load_cached_or_hf(
            task, "test",
            arrow_path("iwslt2017/iwslt2017-en-fr/*/*", "iwslt2017-test.arrow"),
            [("iwslt2017", "iwslt2017-en-fr")], args.allow_download
        )
        return {"dataset": (map_iwslt(train_ds), train_info)}, {"dataset": {**eval_info, "row_count": len(eval_ds)}}

    if task == "squad2":
        eval_ds, eval_info = load_squad_json(Path(get_data_path("./data/SQuAD2.0/dev-v2.0.json", local_mode=True)), "validation")
        train_path = Path(args.squad_train_path)
        if train_path.exists():
            train_ds, train_info = load_squad_json(train_path, "train")
        else:
            train_ds, train_info = load_cached_or_hf(
                task, "train", None,
                [("rajpurkar/squad_v2", None), ("squad_v2", None)], args.allow_download
            )
            converted = []
            for row in train_ds:
                answers = row.get("answers", {}).get("text", [])
                converted.append({
                    "context": row["context"],
                    "question": row["question"],
                    "answers": list(dict.fromkeys(answers)) or ["impossible to answer"],
                })
            train_ds = Dataset.from_list(converted)
        return {"squad2.0": (train_ds, train_info)}, {"squad2.0": {**eval_info, "row_count": len(eval_ds)}}

    if task == "race":
        root = Path(args.race_train_root)
        eval_root = Path(get_data_path("opencompass/race"))
        train_sources = {}
        eval_sources = {}
        for name, abbr in (("middle", "race-middle"), ("high", "race-high")):
            eval_sources[abbr] = load_local_race(eval_root, "test", name)[1]
            if (root / "train" / f"{name}.jsonl").exists():
                train_sources[abbr] = load_local_race(root, "train", name)
                continue
            train_ds, train_info = load_cached_or_hf(
                f"race-{name}", "train", None,
                [("ehovy/race", name), ("race", name)], args.allow_download
            )
            def convert(row):
                return {
                    "article": row["article"], "question": row["question"],
                    "A": row["options"][0], "B": row["options"][1],
                    "C": row["options"][2], "D": row["options"][3],
                    "answer": row["answer"],
                }
            train_sources[abbr] = (Dataset.from_list([convert(row) for row in train_ds]), train_info)
        return train_sources, eval_sources
    raise ValueError(f"Unsupported task: {task}")


def dataset_cfgs_from_file(config_path: str) -> List[Dict]:
    cfg = Config.fromfile(config_path)
    result = []
    for key, value in cfg.items():
        if key.endswith("_datasets") and isinstance(value, list):
            result.extend(value)
    if not result:
        raise ValueError(f"No dataset list in {config_path}")
    return result


def render_prompts(dataset_cfg: Dict, rows: List[Dict]) -> List[str]:
    infer_cfg = dataset_cfg["infer_cfg"]
    retriever_type = infer_cfg["retriever"]["type"]
    retriever_name = getattr(retriever_type, "__name__", str(retriever_type))
    if "ZeroRetriever" not in retriever_name:
        raise ValueError(f"Only ZeroRetriever configs are supported, got {retriever_name}")
    source = Dataset.from_list(rows)
    reader_cfg = copy.deepcopy(dataset_cfg["reader_cfg"])
    wrapper = SimpleNamespace()
    render_views = DatasetDict({
        "train": source,
        "validation": source,
        "test": source,
    })
    wrapper.reader = DatasetReader(render_views, **reader_cfg)
    wrapper.train = wrapper.reader.dataset["train"]
    wrapper.test = wrapper.reader.dataset["test"]
    retriever_cfg = copy.deepcopy(infer_cfg["retriever"])
    retriever_cfg["dataset"] = wrapper
    retriever = ICL_RETRIEVERS.build(retriever_cfg)
    prompt_template = ICL_PROMPT_TEMPLATES.build(infer_cfg["prompt_template"])
    parser = LMTemplateParser(meta_template=None)
    prompts = []
    for idx, ice_idx in enumerate(retriever.retrieve()):
        prompt_obj = retriever.generate_prompt_for_generate_task(
            idx, retriever.generate_ice(ice_idx),
            gen_field_replace_token=infer_cfg["inferencer"].get("gen_field_replace_token", ""),
            prompt_template=prompt_template,
        )
        prompts.append(parser.parse_template(prompt_obj, mode="gen"))
    return prompts


def target_for(task: str, row: Dict) -> str:
    if task == "medmcqa":
        value = row["cop"]
        return "ABCD"[int(value)] if isinstance(value, int) else str(value).upper()[:1]
    if task == "squad2":
        return str(row["answers"][0])
    if task == "iwslt2017":
        return str(row["fr"])
    if task == "piqa":
        return str(row["answer"])
    if task == "race":
        return str(row["answer"])
    if task in {"openbookqa", "arc_c"}:
        return str(row["answerKey"])
    return str(row["label"])


def build_task(task: str, args, seed: int) -> Dict:
    config_path = DATASET_CONFIGS[task]
    config_by_abbr = {
        cfg.get("abbr", "dataset"): cfg for cfg in dataset_cfgs_from_file(config_path)
    }
    train_sources, eval_sources = load_task_sources(task, args)
    candidates = []
    for abbr, (dataset, _info) in train_sources.items():
        for idx in range(len(dataset)):
            candidates.append((abbr, idx))
    if len(candidates) < args.train_samples + args.val_samples:
        raise ValueError(f"{task}: only {len(candidates)} official-train rows available")
    rng = random.Random(seed)
    chosen = rng.sample(candidates, args.train_samples + args.val_samples)
    output = {}
    for split, pairs in (
        ("train", chosen[:args.train_samples]),
        ("validation", chosen[args.train_samples:]),
    ):
        grouped: Dict[str, List[int]] = {}
        for abbr, idx in pairs:
            grouped.setdefault(abbr, []).append(idx)
        generated = {}
        for abbr, indices in grouped.items():
            source_ds, source_info = train_sources[abbr]
            rows = [source_ds[int(idx)] for idx in indices]
            prompts = render_prompts(config_by_abbr[abbr], rows)
            for idx, row, prompt in zip(indices, rows, prompts):
                target = target_for(task, row)
                generated[(abbr, idx)] = {
                    "_sample_id": f"{task}:official_train:{abbr}:{idx}:{sha1_text(prompt + target)[:16]}",
                    "_source_config": config_path,
                    "_opencompass_abbr": abbr,
                    "_source_index": int(idx),
                    "_source_raw_split": "train",
                    "_source_paths": source_info["resolved_paths"],
                    "target": target,
                    "text": prompt,
                    "source_text": prompt,
                    "prompt_text": prompt,
                    "label": task,
                }
        output[split] = [generated[pair] for pair in pairs]
    provenance = {}
    for abbr, (_dataset, train_info) in train_sources.items():
        eval_info = eval_sources[abbr]
        provenance[abbr] = {
            "router_source": train_info,
            "opencompass_eval_source": eval_info,
            "overlap_check": {
                "router_raw_split": train_info["raw_split"],
                "eval_raw_split": eval_info["raw_split"],
                "raw_split_disjoint": train_info["raw_split"] != eval_info["raw_split"],
                "resolved_path_disjoint": not bool(
                    set(train_info["resolved_paths"]) & set(eval_info["resolved_paths"])
                ),
                "status": "pass",
            },
        }
        if train_info["raw_split"] == eval_info["raw_split"]:
            raise ValueError(f"{task}/{abbr}: router and eval use the same raw split")
        if set(train_info["resolved_paths"]) & set(eval_info["resolved_paths"]):
            raise ValueError(f"{task}/{abbr}: router and eval source paths overlap")
    return {
        "train_rows": output["train"],
        "validation_rows": output["validation"],
        "metadata": {
            "config_path": config_path,
            "source_policy": "router rows sampled only from raw train; OpenCompass eval provenance recorded separately",
            "sources_by_abbr": provenance,
            "written_train_rows": len(output["train"]),
            "written_validation_rows": len(output["validation"]),
            "train_sample_ids": [row["_sample_id"] for row in output["train"][:5]],
            "validation_sample_ids": [row["_sample_id"] for row in output["validation"][:5]],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build router datasets from raw training splits while preserving OpenCompass eval prompts."
    )
    parser.add_argument("--output_root", default="0527_router_train_dataset")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    parser.add_argument("--train_samples", type=int, default=200)
    parser.add_argument("--val_samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=527)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--boolq_train_path", default="")
    parser.add_argument("--rte_eval_path", default=str(WORKSPACE_ROOT / "data/SuperGLUE/RTE/val.jsonl"))
    parser.add_argument("--squad_train_path", default=str(Path.home() / ".cache/opencompass/data/SQuAD2.0/train-v2.0.json"))
    parser.add_argument("--race_train_root", default=str(Path.home() / ".cache/opencompass/data/race"))
    parser.add_argument("--arc_c_train_path", default=str(WORKSPACE_ROOT / "data/ARC/ARC-c/ARC-Challenge-Train.jsonl"))
    args = parser.parse_args()
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    unknown = sorted(set(tasks) - set(DATASET_CONFIGS))
    if unknown:
        raise ValueError(f"Unknown tasks: {unknown}")
    output_root = Path(args.output_root)
    if output_root.exists() and not (args.overwrite or args.append):
        raise FileExistsError(f"Output root exists: {output_root}. Use --overwrite.")
    output_root.mkdir(parents=True, exist_ok=True)
    meta_path = output_root / "alignment_meta.json"
    if args.append and meta_path.exists():
        with meta_path.open("r", encoding="utf-8") as f:
            summary = json.load(f)
    else:
        summary = {
            "output_root": str(output_root),
            "train_samples_per_task": args.train_samples,
            "val_samples_per_task": args.val_samples,
            "seed": args.seed,
            "tasks": {},
            "contract": {
                "router_source": "raw train split only",
                "eval_source": "actual OpenCompass eval raw split/path",
                "prompt_source": "OpenCompass dataset config infer_cfg.prompt_template",
                "overlap_requirement": "router raw split/path must be disjoint from eval raw split/path",
            },
        }
    for task_idx, task in enumerate(tasks):
        result = build_task(task, args, args.seed + task_idx * 7919)
        write_jsonl(output_root / task / "train.jsonl", result["train_rows"])
        write_jsonl(output_root / task / "validation.jsonl", result["validation_rows"])
        summary["tasks"][task] = result["metadata"]
        print(f"[OK] {task}: train={len(result['train_rows'])} validation={len(result['validation_rows'])}")
    with (output_root / "alignment_meta.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[OK] wrote {output_root / 'alignment_meta.json'}")


if __name__ == "__main__":
    main()
