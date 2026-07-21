import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import torch
from transformers import AutoTokenizer

from opencompass.models.router_moe_shared import (
    NULL_EXPERT_ID,
    set_all_experts,
    set_layer_range_expert,
)
from router_answer_supervision_core import (
    JointAnswerSupervisionRouterModel,
    build_lm_batch,
    compute_official_evaluator_sample_score,
)
from task_eval_specs import TASK_EVAL_SPECS, normalize_boolq_label
from opencompass.utils.text_postprocessors import first_capital_postprocess


DEFAULT_TASKS = [
    "boolq",
    "rte",
    "sst2",
    "race",
    "medmcqa",
    "piqa",
    "siqa",
    "iwslt2017",
    "squad2",
]
DEFAULT_EXPERTS = ["medmcqa", "race", "sst2"]


def parse_csv(raw: str | None, default: Sequence[str]) -> List[str]:
    if not raw:
        return list(default)
    return [item.strip() for item in str(raw).split(",") if item.strip()]


def write_json(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def discover_task_folders(data_root: Path) -> List[str]:
    return sorted(
        path.name for path in data_root.iterdir()
        if path.is_dir() and (path / "train.jsonl").exists()
    )


def read_rows(data_root: Path, tasks: Sequence[str], samples_per_task: int) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    for task in tasks:
        path = data_root / task / "train.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"Missing train split: {path}")
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
                if len(rows) >= samples_per_task:
                    break
        if len(rows) < samples_per_task:
            raise ValueError(
                f"Task {task!r} only contains {len(rows)} train rows; "
                f"requested {samples_per_task}."
            )
        for task_sample_idx, row in enumerate(rows):
            prompt = row.get("text") or row.get("source_text")
            target = row.get("target")
            if prompt is None or target is None:
                raise ValueError(f"Invalid row in {path}: keys={sorted(row)}")
            records.append(
                {
                    "sample_id": row.get("_sample_id", f"{task}:train:{task_sample_idx}"),
                    "task": task,
                    "task_sample_idx": task_sample_idx,
                    "source_config": row.get("_source_config"),
                    "opencompass_abbr": row.get("_opencompass_abbr"),
                    "source_index": row.get("_source_index"),
                    "prompt": str(prompt),
                    "target": str(target),
                    "pairs": {},
                }
            )
    return records


def pair_rows(expert_names: Sequence[str]) -> List[Dict[str, Any]]:
    pairs = []
    for first_idx, first_name in enumerate(expert_names):
        for mid_idx, mid_name in enumerate(expert_names):
            pairs.append(
                {
                    "pair_idx": first_idx * len(expert_names) + mid_idx,
                    "first_expert": first_name,
                    "mid_expert": mid_name,
                    "pair": f"{first_name}->{mid_name}",
                }
            )
    return pairs


def chunks(items: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[Sequence[Dict[str, Any]]]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


@torch.no_grad()
def generate_batch_for_pair(
    model: JointAnswerSupervisionRouterModel,
    tokenizer,
    records: Sequence[Dict[str, Any]],
    first_expert: str,
    mid_expert: str,
    max_llm_len: int,
    max_new_tokens: int,
) -> List[str]:
    batch = build_lm_batch(
        tokenizer=tokenizer,
        prompts=[record["prompt"] for record in records],
        targets=[""] * len(records),
        max_length=max_llm_len,
        add_eos_to_target=False,
    )
    device = next(model.parameters()).device
    prompt_input_ids = batch["prompt_input_ids"].to(device)
    prompt_attention_mask = batch["prompt_attention_mask"].to(device)

    set_all_experts(model.model, NULL_EXPERT_ID)
    set_layer_range_expert(
        model.model,
        model.first_layer_idx,
        model.middle_layer_idx - 1,
        model.task_to_expert_id[first_expert],
    )
    set_layer_range_expert(
        model.model,
        model.middle_layer_idx,
        model.num_layers - 1,
        model.task_to_expert_id[mid_expert],
    )
    outputs = model.model.generate(
        input_ids=prompt_input_ids,
        attention_mask=prompt_attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        temperature=None,
        top_p=None,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    set_all_experts(model.model, NULL_EXPERT_ID)
    new_tokens = outputs[:, prompt_input_ids.size(1) :]
    return [text.strip() for text in tokenizer.batch_decode(new_tokens, skip_special_tokens=True)]


def opencompass_config_score(prediction: str, record: Dict[str, Any]) -> float:
    # The public BoolQ config intentionally uses first_capital_postprocess.
    # The shared router-scoring helper uses first_option_postprocess instead,
    # so keep this report faithful to SuperGLUE_BoolQ_gen.py.
    if record["task"] == "boolq":
        processed = first_capital_postprocess(prediction)
        return float(processed == normalize_boolq_label(record["target"]))
    cost = compute_official_evaluator_sample_score(
        prediction=prediction,
        target=record["target"],
        task_name=record["task"],
        source_text=record["prompt"],
    )
    return 1.0 - float(cost)


def opencompass_metrics(prediction: str, record: Dict[str, Any]) -> Dict[str, Any]:
    score = opencompass_config_score(prediction, record)
    return {
        "prediction": prediction,
        "raw_exact_target": prediction.strip() == record["target"].strip(),
        "opencompass_config_score": score,
        "opencompass_config_full_credit": score >= 1.0 - 1e-8,
    }


def increment(stats: Dict[str, Any], pair_result: Dict[str, Any]) -> None:
    stats["count"] += 1
    stats["same_generation"] += int(pair_result["same_generation"])
    for token_key in ("4", "64"):
        metric = pair_result[f"max_new_tokens_{token_key}"]
        bucket = stats[f"max_new_tokens_{token_key}"]
        bucket["raw_exact_target"] += int(metric["raw_exact_target"])
        bucket["opencompass_config_full_credit"] += int(metric["opencompass_config_full_credit"])
        bucket["opencompass_config_score_sum"] += float(metric["opencompass_config_score"])


def new_stat_bucket() -> Dict[str, Any]:
    return {
        "count": 0,
        "same_generation": 0,
        "max_new_tokens_4": {
            "raw_exact_target": 0,
            "opencompass_config_full_credit": 0,
            "opencompass_config_score_sum": 0.0,
        },
        "max_new_tokens_64": {
            "raw_exact_target": 0,
            "opencompass_config_full_credit": 0,
            "opencompass_config_score_sum": 0.0,
        },
    }


def finalize_stats(stats: Dict[str, Any]) -> Dict[str, Any]:
    count = int(stats["count"])
    result = {
        "count": count,
        "same_generation": int(stats["same_generation"]),
        "same_generation_rate": float(stats["same_generation"] / count) if count else None,
    }
    for token_key in ("4", "64"):
        source = stats[f"max_new_tokens_{token_key}"]
        result[f"max_new_tokens_{token_key}"] = {
            "raw_exact_target": int(source["raw_exact_target"]),
            "raw_exact_target_rate": float(source["raw_exact_target"] / count) if count else None,
            "opencompass_config_full_credit": int(source["opencompass_config_full_credit"]),
            "opencompass_config_full_credit_rate": float(source["opencompass_config_full_credit"] / count) if count else None,
            "opencompass_config_mean_score": float(source["opencompass_config_score_sum"] / count) if count else None,
        }
    return result


def build_summary(
    records: Sequence[Dict[str, Any]],
    pairs: Sequence[Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    overall = new_stat_bucket()
    task_stats = defaultdict(new_stat_bucket)
    pair_stats = defaultdict(new_stat_bucket)
    task_pair_stats = defaultdict(new_stat_bucket)
    for record in records:
        task = record["task"]
        for pair in pairs:
            pair_name = pair["pair"]
            pair_result = record["pairs"][pair_name]
            increment(overall, pair_result)
            increment(task_stats[task], pair_result)
            increment(pair_stats[pair_name], pair_result)
            increment(task_pair_stats[(task, pair_name)], pair_result)
    return {
        "metadata": metadata,
        "overall": finalize_stats(overall),
        "by_task": {task: finalize_stats(task_stats[task]) for task in metadata["task_names"]},
        "by_pair": {pair["pair"]: finalize_stats(pair_stats[pair["pair"]]) for pair in pairs},
        "by_task_pair": {
            task: {
                pair["pair"]: finalize_stats(task_pair_stats[(task, pair["pair"])])
                for pair in pairs
            }
            for task in metadata["task_names"]
        },
    }


def pct(value: float | None) -> str:
    return "-" if value is None else f"{value * 100:.2f}%"


def write_summary_markdown(summary: Dict[str, Any], path: Path) -> None:
    meta = summary["metadata"]
    lines = [
        "# Router Pair Generation Length Comparison",
        "",
        f"- Data root: `{meta['data_root']}`",
        f"- Split: `{meta['split']}`; samples per task: `{meta['samples_per_task']}`",
        f"- Model: `{meta['base_model_path']}`",
        f"- Experts: `{', '.join(meta['expert_names'])}`; pairs: `{meta['num_pairs']}`",
        f"- Compared generation lengths: `4` and `64` new tokens",
        "",
        "## Overall",
        "",
        "| Rows | Same output | Exact target @4 | Exact target @64 | OpenCompass full @4 | OpenCompass full @64 | Mean score @4 | Mean score @64 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    overall = summary["overall"]
    short = overall["max_new_tokens_4"]
    long = overall["max_new_tokens_64"]
    lines.append(
        f"| {overall['count']} | {pct(overall['same_generation_rate'])} | "
        f"{pct(short['raw_exact_target_rate'])} | {pct(long['raw_exact_target_rate'])} | "
        f"{pct(short['opencompass_config_full_credit_rate'])} | {pct(long['opencompass_config_full_credit_rate'])} | "
        f"{short['opencompass_config_mean_score']:.4f} | {long['opencompass_config_mean_score']:.4f} |"
    )
    lines.extend(
        [
            "",
            "## By Task",
            "",
            "| Task | Rows | Same output | Exact @4 | Exact @64 | OpenCompass full @4 | OpenCompass full @64 | Mean score @4 | Mean score @64 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, row in summary["by_task"].items():
        short = row["max_new_tokens_4"]
        long = row["max_new_tokens_64"]
        lines.append(
            f"| {task} | {row['count']} | {pct(row['same_generation_rate'])} | "
            f"{pct(short['raw_exact_target_rate'])} | {pct(long['raw_exact_target_rate'])} | "
            f"{pct(short['opencompass_config_full_credit_rate'])} | {pct(long['opencompass_config_full_credit_rate'])} | "
            f"{short['opencompass_config_mean_score']:.4f} | {long['opencompass_config_mean_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## By Pair",
            "",
            "| Pair | Rows | Same output | Exact @4 | Exact @64 | OpenCompass full @4 | OpenCompass full @64 | Mean score @4 | Mean score @64 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair, row in summary["by_pair"].items():
        short = row["max_new_tokens_4"]
        long = row["max_new_tokens_64"]
        lines.append(
            f"| {pair} | {row['count']} | {pct(row['same_generation_rate'])} | "
            f"{pct(short['raw_exact_target_rate'])} | {pct(long['raw_exact_target_rate'])} | "
            f"{pct(short['opencompass_config_full_credit_rate'])} | {pct(long['opencompass_config_full_credit_rate'])} | "
            f"{short['opencompass_config_mean_score']:.4f} | {long['opencompass_config_mean_score']:.4f} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- `Exact target` means the raw generated string after outer whitespace stripping exactly matches `target`.",
            "- `OpenCompass full` means the dataset-config evaluator gives full credit for that sample.",
            "- `BoolQ` follows `SuperGLUE_BoolQ_gen.py` exactly and uses `first_capital_postprocess`.",
            "- `Mean score` preserves partial-credit behavior for tasks such as `iwslt2017` BLEU and `squad2` F1-like scoring.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare every fixed expert pair at max_new_tokens=4 versus 64 on "
            "sampled train rows, reporting raw target equality and OpenCompass-aligned scoring."
        )
    )
    parser.add_argument("--data_root", type=Path, default=Path("router_train_datasets_config_zeroshot_0521"))
    parser.add_argument("--split", default="train", choices=["train"])
    parser.add_argument("--task_names", default=None, help="Comma-separated task folders; defaults to every folder with train.jsonl.")
    parser.add_argument("--expert_names", default=None, help="Comma-separated expert names; defaults to medmcqa,race,sst2.")
    parser.add_argument("--samples_per_task", type=int, default=10)
    parser.add_argument("--base_model_path", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--router_bert_init", default="./task_classifier_ckpt")
    parser.add_argument("--max_llm_len", type=int, default=768)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--first_layer_idx", type=int, default=0)
    parser.add_argument("--middle_layer_idx", type=int, default=18)
    parser.add_argument("--router_dim", type=int, default=512)
    parser.add_argument("--router_pooling", default="mean")
    parser.add_argument("--router_pooling_last_k", type=int, default=4)
    parser.add_argument("--dtype", choices=["float16", "bfloat16"], default="float16")
    parser.add_argument("--r", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lora_medmcqa", default="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa")
    parser.add_argument("--lora_race", default="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race")
    parser.add_argument("--lora_sst2", default="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2")
    parser.add_argument("--lora_iwslt2017", default=None)
    parser.add_argument("--lora_squad2", default=None)
    parser.add_argument("--lora_boolq", default=None)
    parser.add_argument("--lora_rte", default=None)
    parser.add_argument("--lora_siqa", default=None)
    parser.add_argument("--lora_piqa", default=None)
    parser.add_argument("--out_dir", type=Path, default=Path("outputs/router_pair_max_tokens_compare_0521_qwen3"))
    args = parser.parse_args()

    task_names = parse_csv(args.task_names, discover_task_folders(args.data_root))
    expert_names = parse_csv(args.expert_names, DEFAULT_EXPERTS)
    missing_specs = [task for task in task_names if task not in TASK_EVAL_SPECS]
    if missing_specs:
        raise ValueError(f"Missing OpenCompass-aligned task evaluation specs: {missing_specs}")
    lora_candidates = {
        "medmcqa": args.lora_medmcqa,
        "race": args.lora_race,
        "sst2": args.lora_sst2,
        "iwslt2017": args.lora_iwslt2017,
        "squad2": args.lora_squad2,
        "boolq": args.lora_boolq,
        "rte": args.lora_rte,
        "siqa": args.lora_siqa,
        "piqa": args.lora_piqa,
    }
    missing_lora = [expert for expert in expert_names if not lora_candidates.get(expert)]
    if missing_lora:
        raise ValueError(f"Missing LoRA paths for expert_names={missing_lora}")
    lora_paths = {expert: lora_candidates[expert] for expert in expert_names}
    records = read_rows(args.data_root, task_names, args.samples_per_task)
    pairs = pair_rows(expert_names)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = JointAnswerSupervisionRouterModel(
        base_model_path=args.base_model_path,
        router_bert_init=args.router_bert_init,
        lora_paths=lora_paths,
        first_layer_idx=args.first_layer_idx,
        middle_layer_idx=args.middle_layer_idx,
        router_dim=args.router_dim,
        dtype=args.dtype,
        r=args.r,
        alpha=args.alpha,
        expert_names=expert_names,
        router_pooling=args.router_pooling,
        router_pooling_last_k=args.router_pooling_last_k,
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    metadata = {
        "data_root": str(args.data_root),
        "split": args.split,
        "samples_per_task": args.samples_per_task,
        "task_names": task_names,
        "expert_names": expert_names,
        "num_pairs": len(pairs),
        "base_model_path": args.base_model_path,
        "lora_paths": lora_paths,
        "max_new_tokens": [4, 64],
        "scoring": {
            "raw_exact_target": "prediction.strip() == target.strip()",
            "opencompass_config": (
                "source config contract; BoolQ uses its first_capital_postprocess, "
                "remaining tasks use router_answer_supervision_core.compute_official_evaluator_sample_score"
            ),
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_json(metadata, args.out_dir / "run_config.json")

    for pair in pairs:
        pair_name = pair["pair"]
        print(f"[PAIR] {pair_name}", flush=True)
        for max_new_tokens in (4, 64):
            outputs: List[str] = []
            for batch in chunks(records, args.batch_size):
                outputs.extend(
                    generate_batch_for_pair(
                        model=model,
                        tokenizer=tokenizer,
                        records=batch,
                        first_expert=pair["first_expert"],
                        mid_expert=pair["mid_expert"],
                        max_llm_len=args.max_llm_len,
                        max_new_tokens=max_new_tokens,
                    )
                )
            for record, prediction in zip(records, outputs):
                pair_result = record["pairs"].setdefault(pair_name, {})
                pair_result[f"max_new_tokens_{max_new_tokens}"] = opencompass_metrics(prediction, record)
        for record in records:
            pair_result = record["pairs"][pair_name]
            pair_result["same_generation"] = (
                pair_result["max_new_tokens_4"]["prediction"]
                == pair_result["max_new_tokens_64"]["prediction"]
            )
        with (args.out_dir / "records.partial.jsonl").open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        write_json({"completed_pairs": pair["pair_idx"] + 1, "last_pair": pair_name}, args.out_dir / "progress.json")

    records_path = args.out_dir / "records.jsonl"
    with records_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    summary = build_summary(records, pairs, metadata)
    write_json(summary, args.out_dir / "summary.json")
    write_summary_markdown(summary, args.out_dir / "summary.md")
    print(f"[DONE] records={records_path}", flush=True)
    print(f"[DONE] summary={args.out_dir / 'summary.md'}", flush=True)


if __name__ == "__main__":
    main()
