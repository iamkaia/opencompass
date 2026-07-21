#!/usr/bin/env python3
"""Summarize best-epoch router distributions and OpenCompass routing flips."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


MRS_TASKS = {"medmcqa", "race", "sst2"}
DATASET_ALIASES = {
    "arc_c": {"ARC-c"},
    "boolq": {"BoolQ"},
    "medmcqa": {"medmcqa"},
    "openbookqa": {"openbookqa"},
    "piqa": {"piqa"},
    "race": {"race-high", "race-middle"},
    "rte": {"RTE"},
    "siqa": {"siqa"},
    "sst2": {"sst2"},
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def router_label(router_dir: Path) -> str:
    name = router_dir.name
    name = name.removeprefix("router_T0602_").removeprefix("router_T0603_")
    name = name.removesuffix("_correct_conf_ce_t1_3expert_sst2words")
    name = name.removesuffix("_correct_conf_ce_t1_3expert")
    return name


def model_family(router_dir: Path) -> str:
    name = router_dir.name
    if "qwen3_fp16" in name:
        return "qwen3_fp16"
    if "llama" in name:
        return "llama"
    return "unknown"


def mode_slug(router_dir: Path) -> str:
    label = router_label(router_dir)
    marker = "_taskcls_trainbert"
    if marker in label:
        label = label.split(marker, 1)[0]
    for prefix in ("qwen3_fp16_", "llama_"):
        if label.startswith(prefix):
            label = label[len(prefix) :]
    return label


def format_dist(items: list[dict], max_items: int = 9) -> str:
    ordered = sorted(
        (x for x in items if int(x.get("count", 0)) > 0),
        key=lambda x: (-float(x.get("rate", 0.0)), str(x.get("name", ""))),
    )
    if not ordered:
        return "-"
    return "<br>".join(
        f"{x['name']} {float(x['rate']) * 100:.1f}% ({int(x['count'])})"
        for x in ordered[:max_items]
    )


def top_pair(items: list[dict]) -> str:
    ordered = sorted(
        (x for x in items if int(x.get("count", 0)) > 0),
        key=lambda x: (-float(x.get("rate", 0.0)), str(x.get("name", ""))),
    )
    return str(ordered[0]["name"]) if ordered else "-"


def dataset_aliases(tasks: list[str]) -> set[str]:
    aliases: set[str] = set()
    for task in tasks:
        aliases.update(DATASET_ALIASES.get(task, {task}))
    return aliases


def summarize_records(
    path: Path, allowed_datasets: set[str] | None = None
) -> tuple[list[dict], dict[str, list[dict]]]:
    counts: Counter[str] = Counter()
    by_dataset: dict[str, Counter[str]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            dataset = str(rec.get("dataset", "unknown"))
            if allowed_datasets is not None and dataset not in allowed_datasets:
                continue
            pair = str(rec["pred_pair"])
            counts[pair] += 1
            by_dataset.setdefault(dataset, Counter())[pair] += 1
    return counter_to_items(counts), {
        dataset: counter_to_items(counter) for dataset, counter in sorted(by_dataset.items())
    }


def counter_to_items(counter: Counter[str]) -> list[dict]:
    total = sum(counter.values())
    return [
        {"name": name, "count": count, "rate": count / total if total else 0.0}
        for name, count in sorted(counter.items())
    ]


def record_path_for(root: Path, router_dir: Path) -> Path | None:
    slug = mode_slug(router_dir)
    family = model_family(router_dir)
    candidates: list[str] = []
    if root.name.startswith("T0603_"):
        candidates.append(f"T0603_{family}_{slug}_taskcls_trainbert_qwenfix_hard_20260603_063927.jsonl")
    else:
        if slug == "mrs_only":
            candidates.append(f"{family}_mrs_only_taskcls_trainbert_chattemplate_alltasks_hard.jsonl")
        candidates.append(f"{family}_{slug}_taskcls_trainbert_chattemplate_hard.jsonl")
    for candidate in candidates:
        path = root / "router_records" / candidate
        if path.exists():
            return path
    return None


def trained_datasets(router_dir: Path) -> list[str]:
    cfg = load_json(router_dir / "train_config.json")
    resolved = cfg.get("resolved_sample_task_names")
    if resolved:
        return [str(x) for x in resolved]
    raw = cfg.get("sample_task_names")
    if raw:
        return [x.strip() for x in str(raw).split(",") if x.strip()]
    return []


def row_for(root: Path, router_dir: Path, include_family: str) -> dict | None:
    if model_family(router_dir) != include_family:
        return None
    best = load_json(router_dir / "best_metrics.json")
    epoch = int(best["best_epoch"])
    train_summary = load_json(router_dir / f"routing_summary_train_eval_epoch{epoch}.json")
    val_summary = load_json(router_dir / f"routing_summary_val_epoch{epoch}.json")
    rec_path = record_path_for(root, router_dir)
    eval_items: list[dict] = []
    per_eval_dataset: dict[str, list[dict]] = {}
    tasks = trained_datasets(router_dir)
    if rec_path:
        eval_items, per_eval_dataset = summarize_records(rec_path, dataset_aliases(tasks))

    train_top = top_pair(train_summary["all_pred_pairs"])
    val_top = top_pair(val_summary["all_pred_pairs"])
    eval_top = top_pair(eval_items)
    flip_bits = []
    if val_top != train_top:
        flip_bits.append(f"VAL: {train_top} -> {val_top}")
    if eval_top != "-" and eval_top != train_top:
        flip_bits.append(f"OC: {train_top} -> {eval_top}")

    trained_eval_tasks = [task for task in tasks if task not in MRS_TASKS]
    if not trained_eval_tasks and "mrs_only" in mode_slug(router_dir):
        trained_eval_tasks = tasks

    return {
        "router": router_label(router_dir),
        "mode": mode_slug(router_dir),
        "trained": ", ".join(tasks),
        "trained_eval_tasks": trained_eval_tasks,
        "epoch": epoch,
        "route_correct": best.get("metrics", {}).get("route_correct_acc"),
        "pair_acc": best.get("metrics", {}).get("pair_acc"),
        "train_top": train_top,
        "val_top": val_top,
        "eval_top": eval_top,
        "flip": "有翻轉: " + "; ".join(flip_bits) if flip_bits else "無",
        "train_dist": format_dist(train_summary["all_pred_pairs"]),
        "val_dist": format_dist(val_summary["all_pred_pairs"]),
        "eval_dist": format_dist(eval_items),
        "eval_record": str(rec_path) if rec_path else "找不到",
        "per_eval_dataset": per_eval_dataset,
    }


def table(rows: list[dict]) -> list[str]:
    out = [
        "| Router | 訓練 datasets | best_epoch | top-1 狀態 | best train_eval pred_pair 分布 | best valid pred_pair 分布 | OpenCompass hard evaluator 分布 |",
        "|---|---|---:|---|---|---|---|",
    ]
    for row in rows:
        metric = ""
        if row["route_correct"] is not None and row["pair_acc"] is not None:
            metric = (
                f"<br>route_correct={float(row['route_correct']) * 100:.1f}%, "
                f"pair_acc={float(row['pair_acc']) * 100:.1f}%"
            )
        status = (
            f"{row['flip']}<br>"
            f"train={row['train_top']}<br>valid={row['val_top']}<br>OC={row['eval_top']}{metric}"
        )
        out.append(
            "| "
            + " | ".join(
                [
                    row["router"],
                    row["trained"],
                    str(row["epoch"]),
                    status,
                    row["train_dist"],
                    row["val_dist"],
                    row["eval_dist"],
                ]
            )
            + " |"
        )
    return out


def per_dataset_section(rows: list[dict]) -> list[str]:
    out = []
    for row in rows:
        selected = []
        for task in row["trained_eval_tasks"]:
            selected.extend(
                alias
                for alias in sorted(DATASET_ALIASES.get(task, {task}))
                if alias in row["per_eval_dataset"]
            )
        if not selected:
            continue
        out.append(f"### {row['router']}")
        out.append("")
        out.append("| evaluator dataset | OpenCompass hard pred_pair 分布 |")
        out.append("|---|---|")
        for task in selected:
            out.append(f"| {task} | {format_dist(row['per_eval_dataset'][task])} |")
        out.append("")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    specs = [
        (
            Path("T0603_qwenfix_trainbert_chattemplate_20260603_063927"),
            "qwen3_fp16",
            "T0603 qwenfix / qwen3_fp16",
        ),
        (
            Path("t0602_taskcls_trainbert_chattemplate_20260603_02"),
            "llama",
            "T0602 / llama",
        ),
    ]

    lines = [
        "# T0603 qwen 與 T0602 llama router top-1 翻轉檢查",
        "",
        "- 翻轉定義：同一 router 的 top-1 `pred_pair` 在 `best_epoch train_eval`、`best_epoch valid`、OpenCompass hard evaluator 三者不完全一致；狀態欄標出相對於 train_eval 的變化。",
        "- 分布皆為 router `pred_pair`，依占比由高到低排序；`route_correct` 是答案正確率，`pair_acc` 才是 pair equality。",
        "- 範圍：`T0603_qwenfix_trainbert_chattemplate_20260603_063927` 只看 qwen/qwenfix；`t0602_taskcls_trainbert_chattemplate_20260603_02` 只看 llama。",
        "",
    ]

    all_rows: list[dict] = []
    for root, family, title in specs:
        rows = []
        for router_dir in sorted((root / "routers").iterdir()):
            if router_dir.is_dir():
                row = row_for(root, router_dir, family)
                if row:
                    rows.append(row)
        all_rows.extend(rows)
        lines.append(f"## {title}")
        lines.append("")
        lines.extend(table(rows))
        lines.append("")
        lines.append("OpenCompass hard evaluator record 來源：")
        for row in rows:
            lines.append(f"- `{row['router']}`: `{row['eval_record']}`")
        lines.append("")
        lines.append("### 針對非 MRS 訓練 dataset 的 OpenCompass 分布")
        lines.append("")
        detail = per_dataset_section(rows)
        lines.extend(detail if detail else ["無非 MRS evaluator dataset 可列。", ""])

    flipped = [row for row in all_rows if row["flip"] != "無"]
    lines.append("## 翻轉總結")
    lines.append("")
    lines.append(f"- 共檢查 {len(all_rows)} 個指定範圍 router，其中 {len(flipped)} 個有 top-1 翻轉。")
    if flipped:
        for row in flipped:
            lines.append(f"- `{row['router']}`: {row['flip']}")
    lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
