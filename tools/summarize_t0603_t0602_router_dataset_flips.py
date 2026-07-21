#!/usr/bin/env python3
"""Dataset-level router flip summary for T0603 qwenfix and T0602 llama."""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path


BASE = Path(__file__).resolve().parent / "summarize_t0603_t0602_router_flips.py"
SPEC = importlib.util.spec_from_file_location("router_flip_base", BASE)
BASE_MOD = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BASE_MOD)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def counter_to_items(counter: Counter[str]) -> list[dict]:
    total = sum(counter.values())
    return [
        {"name": name, "count": count, "rate": count / total if total else 0.0}
        for name, count in sorted(counter.items())
    ]


def top(items: list[dict]) -> tuple[str, float, int]:
    ordered = sorted_nonzero(items)
    if not ordered:
        return "-", 0.0, 0
    item = ordered[0]
    return str(item["name"]), float(item["rate"]), int(item["count"])


def rate_of(items: list[dict], name: str) -> tuple[float, int]:
    for item in items:
        if item["name"] == name:
            return float(item["rate"]), int(item["count"])
    return 0.0, 0


def sorted_nonzero(items: list[dict]) -> list[dict]:
    return sorted(
        (x for x in items if int(x.get("count", 0)) > 0),
        key=lambda x: (-float(x.get("rate", 0.0)), str(x.get("name", ""))),
    )


def per_task_items(summary: dict, task: str) -> list[dict]:
    lookup_task = task
    if task not in BASE_MOD.MRS_TASKS:
        lookup_task = "task_id:-1"
    for row in summary.get("per_task", []):
        if row.get("task") == lookup_task:
            return row.get("all_pred_pairs") or row.get("top_pred_pairs") or []
    return []


def oc_items_for_task(record_path: Path, task: str) -> list[dict]:
    aliases = BASE_MOD.DATASET_ALIASES.get(task, {task})
    counts: Counter[str] = Counter()
    with record_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if str(rec.get("dataset")) in aliases:
                counts[str(rec["pred_pair"])] += 1
    return counter_to_items(counts)


def changed_top_judgement(
    source_items: list[dict],
    target_items: list[dict],
    source_top: str,
    target_top: str,
    label: str,
) -> str:
    source_top_rate, _ = rate_of(source_items, source_top)
    source_new_rate, _ = rate_of(source_items, target_top)
    target_new_rate, _ = rate_of(target_items, target_top)
    target_old_rate, _ = rate_of(target_items, source_top)
    source_margin = (source_top_rate - source_new_rate) * 100
    target_margin = (target_new_rate - target_old_rate) * 100
    if max(source_margin, target_margin) <= 5:
        scale = "top 交換但近乎平手"
    elif target_margin < 10:
        scale = "小幅翻轉"
    elif target_margin < 25:
        scale = "明顯翻轉"
    else:
        scale = "差很多"
    return (
        f"{label} {scale}: {source_top}({target_old_rate * 100:.1f}%)"
        f" -> {target_top}({target_new_rate * 100:.1f}%), "
        f"train margin {source_margin:.1f}pp, {label} margin {target_margin:.1f}pp"
    )


def judgement(
    train_items: list[dict],
    val_items: list[dict],
    oc_items: list[dict],
    train_top: str,
    train_rate: float,
    val_top: str,
    val_rate: float,
    oc_top: str,
    oc_rate: float,
) -> str:
    val_delta = abs(val_rate - train_rate) * 100
    if val_top != train_top:
        val_part = changed_top_judgement(train_items, val_items, train_top, val_top, "valid")
    elif val_delta <= 5:
        val_part = f"valid 很接近, 差 {val_delta:.1f}pp"
    elif val_delta <= 10:
        val_part = f"valid 小差, 差 {val_delta:.1f}pp"
    else:
        val_part = f"valid 比例差較大, 差 {val_delta:.1f}pp"

    old_oc_rate, _ = rate_of(oc_items, train_top)
    if oc_top == train_top:
        oc_delta = abs(oc_rate - train_rate) * 100
        if oc_delta <= 10:
            oc_part = f"OC 同 top, 比例接近, 差 {oc_delta:.1f}pp"
        elif oc_delta <= 25:
            oc_part = f"OC 同 top, 比例中等差, 差 {oc_delta:.1f}pp"
        else:
            oc_part = f"OC 同 top, 但比例差很多, 差 {oc_delta:.1f}pp"
    else:
        oc_part = changed_top_judgement(train_items, oc_items, train_top, oc_top, "OC")
    return f"{val_part}; {oc_part}"


def row_for(root: Path, family: str, router_dir: Path) -> list[dict]:
    best = load_json(router_dir / "best_metrics.json")
    epoch = int(best["best_epoch"])
    train_summary = load_json(router_dir / f"routing_summary_train_eval_epoch{epoch}.json")
    val_summary = load_json(router_dir / f"routing_summary_val_epoch{epoch}.json")
    record_path = BASE_MOD.record_path_for(root, router_dir)
    tasks = BASE_MOD.trained_datasets(router_dir)

    rows = []
    for task in tasks:
        train_items = per_task_items(train_summary, task)
        val_items = per_task_items(val_summary, task)
        oc_items = oc_items_for_task(record_path, task) if record_path else []
        train_top, train_rate, train_count = top(train_items)
        val_top, val_rate, val_count = top(val_items)
        oc_top, oc_rate, oc_count = top(oc_items)
        rows.append(
            {
                "router": BASE_MOD.router_label(router_dir),
                "dataset": task,
                "best_epoch": epoch,
                "train": (train_top, train_rate, train_count),
                "valid": (val_top, val_rate, val_count),
                "oc": (oc_top, oc_rate, oc_count),
                "train_items": train_items,
                "valid_items": val_items,
                "oc_items": oc_items,
                "judgement": judgement(
                    train_items,
                    val_items,
                    oc_items,
                    train_top,
                    train_rate,
                    val_top,
                    val_rate,
                    oc_top,
                    oc_rate,
                ),
            }
        )
    return rows


def fmt(cell: tuple[str, float, int]) -> str:
    name, rate, count = cell
    return f"{name} {rate * 100:.1f}% ({count})"


def fmt_dist(items: list[dict]) -> str:
    ordered = sorted_nonzero(items)
    if not ordered:
        return "-"
    return "<br>".join(
        f"{idx}. {item['name']} {float(item['rate']) * 100:.1f}% ({int(item['count'])})"
        for idx, item in enumerate(ordered, 1)
    )


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
        "# Dataset-level router top-1 比例翻轉檢查",
        "",
        "- 範圍：T0603 只看 qwen/qwenfix；T0602 只看 llama。",
        "- 只檢查 router 訓練過的 datasets；OpenCompass 的 `race-high`/`race-middle` 合併回訓練 dataset `race`。",
        "- 每列比較 best_epoch 的 train_eval、valid、OpenCompass evaluator `pred_pair` 完整非零分布；判斷仍以 top-1 和前幾名 margin 為主。",
        "",
    ]

    all_rows: list[dict] = []
    for root, family, title in specs:
        rows: list[dict] = []
        for router_dir in sorted((root / "routers").iterdir()):
            if router_dir.is_dir() and BASE_MOD.model_family(router_dir) == family:
                rows.extend(row_for(root, family, router_dir))
        all_rows.extend(rows)
        lines.append(f"## {title}")
        lines.append("")
        lines.append("| Router | dataset | best_epoch | train_eval 分布 | valid 分布 | OpenCompass 分布 | 判斷 |")
        lines.append("|---|---|---:|---|---|---|---|")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        row["router"],
                        row["dataset"],
                        str(row["best_epoch"]),
                        fmt_dist(row["train_items"]),
                        fmt_dist(row["valid_items"]),
                        fmt_dist(row["oc_items"]),
                        row["judgement"],
                    ]
                )
                + " |"
            )
        lines.append("")

    flipped = [row for row in all_rows if row["train"][0] != row["oc"][0]]
    big = [row for row in flipped if "差很多" in row["judgement"]]
    lines.append("## 摘要")
    lines.append("")
    lines.append(f"- 共檢查 {len(all_rows)} 個 router-dataset 組合。")
    lines.append(f"- OpenCompass top-1 相對 train_eval 翻轉：{len(flipped)} 個。")
    lines.append(f"- 其中判為差很多：{len(big)} 個。")
    lines.append("")
    if flipped:
        lines.append("### 有翻轉的 dataset")
        lines.append("")
        lines.append("| Router | dataset | train_eval 分布 | valid 分布 | OpenCompass 分布 | 判斷 |")
        lines.append("|---|---|---|---|---|---|")
        for row in flipped:
            lines.append(
                f"| {row['router']} | {row['dataset']} | {fmt_dist(row['train_items'])} | {fmt_dist(row['valid_items'])} | {fmt_dist(row['oc_items'])} | {row['judgement']} |"
            )
        lines.append("")

    args.out.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
