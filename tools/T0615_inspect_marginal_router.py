#!/usr/bin/env python3
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def normalized_entropy(values):
    values = [max(float(x), 0.0) for x in values]
    total = sum(values)
    if total <= 0 or len(values) < 2:
        return 0.0
    probs = [x / total for x in values]
    return -sum(x * math.log(x + 1e-12) for x in probs) / math.log(len(probs))


def mean(rows, index):
    return sum(row[index] for row in rows) / len(rows)


def inspect_records(path):
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["records"] if isinstance(payload, dict) else payload
    by_task = defaultdict(list)

    for record in records:
        pred_first = record.get("pred_first_weights")
        pred_mid = record.get("pred_mid_weights")
        target_first = record.get("target_first_weights")
        target_mid = record.get("target_mid_weights")
        if any(x is None for x in (pred_first, pred_mid, target_first, target_mid)):
            continue

        size = len(target_first)
        uniform = [1.0 / size] * size
        first_mse = sum((a - b) ** 2 for a, b in zip(pred_first, target_first)) / size
        mid_mse = sum((a - b) ** 2 for a, b in zip(pred_mid, target_mid)) / size
        uniform_mse = (
            sum((a - b) ** 2 for a, b in zip(uniform, target_first)) / size
            + sum((a - b) ** 2 for a, b in zip(uniform, target_mid)) / size
        )
        by_task[str(record.get("task", "unknown"))].append(
            (
                first_mse + mid_mse,
                first_mse,
                mid_mse,
                uniform_mse,
                normalized_entropy(pred_first),
                normalized_entropy(target_first),
                normalized_entropy(pred_mid),
                normalized_entropy(target_mid),
            )
        )

    print(f"\n[{path.name}] samples={sum(map(len, by_task.values()))}")
    print("task          n       mse   vs_uniform   first_mse   mid_mse   first_ent(p/t)   mid_ent(p/t)")
    groups = [("ALL", sum(by_task.values(), []))] + sorted(by_task.items())
    for task, rows in groups:
        mse = mean(rows, 0)
        uniform_mse = mean(rows, 3)
        improvement = 100.0 * (1.0 - mse / uniform_mse) if uniform_mse > 0 else 0.0
        print(
            f"{task:<12} {len(rows):>4}  {mse:>8.6f}  {improvement:>9.2f}%"
            f"  {mean(rows, 1):>10.6f}  {mean(rows, 2):>8.6f}"
            f"   {mean(rows, 4):.4f}/{mean(rows, 5):.4f}"
            f"       {mean(rows, 6):.4f}/{mean(rows, 7):.4f}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("router_dir", type=Path)
    args = parser.parse_args()

    best_path = args.router_dir / "best_metrics.json"
    if not best_path.is_file():
        raise SystemExit(f"missing {best_path}")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    epoch = int(best["best_epoch"])
    metrics = best.get("metrics", {})
    print(f"router={args.router_dir}")
    print(f"best_epoch={epoch}")
    print(f"val_marginal_mse={metrics.get('weighted_sum_marginal_mse')}")
    print(f"val_first_mse={metrics.get('weighted_sum_first_mse')}")
    print(f"val_mid_mse={metrics.get('weighted_sum_mid_mse')}")

    for split in ("train_eval", "val"):
        path = args.router_dir / f"route_records_{split}_epoch{epoch}.json"
        if path.is_file():
            inspect_records(path)
        else:
            print(f"missing {path}")


if __name__ == "__main__":
    main()
