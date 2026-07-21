#!/usr/bin/env python

import argparse
import json
from pathlib import Path

from inspect_T0525_qwen3_router_records import (
    format_top,
    print_summary,
    self_pair_details,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=Path)
    parser.add_argument("--split", choices=["val", "train_eval"], default="val")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    if args.eval_only:
        epoch = None
        record_name = "route_records_train_eval_only.json" if args.split == "train_eval" else "route_records_val_eval_only.json"
        records_path = args.run_dir / record_name
    else:
        metrics = json.loads((args.run_dir / "best_metrics.json").read_text())
        epoch = int(metrics["best_epoch"])
        records_path = args.run_dir / f"route_records_{args.split}_epoch{epoch}.json"
    records = json.loads(records_path.read_text())["records"]

    print(f"run_dir={args.run_dir} best_epoch={epoch} records_file={records_path}")
    print_summary(records)
    if args.summary_only:
        return

    for index, record in enumerate(records[: max(args.limit, 0)]):
        print("=" * 100)
        print(
            f"idx={index} task={record['task']} target={record['target']} "
            f"pred_correct={record['pred_correct']} any_pair_correct={record['any_pair_correct']}"
        )
        print(
            f"gold_lowest_loss_pair={record['gold_pair']} loss={record['gold_raw_loss']:.6f} "
            f"router_pred_pair={record['pred_pair']} prob={record['pred_pair_prob']:.4f} "
            f"loss={record['pred_raw_loss']:.6f}"
        )
        print(f"router_target_top{args.topk}: {format_top(record['router_target_matrix'], args.topk)}")
        print(f"router_pred_top{args.topk}:   {format_top(record['pair_prob_matrix'], args.topk)}")
        print(self_pair_details(record))


if __name__ == "__main__":
    main()
