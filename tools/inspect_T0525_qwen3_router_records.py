import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


EXPERT_NAMES = ["medmcqa", "race", "sst2"]
RUN_DIRS = {
    "mrs": "router_0525_qwen3_fp16_correct_conf_ce_t1_mrs_3expert_sst2words",
    "boolq": "router_0525_qwen3_fp16_4sum_boolq_correct_conf_ce_t1_mrs_3expert_sst2words",
    "rte": "router_0525_qwen3_fp16_4sum_rte_correct_conf_ce_t1_mrs_3expert_sst2words",
    "siqa": "router_0525_qwen3_fp16_4sum_siqa_correct_conf_ce_t1_mrs_3expert_sst2words",
    "piqa": "router_0525_qwen3_fp16_4sum_piqa_correct_conf_ce_t1_mrs_3expert_sst2words",
}


def pair_name(pair_id: int) -> str:
    num_experts = len(EXPERT_NAMES)
    return f"{EXPERT_NAMES[pair_id // num_experts]}->{EXPERT_NAMES[pair_id % num_experts]}"


def ranked_matrix(matrix: Sequence[Sequence[float]]) -> List[Tuple[int, float]]:
    flattened = [value for row in matrix for value in row]
    return sorted(enumerate(flattened), key=lambda item: item[1], reverse=True)


def format_top(matrix: Sequence[Sequence[float]], topk: int) -> str:
    return ", ".join(
        f"{pair_name(pair_id)}={prob:.4f}"
        for pair_id, prob in ranked_matrix(matrix)[:topk]
    )


def self_pair_details(record: Dict) -> str:
    task = str(record["task"])
    if task not in EXPERT_NAMES:
        return "self=N/A"
    self_pair = f"{task}->{task}"
    self_pair_id = EXPERT_NAMES.index(task) * len(EXPERT_NAMES) + EXPERT_NAMES.index(task)
    pred_ranking = ranked_matrix(record["pair_prob_matrix"])
    target_ranking = ranked_matrix(record["router_target_matrix"])
    pred_rank = next(rank for rank, (pair_id, _value) in enumerate(pred_ranking, start=1) if pair_id == self_pair_id)
    target_rank = next(rank for rank, (pair_id, _value) in enumerate(target_ranking, start=1) if pair_id == self_pair_id)
    pred_prob = pred_ranking[pred_rank - 1][1]
    target_prob = target_ranking[target_rank - 1][1]
    chosen = record["pred_pair"] == self_pair
    return (
        f"self={self_pair} chosen={chosen} "
        f"pred_rank={pred_rank} pred_prob={pred_prob:.4f} "
        f"target_rank={target_rank} target_prob={target_prob:.4f}"
    )


def load_records(run_key: str, split: str) -> Tuple[Path, int, List[Dict]]:
    run_dir = Path(RUN_DIRS[run_key])
    metrics = json.loads((run_dir / "best_metrics.json").read_text())
    epoch = int(metrics["best_epoch"])
    records_path = run_dir / f"route_records_{split}_epoch{epoch}.json"
    records = json.loads(records_path.read_text())["records"]
    return records_path, epoch, records


def print_summary(records: Sequence[Dict]) -> None:
    selected = [record for record in records if record["task"] in EXPERT_NAMES]
    self_selected = sum(
        record["pred_pair"] == f"{record['task']}->{record['task']}"
        for record in selected
    )
    pred_correct = sum(bool(record["pred_correct"]) for record in records)
    print(f"records={len(records)} pred_correct={pred_correct}/{len(records)}={pred_correct / max(len(records), 1):.4f}")
    print(
        "self_expert_selected="
        f"{self_selected}/{len(selected)}={self_selected / max(len(selected), 1):.4f}"
    )
    for task in sorted(set(record["task"] for record in records)):
        task_rows = [record for record in records if record["task"] == task]
        task_correct = sum(bool(record["pred_correct"]) for record in task_rows)
        if task in EXPERT_NAMES:
            task_self = sum(record["pred_pair"] == f"{task}->{task}" for record in task_rows)
            print(
                f"task={task} n={len(task_rows)} "
                f"pred_correct={task_correct / len(task_rows):.4f} "
                f"self_selected={task_self / len(task_rows):.4f}"
            )
        else:
            print(f"task={task} n={len(task_rows)} pred_correct={task_correct / len(task_rows):.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", choices=list(RUN_DIRS))
    parser.add_argument("--split", choices=["val", "train_eval"], default="val")
    parser.add_argument("--task", default=None)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--summary_only", action="store_true")
    args = parser.parse_args()

    records_path, best_epoch, records = load_records(args.run, args.split)
    if args.task:
        records = [record for record in records if record["task"] == args.task]

    print(f"run={args.run} best_epoch={best_epoch} records_file={records_path}")
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
