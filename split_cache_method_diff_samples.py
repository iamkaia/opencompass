import argparse
import json
import os
from typing import Dict, Iterable, List


def load_jsonl(path: str) -> List[Dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def save_jsonl(rows: Iterable[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def compact_pair_rows(rows: List[Dict], topn: int = 5) -> List[Dict]:
    out = []
    for row in rows[:topn]:
        compact = {
            "rank_by_loss": row.get("rank_by_loss"),
            "pair": row.get("pair"),
            "loss": row.get("loss"),
            "correct": row.get("correct"),
        }
        if "prediction" in row:
            compact["prediction"] = row.get("prediction")
        if "pred_option_label" in row:
            compact["pred_option_label"] = row.get("pred_option_label")
        if "pred_option_prob" in row:
            compact["pred_option_prob"] = row.get("pred_option_prob")
        if "option_probs" in row:
            compact["option_probs"] = row.get("option_probs")
        out.append(compact)
    return out


def compact_router_pairs(record: Dict, topn: int = 10) -> Dict:
    if not record:
        return {}
    pair_probs = record.get("pair_probs") or []
    pair_logits = record.get("pair_logits") or []
    top_router_pairs = record.get("top_router_pairs") or []
    compact = {
        "pred_pair": record.get("pred_pair"),
        "pred_pair_id": record.get("pred_pair_id"),
        "pred_correct": record.get("pred_correct"),
        "gold_pair": record.get("gold_pair"),
        "gold_pair_id": record.get("gold_pair_id"),
        "match": record.get("match"),
        "top_router_pairs": top_router_pairs[:topn],
    }
    if pair_probs:
        compact["pair_probs"] = pair_probs
    if pair_logits:
        compact["pair_logits"] = pair_logits
    return compact


def build_pretty_row(row: Dict) -> Dict:
    old_method = row["old_method"]
    new_method = row["new_method"]
    prompt_text = str(row.get("prompt_text", ""))
    text = str(row.get("text", ""))
    return {
        "key": row.get("key"),
        "task": row.get("task"),
        "target": row.get("target"),
        "prompt_preview": prompt_text[:240],
        "text_preview": text[:240],
        "old": {
            "self_pair": old_method.get("self_pair"),
            "self_loss": old_method.get("self_loss"),
            "self_correct": old_method.get("self_correct"),
            "oracle_pair": old_method.get("oracle_pair"),
            "oracle_loss": old_method.get("oracle_loss"),
            "oracle_correct": old_method.get("oracle_correct"),
            "num_correct_pairs": old_method.get("num_correct_pairs"),
            "any_pair_correct": old_method.get("any_pair_correct"),
            "top5_pairs_by_loss": compact_pair_rows(old_method.get("all_pairs", []), topn=5),
            "all25_pairs_by_loss": compact_pair_rows(old_method.get("all_pairs", []), topn=25),
        },
        "new": {
            "self_pair": new_method.get("self_pair"),
            "self_loss": new_method.get("self_loss"),
            "self_correct": new_method.get("self_correct"),
            "oracle_pair": new_method.get("oracle_pair"),
            "oracle_loss": new_method.get("oracle_loss"),
            "oracle_correct": new_method.get("oracle_correct"),
            "num_correct_pairs": new_method.get("num_correct_pairs"),
            "any_pair_correct": new_method.get("any_pair_correct"),
            "top5_pairs_by_loss": compact_pair_rows(new_method.get("all_pairs", []), topn=5),
            "all25_pairs_by_loss": compact_pair_rows(new_method.get("all_pairs", []), topn=25),
        },
        "old_router": compact_router_pairs(row.get("old_router_prediction", {}), topn=10),
        "new_router": compact_router_pairs(row.get("new_router_prediction", {}), topn=10),
        "diff_flags": {
            "self_correct_changed": bool(old_method.get("self_correct") != new_method.get("self_correct")),
            "oracle_pair_changed": bool(old_method.get("oracle_pair") != new_method.get("oracle_pair")),
            "any_pair_correct_changed": bool(old_method.get("any_pair_correct") != new_method.get("any_pair_correct")),
            "num_correct_pairs_delta": int(new_method.get("num_correct_pairs", 0) - old_method.get("num_correct_pairs", 0)),
            "self_loss_delta": (
                None
                if old_method.get("self_loss") is None or new_method.get("self_loss") is None
                else float(new_method.get("self_loss") - old_method.get("self_loss"))
            ),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--prefix", default="cache_method_diff")
    args = parser.parse_args()

    rows = load_jsonl(args.input)
    os.makedirs(args.out_dir, exist_ok=True)

    by_task: Dict[str, List[Dict]] = {}
    for row in rows:
        task = str(row.get("task", "unknown"))
        by_task.setdefault(task, []).append(row)

    for task, task_rows in sorted(by_task.items()):
        raw_path = os.path.join(args.out_dir, f"{args.prefix}_{task}.jsonl")
        pretty_path = os.path.join(args.out_dir, f"{args.prefix}_{task}_pretty.jsonl")
        summary_path = os.path.join(args.out_dir, f"{args.prefix}_{task}_summary.json")
        pretty_rows = [build_pretty_row(row) for row in task_rows]
        save_jsonl(task_rows, raw_path)
        save_jsonl(pretty_rows, pretty_path)

        n = len(task_rows)
        old_self = sum(int(bool(row["old_method"].get("self_correct"))) for row in task_rows)
        new_self = sum(int(bool(row["new_method"].get("self_correct"))) for row in task_rows)
        old_any = sum(int(bool(row["old_method"].get("any_pair_correct"))) for row in task_rows)
        new_any = sum(int(bool(row["new_method"].get("any_pair_correct"))) for row in task_rows)
        self_changed = sum(int(pretty["diff_flags"]["self_correct_changed"]) for pretty in pretty_rows)
        oracle_changed = sum(int(pretty["diff_flags"]["oracle_pair_changed"]) for pretty in pretty_rows)
        summary = {
            "task": task,
            "num_samples": n,
            "old_self_acc": old_self / max(n, 1),
            "new_self_acc": new_self / max(n, 1),
            "old_any_pair_acc": old_any / max(n, 1),
            "new_any_pair_acc": new_any / max(n, 1),
            "self_correct_changed_count": self_changed,
            "oracle_pair_changed_count": oracle_changed,
            "raw_path": raw_path,
            "pretty_path": pretty_path,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"[DONE] {task}: raw={raw_path} pretty={pretty_path} summary={summary_path}")


if __name__ == "__main__":
    main()
