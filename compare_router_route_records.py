import argparse
import hashlib
import json
from collections import Counter
from typing import Dict, Iterable, List


def load_training_records(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    records = data.get("records", data if isinstance(data, list) else [])
    if not isinstance(records, list):
        raise ValueError(f"Unsupported training route record format: {path}")
    return records


def load_jsonl_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def keyed_records(records: Iterable[Dict], key_field: str) -> Dict[str, Dict]:
    out = {}
    duplicates = Counter()
    for record in records:
        key = record.get(key_field)
        if not key and key_field == "prompt_sha1" and record.get("prompt_text") is not None:
            key = hashlib.sha1(str(record["prompt_text"]).encode("utf-8")).hexdigest()
        if not key:
            continue
        if key in out:
            duplicates[key] += 1
            continue
        out[key] = record
    if duplicates:
        print(f"[WARN] duplicate {key_field} values ignored: {len(duplicates)} unique duplicates")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_records", required=True, help="route_records_val_epoch*.json or route_records_train_eval_epoch*.json")
    parser.add_argument("--opencompass_records", required=True, help="JSONL produced by debug_router_record_path")
    parser.add_argument("--key_field", default="prompt_sha1")
    args = parser.parse_args()

    train_records = load_training_records(args.train_records)
    opencompass_records = load_jsonl_records(args.opencompass_records)
    train_by_key = keyed_records(train_records, args.key_field)
    oc_by_key = keyed_records(opencompass_records, args.key_field)

    shared_keys = sorted(set(train_by_key) & set(oc_by_key))
    missing_in_oc = len(set(train_by_key) - set(oc_by_key))
    missing_in_train = len(set(oc_by_key) - set(train_by_key))
    pair_matches = 0
    mismatches = []
    by_task = Counter()
    by_task_match = Counter()

    for key in shared_keys:
        train_record = train_by_key[key]
        oc_record = oc_by_key[key]
        task = str(train_record.get("task") or oc_record.get("dataset") or "unknown")
        train_pair = str(train_record.get("pred_pair"))
        oc_pair = str(oc_record.get("pred_pair"))
        by_task[task] += 1
        if train_pair == oc_pair:
            pair_matches += 1
            by_task_match[task] += 1
        elif len(mismatches) < 20:
            mismatches.append(
                {
                    "key": key,
                    "task": task,
                    "train_pair": train_pair,
                    "opencompass_pair": oc_pair,
                    "train_item_id": train_record.get("item_id"),
                    "opencompass_sample_index": oc_record.get("sample_index"),
                    "prompt_preview": oc_record.get("prompt_preview") or str(train_record.get("prompt_text", ""))[:240],
                }
            )

    match_rate = pair_matches / len(shared_keys) if shared_keys else 0.0
    print(
        f"[SUMMARY] shared={len(shared_keys)} match={pair_matches} "
        f"match_rate={match_rate:.4f} missing_in_opencompass={missing_in_oc} missing_in_train={missing_in_train}"
    )
    for task, count in sorted(by_task.items()):
        task_rate = by_task_match[task] / count if count else 0.0
        print(f"[TASK][{task}] shared={count} match={by_task_match[task]} match_rate={task_rate:.4f}")
    if mismatches:
        print("[MISMATCH_EXAMPLES]")
        for item in mismatches:
            print(json.dumps(item, ensure_ascii=False))


if __name__ == "__main__":
    main()
