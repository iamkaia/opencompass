#!/usr/bin/env python3
import argparse
import ast
import json
import re
from collections import defaultdict
from pathlib import Path


CKPT_CFG_RE = re.compile(r"\[INFO\] loaded router ckpt config: (.+)")
ROUTING_RE = re.compile(
    r"\[ROUTING\]\[dataset=(?P<dataset>[^\]]+)\]\s+sample_count=(?P<count>\d+)\s+pair_distribution=(?P<dist>.+)"
)


def parse_keyvals(blob: str) -> dict:
    result = {}
    for part in blob.split(", "):
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            result[key] = ast.literal_eval(value)
        except Exception:
            result[key] = value
    return result


def parse_pair_distribution(text: str):
    entries = []
    for item in text.split(", "):
        pair, rest = item.split(":", 1)
        m = re.match(r"(?P<hits>\d+)/(?P<total>\d+)\s+\((?P<ratio>[^)]+)\)", rest.strip())
        if not m:
            continue
        entries.append(
            {
                "pair": pair.strip(),
                "count": int(m.group("hits")),
                "total": int(m.group("total")),
                "ratio": m.group("ratio"),
            }
        )
    return entries


def parse_log(path: Path) -> dict:
    data = {
        "path": str(path),
        "model_tag": infer_model_tag(path),
        "ckpt_config": {},
        "datasets": defaultdict(dict),
    }
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            m = CKPT_CFG_RE.search(line)
            if m:
                data["ckpt_config"] = parse_keyvals(m.group(1))
                continue

            m = ROUTING_RE.search(line)
            if m:
                dataset = m.group("dataset")
                pair_entries = parse_pair_distribution(m.group("dist"))
                data["datasets"][dataset] = {
                    "sample_count": int(m.group("count")),
                    "pair_distribution": pair_entries,
                    "top_pair": pair_entries[0]["pair"] if pair_entries else None,
                    "top_ratio": pair_entries[0]["ratio"] if pair_entries else None,
                }
    data["datasets"] = dict(data["datasets"])
    return data


def infer_model_tag(path: Path) -> str:
    name = path.name
    for tag in ("copa", "boolq", "siqa"):
        if tag in name:
            return tag
    return path.stem


def print_human_summary(records: list[dict]):
    print("== Router Checkpoint Config ==")
    for rec in records:
        cfg = rec["ckpt_config"]
        print(f"[{rec['model_tag']}] {rec['path']}")
        if not cfg:
            print("  ckpt_config: <not found in log>")
            continue
        print(f"  task_names: {cfg.get('task_names')}")
        print(f"  first_layer_idx: {cfg.get('first_layer_idx')}")
        print(f"  middle_layer_idx: {cfg.get('middle_layer_idx')}")
        print(f"  router_max_len: {cfg.get('router_max_len')}")
    print()

    all_datasets = sorted({ds for rec in records for ds in rec["datasets"].keys()})
    print("== Dataset Routing Summary ==")
    for dataset in all_datasets:
        print(f"[dataset={dataset}]")
        for rec in records:
            ds = rec["datasets"].get(dataset)
            if not ds:
                print(f"  {rec['model_tag']}: <missing>")
                continue
            print(
                f"  {rec['model_tag']}: sample_count={ds['sample_count']} "
                f"top_pair={ds['top_pair']} top_ratio={ds['top_ratio']}"
            )
        print()

    print("== Full Pair Distribution ==")
    for rec in records:
        print(f"[{rec['model_tag']}]")
        if not rec["datasets"]:
            print("  <no routing summary found>")
            continue
        for dataset, ds in sorted(rec["datasets"].items()):
            print(f"  dataset={dataset} sample_count={ds['sample_count']}")
            for item in ds["pair_distribution"]:
                print(f"    {item['pair']}: {item['count']}/{item['total']} ({item['ratio']})")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Compare OpenCompass routing summaries across multiple run logs."
    )
    parser.add_argument("logs", nargs="+", help="Paths to run logs.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    records = []
    for log in args.logs:
        path = Path(log)
        if not path.exists():
            raise FileNotFoundError(f"Log file not found: {log}")
        records.append(parse_log(path))

    if args.json:
        print(json.dumps(records, indent=2, ensure_ascii=False))
        return

    print_human_summary(records)


if __name__ == "__main__":
    main()
