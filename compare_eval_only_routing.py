import argparse
import json
import math
import os
import re
from typing import Dict, Iterable, List, Tuple


PAIR_DIST_RE = re.compile(r"^\[ROUTE\]\[(?P<tag>[^\]]+)\](?:\[(?P<task>[^\]]+)\])?\[PAIR_DIST\]\s+(?P<dist>.+)$")


def _read_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _summary_path(path: str) -> str:
    if os.path.isdir(path):
        candidate = os.path.join(path, "routing_summary_val_eval_only.json")
        if os.path.exists(candidate):
            return candidate
        candidate = os.path.join(path, "routing_summary_validation.json")
        if os.path.exists(candidate):
            return candidate
    return path


def _rows_to_dist(rows: Iterable[Dict]) -> Dict[str, float]:
    return {str(row["name"]): float(row.get("rate", 0.0)) for row in rows}


def _parse_percent_dist(text: str) -> Dict[str, float]:
    out = {}
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk or ":" not in chunk:
            continue
        name, value = chunk.rsplit(":", 1)
        value = value.strip().rstrip("%")
        try:
            out[name.strip()] = float(value) / 100.0
        except ValueError:
            continue
    return out


def _load_from_log(path: str) -> Dict[str, Dict[str, float]]:
    dists: Dict[str, Dict[str, float]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            match = PAIR_DIST_RE.match(line.strip())
            if not match:
                continue
            tag = match.group("tag")
            if "EVAL-ONLY-VAL" not in tag and "VAL" not in tag:
                continue
            task = match.group("task") or "__overall__"
            dists[task] = _parse_percent_dist(match.group("dist"))
    return dists


def load_routing_dists(path: str) -> Dict[str, Dict[str, float]]:
    path = _summary_path(path)
    if path.endswith(".log"):
        return _load_from_log(path)

    summary = _read_json(path)
    dists = {"__overall__": _rows_to_dist(summary.get("all_pred_pairs", []))}
    for row in summary.get("per_task", []):
        dists[str(row["task"])] = _rows_to_dist(row.get("all_pred_pairs", []))
    return dists


def l1_distance(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = set(a) | set(b)
    return sum(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys)


def js_divergence(a: Dict[str, float], b: Dict[str, float]) -> float:
    keys = sorted(set(a) | set(b))
    pa = [max(a.get(k, 0.0), 0.0) for k in keys]
    pb = [max(b.get(k, 0.0), 0.0) for k in keys]
    sa = sum(pa) or 1.0
    sb = sum(pb) or 1.0
    pa = [x / sa for x in pa]
    pb = [x / sb for x in pb]
    pm = [(x + y) / 2.0 for x, y in zip(pa, pb)]

    def kl(p: List[float], q: List[float]) -> float:
        total = 0.0
        for x, y in zip(p, q):
            if x > 0 and y > 0:
                total += x * math.log2(x / y)
        return total

    return 0.5 * kl(pa, pm) + 0.5 * kl(pb, pm)


def top_pair(dist: Dict[str, float]) -> Tuple[str, float]:
    if not dist:
        return "-", 0.0
    name, rate = max(dist.items(), key=lambda item: item[1])
    return name, rate


def changed_pairs(
    baseline: Dict[str, float],
    current: Dict[str, float],
    limit: int,
) -> List[Tuple[str, float, float, float]]:
    rows = []
    for pair in sorted(set(baseline) | set(current)):
        b = baseline.get(pair, 0.0)
        c = current.get(pair, 0.0)
        rows.append((pair, b, c, c - b))
    rows.sort(key=lambda row: abs(row[3]), reverse=True)
    return rows[:limit]


def format_rate(value: float) -> str:
    return f"{value * 100:.2f}%"


def compare_one(name: str, baseline: Dict[str, Dict[str, float]], current: Dict[str, Dict[str, float]], top_changes: int):
    print(f"\n=== {name} ===")
    tasks = ["__overall__"] + sorted((set(baseline) | set(current)) - {"__overall__"})
    for task in tasks:
        b = baseline.get(task, {})
        c = current.get(task, {})
        b_top, b_rate = top_pair(b)
        c_top, c_rate = top_pair(c)
        l1 = l1_distance(b, c)
        jsd = js_divergence(b, c)
        label = "overall" if task == "__overall__" else task
        flag = ""
        if b_top != c_top:
            flag = "  TOP_CHANGED"
        elif l1 >= 0.20:
            flag = "  DRIFT_HIGH"
        elif l1 >= 0.10:
            flag = "  drift_mid"
        print(
            f"[{label}] l1={l1:.4f} js={jsd:.4f} "
            f"base_top={b_top}:{format_rate(b_rate)} "
            f"cur_top={c_top}:{format_rate(c_rate)}{flag}"
        )
        for pair, base_rate, cur_rate, delta in changed_pairs(b, c, top_changes):
            if abs(delta) <= 1e-9:
                continue
            print(
                f"  {pair}: {format_rate(base_rate)} -> {format_rate(cur_rate)} "
                f"({delta * 100:+.2f}%)"
            )


def main():
    parser = argparse.ArgumentParser(
        description="Compare eval-only router distributions before and after CL."
    )
    parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline eval-only out_dir, routing_summary_val_eval_only.json, or log file.",
    )
    parser.add_argument(
        "--compare",
        nargs="+",
        required=True,
        help="CL eval-only out_dirs, routing_summary_val_eval_only.json files, or log files.",
    )
    parser.add_argument("--top_changes", type=int, default=5)
    args = parser.parse_args()

    baseline = load_routing_dists(args.baseline)
    for path in args.compare:
        name = os.path.basename(path.rstrip("/"))
        compare_one(name, baseline, load_routing_dists(path), args.top_changes)


if __name__ == "__main__":
    main()
