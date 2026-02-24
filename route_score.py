import os, json, glob
from collections import defaultdict

EXP_DIR = "./outputs/default/20260224_184742"  # 改成你的
PRED_DIR = os.path.join(EXP_DIR, "predictions")

# stats[kind][dataset] = accumulators
stats = defaultdict(lambda: defaultdict(lambda: {
    "n": 0,
    "ok": 0,
    "ok_den": 0,          # routing acc denominator (non-null)
    "match_sum": 0.0,
    "match_den": 0,
    "cov_sum": 0.0,
    "cov_den": 0,
    "cov_gt0": 0,
    "ok_cov_ok": 0,
    "ok_cov_den": 0,      # routing acc when coverage>0
    "conf": defaultdict(int),  # (gt, pred) -> count
}))

def infer_dataset_name(fp, rec):
    # 優先用檔名 routing_log__<ds>.jsonl
    base = os.path.basename(fp)
    if base.startswith("routing_log__") and base.endswith(".jsonl"):
        return base[len("routing_log__"):-len(".jsonl")]
    # fallback: 用 gt_task
    return rec.get("gt_task", "unknown")

for fp in glob.glob(os.path.join(PRED_DIR, "routing_log__*.jsonl")) + \
         glob.glob(os.path.join(PRED_DIR, "*", "routing_log__*.jsonl")) + \
         glob.glob(os.path.join(PRED_DIR, "*", "routing_log.jsonl")) + \
         glob.glob(os.path.join(PRED_DIR, "routing_log.jsonl")):

    with open(fp, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)

            kind = r.get("kind", "unknown")  # external / layer_router / dmole
            ds = infer_dataset_name(fp, r)

            s = stats[kind][ds]
            s["n"] += 1

            gt = r.get("gt_task", None)

            # ---- choose predicted task field ----
            pred = None
            if "routed_task" in r:
                pred = r.get("routed_task")
            elif "major_task" in r:
                pred = r.get("major_task")

            if gt is not None and pred is not None:
                s["conf"][(gt, pred)] += 1

            # ---- routing accuracy ----
            if "route_ok" in r:
                v = r.get("route_ok")
            else:
                v = r.get("route_ok_major")

            if v is not None:
                s["ok_den"] += 1
                s["ok"] += 1 if v else 0

            # ---- match rate ----
            mr = r.get("match_rate", None)
            if mr is not None:
                s["match_den"] += 1
                s["match_sum"] += float(mr)

            # ---- coverage ----
            cov = r.get("coverage", None)
            if cov is not None:
                cov = float(cov)
                s["cov_den"] += 1
                s["cov_sum"] += cov
                if cov > 0:
                    s["cov_gt0"] += 1

                # dmole only: routing acc when coverage>0
                if cov > 0 and v is not None:
                    s["ok_cov_den"] += 1
                    s["ok_cov_ok"] += 1 if v else 0

# ---- print summary ----
for kind in sorted(stats.keys()):
    print("=" * 90)
    print("KIND:", kind)
    for ds in sorted(stats[kind].keys()):
        s = stats[kind][ds]
        routing_acc = (s["ok"] / s["ok_den"]) if s["ok_den"] else None
        match_mean = (s["match_sum"] / s["match_den"]) if s["match_den"] else None
        cov_mean = (s["cov_sum"] / s["cov_den"]) if s["cov_den"] else None
        cov_gt0_frac = (s["cov_gt0"] / s["cov_den"]) if s["cov_den"] else None
        routing_acc_when_cov = (s["ok_cov_ok"] / s["ok_cov_den"]) if s["ok_cov_den"] else None

        print(f"- {ds}")
        print(f"    samples = {s['n']}")
        print(f"    routing_acc = {routing_acc}  (ok={s['ok']}/{s['ok_den']})")
        if match_mean is not None:
            print(f"    match_rate_mean = {match_mean:.4f}  (n={s['match_den']})")
        if cov_mean is not None:
            print(f"    coverage_mean = {cov_mean:.4f}  (n={s['cov_den']})")
            print(f"    coverage>0 frac = {cov_gt0_frac:.4f}")
            if routing_acc_when_cov is not None:
                print(f"    routing_acc | coverage>0 = {routing_acc_when_cov:.4f}  (n={s['ok_cov_den']})")
