import argparse
import json
import os
import re
from collections import Counter, defaultdict
from typing import Dict, List, Optional, Sequence, Tuple

import torch


LETTER_TO_SST2 = {"A": "negative", "B": "positive"}


def load_manifest(feature_root: str, split: str) -> Dict:
    path = os.path.join(feature_root, split, "manifest.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def normalize_prompt(text: str) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def item_key(task: str, target: str, prompt: str) -> Tuple[str, str, str]:
    return str(task), str(target), normalize_prompt(prompt)


def parse_output_answer(output: str, task: str) -> Optional[str]:
    s = str(output or "").strip()
    low = s.lower().strip()
    compact = re.sub(r"\s+", " ", low)

    if re.search(r"answer:\s*$", compact):
        return None

    if task == "sst2":
        m = re.fullmatch(r"[-\s]*(positive|negative)[\.\s]*", low)
        if m:
            return m.group(1)
        m = re.fullmatch(r"[-\s]*([ab])[:\.\)\s]*", low)
        if m:
            return LETTER_TO_SST2[m.group(1).upper()]
        m = re.search(r"answer\s*:\s*[-\s]*(positive|negative)\b", low)
        if m:
            return m.group(1)
        m = re.search(r"answer\s*:\s*([ab])\b", low)
        if m:
            return LETTER_TO_SST2[m.group(1).upper()]
        return None

    m = re.fullmatch(r"\s*([ABCD])\s*[:\.\)]?(?:\s+.*)?", s, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"answer\s*:\s*([ABCD])\b", s, flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def load_generation_correct_sets(path: str) -> Dict[Tuple[str, str, str], Dict]:
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            sample = json.loads(line)
            task = str(sample["task"])
            target = str(sample["target"])
            correct_pair_indices = []
            correct_pair_names = []
            for pair in sample["pairs"]:
                answer = parse_output_answer(pair.get("output", ""), task)
                if task == "sst2":
                    ok = answer == target.lower()
                else:
                    ok = answer == target.upper()[:1]
                if ok:
                    correct_pair_indices.append(int(pair["pair_idx"]))
                    correct_pair_names.append(str(pair["name"]))
            key = item_key(task, target, sample["llama_prompt"])
            rows[key] = {
                "sample_idx": int(sample["sample_idx"]),
                "task": task,
                "target": target,
                "correct_pair_indices": correct_pair_indices,
                "correct_pair_names": correct_pair_names,
            }
    return rows


def mean(values: Sequence[float]) -> float:
    return float(sum(values) / max(len(values), 1))


def pairwise_correct_loss_rank(correct_losses: Sequence[float], wrong_losses: Sequence[float]) -> Optional[float]:
    total = len(correct_losses) * len(wrong_losses)
    if total <= 0:
        return None
    wins = 0.0
    for c in correct_losses:
        for w in wrong_losses:
            if c < w:
                wins += 1.0
            elif c == w:
                wins += 0.5
    return float(wins / total)


def analyze(feature_root: str, split: str, generation_jsonl: str, top_ks: Sequence[int]) -> Dict:
    manifest = load_manifest(feature_root, split)
    expert_names = list(manifest.get("expert_names") or manifest.get("task_names") or [])
    if not expert_names:
        raise ValueError("manifest must contain expert_names or task_names")
    num_pairs = len(expert_names) * len(expert_names)
    gen_rows = load_generation_correct_sets(generation_jsonl)

    totals = Counter()
    by_task = defaultdict(Counter)
    margin_values = []
    correct_minus_wrong_values = []
    pairwise_rank_values = []
    detail_rows = []

    for fn in manifest["files"]:
        payload = torch.load(os.path.join(feature_root, split, fn), map_location="cpu")
        for item in payload["items"]:
            key = item_key(item["task"], item["target"], item.get("prompt_text") or item.get("text"))
            gen = gen_rows.get(key)
            if gen is None:
                totals["unmatched"] += 1
                continue

            loss_matrix = item["loss_matrix"].to(torch.float32)
            flat_loss = loss_matrix.view(-1)
            if flat_loss.numel() != num_pairs:
                raise ValueError(f"Unexpected loss_matrix shape={tuple(loss_matrix.shape)} in {fn}")

            sorted_loss, sorted_idx = flat_loss.sort()
            top1 = int(sorted_idx[0].item())
            correct = set(gen["correct_pair_indices"])
            wrong = [idx for idx in range(num_pairs) if idx not in correct]

            totals["matched"] += 1
            totals["with_any_correct"] += int(bool(correct))
            totals["with_no_correct"] += int(not correct)
            totals["top1_in_correct"] += int(top1 in correct)
            task_counter = by_task[str(item["task"])]
            task_counter["matched"] += 1
            task_counter["with_any_correct"] += int(bool(correct))
            task_counter["top1_in_correct"] += int(top1 in correct)

            for k in top_ks:
                topk = set(int(idx.item()) for idx in sorted_idx[:k])
                hit = bool(topk & correct)
                totals[f"top{k}_hit_correct"] += int(hit)
                task_counter[f"top{k}_hit_correct"] += int(hit)

            if len(sorted_loss) > 1:
                margin_values.append(float((sorted_loss[1] - sorted_loss[0]).item()))

            if correct and wrong:
                correct_losses = [float(flat_loss[idx].item()) for idx in sorted(correct)]
                wrong_losses = [float(flat_loss[idx].item()) for idx in wrong]
                correct_mean = mean(correct_losses)
                wrong_mean = mean(wrong_losses)
                diff = correct_mean - wrong_mean
                rank_score = pairwise_correct_loss_rank(correct_losses, wrong_losses)
                correct_minus_wrong_values.append(diff)
                if rank_score is not None:
                    pairwise_rank_values.append(rank_score)
                totals["correct_mean_loss_lower_than_wrong"] += int(correct_mean < wrong_mean)
                task_counter["correct_mean_loss_lower_than_wrong"] += int(correct_mean < wrong_mean)
                detail_rows.append({
                    "sample_idx": gen["sample_idx"],
                    "task": item["task"],
                    "target": item["target"],
                    "correct_count": len(correct),
                    "loss_top1_pair_idx": top1,
                    "loss_top1_pair_name": f"{expert_names[top1 // len(expert_names)]}->{expert_names[top1 % len(expert_names)]}",
                    "loss_top1_in_correct": top1 in correct,
                    "correct_mean_loss": correct_mean,
                    "wrong_mean_loss": wrong_mean,
                    "correct_minus_wrong_mean_loss": diff,
                    "pairwise_correct_lower_than_wrong_rate": rank_score,
                    "correct_pair_names": gen["correct_pair_names"],
                })

    matched = max(totals["matched"], 1)
    summary = {
        "feature_root": feature_root,
        "split": split,
        "generation_jsonl": generation_jsonl,
        "expert_names": expert_names,
        "matched": int(totals["matched"]),
        "unmatched": int(totals["unmatched"]),
        "with_any_correct": int(totals["with_any_correct"]),
        "with_no_correct": int(totals["with_no_correct"]),
        "top1_in_correct_rate": float(totals["top1_in_correct"] / matched),
        "avg_best_second_loss_margin": mean(margin_values),
        "avg_correct_minus_wrong_mean_loss": mean(correct_minus_wrong_values),
        "avg_pairwise_correct_lower_than_wrong_rate": mean(pairwise_rank_values),
        "topk_hit_correct_rates": {
            str(k): float(totals[f"top{k}_hit_correct"] / matched) for k in top_ks
        },
        "by_task": {},
        "details": detail_rows,
    }
    for task, counter in sorted(by_task.items()):
        denom = max(counter["matched"], 1)
        summary["by_task"][task] = {
            "matched": int(counter["matched"]),
            "with_any_correct": int(counter["with_any_correct"]),
            "top1_in_correct_rate": float(counter["top1_in_correct"] / denom),
            "topk_hit_correct_rates": {
                str(k): float(counter[f"top{k}_hit_correct"] / denom) for k in top_ks
            },
            "correct_mean_loss_lower_than_wrong_rate": float(
                counter["correct_mean_loss_lower_than_wrong"] / denom
            ),
        }
    return summary


def print_summary(summary: Dict):
    print(
        f"[GEN-VS-LOSS] matched={summary['matched']} unmatched={summary['unmatched']} "
        f"with_any_correct={summary['with_any_correct']} with_no_correct={summary['with_no_correct']}"
    )
    print(
        f"[GEN-VS-LOSS] top1_in_correct={summary['top1_in_correct_rate']:.2%} "
        f"avg_margin={summary['avg_best_second_loss_margin']:.4f} "
        f"avg_correct_minus_wrong_loss={summary['avg_correct_minus_wrong_mean_loss']:.4f} "
        f"pairwise_correct_lower_wrong={summary['avg_pairwise_correct_lower_than_wrong_rate']:.2%}"
    )
    topk = " ".join(f"top{k}={v:.2%}" for k, v in summary["topk_hit_correct_rates"].items())
    print(f"[GEN-VS-LOSS] {topk}")
    for task, row in summary["by_task"].items():
        task_topk = " ".join(f"top{k}={v:.2%}" for k, v in row["topk_hit_correct_rates"].items())
        print(
            f"[GEN-VS-LOSS][{task}] n={row['matched']} "
            f"top1={row['top1_in_correct_rate']:.2%} {task_topk} "
            f"correct_mean_lower={row['correct_mean_loss_lower_than_wrong_rate']:.2%}"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature_root", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--generation_jsonl", type=str, required=True)
    parser.add_argument("--top_ks", type=str, default="1,3,5")
    parser.add_argument("--output_json", type=str, default=None)
    args = parser.parse_args()

    top_ks = [int(part) for part in str(args.top_ks).split(",") if part.strip()]
    summary = analyze(
        feature_root=args.feature_root,
        split=args.split,
        generation_jsonl=args.generation_jsonl,
        top_ks=top_ks,
    )
    print_summary(summary)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
            f.write("\n")


if __name__ == "__main__":
    main()
