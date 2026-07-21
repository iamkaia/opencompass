#!/usr/bin/env python3
import os
import subprocess
import sys
from pathlib import Path

from tqdm.auto import tqdm


ROOT = Path(__file__).resolve().parents[1]


def cached_qwen3_model_path() -> str:
    snapshot_root = Path.home() / ".cache/huggingface/hub/models--Qwen--Qwen3-4B-Instruct-2507/snapshots"
    if snapshot_root.is_dir():
        snapshots = sorted(p for p in snapshot_root.iterdir() if p.is_dir())
        if snapshots:
            return str(snapshots[-1])
    return "Qwen/Qwen3-4B-Instruct-2507"


def build_env() -> dict:
    env = os.environ.copy()
    env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    return env


def split_csv(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [part.strip() for part in raw.split(",") if part.strip()]


def export_output_name(cache_root: str, split: str) -> str:
    root_name = Path(cache_root).name
    return f"{root_name}_{split}_all_pairs.json"


def is_truthy(raw: str | None) -> bool:
    if raw is None:
        return False
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def run_cmd(cmd: list[str], *, env: dict, log_path: Path | None = None) -> None:
    if log_path is None:
        subprocess.run(cmd, cwd=ROOT, env=env, check=True)
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        proc = subprocess.Popen(
            cmd,
            cwd=ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            log_file.write(line)
            log_file.flush()
        rc = proc.wait()
        if rc != 0:
            raise subprocess.CalledProcessError(rc, cmd)


def main() -> int:
    mode = os.environ.get("MODE", "qwen3")
    data_root = os.environ.get("DATA_ROOT", "./0527_router_train_dataset")
    bert = os.environ.get("BERT", "./task_classifier_ckpt")
    build_cache = is_truthy(os.environ.get("BUILD_CACHE"))

    if mode == "qwen3":
        model_path = os.environ.get("MODEL_PATH", cached_qwen3_model_path())
        lora_root = os.environ.get("LORA_ROOT", "./saves/Qwen/Qwen3-4B-Instruct-2507/lora")
        feature_root = os.environ.get("FEATURE_ROOT", "./0530_qwen3_fp16_cache_all_3expert_official_eval_aligned_sst2words")
        expert_names = os.environ.get("EXPERT_NAMES", "medmcqa,race,sst2")
        lora_medmcqa = os.environ.get("LORA_MEDMCQA", f"{lora_root}/sft_medmcqa")
        lora_race = os.environ.get("LORA_RACE", f"{lora_root}/sft_race")
        lora_sst2 = os.environ.get("LORA_SST2", f"{lora_root}/sft_sst2")
        lora_iwslt = os.environ.get("LORA_IWSLT", "")
        lora_squad2 = os.environ.get("LORA_SQUAD2", "")
        default_export_roots = [
            "./0527_qwen3_fp16_cache_mrs_3expert_official_eval_aligned_sst2words",
            "./0527_qwen3_fp16_cache_4other_3expert_official_eval_aligned_sst2words",
        ]
    elif mode == "llama":
        model_path = os.environ.get("MODEL_PATH", "meta-llama/Llama-2-7b-chat-hf")
        lora_root = os.environ.get("LORA_ROOT", "./saves/llama2-7b-chat-hf/lora")
        feature_root = os.environ.get("FEATURE_ROOT", "./0530_llama_cache_all_3expert_official_eval_aligned")
        expert_names = os.environ.get("EXPERT_NAMES", "medmcqa,race,sst2")
        lora_medmcqa = os.environ.get("LORA_MEDMCQA", f"{lora_root}/sft_medmcqa")
        lora_race = os.environ.get("LORA_RACE", f"{lora_root}/sft_race")
        lora_sst2 = os.environ.get("LORA_SST2", f"{lora_root}/sft_sst2")
        lora_iwslt = os.environ.get("LORA_IWSLT", "")
        lora_squad2 = os.environ.get("LORA_SQUAD2", "")
        default_export_roots = [
            "./0527_llama_cache_mrs_3expert_official_eval_aligned",
            "./0527_llama_cache_4other_3expert_official_eval_aligned",
            "./0528_llama_cache_arc_c_openbookqa_3expert_official_eval_aligned",
        ]
    else:
        print(f"unknown MODE: {mode}", file=sys.stderr)
        return 2

    export_roots = split_csv(os.environ.get("CACHE_ROOTS")) or default_export_roots
    extra_export_roots = split_csv(os.environ.get("EXTRA_EXPORT_ROOTS"))
    if build_cache:
        export_roots = [feature_root] + export_roots

    train_json = os.environ.get("TRAIN_JSON", f"{feature_root}_train_all_pairs.json")
    validation_json = os.environ.get("VALIDATION_JSON", f"{feature_root}_validation_all_pairs.json")
    log_file = os.environ.get("LOG_FILE", f"{feature_root}.log")

    env = build_env()
    print(f"[RUN] mode={mode} build_cache={build_cache} data_root={data_root}")
    print(f"[RUN] feature_root={feature_root}")
    print(f"[RUN] export_roots={export_roots}")
    if extra_export_roots:
        print(f"[RUN] extra_export_roots={extra_export_roots}")

    stages = []
    if build_cache:
        stages.append((
            "building cache",
            [
                sys.executable,
                "-u",
                "build_cached_router_pair_dataset.py",
                "--data_root", data_root,
                "--feature_root", feature_root,
                "--base_model_path", model_path,
                "--router_bert_init", bert,
                "--expert_names", expert_names,
                "--lora_iwslt", lora_iwslt,
                "--lora_medmcqa", lora_medmcqa,
                "--lora_race", lora_race,
                "--lora_squad2", lora_squad2,
                "--lora_sst2", lora_sst2,
            ],
            log_file,
        ))
        stages.append((
            "exporting built train",
            [
                sys.executable,
                "tools/export_router_sample_outputs.py",
                "--cache_root", feature_root,
                "--split", "train",
                "--output_json", train_json,
            ],
            None,
        ))
        stages.append((
            "exporting built validation",
            [
                sys.executable,
                "tools/export_router_sample_outputs.py",
                "--cache_root", feature_root,
                "--split", "validation",
                "--output_json", validation_json,
            ],
            None,
        ))

    for cache_root in export_roots:
        stages.append((
            f"exporting {Path(cache_root).name} train",
            [
                sys.executable,
                "tools/export_router_sample_outputs.py",
                "--cache_root", cache_root,
                "--split", "train",
                "--output_json", export_output_name(cache_root, "train"),
            ],
            None,
        ))
        stages.append((
            f"exporting {Path(cache_root).name} validation",
            [
                sys.executable,
                "tools/export_router_sample_outputs.py",
                "--cache_root", cache_root,
                "--split", "validation",
                "--output_json", export_output_name(cache_root, "validation"),
            ],
            None,
        ))

    with tqdm(total=len(stages), desc="0527 router export", dynamic_ncols=True) as pbar:
        for desc, cmd, log_path in stages:
            pbar.set_description(desc)
            run_cmd(cmd, env=env, log_path=ROOT / log_path if log_path else None)
            pbar.update(1)

    if build_cache:
        print(f"[DONE] cache_root={feature_root}")
        print(f"[DONE] train_json={train_json}")
        print(f"[DONE] validation_json={validation_json}")
    else:
        print("[DONE] export-only mode")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
