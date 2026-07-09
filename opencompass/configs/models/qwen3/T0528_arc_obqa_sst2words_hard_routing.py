import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


task = os.environ.get("T0528_ROUTER_TASK", "").lower()
mode = os.environ.get("T0528_ROUTER_MODE", "").lower()
valid_tasks = {"openbookqa", "arc_c"}
valid_modes = {"new_only", "replay"}
if task not in valid_tasks:
    raise ValueError(f"Set T0528_ROUTER_TASK to one of: {sorted(valid_tasks)}")
if mode not in valid_modes:
    raise ValueError(f"Set T0528_ROUTER_MODE to one of: {sorted(valid_modes)}")

if mode == "new_only":
    router_ckpt_dir = (
        f"./router_T0528_qwen3_fp16_new_only_{task}_from_mrs_"
        "correct_conf_ce_t1_3expert_sst2words"
    )
else:
    router_ckpt_dir = (
        f"./router_T0528_qwen3_fp16_replay_mrs_{task}_"
        "correct_conf_ce_t1_3expert_sst2words"
    )

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_hard_routing",
        path="Qwen/Qwen3-4B-Instruct-2507",
        router_ckpt_dir=router_ckpt_dir,
        router_bert_init="./task_classifier_ckpt",
        lora_paths=dict(
            medmcqa="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa",
            race="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race",
            sst2="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2",
        ),
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=18,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=64,
        routing_mode="hard",
        debug_router_record_path=(
            f"T0528_opencompass_router_records_qwen3_{mode}_{task}_"
            "sst2words_hard_routing.jsonl"
        ),
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
