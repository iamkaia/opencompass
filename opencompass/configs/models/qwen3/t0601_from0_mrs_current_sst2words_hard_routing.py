import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


task = os.environ.get("T0601_CURRENT_TASK", "boolq").lower()
record_tag = os.environ.get("T0601_ROUTER_RECORD_TAG", "").strip()
record_tag_part = f"{record_tag}_" if record_tag else ""
router_bert_init = os.environ.get("T0601_ROUTER_BERT_INIT", "./task_classifier_ckpt")
routing_mode = os.environ.get("T0601_ROUTING_MODE", "hard")
valid_tasks = {"boolq", "rte", "siqa", "piqa"}
if task not in valid_tasks:
    raise ValueError(
        f"Unknown T0601_CURRENT_TASK={task!r}; expected one of {sorted(valid_tasks)}"
    )

router_ckpt_dir = os.environ.get("T0601_ROUTER_CKPT") or (
    f"./router_t0601_qwen3_fp16_from0_mrs_current_trainbert_step01_"
    f"{task}_correct_conf_ce_t1_3expert_sst2words"
)

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_hard_routing",
        path="Qwen/Qwen3-4B-Instruct-2507",
        router_ckpt_dir=router_ckpt_dir,
        router_bert_init=router_bert_init,
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
        routing_mode=routing_mode,
        debug_router_record_path=(
            f"t0601_opencompass_router_records_qwen3_from0_mrs_current_"
            f"{record_tag_part}{task}_sst2words_hard_routing.jsonl"
        ),
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
