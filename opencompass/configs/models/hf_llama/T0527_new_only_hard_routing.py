import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


task = os.environ.get("T0527_NEW_ONLY_TASK", "mrs").lower()
valid_tasks = {"mrs", "boolq", "rte", "siqa", "piqa"}
if task not in valid_tasks:
    raise ValueError(
        f"Unknown T0527_NEW_ONLY_TASK={task!r}; expected one of {sorted(valid_tasks)}"
    )

if task == "mrs":
    router_ckpt_dir = "./router_T0527_llama_mrs_correct_conf_ce_t1_3expert"
else:
    router_ckpt_dir = (
        f"./router_T0527_llama_new_only_{task}_from_mrs_correct_conf_ce_t1_3expert"
    )

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_hard_routing",
        path="meta-llama/Llama-2-7b-chat-hf",
        router_ckpt_dir=router_ckpt_dir,
        router_bert_init="./task_classifier_ckpt",
        lora_paths=dict(
            medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
            race="./saves/llama2-7b-chat-hf/lora/sft_race",
            sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
        ),
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=64,
        routing_mode="hard",
        debug_router_record_path=(
            f"T0527_opencompass_router_records_llama_new_only_sequence_"
            f"{task}_hard_routing.jsonl"
        ),
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
