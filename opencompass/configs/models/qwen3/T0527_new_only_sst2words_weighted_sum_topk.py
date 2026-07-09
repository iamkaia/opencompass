import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)

task = os.environ.get("T0527_NEW_ONLY_TASK", "")
if task not in {"mrs", "boolq", "rte", "siqa", "piqa"}:
    raise ValueError("Set T0527_NEW_ONLY_TASK to one of: mrs, boolq, rte, siqa, piqa")

if task == "mrs":
    router_ckpt_dir = "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words"
else:
    router_ckpt_dir = f"./router_T0527_qwen3_fp16_new_only_{task}_from_mrs_correct_conf_ce_t1_3expert_sst2words"

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_weighted_sum_top3_s1",
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
        routing_mode="weighted_sum",
        routing_sharpness=1.0,
        routing_topk=3,
        debug_router_record_path=f"T0527_opencompass_router_records_qwen3_new_only_sequence_{task}_sst2words_weighted_sum_top3.jsonl",
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
