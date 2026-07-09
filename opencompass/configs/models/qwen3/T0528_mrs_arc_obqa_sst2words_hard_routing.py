import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


router_ckpt_dir = "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words"

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_T0528_arc_obqa_hard_routing",
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
            "T0528_opencompass_router_records_qwen3_mrs_arc_obqa_"
            "sst2words_hard_routing.jsonl"
        ),
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
