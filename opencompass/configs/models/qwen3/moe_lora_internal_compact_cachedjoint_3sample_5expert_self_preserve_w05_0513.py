from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr="qwen3_4b_2507_router_moe_cachedjoint_3sample_5expert_self_preserve_w05_0513",
        path="Qwen/Qwen3-4B-Instruct-2507",
        router_ckpt_dir="./router_qwen3_4b_2507_3sample_5expert_self_preserve_w05_0513",
        router_bert_init="./task_classifier_ckpt",
        lora_paths=dict(
            iwslt2017="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_iwslt",
            medmcqa="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa",
            race="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race",
            squad2="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_squad2",
            sst2="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2",
        ),
        dtype="bfloat16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=18,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=128,
        debug_router_record_path="0513_opencompass_router_records_qwen3_3sample_5expert_w05.jsonl",
        debug_router_topk=5,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
