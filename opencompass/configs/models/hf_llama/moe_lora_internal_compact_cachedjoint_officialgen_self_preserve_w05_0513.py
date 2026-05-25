from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr="router_moe_internal_compact_cachedjoint_officialgen_self_preserve_w05_0513",
        path="meta-llama/Llama-2-7b-chat-hf",
        router_ckpt_dir="./router_debug_500_250_officialgen_self_preserve_w05_0513",
        router_bert_init="./task_classifier_ckpt",
        lora_paths=dict(
            iwslt2017="./saves/llama2-7b-chat-hf/lora/sft_iwslt",
            medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
            race="./saves/llama2-7b-chat-hf/lora/sft_race",
            squad2="./saves/llama2-7b-chat-hf/lora/sft_squad20",
            sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
        ),
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=15,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=128,
        debug_router_record_path="0513_opencompass_router_records_w05.jsonl",
        debug_router_topk=5,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
