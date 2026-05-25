from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


_lora_paths = dict(
    iwslt2017="./saves/llama2-7b-chat-hf/lora/sft_iwslt",
    medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
    race="./saves/llama2-7b-chat-hf/lora/sft_race",
    squad2="./saves/llama2-7b-chat-hf/lora/sft_squad20",
    sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
)


models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr="router_aligncheck_expected",
        path="meta-llama/Llama-2-7b-chat-hf",
        router_ckpt_dir="./router_ckpt_officialgen_debug_bs16_expected_0519_195419",
        router_bert_init="./task_classifier_ckpt",
        lora_paths=_lora_paths,
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=15,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=128,
        generation_kwargs=dict(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        ),
        debug_router_record_path="router_aligncheck_expected_records.jsonl",
        debug_router_topk=0,
        debug_router_max_prints=0,
        run_cfg=dict(num_gpus=1),
    )
]
