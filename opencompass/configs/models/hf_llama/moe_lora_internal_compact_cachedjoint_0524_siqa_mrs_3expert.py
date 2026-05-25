from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)
import os

router_ckpt_dir = "./router_0524_correct_conf_ce_t1_siqa_mrs_3expert"

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=os.path.basename(os.path.normpath(router_ckpt_dir)),
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
        debug_router_record_path="0524_opencompass_router_records_siqa_mrs_3expert.jsonl",
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
