from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
            RouterMoELlamaInternalCompactCachedJoint,
)
import os

router_ckpt_dir = "./router_ckpt_cl_copa_from_5task_tokennll_mixed_norm_expected"
####router_ckpt_dir = "./router_ckpt_cl_copa_expected_taskaware"
####outer_ckpt_dir="./router_ckpt_newonly_copa_from_5expert_freezebert"

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=os.path.basename(os.path.normpath(router_ckpt_dir)),
        path="meta-llama/Llama-2-7b-chat-hf",
        router_ckpt_dir=router_ckpt_dir,
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
        batch_size=128,
        max_seq_len=2048,
        max_out_len=64,
        run_cfg=dict(num_gpus=1),
    )
]
