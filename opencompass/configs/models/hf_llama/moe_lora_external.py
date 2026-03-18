from opencompass.models.router_moe_llama import RouterMoELlama

models = [
    dict(
        type=RouterMoELlama,
        abbr="router_moe_lora_v2（external)",
        path="meta-llama/Llama-2-7b-chat-hf",
        #cls_dir="./task_classifier_ckpt",
        router_ckpt_dir="./router_ckpt_stage2",
        lora_paths=dict(
            iwslt2017="./saves/llama2-7b-chat-hf/lora/sft_iwslt",
            medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
            race="./saves/llama2-7b-chat-hf/lora/sft_race",
            squad2="./saves/llama2-7b-chat-hf/lora/sft_squad20",
            sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
        ),
        dtype='float16',
        r=8,
        alpha=32,  # 會被 adapter_config 覆蓋，但保留一致性
        batch_size=64,###原本是8
        max_out_len=128,
        run_cfg=dict(num_gpus=1),
    )
]

