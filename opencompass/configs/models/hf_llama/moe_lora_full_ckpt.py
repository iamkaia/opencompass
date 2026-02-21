from opencompass.models.router_moe_llama_layer import RouterMoELlama
models = [
    dict(
        type=RouterMoELlama,
        abbr="router_moe_lora_layer_full_ckpt",
        #path="meta-llama/Llama-2-7b-chat-hf",
        path = "./Llama-2-7b-chat-hf",

        # ★ 一定要有
        layer_router_ckpt='./layer_router_full_ckpt',

        # ★ 不可以再有 cls_dir
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
        batch_size=128,###原本是8
        ##max_out_len=2048,
        max_out_len=128,
        run_cfg=dict(num_gpus=1),
    )
]


