'''
from opencompass.models.router_moe_llama import RouterMoELlama

models = [
    dict(
        type=RouterMoELlama,
        abbr="router_moe_lora_v2",
        path="meta-llama/Llama-2-7b-chat-hf",
        cls_dir="./task_classifier_ckpt",
        lora_paths=dict(
            iwslt2017="./saves/llama2-7b-chat-hf/lora/sft_iwslt",
            medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
            race="./saves/llama2-7b-chat-hf/lora/sft_race",
            squad2="./saves/llama2-7b-chat-hf/lora/sft_squad20",
            sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
        ),
        batch_size=8,
        max_out_len=2048,
        run_cfg=dict(num_gpus=1),
    )
]
'''

from opencompass.models.router_moe_llama_layer import RouterMoELlama

models = [
    dict(
        type=RouterMoELlama,
        abbr="router_moe_lora_layer",
        #path="meta-llama/Llama-2-7b-chat-hf",
        path = "/home/kaia/opencompass/Llama-2-7b-chat-hf",

        # ★ 一定要有
        layer_router_ckpt="/home/kaia/recall_1108/layer_router_full_ckpt",

        # ★ 不可以再有 cls_dir
        lora_paths=dict(
            iwslt2017="./saves/llama2-7b-chat-hf/lora/sft_iwslt",
            medmcqa="./saves/llama2-7b-chat-hf/lora/sft_medmcqa",
            race="./saves/llama2-7b-chat-hf/lora/sft_race",
            squad2="./saves/llama2-7b-chat-hf/lora/sft_squad20",
            sst2="./saves/llama2-7b-chat-hf/lora/sft_sst2",
        ),

        batch_size=8,
        max_out_len=2048,
        run_cfg=dict(num_gpus=1),
    )
]

