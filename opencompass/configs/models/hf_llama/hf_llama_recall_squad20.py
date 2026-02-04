from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="squad2_lora",
        path='meta-llama/Llama-2-7b-chat-hf',
        peft_path="./saves/llama2-7b-chat-hf/lora/sft_squad20",
        peft_kwargs=dict(local_files_only=True),   # ⭐ 這行非常關鍵
        max_out_len=2048, ###原本是1024
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]