from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="squad2_fused_lora",
        path='meta-llama/Llama-2-7b-chat-hf',
        peft_path="./fused_models/recall_fused_squad20",
        peft_kwargs=dict(local_files_only=True),   # ⭐ 這行非常關鍵
        max_out_len=2048, ###原本是1024
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]