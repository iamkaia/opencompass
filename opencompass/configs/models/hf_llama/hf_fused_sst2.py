from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="sst2_fused_lora",
        #path='meta-llama/Llama-2-7b-chat-hf',
        path="../recall/fullweight_models/recall_sst2_a02",
        peft_kwargs=dict(local_files_only=True),   # ⭐ 這行非常關鍵
        max_out_len=2048, ###原本是1024
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
