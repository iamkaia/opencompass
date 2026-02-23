from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-2-7b-chat-hf',
        path='meta-llama/Llama-2-7b-chat-hf',
        max_out_len=128, ###原本是2048###原本是1024
        batch_size=64,
        run_cfg=dict(num_gpus=1),
        use_cache=False
    )
]
