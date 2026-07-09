from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen3-4b-instruct-2507-hf',
        path='Qwen/Qwen3-4B-Instruct-2507',
        tokenizer_path='Qwen/Qwen3-4B-Instruct-2507',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        model_kwargs=dict(
            torch_dtype='torch.float16',
            device_map='auto',
            trust_remote_code=True,
        ),
        max_seq_len=32768,
        max_out_len=128,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
    )
]
