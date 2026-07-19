from opencompass.models import HuggingFacewithChatTemplate


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='gemma-4-e4b-it-hf-bf16',
        path='google/gemma-4-e4b-it',
        tokenizer_path='google/gemma-4-e4b-it',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        model_kwargs=dict(
            torch_dtype='torch.bfloat16',
            device_map='auto',
            trust_remote_code=True,
        ),
        max_seq_len=2048,
        max_out_len=128,
        min_out_len=64,
        batch_size=1,
        stop_words=[],
        disable_auto_stop_words=True,
        run_cfg=dict(num_gpus=1),
    ),
]
