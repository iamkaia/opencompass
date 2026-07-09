from opencompass.models import HuggingFaceMistral3


models = [
    dict(
        type=HuggingFaceMistral3,
        abbr='ministral-3-8b-instruct-2512-hf',
        path='mistralai/Ministral-3-8B-Instruct-2512',
        tokenizer_path='mistralai/Ministral-3-8B-Instruct-2512',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            fix_mistral_regex=True,
        ),
        model_kwargs=dict(
            torch_dtype='auto',
            device_map='auto',
            trust_remote_code=True,
        ),
        max_seq_len=32768,
        max_out_len=128,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
    ),
]
