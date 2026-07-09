from opencompass.models import (HuggingFaceMistral3,
                                HuggingFacewithChatTemplate)


common_model_kwargs = dict(
    torch_dtype='torch.float16',
    device_map='auto',
    trust_remote_code=True,
)

common_tokenizer_kwargs = dict(
    padding_side='left',
    truncation_side='left',
    trust_remote_code=True,
)


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='qwen2.5-7b-instruct-hf-fp16',
        path='Qwen/Qwen2.5-7B-Instruct',
        tokenizer_path='Qwen/Qwen2.5-7B-Instruct',
        tokenizer_kwargs=common_tokenizer_kwargs,
        model_kwargs=common_model_kwargs,
        max_seq_len=32768,
        max_out_len=128,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-3.1-8b-instruct-hf-fp16',
        path='meta-llama/Llama-3.1-8B-Instruct',
        tokenizer_path='meta-llama/Llama-3.1-8B-Instruct',
        tokenizer_kwargs=common_tokenizer_kwargs,
        model_kwargs=common_model_kwargs,
        max_seq_len=32768,
        max_out_len=128,
        batch_size=8,
        stop_words=['<|end_of_text|>', '<|eot_id|>'],
        run_cfg=dict(num_gpus=1),
    ),
    dict(
        type=HuggingFaceMistral3,
        abbr='ministral-3-8b-instruct-2512-hf',
        path='mistralai/Ministral-3-8B-Instruct-2512',
        tokenizer_path='mistralai/Ministral-3-8B-Instruct-2512',
        tokenizer_kwargs=dict(
            **common_tokenizer_kwargs,
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
