from opencompass.models import HuggingFacewithChatTemplate

models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-2-7b-chat-hf',
        path='meta-llama/Llama-2-7b-chat-hf',
        max_out_len=64,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            torch_dtype='torch.float16',
            device_map='auto',
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
        ),
        generation_kwargs=dict(
            do_sample=False,
            num_beams=1,
            temperature=None,
            top_p=None,
            top_k=None,
        ),
    )
]
