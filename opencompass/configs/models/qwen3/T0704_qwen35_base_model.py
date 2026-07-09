import os

from opencompass.models import HuggingFacewithChatTemplate


max_out_len = int(os.environ.get("T0704_MAX_OUT_LEN", "128"))


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="qwen35-4b-hf",
        path="Qwen/Qwen3.5-4B",
        tokenizer_path="Qwen/Qwen3.5-4B",
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
            trust_remote_code=True,
        ),
        model_kwargs=dict(
            torch_dtype="torch.float16",
            device_map="auto",
            trust_remote_code=True,
        ),
        max_seq_len=32768,
        max_out_len=max_out_len,
        batch_size=64,
        run_cfg=dict(num_gpus=1),
    )
]
