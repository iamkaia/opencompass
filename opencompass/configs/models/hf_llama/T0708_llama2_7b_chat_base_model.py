import os

from opencompass.models import HuggingFacewithChatTemplate


max_out_len = int(os.environ.get("T0704_MAX_OUT_LEN", "128"))
batch_size = int(os.environ.get("T0708_LLAMA2_BASE_BATCH_SIZE", "8"))


models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr="llama-2-7b-chat-hf",
        path="meta-llama/Llama-2-7b-chat-hf",
        max_seq_len=2048,
        max_out_len=max_out_len,
        batch_size=batch_size,
        run_cfg=dict(num_gpus=1),
        model_kwargs=dict(
            torch_dtype="torch.float16",
            device_map="auto",
            trust_remote_code=True,
        ),
        tokenizer_kwargs=dict(
            padding_side="left",
            truncation_side="left",
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
