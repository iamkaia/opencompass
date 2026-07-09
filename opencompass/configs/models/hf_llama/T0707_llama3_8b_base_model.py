import os

from opencompass.models import HuggingFaceBaseModel


max_out_len = int(os.environ.get("T0704_MAX_OUT_LEN", "128"))


models = [
    dict(
        type=HuggingFaceBaseModel,
        abbr="llama3-8b-hf",
        path="meta-llama/Meta-Llama-3-8B",
        max_seq_len=2048,
        max_out_len=max_out_len,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]
