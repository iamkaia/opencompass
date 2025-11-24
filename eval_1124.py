from opencompass.models import HuggingFacewithChatTemplate
from mmengine.config import read_base

'''
with read_base():
    from .race_gen_69ee4f import race_datasets  # noqa: F401, F403
'''

with read_base():
    from opencompass.configs.datasets.iwslt2017.iwslt2017_gen import iwslt2017_datasets  # noqa: F401, F403
models = [
    dict(
        type=HuggingFacewithChatTemplate,
        abbr='llama-2-7b-chat-hf',
        path='meta-llama/Llama-2-7b-chat-hf',
        max_out_len=2048,
        batch_size=8,
        run_cfg=dict(num_gpus=1),
    )
]

datasets = iwslt2017_datasets
