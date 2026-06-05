from mmengine.config import read_base

with read_base():
    from .obqa_gen_9069e4 import obqa_datasets

obqa_datasets = [
    dataset for dataset in obqa_datasets
    if dataset.get('abbr') == 'openbookqa'
]
