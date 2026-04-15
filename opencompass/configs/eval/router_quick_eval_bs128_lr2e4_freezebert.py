from mmengine.config import read_base

with read_base():
    from ..datasets.collections.router_quick_eval_small import datasets
    from ..models.hf_llama.moe_lora_internal_compact_bs128_lr2e4_freezebert import models

work_dir = './outputs/router_quick_eval_bs128_lr2e4_freezebert'
