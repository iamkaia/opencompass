import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


router_ckpt_dir = os.environ.get(
    "T0531_MRS_ROUTER_CKPT",
    "./router_T0704_qwen35_mrs_only_single_all_layers",
)
router_bert_init = os.environ.get("T0531_ROUTER_BERT_INIT", "./task_classifier_ckpt")
routing_mode = os.environ.get("T0531_ROUTING_MODE", "weighted_sum")
routing_sharpness = float(os.environ.get("T0531_ROUTING_SHARPNESS", "3.0"))
routing_topk_raw = os.environ.get("T0531_ROUTING_TOPK", "").strip()
routing_topk = int(routing_topk_raw) if routing_topk_raw else None
max_out_len = int(os.environ.get("T0704_MAX_OUT_LEN", "64"))
middle_layer_idx = int(os.environ.get("MIDDLE_LAYER_IDX", "18"))
record_tag = os.environ.get("T0531_ROUTER_RECORD_TAG", "").strip()
record_tag_part = f"{record_tag}_" if record_tag else ""
router_record_path = os.environ.get(
    "T0601_ROUTER_RECORD_PATH",
    f"T0704_opencompass_router_records_qwen35_single_"
    f"{record_tag_part}{routing_mode}.jsonl",
)

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_{routing_mode}",
        path="Qwen/Qwen3.5-4B",
        router_ckpt_dir=router_ckpt_dir,
        router_bert_init=router_bert_init,
        lora_paths=dict(
            medmcqa="./saves/Qwen/Qwen3.5-4B/lora/sft_medmcqa",
            race="./saves/Qwen/Qwen3.5-4B/lora/sft_race",
            sst2="./saves/Qwen/Qwen3.5-4B/lora/sft_sst2",
        ),
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=middle_layer_idx,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=max_out_len,
        routing_mode=routing_mode,
        routing_sharpness=routing_sharpness,
        routing_topk=routing_topk,
        share_first_weights_all_layers=True,
        debug_router_record_path=router_record_path,
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
