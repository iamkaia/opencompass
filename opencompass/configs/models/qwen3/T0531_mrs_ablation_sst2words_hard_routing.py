import os

from opencompass.models.router_moe_llama_internal_compact_cached_joint import (
    RouterMoELlamaInternalCompactCachedJoint,
)


router_ckpt_dir = os.environ.get(
    "T0531_MRS_ROUTER_CKPT",
    "./router_T0527_qwen3_fp16_mrs_correct_conf_ce_t1_3expert_sst2words",
)
router_bert_init = os.environ.get("T0531_ROUTER_BERT_INIT", "./task_classifier_ckpt")
routing_mode = os.environ.get("T0531_ROUTING_MODE", "hard")
routing_sharpness = float(os.environ.get("T0531_ROUTING_SHARPNESS", "1.0"))
routing_topk_raw = os.environ.get("T0531_ROUTING_TOPK", "").strip()
routing_topk = int(routing_topk_raw) if routing_topk_raw else None
share_first_weights_all_layers = os.environ.get(
    "T0531_SHARE_FIRST_WEIGHTS_ALL_LAYERS", "0"
).strip().lower() in {"1", "true", "yes", "on"}
oracle_weight_path = os.environ.get("T0610_ORACLE_WEIGHT_PATH", "").strip() or None
static_weight_path = os.environ.get("T0616_STATIC_WEIGHT_PATH", "").strip() or None
record_tag = os.environ.get("T0531_ROUTER_RECORD_TAG", "").strip()
record_tag_part = f"{record_tag}_" if record_tag else ""
router_record_path = os.environ.get(
    "T0601_ROUTER_RECORD_PATH",
    f"T0531_opencompass_router_records_qwen3_mrs_"
    f"{record_tag_part}{routing_mode}.jsonl",
)

models = [
    dict(
        type=RouterMoELlamaInternalCompactCachedJoint,
        abbr=f"{os.path.basename(os.path.normpath(router_ckpt_dir))}_{routing_mode}",
        path="Qwen/Qwen3-4B-Instruct-2507",
        router_ckpt_dir=router_ckpt_dir,
        router_bert_init=router_bert_init,
        lora_paths=dict(
            medmcqa="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_medmcqa",
            race="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_race",
            sst2="./saves/Qwen/Qwen3-4B-Instruct-2507/lora/sft_sst2",
        ),
        dtype="float16",
        r=8,
        alpha=32,
        router_dim=512,
        first_layer_idx=0,
        middle_layer_idx=18,
        batch_size=32,
        max_seq_len=2048,
        max_out_len=64,
        routing_mode=routing_mode,
        routing_sharpness=routing_sharpness,
        routing_topk=routing_topk,
        share_first_weights_all_layers=share_first_weights_all_layers,
        oracle_weight_path=oracle_weight_path,
        static_weight_path=static_weight_path,
        debug_router_record_path=router_record_path,
        debug_router_topk=3,
        debug_router_max_prints=20,
        run_cfg=dict(num_gpus=1),
    )
]
