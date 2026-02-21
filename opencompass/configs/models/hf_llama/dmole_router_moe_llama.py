from opencompass.models.dmole_router_moe_llama import DMoLERouterMoELlama

models = [
    dict(
        type=DMoLERouterMoELlama,
        abbr='dmole_router_moe',

        # base
        path='./Llama-2-7b-chat-hf',

        # allocations（你跑 zero-cost 的輸出）
        allocations_json='./dmole_alloc/allocations.json',

        # 如果你要「不用 task label，自動路由」才需要這個
        # 先不確定你有沒有準備好 layer-router ckpt；沒有就先註解
        # layer_router_ckpt='/home/kaia/recall_1108/layer_router_ckpt',

        # 你剛訓練出的 LoRA（照你 log，是在 ./dmole_loras）
        lora_paths=dict(
            sst2='./dmole_loras/lora_dmole_sst2',
            squad2='./dmole_loras/lora_dmole_squad2',
            race='./dmole_loras/lora_dmole_race',
            medmcqa='./dmole_loras/lora_dmole_medmcqa',
            iwslt2017='./dmole_loras/lora_dmole_iwslt2017',
        ),

        dtype='float16',
        r=8,
        alpha=32, ###16跟32在哪裡啊？原本是32
        max_seq_len=128,
    )
]

