from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.sst2_ab import SST2_convert_np
from opencompass.datasets.jsonl import JsonlDataset
from opencompass.utils.text_postprocessors import first_option_postprocess, sst2_postprocess_in_routed

ROUTED_PATH = './routed_eval/sst2/test.jsonl'

reader_cfg = dict(
    input_columns=['input'],
    output_column='target',      # HF SST-2: "0" / "1"
    #test_split='validation'
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""{input}"""
                ),
                dict(
                    role='BOT',
                    prompt=''   # ⭐ 關鍵：留空，讓模型填 ###不知道其他幾個原版有沒有留?
                )
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer), ####不能加max_out_len, model會只輸出奇怪的字

)

eval_cfg = dict(
    evaluator=dict(
        type=AccEvaluator,
    ),
    pred_role='BOT',
    pred_postprocessor=dict(
        type=sst2_postprocess_in_routed, #####抓預測字的東西
    ),
)

routed_sst2_datasets = [
    dict(
        abbr='routed_sst2',
        type=JsonlDataset, ####使0/1轉成negetive/positive
        #path='glue',
        path = ROUTED_PATH,
        #name='routed_sst2',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]