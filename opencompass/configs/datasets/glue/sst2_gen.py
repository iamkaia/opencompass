from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.datasets.sst2_ab import SST2_convert_np
from opencompass.utils.text_postprocessors import  sst2_postprocess


###讀sentence當輸入，label當標準答案，validation 的 datasets split 當評估
reader_cfg = dict(
    input_columns=['sentence'],
    output_column='label',      # HF SST-2: "0" / "1"
    test_split='validation'
)

###套PromptTemplate
infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="""Statement: {sentence} What’s sentiment should the above sentence be?\nOPTIONS:-negative.-positive. Answer:"""
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

####用Acc evaluator, 前處理用sst2_postprocess回答整理成可比對 label
eval_cfg = dict(
    evaluator=dict(
        type=AccEvaluator,
    ),
    pred_role='BOT',
    pred_postprocessor=dict(
        type=sst2_postprocess, #####抓預測字的東西
    ),
)

###註冊成一個datasets, abbr是用sst2, type是用SST2_convert_np
sst2_datasets = [
    dict(
        abbr='sst2',
        type=SST2_convert_np, ####使0/1轉成negetive/positive
        path='glue',
        name='sst2',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]