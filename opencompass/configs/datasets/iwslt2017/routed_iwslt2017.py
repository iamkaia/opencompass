from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import BM25Retriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.datasets import IWSLT2017Dataset, JsonlDataset
from opencompass.utils.text_postprocessors import general_cn_postprocess


ROUTED_PATH = './routed_eval/iwslt2017/test.jsonl'

iwslt2017_reader_cfg = dict(
    input_columns='input', output_column='target', #train_split='validation'
)

iwslt2017_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt='{input}'),  ###iwslt2017似乎沒有寫prompt要怎麼寫
            ]
        )
    ),
    retriever=dict(type="ZeroRetriever"),
    inferencer=dict(
        type="GenInferencer",
        generation_kwargs=dict(
            do_sample=False,
            num_beams=2,
            max_new_tokens=128,
        )
    )
)


iwslt2017_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=general_cn_postprocess),
    dataset_postprocessor=dict(type=general_cn_postprocess)
)

routed_iwslt2017_datasets = [
    dict(
        type=JsonlDataset,
        path = ROUTED_PATH,
        abbr='routed_iwslt2017-en-fr',  ###原本是en-de
        reader_cfg=iwslt2017_reader_cfg,
        infer_cfg=iwslt2017_infer_cfg,
        eval_cfg=iwslt2017_eval_cfg)
]