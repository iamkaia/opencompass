from opencompass.datasets.router_quick_eval import BoolQDatasetV2Small
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import first_capital_postprocess

BoolQ_reader_cfg = dict(
    input_columns=['question', 'passage'],
    output_column='label',
)

BoolQ_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{passage}\nQuestion: {question}\nA. Yes\nB. No\nAnswer:',
                ),
            ]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

BoolQ_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_capital_postprocess),
)

BoolQ_small_datasets = [
    dict(
        abbr='BoolQ-small',
        type=BoolQDatasetV2Small,
        path='opencompass/boolq',
        max_samples=500,
        reader_cfg=BoolQ_reader_cfg,
        infer_cfg=BoolQ_infer_cfg,
        eval_cfg=BoolQ_eval_cfg,
    )
]
