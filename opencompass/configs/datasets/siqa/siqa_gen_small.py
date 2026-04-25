from opencompass.datasets.router_quick_eval import SiqaDatasetV2Small
from opencompass.openicl.icl_evaluator import EDAccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

siqa_reader_cfg = dict(
    input_columns=['context', 'question', 'answerA', 'answerB', 'answerC'],
    output_column='all_labels',
    test_split='validation',
)

siqa_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='{context}\nQuestion: {question}\nA. {answerA}\nB. {answerB}\nC. {answerC}\nAnswer:',
                )
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

siqa_eval_cfg = dict(
    evaluator=dict(type=EDAccEvaluator),
    pred_role='BOT',
)

siqa_small_datasets = [
    dict(
        abbr='siqa-small',
        type=SiqaDatasetV2Small,
        path='opencompass/siqa',
        max_samples=500,
        reader_cfg=siqa_reader_cfg,
        infer_cfg=siqa_infer_cfg,
        eval_cfg=siqa_eval_cfg,
    )
]
