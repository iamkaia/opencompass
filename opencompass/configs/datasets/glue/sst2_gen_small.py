from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.datasets.router_quick_eval import SST2ConvertNPSmall
from opencompass.utils.text_postprocessors import sst2_postprocess


reader_cfg = dict(
    input_columns=['sentence'],
    output_column='label',
    test_split='validation',
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt="Statement: {sentence} What’s sentiment should the above sentence be?\nOPTIONS:-negative.-positive. Answer:",
                ),
                dict(role='BOT', prompt=''),
            ],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=sst2_postprocess),
)

sst2_small_datasets = [
    dict(
        abbr='sst2-small',
        type=SST2ConvertNPSmall,
        path='glue',
        name='sst2',
        max_samples=500,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
