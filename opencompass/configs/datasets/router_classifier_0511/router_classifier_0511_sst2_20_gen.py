from opencompass.datasets import RouterClassifier0511Dataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import sst2_postprocess


sst2_reader_cfg = dict(
    input_columns=['text'],
    output_column='sst2_label',
    test_range='[:20]',
)

sst2_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{text}')]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

sst2_eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=sst2_postprocess),
)

router_classifier_0511_sst2_20_datasets = [
    dict(
        abbr='classifier0511_sst2_20',
        type=RouterClassifier0511Dataset,
        path='datasets_classifier_0511/sst2/validation.jsonl',
        task_name='sst2',
        reader_cfg=sst2_reader_cfg,
        infer_cfg=sst2_infer_cfg,
        eval_cfg=sst2_eval_cfg,
    )
]
