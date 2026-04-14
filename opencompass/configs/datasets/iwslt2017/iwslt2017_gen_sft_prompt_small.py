from opencompass.openicl.icl_evaluator import BleuEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.datasets.router_quick_eval import IWSLT2017DatasetSmall
from opencompass.utils.text_postprocessors import general_cn_postprocess

iwslt2017_reader_cfg = dict(
    input_columns='en',
    output_column='fr',
    train_split='validation',
)

iwslt2017_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(
                    role='HUMAN',
                    prompt='Please translate the following English statements to French:\n{en}',
                )
            ]
        ),
    ),
    retriever=dict(type='ZeroRetriever'),
    inferencer=dict(
        type='GenInferencer',
        generation_kwargs=dict(do_sample=False, num_beams=2, max_new_tokens=128),
    ),
)

iwslt2017_eval_cfg = dict(
    evaluator=dict(type=BleuEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=general_cn_postprocess),
    dataset_postprocessor=dict(type=general_cn_postprocess),
)

iwslt2017_small_datasets = [
    dict(
        type=IWSLT2017DatasetSmall,
        path='iwslt2017',
        name='iwslt2017-en-fr',
        max_samples=100,
        reader_cfg=iwslt2017_reader_cfg,
        infer_cfg=iwslt2017_infer_cfg,
        eval_cfg=iwslt2017_eval_cfg,
    )
]
