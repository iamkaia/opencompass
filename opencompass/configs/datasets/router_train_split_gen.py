from opencompass.datasets import JsonlDataset
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import router_train_answer_postprocess


DATA_ROOT = './0602_router_train_dataset'
TASKS = [
    'boolq',
    'medmcqa',
    'race',
    'sst2',
    'rte',
    'siqa',
    'piqa',
    'arc_c',
    'openbookqa',
]

reader_cfg = dict(
    input_columns=['prompt_text'],
    output_column='target',
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role='HUMAN', prompt='{prompt_text}')],
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=16),
)

eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=router_train_answer_postprocess),
)

router_train_split_datasets = [
    dict(
        abbr=f'{task}_router_train',
        type=JsonlDataset,
        path=f'{DATA_ROOT}/{task}/train.jsonl',
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
    for task in TASKS
]

datasets = router_train_split_datasets
