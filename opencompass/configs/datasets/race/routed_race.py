from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_evaluator import AccwithDetailsEvaluator
from opencompass.datasets import RaceDataset, JsonlDataset
from opencompass.utils.text_postprocessors import first_option_postprocess

ROUTED_PATH = './routed_eval/race/test.jsonl'

RACE_PROMPT = (
    "Read the article, and answer the question by replying A, B, C or D.\n"
    "Article: {article}\n"
    "Q:{question}"
    " Options: "
    "A: {A} "
    "B: {B} "
    "C: {C} "
    "D: {D}"
)


race_reader_cfg = dict(
    input_columns=['input'],
    output_column='target',
    #train_split='validation',
    #test_split='test'
)


race_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt="{input}")
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer)
)

race_eval_cfg = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
    pred_role='BOT')

routed_race_datasets = [
    dict(
        abbr='race-middle',
        type=JsonlDataset,
        #path='opencompass/race',
        path = ROUTED_PATH,
        reader_cfg=race_reader_cfg,
        infer_cfg=race_infer_cfg,
        eval_cfg=race_eval_cfg
    )
]
