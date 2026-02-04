from opencompass.datasets import MedmcqaDataset, MedmcqaEvaluator, JsonlDataset
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_evaluator import AccEvaluator
from opencompass.utils.text_postprocessors import mcqa_choice_postprocess
import re
from opencompass.registry import TEXT_POSTPROCESSORS
ROUTED_PATH = './routed_eval/medmcqa/test.jsonl'


MEDMCQA_PROMPT = (
    "Instruction: Question: {question} "
    "Options: "
    "A: {opa} "
    "B: {opb} "
    "C: {opc} "
    "D: {opd} "
    "Choose a correct answer from A/B/C/D."
    "Answer:"
)

reader_cfg = dict(
    input_columns=['input'],
    output_column='target',
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt="{input}")]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role="BOT",
    pred_postprocessor=dict(type=mcqa_choice_postprocess),
)

routed_medmcqa_datasets = [
    dict(
        type=JsonlDataset,
        abbr='routed_medmcqa',
        #path='openlifescienceai/medmcqa',
        path = ROUTED_PATH,
        #prompt_mode="zero-shot",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg
    )
]
