from opencompass.datasets import MedmcqaDataset, MedmcqaEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
import re

'''
def preprocess(example):
    
    q = example['question']

    # remove any lines that start with A/B/C/D
    q = re.sub(r'^[ \t]*[A-D][\.\:][^\n]*\n?', '', q, flags=re.MULTILINE)

    # remove Question: prefix
    q = re.sub(r'^Question:\s*', '', q, flags=re.IGNORECASE)

    example['question'] = q.strip()
    return example
'''

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
    input_columns=['question', 'opa', 'opb', 'opc', 'opd'],
    output_column='cop',
)

infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[dict(role="HUMAN", prompt=MEDMCQA_PROMPT)]
        ),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=64),
)

eval_cfg = dict(
    evaluator=dict(type=MedmcqaEvaluator),
    pred_role="BOT",
)

medmcqa_datasets = [
    dict(
        type=MedmcqaDataset,
        abbr='medmcqa',
        path='openlifescienceai/medmcqa',
        prompt_mode="zero-shot",
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg
    )
]

