from opencompass.datasets import MedmcqaEvaluator
from opencompass.datasets.router_quick_eval import MedmcqaDatasetSmall
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever

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
        template=dict(round=[dict(role='HUMAN', prompt=MEDMCQA_PROMPT)]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)

eval_cfg = dict(
    evaluator=dict(type=MedmcqaEvaluator),
    pred_role='BOT',
)

medmcqa_small_datasets = [
    dict(
        type=MedmcqaDatasetSmall,
        abbr='medmcqa-small',
        path='openlifescienceai/medmcqa',
        prompt_mode='zero-shot',
        max_samples=100,
        reader_cfg=reader_cfg,
        infer_cfg=infer_cfg,
        eval_cfg=eval_cfg,
    )
]
