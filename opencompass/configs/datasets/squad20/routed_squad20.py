from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SQuAD20Dataset, SQuAD20Evaluator, JsonlDataset

ROUTED_PATH = './routed_eval/squad20/test.jsonl'

SQUAD_PROMPT = (
    "{input}"
)


squad20_reader_cfg = dict(
    input_columns=['input'],
    output_column='target')

squad20_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            round=[
                dict(role='HUMAN', prompt=SQUAD_PROMPT),
            ]
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=50),
)

squad20_eval_cfg = dict(
    evaluator=dict(type=SQuAD20Evaluator), metric="em", pred_role='BOT') ###metric是多加的

routed_squad20_datasets = [
    dict(
        type=JsonlDataset,
        abbr='routed_squad20',
        #name='routed_squad20',
        #path='./data/SQuAD2.0/dev-v2.0.json',
        path = ROUTED_PATH,
        reader_cfg=squad20_reader_cfg,
        infer_cfg=squad20_infer_cfg,
        eval_cfg=squad20_eval_cfg)
]
