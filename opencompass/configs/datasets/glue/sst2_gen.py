# configs/datasets/glue/sst2_gen.py

from opencompass.registry import LOAD_DATASET
from datasets import load_dataset

# ---- Dataset Loader ----
@LOAD_DATASET.register_module()
def load_sst2(split='validation'):
    dataset = load_dataset('glue', 'sst2', split=split)
    return dataset

# ---- Reader Config ----
sst2_reader_cfg = dict(
    input_columns=['sentence'],
    output_column='label'
)

# ---- Prompt / Infer Config ----
sst2_infer_cfg = dict(
    type='GenInferencer',
    prompt_template=dict(
        type='PromptTemplate',
        template="""Classify the sentiment of the sentence as positive or negative.

Sentence: {sentence}
Answer:"""
    ),
    max_out_len=32,
)

# ---- Evaluation ----
sst2_eval_cfg = dict(
    evaluator=dict(type='AccEvaluator')
)

# ---- Dataset name registry ----
sst2_datasets = [
    dict(
        type='sst2',
        abbr='sst2',
        path='glue',
        subset='sst2',
        reader_cfg=sst2_reader_cfg,
        infer_cfg=sst2_infer_cfg,
        eval_cfg=sst2_eval_cfg,
    )
]
