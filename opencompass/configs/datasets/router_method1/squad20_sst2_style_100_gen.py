from opencompass.datasets import CustomDataset, OptionSimAccEvaluator
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever


datasets = [
    dict(
        abbr='squad20-sst2-style-100',
        type=CustomDataset,
        path='./data/router_method1/squad20_sst2_style_100',
        local_mode=True,
        file_name='validation.jsonl',
        reader_cfg=dict(
            input_columns=['prompt', 'A', 'B'],
            output_column='label',
        ),
        infer_cfg=dict(
            prompt_template=dict(
                type=PromptTemplate,
                template=dict(
                    round=[
                        dict(role='HUMAN', prompt='{prompt}'),
                        dict(role='BOT', prompt=''),
                    ],
                ),
            ),
            retriever=dict(type=ZeroRetriever),
            inferencer=dict(type=GenInferencer, max_out_len=8),
        ),
        eval_cfg=dict(
            evaluator=dict(type=OptionSimAccEvaluator, options=['A', 'B']),
            pred_role='BOT',
        ),
    )
]
