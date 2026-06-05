from opencompass.datasets import (MedmcqaEvaluator,
                                  RouterClassifier0511Dataset,
                                  SQuAD20Evaluator)
from opencompass.openicl.icl_evaluator import (AccEvaluator,
                                               AccwithDetailsEvaluator,
                                               BleuEvaluator,
                                               EDAccEvaluator)
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.utils.text_postprocessors import (
    first_capital_postprocess,
    first_option_postprocess,
    general_cn_postprocess,
    sst2_postprocess,
)


def _reader(output_column='target'):
    return dict(input_columns=['text'], output_column=output_column)


_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(round=[dict(role='HUMAN', prompt='{text}')]),
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer),
)


_acc_ab_eval = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='AB'),
)

_acc_abcde_eval = dict(
    evaluator=dict(type=AccEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCDE'),
)

_acc_abcd_details_eval = dict(
    evaluator=dict(type=AccwithDetailsEvaluator),
    pred_role='BOT',
    pred_postprocessor=dict(type=first_option_postprocess, options='ABCD'),
)


def _dataset(task, abbr, output_column='target', eval_cfg=None):
    return dict(
        abbr=abbr,
        type=RouterClassifier0511Dataset,
        path=f'datasets_classifier_0511/{task}/validation.jsonl',
        task_name=task,
        reader_cfg=_reader(output_column),
        infer_cfg=_infer_cfg,
        eval_cfg=eval_cfg,
    )


router_classifier_0511_datasets = [
    _dataset('siqa', 'classifier0511_siqa', 'all_labels',
             dict(evaluator=dict(type=EDAccEvaluator), pred_role='BOT')),
    _dataset('commonsenseqa', 'classifier0511_commonsenseqa', 'target',
             _acc_abcde_eval),
    _dataset('boolq', 'classifier0511_boolq', 'target', _acc_ab_eval),
    _dataset('hellaswag', 'classifier0511_hellaswag', 'target',
             _acc_abcd_details_eval),
    _dataset('rte', 'classifier0511_rte', 'target', _acc_ab_eval),
    _dataset('piqa', 'classifier0511_piqa', 'target', _acc_ab_eval),
    _dataset('race', 'classifier0511_race', 'target',
             _acc_abcd_details_eval),
    _dataset(
        'sst2',
        'classifier0511_sst2',
        'sst2_label',
        dict(
            evaluator=dict(type=AccEvaluator),
            pred_role='BOT',
            pred_postprocessor=dict(type=sst2_postprocess),
        ),
    ),
    _dataset(
        'medmcqa',
        'classifier0511_medmcqa',
        'cop',
        dict(evaluator=dict(type=MedmcqaEvaluator), pred_role='BOT'),
    ),
    _dataset(
        'iwslt2017',
        'classifier0511_iwslt2017',
        'target',
        dict(
            evaluator=dict(type=BleuEvaluator),
            pred_role='BOT',
            pred_postprocessor=dict(type=general_cn_postprocess),
            dataset_postprocessor=dict(type=general_cn_postprocess),
        ),
    ),
    _dataset(
        'squad2',
        'classifier0511_squad2',
        'answers',
        dict(evaluator=dict(type=SQuAD20Evaluator), pred_role='BOT'),
    ),
]
