import argparse
import copy
import json

from datasets import load_dataset

from opencompass.configs.datasets.medmcqa.medmcqa_gen_sft_prompt import (
    MEDMCQA_PROMPT,
)
from opencompass.datasets.medmcqa import _parse, preprocess
from opencompass.openicl.icl_prompt_template import PromptTemplate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Inspect how medmcqa_gen_sft_prompt transforms each MedMCQA "
            "sample into the final OpenCompass prompt."
        ))
    parser.add_argument('--index',
                        type=int,
                        default=0,
                        help='Start index in the validation split.')
    parser.add_argument('--count',
                        type=int,
                        default=1,
                        help='Number of samples to print.')
    parser.add_argument('--path',
                        type=str,
                        default='openlifescienceai/medmcqa',
                        help='HF dataset path.')
    return parser.parse_args()


def render_prompt(sample):
    template = PromptTemplate(
        template=dict(round=[dict(role='HUMAN', prompt=MEDMCQA_PROMPT)]))
    return template.generate_item(sample)


def inspect_one(raw_sample, idx):
    parsed_sample = _parse(copy.deepcopy(raw_sample), 'zero-shot')
    preprocessed_sample = preprocess(copy.deepcopy(parsed_sample))
    final_prompt = render_prompt(preprocessed_sample)

    payload = {
        'index': idx,
        'raw_sample': raw_sample,
        'after_parse': parsed_sample,
        'after_preprocess': preprocessed_sample,
        'final_prompt': final_prompt,
    }
    print('=' * 120)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    args = parse_args()
    dataset = load_dataset(
        path=args.path,
        split='validation',
        trust_remote_code=True,
    )

    end = min(args.index + args.count, len(dataset))
    for idx in range(args.index, end):
        inspect_one(dataset[idx], idx)


if __name__ == '__main__':
    main()
