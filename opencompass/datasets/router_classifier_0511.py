import json
import re
from typing import Dict, List

from datasets import Dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from .base import BaseDataset


_OPTION_RE = re.compile(
    r'(?P<label>[A-E])\s*[\.:]\s*(?P<text>.*?)(?=(?:\s+[A-E]\s*[\.:]\s*)|$)',
    re.S,
)


def _read_jsonl(path: str) -> List[Dict]:
    path = get_data_path(path, local_mode=True)
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _target_letter(value: str) -> str:
    value = str(value).strip().upper()
    return value[:1] if value else ''


def _parse_options(text: str, labels: str) -> Dict[str, str]:
    found = {}
    for match in _OPTION_RE.finditer(str(text)):
        label = match.group('label')
        if label in labels:
            found[label] = re.sub(r'\s+', ' ', match.group('text')).strip()
    return found


def _siqa_reference(text: str, target: str) -> Dict:
    options = _parse_options(text, 'ABC')
    label = _target_letter(target)
    if len(options) != 3 or label not in options:
        return {
            'candidates': [['A', 'A. '], ['B', 'B. '], ['C', 'C. ']],
            'label': max(0, min(2, ord(label or 'A') - ord('A'))),
        }
    return {
        'candidates': [
            [f'A. {options["A"]}', 'A', options['A']],
            [f'B. {options["B"]}', 'B', options['B']],
            [f'C. {options["C"]}', 'C', options['C']],
        ],
        'label': ord(label) - ord('A'),
    }


def _medmcqa_fields(text: str, target: str) -> Dict:
    options = _parse_options(text, 'ABCD')
    label = _target_letter(target)
    option_list = [options.get(key, '') for key in 'ABCD']
    return {
        'cop': max(0, min(3, ord(label or 'A') - ord('A'))),
        'prompt_mode': 'zero-shot',
        'options': option_list,
        'label': label,
        'subject_name': '',
        'topic_name': '',
        'choice_type': '',
    }


@LOAD_DATASET.register_module()
class RouterClassifier0511Dataset(BaseDataset):
    """Load the sampled router dataset used by cached router training.

    The source jsonl rows already contain the exact prompt text in ``text`` and
    the answer in ``target``. This loader preserves that prompt and only adds
    the extra fields required by a few OpenCompass evaluators.
    """

    @staticmethod
    def load(path: str, task_name: str = ''):
        rows = []
        task_name = str(task_name or '').lower()
        for row in _read_jsonl(path):
            item = dict(row)
            item['prompt_text'] = str(item.get('text', ''))
            item['target'] = str(item.get('target', ''))
            item['origin_prompt_text'] = item['prompt_text']
            if task_name == 'siqa':
                item['all_labels'] = _siqa_reference(
                    item['prompt_text'], item['target'])
            elif task_name == 'medmcqa':
                item.update(_medmcqa_fields(item['prompt_text'],
                                            item['target']))
            elif task_name == 'sst2':
                target = str(item['target']).strip().lower()
                item['sst2_label'] = '1' if target == 'positive' else '0'
            elif task_name in {'squad2', 'squad20', 'squad2.0'}:
                item['answers'] = [item['target']]
            rows.append(item)
        return Dataset.from_list(rows)
