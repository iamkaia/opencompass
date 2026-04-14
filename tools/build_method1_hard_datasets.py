import json
import os
import random
import re
from collections import defaultdict


SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SRC_ROOT = os.path.join(ROOT, 'datasets_classifier')
OUT_ROOT = os.path.join(ROOT, 'data', 'router_method1')


def read_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def parse_sst2_sentence(text):
    prefix = 'Statement: '
    suffix = ' What’s sentiment should the above sentence be?'
    if not text.startswith(prefix):
        raise ValueError(f'Unexpected SST2 prompt: {text[:120]}')
    middle = text[len(prefix):]
    idx = middle.find(suffix)
    if idx == -1:
        raise ValueError(f'Unexpected SST2 prompt: {text[:120]}')
    return middle[:idx].strip()


def parse_squad_prompt(text):
    marker = ('According to the passage, answer the following question.'
              'If it is impossible to answer according to the passage, '
              "answer'impossible to answer': Question: ")
    if marker not in text:
        marker = ('According to the passage, answer the following question.'
                  'If it is impossible to answer according to the passage, '
                  "answer 'impossible to answer': Question: ")
    if marker not in text:
        raise ValueError(f'Unexpected SQuAD prompt: {text[:160]}')
    context, question = text.split(marker, 1)
    return context.strip(), question.strip()


def parse_race_prompt(text):
    prefix = 'Read the article and answer the question by replying A, B, C, or D.Article:'
    q_marker = 'Q:'
    opt_marker = 'OPTIONS:\n'
    if not text.startswith(prefix) or q_marker not in text or opt_marker not in text:
        raise ValueError(f'Unexpected RACE prompt: {text[:200]}')
    body = text[len(prefix):]
    article, rest = body.split(q_marker, 1)
    question, options_blob = rest.split(opt_marker, 1)
    options = {}
    matches = list(re.finditer(r'([ABCD]): ', options_blob))
    for i, m in enumerate(matches):
        letter = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(options_blob)
        options[letter] = options_blob[start:end].strip()
    if set(options) != {'A', 'B', 'C', 'D'}:
        raise ValueError(f'Unexpected RACE options: {options_blob[:200]}')
    return article.strip(), question.strip(), options


def build_sst2_fr_mcq(rows, num_samples=100):
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['target']].append(row)

    # Keep a balanced 50/50 split for this small synthetic set.
    per_label = num_samples // 2
    picked = []
    for target in ['negative', 'positive']:
        bucket = grouped[target]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_label])
    rng.shuffle(picked)

    out = []
    for idx, row in enumerate(picked):
        sentence = parse_sst2_sentence(row['text'])
        label = 'A' if row['target'] == 'negative' else 'B'
        prompt = (
            'Instruction: Lisez l’énoncé suivant et choisissez le bon '
            'sentiment.\n'
            f'Énoncé: {sentence}\n'
            'OPTIONS: A: negatif B: positif\n'
            'Choisissez la bonne réponse (A/B). Réponse:'
        )
        out.append({
            'id': idx,
            'sentence': sentence,
            'prompt': prompt,
            'A': 'negatif',
            'B': 'positif',
            'label': label,
            'target_text': row['target'],
            'source_task': 'sst2',
        })
    return out


def build_squad_medmcqa_style(rows, num_samples=100):
    rng = random.Random(SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    picked = shuffled[:num_samples]

    answer_pool = [row['target'].strip() for row in rows if row['target'].strip()]
    out = []

    for idx, row in enumerate(picked):
        context, question = parse_squad_prompt(row['text'])
        correct = row['target'].strip()

        distractors = []
        seen = {correct}
        pool = list(answer_pool)
        rng.shuffle(pool)
        for cand in pool:
            cand = cand.strip()
            if not cand or cand in seen:
                continue
            seen.add(cand)
            distractors.append(cand)
            if len(distractors) == 3:
                break
        if len(distractors) < 3:
            filler_bank = [
                'impossible to answer',
                'not stated in the passage',
                'the passage does not specify',
                'none of the above',
            ]
            for cand in filler_bank:
                if cand not in seen:
                    distractors.append(cand)
                    seen.add(cand)
                if len(distractors) == 3:
                    break

        options = [correct] + distractors[:3]
        letters = ['A', 'B', 'C', 'D']
        rng.shuffle(options)
        letter_to_text = dict(zip(letters, options))
        label = next(letter for letter, text in letter_to_text.items()
                     if text == correct)

        prompt = (
            'Instruction: Read the context and answer the question.\n'
            f'Context: {context}\n'
            f'Question: {question}\n'
            f'OPTIONS: A: {letter_to_text["A"]} '
            f'B: {letter_to_text["B"]} '
            f'C: {letter_to_text["C"]} '
            f'D: {letter_to_text["D"]} '
            'Choose the correct answer (A/B/C/D). Answer:'
        )
        out.append({
            'id': idx,
            'context': context,
            'question': question,
            'prompt': prompt,
            'A': letter_to_text['A'],
            'B': letter_to_text['B'],
            'C': letter_to_text['C'],
            'D': letter_to_text['D'],
            'label': label,
            'target_text': correct,
            'source_task': 'squad2',
        })
    return out


def build_race_medmcqa_style(rows, num_samples=100):
    rng = random.Random(SEED)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    picked = shuffled[:num_samples]

    out = []
    for idx, row in enumerate(picked):
        article, question, options = parse_race_prompt(row['text'])
        prompt = (
            'Instruction: Read the passage and answer the multiple-choice '
            'question.\n'
            f'Passage: {article}\n'
            f'Question: {question}\n'
            f'OPTIONS: A: {options["A"]} '
            f'B: {options["B"]} '
            f'C: {options["C"]} '
            f'D: {options["D"]} '
            'Choose the correct answer (A/B/C/D). Answer:'
        )
        out.append({
            'id': idx,
            'article': article,
            'question': question,
            'prompt': prompt,
            'A': options['A'],
            'B': options['B'],
            'C': options['C'],
            'D': options['D'],
            'label': row['target'].strip(),
            'target_text': options[row['target'].strip()],
            'source_task': 'race',
        })
    return out


def build_squad_sst2_style(rows, num_samples=100):
    rng = random.Random(SEED)
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['target'].strip() == 'impossible to answer'].append(row)

    per_label = num_samples // 2
    picked = []
    for is_impossible in [True, False]:
        bucket = grouped[is_impossible]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_label])
    rng.shuffle(picked)

    out = []
    for idx, row in enumerate(picked):
        context, question = parse_squad_prompt(row['text'])
        is_impossible = row['target'].strip() == 'impossible to answer'
        label = 'A' if is_impossible else 'B'
        statement = (
            f'Passage: {context} '
            f'Question: {question}'
        )
        prompt = (
            f'Statement: {statement} '
            'Can the question be answered according to the above passage?\n'
            'OPTIONS:-impossible to answer.-answerable. Answer:'
        )
        out.append({
            'id': idx,
            'context': context,
            'question': question,
            'prompt': prompt,
            'A': 'impossible to answer',
            'B': 'answerable',
            'label': label,
            'target_text': row['target'].strip(),
            'source_task': 'squad2',
        })
    return out


def main():
    sst2_rows = read_jsonl(os.path.join(SRC_ROOT, 'sst2', 'train.jsonl'))
    squad_rows = read_jsonl(os.path.join(SRC_ROOT, 'squad2', 'validation.jsonl'))
    race_rows = read_jsonl(os.path.join(SRC_ROOT, 'race', 'validation.jsonl'))

    sst2_fr = build_sst2_fr_mcq(sst2_rows, num_samples=100)
    squad_med = build_squad_medmcqa_style(squad_rows, num_samples=100)
    race_med = build_race_medmcqa_style(race_rows, num_samples=100)
    squad_sst2 = build_squad_sst2_style(squad_rows, num_samples=100)

    sst2_dir = os.path.join(OUT_ROOT, 'sst2_fr_mcq_100')
    squad_dir = os.path.join(OUT_ROOT, 'squad20_medmcqa_style_100')
    race_dir = os.path.join(OUT_ROOT, 'race_medmcqa_style_100')
    squad_sst2_dir = os.path.join(OUT_ROOT, 'squad20_sst2_style_100')

    write_jsonl(os.path.join(sst2_dir, 'validation.jsonl'), sst2_fr)
    write_jsonl(os.path.join(squad_dir, 'validation.jsonl'), squad_med)
    write_jsonl(os.path.join(race_dir, 'validation.jsonl'), race_med)
    write_jsonl(os.path.join(squad_sst2_dir, 'validation.jsonl'), squad_sst2)

    meta = {
        'seed': SEED,
        'datasets': {
            'sst2_fr_mcq_100': {
                'size': len(sst2_fr),
                'source': 'datasets_classifier/sst2/train.jsonl',
                'note': ('French instruction and labels with original SST2 '
                         'sentence content preserved.'),
            },
            'squad20_medmcqa_style_100': {
                'size': len(squad_med),
                'source': 'datasets_classifier/squad2/validation.jsonl',
                'note': ('SQuAD-style QA converted to 4-option MCQ with '
                         'synthetic distractors.'),
            },
            'race_medmcqa_style_100': {
                'size': len(race_med),
                'source': 'datasets_classifier/race/validation.jsonl',
                'note': 'RACE content rendered with a MedMCQA-style MCQ prompt.',
            },
            'squad20_sst2_style_100': {
                'size': len(squad_sst2),
                'source': 'datasets_classifier/squad2/validation.jsonl',
                'note': ('SQuAD2 answerability converted to an SST2-style '
                         'binary classification prompt.'),
            },
        },
    }
    with open(os.path.join(OUT_ROOT, 'build_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
