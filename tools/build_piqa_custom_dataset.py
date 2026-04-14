import argparse
import json
import os


def read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def load_split(src_dir, data_file, label_file):
    data_path = os.path.join(src_dir, data_file)
    label_path = os.path.join(src_dir, label_file)
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Missing PIQA data file: {data_path}')
    if not os.path.exists(label_path):
        raise FileNotFoundError(f'Missing PIQA label file: {label_path}')

    data_lines = read_lines(data_path)
    label_lines = read_lines(label_path)
    if len(data_lines) != len(label_lines):
        raise ValueError(
            f'Line count mismatch: {data_path} has {len(data_lines)} lines '
            f'but {label_path} has {len(label_lines)} lines')

    rows = []
    for idx, (data_line, label_line) in enumerate(zip(data_lines, label_lines)):
        item = json.loads(data_line)
        label_int = int(label_line)
        label = 'AB'[label_int]
        prompt = f"{item['goal']}\nA. {item['sol1']}\nB. {item['sol2']}\nAnswer:"
        rows.append({
            'id': idx,
            'goal': item['goal'],
            'A': item['sol1'],
            'B': item['sol2'],
            'prompt': prompt,
            'label': label,
            'source_task': 'piqa',
        })
    return rows


def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--src_dir', type=str, default='./data/piqa')
    parser.add_argument('--out_dir', type=str, default='./data/router_piqa/piqa_custom')
    args = parser.parse_args()

    train_rows = load_split(args.src_dir, 'train.jsonl', 'train-labels.lst')
    val_rows = load_split(args.src_dir, 'dev.jsonl', 'dev-labels.lst')

    write_jsonl(os.path.join(args.out_dir, 'train.jsonl'), train_rows)
    write_jsonl(os.path.join(args.out_dir, 'validation.jsonl'), val_rows)

    meta = {
        'src_dir': os.path.abspath(args.src_dir),
        'out_dir': os.path.abspath(args.out_dir),
        'train_size': len(train_rows),
        'validation_size': len(val_rows),
        'prompt_format': '{goal}\\nA. {sol1}\\nB. {sol2}\\nAnswer:',
    }
    with open(os.path.join(args.out_dir, 'build_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps(meta, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
