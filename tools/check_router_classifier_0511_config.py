import json

from mmengine.config import Config

from opencompass.utils import build_dataset_from_cfg


def main():
    cfg = Config.fromfile(
        'opencompass/configs/datasets/router_classifier_0511/'
        'router_classifier_0511_gen.py')
    print('num_datasets =', len(cfg.router_classifier_0511_datasets))

    for ds_cfg in cfg.router_classifier_0511_datasets:
        ds = build_dataset_from_cfg(ds_cfg)
        row = ds.test[0]
        with open(ds_cfg.path, encoding='utf-8') as f:
            raw = json.loads(f.readline())
        prompt = ds.reader.generate_input_field_prompt(row)
        print(
            ds_cfg.abbr,
            'n=', len(ds.test),
            'prompt_equal_raw_text=', prompt == raw['text'],
            'output_col=', ds.reader.output_column,
            'target=', row[ds.reader.output_column],
        )


if __name__ == '__main__':
    main()
