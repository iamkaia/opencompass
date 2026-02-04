from opencompass.datasets import HFDataset

class SST2_convert_np(HFDataset):
    def load(self, *args, **kwargs):
        dataset = super().load(*args, **kwargs)

        def convert(example):
            # HF SST-2: label = "0" or "1"
            if example["label"] == "0":
                example["label"] = "negative"   # negative
            elif example["label"] == "1":
                example["label"] = "positive"   # positive
            return example

        return dataset.map(convert)
