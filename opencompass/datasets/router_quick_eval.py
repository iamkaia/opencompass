from datasets import Dataset, DatasetDict

from opencompass.registry import LOAD_DATASET

from .copa import COPADatasetV2
from .boolq import BoolQDatasetV2
from .hellaswag import HellaswagDataset_V2
from .iwslt2017 import IWSLT2017Dataset
from .medmcqa import MedmcqaDataset
from .piqa import PIQADatasetV2
from .race import RaceDataset
from .siqa import siqaDataset_V2
from .squad20 import SQuAD20Dataset
from .sst2_ab import SST2_convert_np


def _truncate_dataset(dataset, max_samples: int):
    max_samples = int(max_samples)
    if max_samples <= 0:
        return dataset

    if isinstance(dataset, DatasetDict):
        truncated = {}
        for split, split_dataset in dataset.items():
            keep = min(len(split_dataset), max_samples)
            truncated[split] = split_dataset.select(range(keep))
        return DatasetDict(truncated)

    if isinstance(dataset, Dataset):
        keep = min(len(dataset), max_samples)
        return dataset.select(range(keep))

    return dataset


@LOAD_DATASET.register_module()
class SST2ConvertNPSmall(SST2_convert_np):

    @staticmethod
    def load(*args, max_samples: int = 100, **kwargs):
        dataset = SST2_convert_np.load(*args, **kwargs)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class SQuAD20DatasetSmall(SQuAD20Dataset):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = SQuAD20Dataset.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class IWSLT2017DatasetSmall(IWSLT2017Dataset):

    @staticmethod
    def load(max_samples: int = 100, **kwargs):
        dataset = IWSLT2017Dataset.load(**kwargs)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class MedmcqaDatasetSmall(MedmcqaDataset):

    @staticmethod
    def load(path: str, prompt_mode: str = 'zero-shot', max_samples: int = 100, **kwargs):
        dataset = MedmcqaDataset.load(path=path, prompt_mode=prompt_mode, **kwargs)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class RaceDatasetSmall(RaceDataset):

    @staticmethod
    def load(path: str, name: str, max_samples: int = 100):
        dataset = RaceDataset.load(path=path, name=name)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class PIQADatasetV2Small(PIQADatasetV2):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = PIQADatasetV2.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class COPADatasetV2Small(COPADatasetV2):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = COPADatasetV2.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class SiqaDatasetV2Small(siqaDataset_V2):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = siqaDataset_V2.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class BoolQDatasetV2Small(BoolQDatasetV2):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = BoolQDatasetV2.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)


@LOAD_DATASET.register_module()
class HellaswagDatasetV2Small(HellaswagDataset_V2):

    @staticmethod
    def load(path: str, max_samples: int = 100):
        dataset = HellaswagDataset_V2.load(path=path)
        return _truncate_dataset(dataset, max_samples=max_samples)
