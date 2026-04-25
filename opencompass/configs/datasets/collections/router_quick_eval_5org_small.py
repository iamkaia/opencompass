from ..glue.sst2_gen_small import sst2_small_datasets
from ..iwslt2017.iwslt2017_gen_sft_prompt_small import iwslt2017_small_datasets
from ..medmcqa.medmcqa_gen_sft_prompt_small import medmcqa_small_datasets
from ..race.race_gen_sft_prompt_small import race_small_datasets
from ..squad20.squad20_gen_sft_prompt_small import squad20_small_datasets

datasets = []
datasets += iwslt2017_small_datasets
datasets += medmcqa_small_datasets
datasets += race_small_datasets
datasets += squad20_small_datasets
datasets += sst2_small_datasets
