"""image2text dataset config using LLaVA-NeXT-Data (738k)."""
from src.datasets.understanding.caption_datasets import CaptionDataset
from mmengine.config import read_base
from src.datasets.collate_functions import collate_func_und, CollateConcat
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
from src.datasets.understanding.llava_datasets import LLaVADataset

from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from xtuner.dataset.utils import expand2square

with read_base():
    from .processors import prompt_template, tokenizer, image_size, pad_index, image_length


data_path = '/data3/data/LLaVA-NeXT-Data/llava_next_738k.json'
image_folder = '/data3/data/LLaVA-NeXT-Data/images'
max_length = int(2048 - (336 / 14) ** 2)

dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=False,
)

group_keys = ['image2text']
repeat = [1]
batch_size = 32
train_dataloader = dict(
    batch_size=batch_size,
    num_workers=4,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=FixedBatchMultiSourceSampler,
                 repeat=repeat,
                 batch_size=batch_size,
                 shuffle=True, seed=42),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate_func_und, pad_index=pad_index)],
                    keys=group_keys)
)
