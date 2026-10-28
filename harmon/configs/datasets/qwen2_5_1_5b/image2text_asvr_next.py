"""Dataset config for ASVR with LLaVA-NeXT: only image2text, wrapped in ConcatDataset."""
from mmengine.config import read_base
from xtuner.dataset import ConcatDataset
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
from src.datasets.collate_functions import collate_func_und, CollateConcat
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

with read_base():
    from .image2text_next import dataset as und_data
    from .processors import *


dataset = dict(
    type=ConcatDataset,
    datasets=[und_data]
)

group_keys = ['image2text']
repeat = [1]
# LLaVA-NeXT + ASVR (DualToken + 1088 image tokens) is very heavy; 8 OOMs on 80GB.
# Micro-batch 4 × grad_accum 8 ≈ effective 32 (see configs/examples/asvr_next.py).
batch_size = 4

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=2,
    prefetch_factor=1,
    persistent_workers=False,
    pin_memory=True,
    dataset=dataset,
    sampler=dict(type=FixedBatchMultiSourceSampler,
                 repeat=repeat,
                 batch_size=batch_size,
                 shuffle=True, seed=42),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[
                        dict(type=collate_func_und,
                             pad_index=pad_index),
                    ],
                    keys=group_keys)
)
