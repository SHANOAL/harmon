"""
Harmon QCR: Question-Conditioned Reconstruction

During understanding training, also compute reconstruction loss
via the MAR generation pathway. Forces LLM to retain fine-grained
visual details needed for image reconstruction.
"""
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from torch.optim import AdamW
from mmengine.config import read_base
from src.models.harmon_dev import HarmonDev
from src.datasets.understanding.inline_caption_dataset import InlineCaptionDataset
from src.datasets.collate_functions import collate_func_und, CollateConcat
from src.datasets.samplers.multi_source_sampler import FixedBatchMultiSourceSampler
from xtuner.dataset.map_fns import template_map_fn_factory
from xtuner.dataset import ConcatDataset

with read_base():
    from ..models.qwen2_5_1_5b_kl16_mar_h import model
    from ..datasets.qwen2_5_1_5b.processors import (
        prompt_template, tokenizer, image_size, pad_index, image_length
    )

# Model
model.update(
    type=HarmonDev,
    pretrained_pth='checkpoints/harmon_1.5b.pth',
    freeze_llm=False,
    loss_weights={'image2text': 1.0, 'text2image': 1.0, 'qcr': 0.1},
)

# Override LLM and tokenizer paths to local
_qwen_local = '/home/szgx/Desktop/ctl/Show-o/show-o2/weights/Qwen2.5-1.5B-Instruct'
model['llm']['pretrained_model_name_or_path'] = _qwen_local
model['llm']['attn_implementation'] = 'sdpa'
tokenizer['pretrained_model_name_or_path'] = _qwen_local

# Dataset: ALLaVA captions (understanding only, QCR adds recon loss internally)
und_dataset = dict(
    type=InlineCaptionDataset,
    data_path='data/allava/allava_50k.json',
    local_folder='/home/szgx/Desktop/ctl/reconstruction-alignment/Show-o/data/ALLaVA-4V',
    image_size=image_size,
    tokenizer=tokenizer,
    template_map_fn=dict(type=template_map_fn_factory, template=prompt_template),
    max_length=2048,
    image_length=image_length,
)

dataset = dict(type=ConcatDataset, datasets=[und_dataset])

group_keys = ['image2text']
repeat = [1]
batch_size = 6

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
                 shuffle=True),
    collate_fn=dict(type=CollateConcat,
                    collate_fns=[dict(type=collate_func_und, pad_index=pad_index)],
                    keys=group_keys)
)

# Training
accumulative_counts = 1
max_iters = 10000
optim_type = AdamW
lr = 1e-5
betas = (0.9, 0.95)
weight_decay = 0.02
max_norm = 1.0
warmup_ratio = 0.01
save_steps = 2500
save_total_limit = 2

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    constructor='MAROptimWrapperConstructor',
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='bfloat16',
)

param_scheduler = [
    dict(type=LinearLR, start_factor=1e-5, by_epoch=False,
         begin=0, end=warmup_ratio * max_iters),
    dict(type=CosineAnnealingLR, eta_min=0.0, by_epoch=False,
         begin=warmup_ratio * max_iters, end=max_iters),
]

train_cfg = dict(type=TrainLoop, max_iters=max_iters)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(type=CheckpointHook, by_epoch=False,
                    interval=save_steps, max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
)

strategy = dict(
    type='DDPStrategy',
    model_wrapper=dict(type='MMDistributedDataParallel', find_unused_parameters=True),
)

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'),
)

visualizer = None
log_level = 'INFO'
load_from = None
resume = False
randomness = dict(seed=None, deterministic=False)
log_processor = dict(by_epoch=False)
