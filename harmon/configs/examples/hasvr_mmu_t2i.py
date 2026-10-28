"""
Hierarchical ASVR on Harmon — MMU (LLaVA-NeXT 738k) + T2I (BLIP3o-60k).

Progressive cumulative: deeper LLM layers predict more RQ depths.
  - Layer 9  (shallow) → depths [0, 2) (coarse only, deep supervision)
  - Layer 18 (mid)     → depths [0, 5) (coarse+mid)
  - Layer 28 (deep)    → depths [0, 8) (full prediction, main signal)

Learnable uncertainty weights (Kendall et al.) balance the groups.

Losses: loss_image2text (MMU) + loss_text2image (T2I) + loss_hasvr
"""

from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from xtuner.engine.runner import TrainLoop
from torch.optim import AdamW
from mmengine.config import read_base
from src.models.harmon_dev import HarmonDev

with read_base():
    from ..models.qwen2_5_1_5b_kl16_mar_h import model
    from ..datasets.qwen2_5_1_5b.image2text_text2image_next import train_dataloader, repeat

model.update(
    type=HarmonDev,
    pretrained_pth='checkpoints/harmon_1.5b.pth',
    freeze_llm=False,
    loss_weights={
        'image2text': 1.0,
        'text2image': 1.0,
        'hasvr': 1.0,
    },
    vision_tokenizer_config=dict(
        config_path='/data1/model_weight/vision_tower_RVQ8_False_1152_epoch_6',
        weights_path='/data1/model_weight/epoch_6.pt',
        num_semantic_tokens=729,
        semantic_depth=8,
    ),
    asvr_hidden_dim=2048,
    hierarchical_asvr=dict(
        layer_indices=[9, 18, 28],
        depth_ranges=[[0, 2], [0, 5], [0, 8]],
    ),
)

accumulative_counts = sum(repeat) * 8
dataloader_num_workers = 4
max_iters = 50000
optim_type = AdamW
lr = 1e-5
betas = (0.9, 0.95)
weight_decay = 0.02
max_norm = 1.0
warmup_ratio = 0.01
save_steps = 1000
save_total_limit = -1

train_dataloader = train_dataloader

optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    constructor='MAROptimWrapperConstructor',
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale=dict(enabled=False),
    dtype='bfloat16')

param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=False,
        begin=0,
        end=warmup_ratio * max_iters),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=False,
        begin=warmup_ratio * max_iters,
        end=max_iters)
]

train_cfg = dict(type=TrainLoop, max_iters=max_iters)

default_hooks = dict(
    timer=dict(type=IterTimerHook),
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    param_scheduler=dict(type=ParamSchedulerHook),
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    sampler_seed=dict(type=DistSamplerSeedHook),
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
randomness = dict(seed=42, deterministic=False)
log_processor = dict(by_epoch=False)
work_dir = '/data1/harmon_work_dirs/hasvr_mmu_t2i'
