# HiSVR: Hierarchical Semantic Visual Reconstruction for Unified Multimodal Models

HiSVR is a training-time objective for post-training unified multimodal
models (UMMs). A frozen semantic visual tokenizer turns each training image
into residual semantic codes (27x27 positions x 8 residual depths). Hidden
states at the image token positions of several Transformer layers are then
trained to reconstruct prefixes of those codes: shallow layers predict short
prefixes, deeper layers predict longer ones. An uncertainty-weighted loss
(Kendall et al., 2018) balances the per-layer, per-depth terms. All auxiliary
modules are dropped after training, so inference cost is unchanged.

> **Status:** the code was recovered from the original experiment machine and
> is being re-run from scratch. Results will be added to this README once the
> reproduction finishes. Until then, do not treat any number as final.

## Backbones

| Backbone | Generation paradigm | HiSVR layers | Status |
|---|---|---|---|
| Show-o-1.3B | discrete diffusion (MAGVIT-v2 tokens) | 8 / 16 / 24 of 24 | code complete, re-running |
| Harmon-1.5B | masked autoregressive (MAR) | 9 / 18 / 28 of 28 | code being fixed, re-running |
| Janus-Pro-1B | autoregressive (VQ tokens) | to be decided | planned |
| Janus-Pro-7B | autoregressive (VQ tokens) | to be decided | planned |

## Repository layout

```
harmon/   Harmon-1.5B (wusize/Harmon@4eb6087) + HiSVR      — mmengine / xtuner
showo/    Show-o-1.3B (showlab/Show-o@45a5a2d) + HiSVR     — accelerate
docs/     implementation notes and the exact configs of past runs
```

Each backbone keeps its upstream README as `README_upstream.md` and its
upstream license.

## Where the method lives

| Component | Show-o | Harmon |
|---|---|---|
| Semantic tokenizer (DualToken RQ-VAE SigLIP) | `showo/models/rqvaesigliptransformer/`, `showo/models/rqsiglip_encoder.py` | `harmon/src/models/multimodal_encoder/` |
| Hierarchical reconstruction + uncertainty weighting | `showo/training/train_asvr.py` | `harmon/src/models/harmon_dev.py` |
| Main configuration | `showo/configs/showo_hasvr_15k.yaml` | `harmon/configs/examples/hasvr_mmu_t2i.py` |
| SFT baseline | `showo/training/train_sft.py`, `showo/configs/showo_sft_15k.yaml` | — |

Internally the method was called `hasvr` (hierarchical ASVR); the names in
the code are kept as they were.

Default schedule: layers 8/16/24 (Show-o, 24 layers) or 9/18/28 (Harmon, 28
layers) reconstruct cumulative RQ prefixes `[0,2)`, `[0,5)`, `[0,8)`.
See [docs/IMPLEMENTATION.md](docs/IMPLEMENTATION.md) for details.

## Setup

### Weights

| What | Source |
|---|---|
| Semantic tokenizer config + checkpoint (`vision_tower_RVQ8_False_1152_epoch_6/`, `epoch_6.pt`) | DualToken, via the [ASVR README](https://github.com/AlenjandroWang/ASVR) |
| Show-o 512x512, MAGVIT-v2, Phi-1.5 | `showlab/show-o-512x512`, `showlab/magvitv2`, `microsoft/phi-1_5` on Hugging Face |
| Harmon-1.5B (`harmon_1.5b.pth`) and the MAR VAE (`kl16.ckpt`) | see `harmon/README_upstream.md` |

### Data

| Use | Dataset |
|---|---|
| Understanding (Show-o) | LLaVA-v1.5 mix 665K |
| Understanding (Harmon) | LLaVA-NeXT 738K |
| Generation (both) | BLIP3o-60K, all shards |

Paths in the configs still point at the original machine (`/data1/...`,
`/home/szgx/...`); edit them for your setup.

## Training

Show-o (one GPU, 15K steps):

```bash
cd showo
accelerate launch --config_file accelerate_configs/1_gpu.yaml \
    training/train_asvr.py config=configs/showo_hasvr_15k.yaml
```

Harmon:

```bash
cd harmon
export PYTHONPATH=$PWD
torchrun --nproc-per-node=4 scripts/train.py configs/examples/hasvr_mmu_t2i.py \
    --work-dir work_dirs/hasvr_mmu_t2i --launcher pytorch
```

## Evaluation

- Understanding: [VLMEvalKit](https://github.com/open-compass/VLMEvalKit)
  (POPE, MME, MMBench, SEED-Bench, GQA, MMMU, MMVP, VSR, RealWorldQA,
  TextVQA, ScienceQA).
- Generation: [GenEval](https://github.com/djghosh13/geneval).

## Acknowledgements

Built on [Harmon](https://github.com/wusize/Harmon),
[Show-o](https://github.com/showlab/Show-o),
[RecA](https://github.com/HorizonWind2004/reconstruction-alignment),
[ASVR](https://github.com/AlenjandroWang/ASVR) and
[DualToken](https://arxiv.org/abs/2503.14324).
