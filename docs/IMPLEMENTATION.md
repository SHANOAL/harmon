# Implementation notes

What the code actually does, for the parts the paper leaves loose.

## Semantic targets

- Tokenizer: DualToken RQ-VAE with a SigLIP-SO400M-patch14-384 encoder,
  frozen. Images are resized to 384x384 and encoded into 27x27 = 729
  positions with 8 residual depths.
- The tokenizer's own depth decoder (`rqtransformer_semantic`) is reused as
  the "semantic decoder". It is **trainable** and **shared** by all
  supervised layers. It predicts depth `d` conditioned on the ground-truth
  codes of depths `< d` (teacher forcing), so it is an autoregressive depth
  decoder rather than a set of independent per-depth classifiers.

## Hierarchical Semantic Code Alignment (HSCA)

- For each supervised layer, image-token hidden states (32x32 grid) are
  reshaped to 2D and bilinearly resized to 27x27.
- Each layer has its own projector: `Linear(d_model, 2048) -> GELU ->
  Linear(2048, 2048)`.
- Prefix schedule is cumulative: layer groups reconstruct depths
  `[0,2)`, `[0,5)` and `[0,8)`.

## Uncertainty-weighted progressive reconstruction (UPR)

- One learnable log-variance `s` per (layer, depth) pair: 2 + 5 + 8 = 15
  parameters, initialised to 0.
- `L_UPR = sum_{l,j} exp(-s_lj) * CE_lj + 0.5 * s_lj`.
- Show-o implements this. The recovered Harmon code used one log-variance per
  layer group (3 parameters); it is being changed to match.

## Which samples receive the HiSVR loss

- Show-o: only image-understanding (MMU) samples. The text-to-image samples in
  the same batch are skipped (`all_hidden_states[layer][batch_size_t2i:]`).
- Harmon: samples that carry `pixel_values`.

## Optimisation (Show-o run `show-o-hasvr-discrete`)

| | value |
|---|---|
| backbone lr / HiSVR lr | 5e-7 / 5e-5 |
| AdamW betas, weight decay | (0.9, 0.999), 0.01 |
| schedule | cosine, 500 warmup steps |
| steps | 15,000 |
| batch | T2I 1 + MMU 2 per step, gradient accumulation 5 |
| precision | bf16, TF32 enabled |
| loss weights | T2I 1.0, MMU 1.0, HiSVR 1.0 |

The full configuration is in `docs/run_configs/`.
