# Copyright (c) OpenMMLab. All rights reserved.
"""ASVR (Autoregressive Semantic Visual Reconstruction) loss helpers for Harmon.

Depth-weighted CE over RVQ code levels implements *hierarchical* alignment
(shallow = coarse, deep = fine). Shared by MMU (image2text) and T2I branches.
"""
import torch
import torch.nn.functional as F


def normalize_depth_weights(depth_weights, num_depths, device, dtype=torch.float32):
    """Positive per-depth weights; ``None`` means uniform."""
    if depth_weights is None:
        return torch.ones(num_depths, device=device, dtype=dtype)
    if isinstance(depth_weights, torch.Tensor):
        w = depth_weights.to(device=device, dtype=dtype).view(-1)
    else:
        w = torch.tensor(list(depth_weights), device=device, dtype=dtype).view(-1)
    if w.numel() == 1:
        w = w.expand(num_depths)
    if w.numel() != num_depths:
        raise ValueError(
            'asvr_depth_weights length %d must match semantic depth %d'
            % (w.numel(), num_depths))
    return torch.clamp(w, min=1e-6)


def compute_asvr_semantic_ce_loss(
    asvr_projector,
    rqtransformer_semantic,
    vision_tokenizer,
    hidden_states_img,
    tokens_semantic,
    depth_weights=None,
):
    """Weighted mean cross-entropy over RVQ depth levels."""
    _b, _seq_len, depth = tokens_semantic.shape
    img_out = asvr_projector(hidden_states_img)
    outs_semantic = rqtransformer_semantic(
        img_out,
        tokens_semantic,
        vision_tokenizer.vision_tower.rqvaesiglip,
        mode='semantic',
    )
    n_out = len(outs_semantic)
    if n_out != depth:
        depth = n_out
        tokens_semantic = tokens_semantic[..., :depth]
    tokens_flat = tokens_semantic.reshape(-1, depth).contiguous()
    w = normalize_depth_weights(depth_weights, n_out, hidden_states_img.device)
    w = w.to(dtype=torch.float32)
    parts = []
    for i in range(n_out):
        li = F.cross_entropy(
            outs_semantic[i].to(tokens_flat[..., i].device),
            tokens_flat[..., i],
        )
        parts.append(li * w[i])
    return torch.stack(parts).sum() / w.sum()


def default_hierarchical_depth_weights(depth=8):
    """Linear ramp: deeper RQ levels get higher weight (finer semantics)."""
    return [0.25 + 0.75 * (i / max(depth - 1, 1)) for i in range(depth)]
