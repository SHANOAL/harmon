"""Harmon training extensions: RecA-style recon, DualToken ASVR, self-ASVR.

Layout:
  - Small helpers (_ScaleGradient, interpolate_hidden_states)
  - HarmonDev: loss branches (text2image / image2text / recon) + ASVR hooks
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import T
from mmengine.model import BaseModel
from torch.autograd.function import Function
from mmengine.logging import print_log
from xtuner.model.utils import guess_load_checkpoint
from xtuner.utils import IMAGE_TOKEN_INDEX

from .asvr_utils import compute_asvr_semantic_ce_loss, default_hierarchical_depth_weights
from .harmon import Harmon

NUM_IMAGE_EMBED_TOKENS = 1088  # 1024 image + 64 MAR buffer
IMAGE_TOKENS_ONLY = 1024
IMAGE_SPATIAL_SIZE = 32


class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


def interpolate_hidden_states(hidden_states, src_size, tgt_size):
    """Bilinear resize of token grid (src_size x src_size) -> (tgt x tgt)."""
    if src_size == tgt_size:
        return hidden_states
    b, n, d = hidden_states.shape
    h = hidden_states.reshape(b, src_size, src_size, d).permute(0, 3, 1, 2)
    h = F.interpolate(h.float(), size=(tgt_size, tgt_size), mode='bilinear', align_corners=False)
    h = h.to(hidden_states.dtype)
    return h.permute(0, 2, 3, 1).reshape(b, tgt_size * tgt_size, d)


class HarmonDev(Harmon, BaseModel):
    """Harmon + optional DualToken ASVR / self-ASVR / bidirectional semantic alignment."""

    def __init__(
        self,
        grad_scale=0.1,
        loss_weights=None,
        pretrained_pth=None,
        freeze_llm=False,
        gradient_checkpointing=True,
        vision_tokenizer_config=None,
        self_asvr=False,
        self_asvr_loss_type='cosine',
        asvr_hidden_dim=2048,
        bidirectional_asvr=False,
        asvr_depth_weights=None,
        use_hierarchical_asvr_weights=False,
        hierarchical_asvr=None,
        **kwargs,
    ):
        if loss_weights is None:
            loss_weights = {'image2text': 1.0, 'text2image': 1.0, 'recon': 1.0}
        super().__init__(**kwargs)
        self.grad_scale = grad_scale
        self.loss_weights = loss_weights
        self.bidirectional_asvr = bidirectional_asvr
        self.debuged = False
        self.vision_tokenizer = None
        self.asvr_projector = None
        self.rqtransformer_semantic = None
        self.self_asvr_projector = None
        self.self_asvr_loss_type = self_asvr_loss_type
        self.asvr_semantic_depth = 8
        self.hierarchical_asvr_cfg = hierarchical_asvr

        if hierarchical_asvr is not None and vision_tokenizer_config is not None:
            self._init_hierarchical_asvr(vision_tokenizer_config, asvr_hidden_dim, hierarchical_asvr)
        elif vision_tokenizer_config is not None:
            self._init_asvr(vision_tokenizer_config, asvr_hidden_dim)
            self._setup_asvr_depth_weights(
                vision_tokenizer_config,
                use_hierarchical_asvr_weights,
                asvr_depth_weights,
            )

        if self_asvr:
            self._init_self_asvr(asvr_hidden_dim)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            info = self.load_state_dict(pretrained_state_dict, strict=False)
            print_log(f'Load pretrained weight from {pretrained_pth}')
            if info.missing_keys:
                print_log(f'Missing keys ({len(info.missing_keys)}): {info.missing_keys[:10]}...')
            if info.unexpected_keys:
                print_log(f'Unexpected keys ({len(info.unexpected_keys)}): {info.unexpected_keys[:10]}...')

        if freeze_llm:
            self.llm.requires_grad_(False)

        if gradient_checkpointing:
            self.gradient_checkpointing_enable()
        else:
            self.gradient_checkpointing_disable()

    def _setup_asvr_depth_weights(self, vt_config, use_hierarchical, explicit_weights):
        """Hierarchical = linear ramp over RQ depth; else uniform or user list."""
        depth = int(vt_config.get('semantic_depth', 8))
        self.asvr_semantic_depth = depth
        if explicit_weights is not None:
            w = list(explicit_weights)
        elif use_hierarchical:
            w = default_hierarchical_depth_weights(depth)
        else:
            w = [1.0] * depth
        self.register_buffer(
            '_asvr_depth_w', torch.tensor(w, dtype=torch.float32), persistent=False)

    def _init_asvr(self, vt_config, asvr_hidden_dim):
        from .multimodal_encoder.rqsiglip_encoder import RQVAESIGLIPTransformerVisionTower

        self.vision_tokenizer = RQVAESIGLIPTransformerVisionTower(
            vt_config['config_path'],
            vt_config['weights_path'],
            delay_load=False,
        )
        self.vision_tokenizer.eval()
        self.vision_tokenizer.requires_grad_(False)

        self.rqtransformer_semantic = self.vision_tokenizer.vision_tower.rqtransformer_semantic
        for p in self.rqtransformer_semantic.parameters():
            p.requires_grad = True

        llm_hidden_size = self.llm.config.hidden_size
        self.asvr_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, asvr_hidden_dim),
            nn.GELU(),
            nn.Linear(asvr_hidden_dim, asvr_hidden_dim),
        )

        self.asvr_num_semantic_tokens = vt_config.get('num_semantic_tokens', 729)
        self.asvr_semantic_spatial_size = int(self.asvr_num_semantic_tokens ** 0.5)
        assert self.asvr_semantic_spatial_size ** 2 == self.asvr_num_semantic_tokens

        print_log(
            f'ASVR initialized: semantic tokens={self.asvr_num_semantic_tokens}, '
            f'projector dim={asvr_hidden_dim}, bidirectional={self.bidirectional_asvr}'
        )

    def _init_hierarchical_asvr(self, vt_config, asvr_hidden_dim, hasvr_cfg):
        """Multi-layer hierarchical ASVR: different LLM layers predict different RQ depths."""
        from .multimodal_encoder.rqsiglip_encoder import RQVAESIGLIPTransformerVisionTower

        self.vision_tokenizer = RQVAESIGLIPTransformerVisionTower(
            vt_config['config_path'],
            vt_config['weights_path'],
            delay_load=False,
        )
        self.vision_tokenizer.eval()
        self.vision_tokenizer.requires_grad_(False)

        self.rqtransformer_semantic = self.vision_tokenizer.vision_tower.rqtransformer_semantic
        for p in self.rqtransformer_semantic.parameters():
            p.requires_grad = True

        self.asvr_num_semantic_tokens = vt_config.get('num_semantic_tokens', 729)
        self.asvr_semantic_spatial_size = int(self.asvr_num_semantic_tokens ** 0.5)
        self.asvr_semantic_depth = int(vt_config.get('semantic_depth', 8))

        self.hasvr_layer_indices = hasvr_cfg['layer_indices']
        self.hasvr_depth_ranges = hasvr_cfg['depth_ranges']
        n_groups = len(self.hasvr_layer_indices)
        assert n_groups == len(self.hasvr_depth_ranges)

        llm_hidden_size = self.llm.config.hidden_size
        self.hasvr_projectors = nn.ModuleDict()
        for i, li in enumerate(self.hasvr_layer_indices):
            self.hasvr_projectors[f'layer_{li}'] = nn.Sequential(
                nn.Linear(llm_hidden_size, asvr_hidden_dim),
                nn.GELU(),
                nn.Linear(asvr_hidden_dim, asvr_hidden_dim),
            )

        self.hasvr_log_var = nn.Parameter(torch.zeros(n_groups))

        print_log(
            f'Hierarchical ASVR: layers={self.hasvr_layer_indices}, '
            f'depth_ranges={self.hasvr_depth_ranges}, '
            f'projector_dim={asvr_hidden_dim}'
        )

    @property
    def has_hierarchical_asvr(self):
        return self.hierarchical_asvr_cfg is not None

    def _init_self_asvr(self, asvr_hidden_dim):
        llm_hidden_size = self.llm.config.hidden_size
        mar_hidden_size = self.mar.encoder_embed_dim
        self.self_asvr_projector = nn.Sequential(
            nn.Linear(llm_hidden_size, asvr_hidden_dim),
            nn.GELU(),
            nn.Linear(asvr_hidden_dim, mar_hidden_size),
        )
        print_log(
            f'Self-ASVR: LLM {llm_hidden_size} -> MAR {mar_hidden_size}, '
            f'hidden={asvr_hidden_dim}, loss={self.self_asvr_loss_type}'
        )

    def gradient_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        self.mar.gradient_checkpointing_disable()

    def gradient_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        self.mar.gradient_checkpointing_enable()

    @property
    def has_asvr(self):
        return self.vision_tokenizer is not None

    @property
    def has_self_asvr(self):
        return self.self_asvr_projector is not None

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        metadata = getattr(state_dict, '_metadata', OrderedDict())
        exclude_prefixes = ['vae.']
        if self.has_asvr or self.has_hierarchical_asvr:
            exclude_prefixes.append('vision_tokenizer.')
        out = OrderedDict(
            (k, v) for k, v in state_dict.items()
            if not any(k.startswith(p) for p in exclude_prefixes))
        out._metadata = metadata
        return out

    def train(self: T, mode: bool = True) -> T:
        super().train(mode=mode)
        self.vae.train(mode=False)
        if self.has_asvr or self.has_hierarchical_asvr:
            self.vision_tokenizer.train(mode=False)
            self.rqtransformer_semantic.train(mode=mode)
        return self

    # ------------------------------------------------------------------ #
    # T2I
    # ------------------------------------------------------------------ #
    def text2image_loss(self, data_dict, return_gen_asvr_hidden=False):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m * n, -1)

        orders = self.mar.sample_orders(bsz=b, seq_len=m * n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        enc_out = self.forward_mae_encoder(
            x, mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_llm_hidden=return_gen_asvr_hidden,
        )
        if return_gen_asvr_hidden:
            x_enc, hidden_gen = enc_out
        else:
            x_enc = enc_out

        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        loss = self.mar.forward_loss(z=z, target=gt_latents, mask=mask)
        if return_gen_asvr_hidden:
            return loss, hidden_gen
        return loss

    # ------------------------------------------------------------------ #
    # MMU (image2text)
    # ------------------------------------------------------------------ #
    def image2text_loss(self, data_dict, return_hidden_for_asvr=False,
                        return_all_hidden_states=False):
        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)
        labels = data_dict['labels'].to(self.device)
        pixel_values = data_dict.get('pixel_values', None)
        if pixel_values is None:
            raise ValueError('image2text batch requires pixel_values')

        x = pixel_values.to(dtype=self.dtype, device=self.device)
        x = self.encode(x)
        x_enc, z_enc = self.extract_visual_feature(x)

        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX
        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], NUM_IMAGE_EMBED_TOKENS,
                    dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], NUM_IMAGE_EMBED_TOKENS,
                    dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ),
            dim=1,
        )
        labels = torch.cat(
            (
                labels[:, :3],
                labels[0, 3] * torch.ones(
                    labels.shape[0], NUM_IMAGE_EMBED_TOKENS,
                    dtype=torch.long, device=labels.device),
                labels[:, 4:],
            ),
            dim=1,
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        output = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=return_all_hidden_states,
        )

        full_hidden_state = output.last_hidden_state
        last_hidden_state = full_hidden_state[:, :-1]
        labels_shifted = labels[:, 1:]
        last_hidden_state_valid = last_hidden_state[labels_shifted >= 0]
        labels_valid = labels_shifted[labels_shifted >= 0]
        logits = self.llm.get_output_embeddings()(last_hidden_state_valid)

        loss_i2t = F.cross_entropy(input=logits, target=labels_valid)

        if return_all_hidden_states:
            img_start = 3
            img_end = img_start + IMAGE_TOKENS_ONLY
            all_hs = output.hidden_states
            return loss_i2t, all_hs, img_start, img_end

        if return_hidden_for_asvr:
            img_start = 3
            img_end = img_start + IMAGE_TOKENS_ONLY
            hidden_states_img = full_hidden_state[:, img_start:img_end, :]
            x_enc_img = x_enc[:, self.mar.buffer_size:, :].detach()
            return loss_i2t, hidden_states_img, x_enc_img

        return loss_i2t

    def _semantic_targets(self, pixel_values):
        with torch.no_grad():
            pv_384 = F.interpolate(
                pixel_values.to(dtype=torch.bfloat16, device=self.device),
                size=(384, 384), mode='bilinear', align_corners=False,
            )
            vt_out = self.vision_tokenizer(pv_384)
            return vt_out[3]

    def asvr_loss(self, hidden_states_img, pixel_values):
        tokens_semantic = self._semantic_targets(pixel_values)
        hidden_states_img = interpolate_hidden_states(
            hidden_states_img,
            IMAGE_SPATIAL_SIZE,
            self.asvr_semantic_spatial_size,
        )
        w = self._asvr_depth_w.to(device=hidden_states_img.device, dtype=torch.float32)
        return compute_asvr_semantic_ce_loss(
            self.asvr_projector,
            self.rqtransformer_semantic,
            self.vision_tokenizer,
            hidden_states_img.to(tokens_semantic.device),
            tokens_semantic,
            depth_weights=w,
        )

    def hierarchical_asvr_loss(self, all_hidden_states, img_start, img_end, pixel_values):
        """Compute hierarchical ASVR: each LLM layer predicts a subset of RQ depths."""
        tokens_semantic = self._semantic_targets(pixel_values)

        loss_parts = []
        for i, (layer_idx, depth_range) in enumerate(
            zip(self.hasvr_layer_indices, self.hasvr_depth_ranges)
        ):
            hs = all_hidden_states[layer_idx][:, img_start:img_end, :]
            hs = interpolate_hidden_states(
                hs, IMAGE_SPATIAL_SIZE, self.asvr_semantic_spatial_size)

            proj = self.hasvr_projectors[f'layer_{layer_idx}']
            img_out = proj(hs)

            outs_semantic = self.rqtransformer_semantic(
                img_out,
                tokens_semantic,
                self.vision_tokenizer.vision_tower.rqvaesiglip,
                mode='semantic',
                depth_range=depth_range,
            )

            ds, de = depth_range
            tokens_flat = tokens_semantic.reshape(-1, tokens_semantic.shape[-1]).contiguous()
            group_loss = []
            for d in range(ds, min(de, len(outs_semantic))):
                if outs_semantic[d] is None:
                    continue
                ce = F.cross_entropy(
                    outs_semantic[d].to(tokens_flat[..., d].device),
                    tokens_flat[..., d],
                )
                group_loss.append(ce)
            if group_loss:
                loss_parts.append(torch.stack(group_loss).mean())

        log_var = self.hasvr_log_var
        total_loss = torch.tensor(0.0, device=log_var.device)
        detail = {}
        labels = ['shallow', 'mid', 'deep']
        for i, part in enumerate(loss_parts):
            precision = torch.exp(-log_var[i])
            total_loss = total_loss + precision * part + 0.5 * log_var[i]
            label = labels[i] if i < len(labels) else f'g{i}'
            detail[label] = part.item()
        return total_loss, detail

    def self_asvr_loss(self, hidden_states_img, x_enc_img):
        projected = self.self_asvr_projector(hidden_states_img)
        if self.self_asvr_loss_type == 'cosine':
            proj_n = F.normalize(projected, dim=-1)
            tgt_n = F.normalize(x_enc_img.to(projected.dtype), dim=-1)
            return 1.0 - (proj_n * tgt_n).sum(dim=-1).mean()
        return F.mse_loss(projected, x_enc_img.to(projected.dtype))

    # ------------------------------------------------------------------ #
    # RecA-style reconstruction branch
    # ------------------------------------------------------------------ #
    def recon_loss(self, data_dict):
        x = data_dict['pixel_values'].to(dtype=self.dtype, device=self.device)
        x = self.encode(x)
        b, m, n, _ = x.shape
        gt_latents = x.clone().detach().view(b, m * n, -1)

        input_ids = data_dict['input_ids'].to(self.device)
        attention_mask = data_dict['attention_mask'].to(self.device)

        _, z_enc = self.extract_visual_feature(x)

        if self.grad_scale is not None:
            z_enc = _ScaleGradient.apply(z_enc, self.grad_scale)

        for i in range(input_ids.shape[0]):
            assert input_ids[i, 3] == IMAGE_TOKEN_INDEX

        input_ids = torch.cat(
            (
                input_ids[:, :3],
                IMAGE_TOKEN_INDEX * torch.ones(
                    input_ids.shape[0], NUM_IMAGE_EMBED_TOKENS,
                    dtype=torch.long, device=input_ids.device),
                input_ids[:, 4:],
            ),
            dim=1,
        )
        attention_mask = torch.cat(
            (
                attention_mask[:, :3],
                torch.ones(
                    attention_mask.shape[0], NUM_IMAGE_EMBED_TOKENS,
                    dtype=torch.long, device=attention_mask.device),
                attention_mask[:, 4:],
            ),
            dim=1,
        )
        inputs_embeds = z_enc.new_zeros(*input_ids.shape, self.llm.config.hidden_size)
        inputs_embeds[input_ids == IMAGE_TOKEN_INDEX] = z_enc.flatten(0, 1)
        inputs_embeds[input_ids != IMAGE_TOKEN_INDEX] = self.llm.get_input_embeddings()(
            input_ids[input_ids != IMAGE_TOKEN_INDEX])

        orders = self.mar.sample_orders(bsz=b, seq_len=m * n)
        mask = self.mar.random_masking(x.flatten(1, 2), orders)

        x_enc = self.forward_mae_encoder(
            x, mask,
            inputs_embeds=inputs_embeds,
            input_ids=None,
            attention_mask=attention_mask,
        )

        z = self.mar.forward_mae_decoder(x_enc, mask, image_shape=(m, n))
        return self.mar.forward_loss(z=z, target=gt_latents, mask=mask)

    def forward(self, data, data_samples=None, mode='loss'):
        if mode == 'loss':
            return self.compute_loss(data_dict=data)
        raise NotImplementedError

    def compute_loss(self, data_dict):
        losses = {}
        for data_type, batch_data in data_dict.items():
            if 'text2image' in data_type:
                gen_asvr = (
                    self.bidirectional_asvr and self.has_asvr
                    and 'pixel_values' in batch_data
                )
                if gen_asvr:
                    loss_t2i, h_gen = self.text2image_loss(
                        batch_data, return_gen_asvr_hidden=True)
                    losses[f'loss_{data_type}'] = (
                        loss_t2i * self.loss_weights.get('text2image', 1.0))
                    losses['loss_asvr_gen'] = (
                        self.asvr_loss(h_gen, batch_data['pixel_values'])
                        * self.loss_weights.get('asvr_gen', 1.0))
                    continue

                loss = self.text2image_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get('text2image', 1.0)

            elif 'image2text' in data_type:
                need_hasvr = self.has_hierarchical_asvr and 'pixel_values' in batch_data
                need_asvr = (self.has_asvr or self.has_self_asvr) and 'pixel_values' in batch_data

                if need_hasvr:
                    loss_i2t, all_hs, img_s, img_e = self.image2text_loss(
                        batch_data, return_all_hidden_states=True)
                    losses[f'loss_{data_type}'] = (
                        loss_i2t * self.loss_weights.get('image2text', 1.0))
                    hasvr_loss, hasvr_detail = self.hierarchical_asvr_loss(
                        all_hs, img_s, img_e, batch_data['pixel_values'])
                    losses['loss_hasvr'] = hasvr_loss * self.loss_weights.get('hasvr', 1.0)
                    for k, v in hasvr_detail.items():
                        losses[f'loss_hasvr_{k}'] = torch.tensor(v, device=hasvr_loss.device)
                    continue

                if need_asvr:
                    loss_i2t, hidden_img, x_enc_img = self.image2text_loss(
                        batch_data, return_hidden_for_asvr=True)
                    losses[f'loss_{data_type}'] = (
                        loss_i2t * self.loss_weights.get('image2text', 1.0))
                    if self.has_asvr:
                        la = self.asvr_loss(hidden_img, batch_data['pixel_values'])
                        losses['loss_asvr'] = la * self.loss_weights.get('asvr', 1.0)
                    if self.has_self_asvr:
                        ls = self.self_asvr_loss(hidden_img, x_enc_img)
                        losses['loss_self_asvr'] = ls * self.loss_weights.get('self_asvr', 1.0)
                    continue

                loss = self.image2text_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get('image2text', 1.0)

            elif 'recon' in data_type:
                loss = self.recon_loss(batch_data)
                losses[f'loss_{data_type}'] = loss * self.loss_weights.get(data_type, 1.0)
            else:
                raise NotImplementedError('Unknown data type: %s' % data_type)
        return losses
