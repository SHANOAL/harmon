# coding=utf-8
"""
Show-o Option (a) + Hierarchical ASVR training script.

Different LLM hidden-state layers predict different RQ depth levels:
  - Shallow LLM layers -> RQ deep depths (fine details)
  - Mid LLM layers     -> RQ mid depths  (mid-level structure)
  - Deep LLM layers    -> RQ shallow depths (coarse semantics)

Each group has an independent MLP projector. Per-depth losses are balanced
via learnable uncertainty weights (Kendall et al.).

Option (a): no CLIP ViT, pure VQ discrete tokens for image understanding.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json, logging, math, random, shutil, time
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed
import torch.distributed as dist

from training.data import Text2ImageDataset
from models import Showo, MAGVITv2, RQVAESIGLIPTransformerVisionTower, get_mask_chedule
from training.prompting_utils import (
    UniversalPrompting,
    create_attention_mask_predict_next,
    create_attention_mask_for_mmu,
)
from models.lr_schedulers import get_scheduler
from models.logging import set_verbosity_info, set_verbosity_error
from llava.llava_data_vq_unified import get_instruct_data_loader
from training.utils import get_config, flatten_omega_conf, mask_or_random_replace_tokens, AverageMeter

try:
    import apex; is_apex_available = True
except ImportError:
    is_apex_available = False

logger = get_logger(__name__, log_level="INFO")
SYSTEM_PROMPT_LEN = 28
GROUP_NAMES = ["shallow", "mid", "deep"]


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    raise ValueError(f"model_type {model_type} not supported.")


def interpolate_hidden_states(hidden_states, src_size, tgt_size):
    if src_size == tgt_size:
        return hidden_states
    B, N, D = hidden_states.shape
    h = hidden_states.reshape(B, src_size, src_size, D).permute(0, 3, 1, 2)
    h = F.interpolate(h.float(), size=(tgt_size, tgt_size), mode='bilinear', align_corners=False)
    h = h.to(hidden_states.dtype)
    return h.permute(0, 2, 3, 1).reshape(B, tgt_size * tgt_size, D)


def build_hasvr_projectors(hidden_size, asvr_dim, device):
    projectors = nn.ModuleDict()
    for name in GROUP_NAMES:
        projectors[name] = nn.Sequential(
            nn.Linear(hidden_size, asvr_dim), nn.GELU(), nn.Linear(asvr_dim, asvr_dim),
        )
    return projectors.to(device)


def compute_hasvr_loss(
    hasvr_projectors, rqtransformer_semantic, vision_tokenizer,
    all_hidden_states, tokens_semantic,
    layer_indices, depth_ranges, hasvr_log_var,
    batch_size_t2i, img_start, img_end, vq_spatial, sem_spatial,
):
    B_sem, SEQ_LEN, total_depth = tokens_semantic.shape
    model_aux = vision_tokenizer.vision_tower.rqvaesiglip
    group_losses, depth_losses = {}, []
    var_idx = 0

    for group_name, layer_idx, (ds, de) in zip(GROUP_NAMES, layer_indices, depth_ranges):
        h = all_hidden_states[layer_idx][batch_size_t2i:][:, img_start:img_end, :]
        h = interpolate_hidden_states(h, vq_spatial, sem_spatial)
        proj = hasvr_projectors[group_name](h)

        logits_list = rqtransformer_semantic(
            proj.to(tokens_semantic.device), tokens_semantic, model_aux,
            mode="semantic", depth_range=(ds, de),
        )
        tokens_flat = tokens_semantic.reshape(B_sem * SEQ_LEN, total_depth).contiguous()
        parts = []
        for i in range(ds, de):
            if logits_list[i] is None:
                continue
            loss_i = F.cross_entropy(logits_list[i].to(tokens_flat[:, i].device), tokens_flat[:, i])
            log_v = hasvr_log_var[var_idx]
            depth_losses.append(loss_i * torch.exp(-log_v) + 0.5 * log_v)
            parts.append(loss_i.detach())
            var_idx += 1
        group_losses[group_name] = torch.stack(parts).mean() if parts else torch.tensor(0.0, device=tokens_semantic.device)

    return (torch.stack(depth_losses).sum() if depth_losses else torch.tensor(0.0, device=tokens_semantic.device)), group_losses


def save_checkpoint(model, config, accelerator, global_step,
                    rqtransformer_semantic=None, hasvr_projectors=None, hasvr_log_var=None):
    output_dir = config.experiment.output_dir
    limit = config.experiment.get("checkpoints_total_limit", None)
    if accelerator.is_main_process and limit is not None:
        cps = sorted([d for d in os.listdir(output_dir) if d.startswith("checkpoint")],
                     key=lambda x: int(x.split("-")[1]))
        for cp in cps[:max(0, len(cps) - limit + 1)]:
            shutil.rmtree(os.path.join(output_dir, cp))
    save_path = Path(output_dir) / f"checkpoint-{global_step}"
    state_dict = accelerator.get_state_dict(model)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(
            save_path / "unwrapped_model", save_function=accelerator.save,
            state_dict=state_dict, safe_serialization=False)
        if rqtransformer_semantic is not None:
            torch.save(rqtransformer_semantic.state_dict(), save_path / "rqtransformer_semantic.pt")
        if hasvr_projectors is not None:
            torch.save(accelerator.unwrap_model(hasvr_projectors).state_dict(), save_path / "hasvr_projectors.pt")
        if hasvr_log_var is not None:
            torch.save(hasvr_log_var.data, save_path / "hasvr_log_var.pt")
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


def main():
    config = get_config()
    random.seed(config.training.seed)
    if config.training.enable_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    config.experiment.logging_dir = str(Path(config.experiment.output_dir) / "logs")
    accelerator = Accelerator(
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        mixed_precision=config.training.mixed_precision, log_with="wandb",
        project_dir=config.experiment.logging_dir, split_batches=True)

    total_batch_size_per_gpu = config.training.batch_size_t2i + config.training.batch_size_mmu
    total_batch_size = total_batch_size_per_gpu * accelerator.num_processes * config.training.gradient_accumulation_steps
    if accelerator.distributed_type == DistributedType.DEEPSPEED:
        accelerator.state.deepspeed_plugin.deepspeed_config["train_micro_batch_size_per_gpu"] = total_batch_size_per_gpu

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S", level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        set_verbosity_info()
    else:
        set_verbosity_error()

    if accelerator.is_main_process:
        resume_wandb = config.wandb.resume
        run_id = config.wandb.get("run_id", None)
        if run_id is None:
            resume_wandb = False; run_id = wandb.util.generate_id(); config.wandb.run_id = run_id
        wc = {k: v for k, v in flatten_omega_conf(config, resolve=True)}
        wc.pop("experiment.resume_from_checkpoint", None)
        accelerator.init_trackers(config.experiment.project, config=wc,
            init_kwargs={"wandb": dict(name=config.experiment.name, id=run_id, resume=resume_wandb,
                                       entity=config.wandb.get("entity", None), config_exclude_keys=[])})
        os.makedirs(config.experiment.output_dir, exist_ok=True)
        OmegaConf.save(config, Path(config.experiment.output_dir) / "config.yaml")

    if config.training.seed is not None:
        set_seed(config.training.seed)

    # --- Hierarchical ASVR config ---
    hasvr_cfg = config.training.hierarchical_asvr
    layer_indices = list(hasvr_cfg.layer_indices)
    depth_ranges = [list(dr) for dr in hasvr_cfg.depth_ranges]
    total_depths = sum(de - ds for ds, de in depth_ranges)
    logger.info(f"HASVR: layers={layer_indices}, depths={depth_ranges}, total={total_depths}")

    # --- Models ---
    tokenizer = AutoTokenizer.from_pretrained(config.model.showo.llm_model_path, padding_side="left")
    uni_prompting = UniversalPrompting(
        tokenizer, max_text_len=config.dataset.preprocessing.max_seq_length,
        special_tokens=("<|soi|>","<|eoi|>","<|sov|>","<|eov|>","<|t2i|>","<|mmu|>","<|t2v|>","<|v2v|>","<|lvg|>"),
        ignore_id=-100, cond_dropout_prob=config.training.cond_dropout_prob)

    vq_model = get_vq_model_class(config.model.vq_model.type).from_pretrained(
        config.model.vq_model.vq_model_name).to(accelerator.device)
    vq_model.eval(); vq_model.requires_grad_(False)

    if config.model.showo.load_from_showo:
        model = Showo.from_pretrained(config.model.showo.pretrained_model_path).to(accelerator.device)
        if config.model.showo.vocab_size != model.vocab_size:
            model.showo.resize_token_embeddings(config.model.showo.vocab_size)
            model.config.codebook_size = config.model.showo.codebook_size
            model.config.vocab_size = config.model.showo.vocab_size
            model.vocab_size = config.model.showo.vocab_size
            model.output_size = config.model.showo.vocab_size
            model.config.mask_token_id = model.config.vocab_size - 1
    else:
        model = Showo(**config.model.showo).to(accelerator.device)
    mask_id = model.config.mask_token_id

    vt_config = config.model.vision_tokenizer
    vision_tokenizer = RQVAESIGLIPTransformerVisionTower(
        vt_config.config_path, vt_config.weights_path, delay_load=False).to(accelerator.device)
    vision_tokenizer.eval()
    for p in vision_tokenizer.parameters():
        p.requires_grad = False
    rqt_sem = vision_tokenizer.vision_tower.rqtransformer_semantic
    for p in rqt_sem.parameters():
        p.requires_grad = True

    num_vq_tokens = config.model.showo.num_vq_tokens
    vq_spatial = int(math.sqrt(num_vq_tokens))
    sem_spatial = int(math.sqrt(vt_config.get("num_semantic_tokens", 729)))

    asvr_dim = config.model.showo.get("asvr_hidden_dim", 2048)
    hasvr_proj = build_hasvr_projectors(model.showo.config.hidden_size, asvr_dim, accelerator.device)
    hasvr_log_var = nn.Parameter(torch.zeros(total_depths, device=accelerator.device))

    # --- Optimizer ---
    oc = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    asvr_lr = oc.get("asvr_learning_rate", oc.learning_rate * 100)

    mpd, mpnd, apd, apnd = [], [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        (mpnd if any(nd in n for nd in no_decay) else mpd).append(p)
    for n, p in rqt_sem.named_parameters():
        if not p.requires_grad: continue
        (apnd if any(nd in n for nd in no_decay) else apd).append(p)
    for n, p in hasvr_proj.named_parameters():
        if not p.requires_grad: continue
        (apnd if any(nd in n for nd in no_decay) else apd).append(p)
    apnd.append(hasvr_log_var)

    optimizer = AdamW([
        {"params": mpd, "weight_decay": oc.weight_decay, "lr": oc.learning_rate},
        {"params": mpnd, "weight_decay": 0.0, "lr": oc.learning_rate},
        {"params": apd, "weight_decay": oc.weight_decay, "lr": asvr_lr},
        {"params": apnd, "weight_decay": 0.0, "lr": asvr_lr},
    ], lr=oc.learning_rate, betas=(oc.beta1, oc.beta2), weight_decay=oc.weight_decay, eps=oc.epsilon)

    mask_schedule = get_mask_chedule(config.training.get("mask_schedule", "cosine"))
    lr_scheduler = get_scheduler(config.lr_scheduler.scheduler, optimizer=optimizer,
        num_training_steps=config.training.max_train_steps, num_warmup_steps=config.lr_scheduler.params.warmup_steps)

    # --- Data ---
    pc = config.dataset.preprocessing; dc = config.dataset.params
    bs_t2i_no_acc = config.training.batch_size_t2i * accelerator.num_processes
    dataset = Text2ImageDataset(
        train_shards_path_or_url=dc.train_t2i_shards_path_or_url, tokenizer=None,
        max_seq_length=pc.max_seq_length, num_train_examples=config.experiment.max_train_examples_t2i,
        per_gpu_batch_size=config.training.batch_size_t2i, global_batch_size=bs_t2i_no_acc,
        num_workers=dc.num_workers, resolution=pc.resolution,
        shuffle_buffer_size=dc.shuffle_buffer_size, pin_memory=dc.pin_memory,
        persistent_workers=dc.persistent_workers, external_caption_path=dc.get("external_caption_path",""),
        external_journeydb_caption_path=dc.get("external_journeydb_caption_path",""),
        external_laion12m_caption_path=dc.get("external_laion12m_caption_path",""),
        external_cc12m_caption_path=dc.get("external_cc12m_caption_path",""))
    dl_t2i = dataset.train_dataloader
    n_upd = math.ceil(dl_t2i.num_batches / config.training.gradient_accumulation_steps)
    n_epochs = math.ceil(config.training.max_train_steps / n_upd)

    dl_mmu = get_instruct_data_loader(tokenizer, resolution=pc.resolution, batch_size=config.training.batch_size_mmu,
        num_workers=dc.num_workers, world_size=accelerator.num_processes,
        local_rank=accelerator.process_index, max_length=pc.max_seq_length + SYSTEM_PROMPT_LEN, phase="tuning")
    combined_dl = CombinedLoader({"t2i_flow": dl_t2i, "mmu_flow": dl_mmu}, mode=config.dataset.combined_loader_mode)

    # --- Resume ---
    global_step, first_epoch = 0, 0
    if config.experiment.resume_from_checkpoint:
        dirs = sorted([d for d in os.listdir(config.experiment.output_dir) if d.startswith("checkpoint")],
                      key=lambda x: int(x.split("-")[1]))
        if dirs:
            path = os.path.join(config.experiment.output_dir, dirs[-1])
            global_step = int(os.path.basename(path).split("-")[1])
            first_epoch = global_step // n_upd
            model.load_state_dict(torch.load(f'{path}/unwrapped_model/pytorch_model.bin', map_location="cpu"), strict=True)
            for fn, tgt in [("rqtransformer_semantic.pt", rqt_sem), ("hasvr_projectors.pt", hasvr_proj)]:
                fp = os.path.join(path, fn)
                if os.path.exists(fp):
                    tgt.load_state_dict(torch.load(fp, map_location="cpu"), strict=True)
            lv_p = os.path.join(path, "hasvr_log_var.pt")
            if os.path.exists(lv_p):
                hasvr_log_var.data.copy_(torch.load(lv_p, map_location="cpu"))
            accelerator.print(f"Resumed from step {global_step}")

    # --- Prepare ---
    model, hasvr_proj, optimizer, lr_scheduler = accelerator.prepare(model, hasvr_proj, optimizer, lr_scheduler)
    vq_model.to(accelerator.device); vision_tokenizer.to(accelerator.device)
    mask_dtype = (model.module if hasattr(model, 'module') else model).showo.model.embed_tokens.weight.dtype

    img_start, img_end = 2, 2 + num_vq_tokens
    logger.info(f"VQ spatial={vq_spatial}, Sem spatial={sem_spatial}, img_range=[{img_start},{img_end})")

    @torch.no_grad()
    def prep_t2i(pv, texts):
        it = vq_model.get_code(pv) + len(uni_prompting.text_tokenizer)
        ids, lbl, lw, mp = mask_or_random_replace_tokens(it, mask_id, config, mask_schedule=mask_schedule, is_train=True)
        ids, _, lbl = uni_prompting((texts, ids, lbl), 't2i')
        return ids, lbl, mp

    @torch.no_grad()
    def sem_targets(pv):
        inp = F.interpolate(pv, size=(384,384), mode='bilinear', align_corners=False)
        inp = inp.to(next(vision_tokenizer.parameters()).dtype)
        return vision_tokenizer(inp)[3]

    btm, dtm = AverageMeter(), AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, n_epochs):
        model.train(); rqt_sem.train()
        proj_uw = accelerator.unwrap_model(hasvr_proj)

        for batch, batch_idx, _ in combined_dl:
            bst = batch["t2i_flow"]["images"].shape[0]
            bsm = batch["mmu_flow"]["images"].shape[0]
            dtm.update(time.time() - end)

            pv = batch["t2i_flow"]["images"].to(accelerator.device, non_blocking=True)
            ids, lbl, mp = prep_t2i(pv, batch["t2i_flow"]["input_ids"])
            am = create_attention_mask_predict_next(ids,
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True, return_inverse_mask=True).to(mask_dtype)

            pvm = batch["mmu_flow"]["images"].to(accelerator.device, non_blocking=True)
            idm = batch["mmu_flow"]["input_ids"].to(accelerator.device, non_blocking=True)
            lbm = batch["mmu_flow"]["labels"]
            itm = vq_model.get_code(pvm) + len(uni_prompting.text_tokenizer)
            dev = accelerator.device
            idm = torch.cat([
                torch.full((bsm,1), int(uni_prompting.sptids_dict['<|mmu|>']), device=dev),
                torch.full((bsm,1), int(uni_prompting.sptids_dict['<|soi|>']), device=dev),
                itm,
                torch.full((bsm,1), int(uni_prompting.sptids_dict['<|eoi|>']), device=dev),
                idm], dim=1).long()
            lbm = torch.cat([
                torch.full((bsm,1), uni_prompting.ignore_id, device=dev),
                torch.full((bsm,1), uni_prompting.ignore_id, device=dev),
                torch.full_like(itm, uni_prompting.ignore_id),
                torch.full((bsm,1), uni_prompting.ignore_id, device=dev),
                lbm.to(dev)], dim=1).long()
            amm = create_attention_mask_for_mmu(idm, eoi_id=int(uni_prompting.sptids_dict['<|eoi|>'])).to(mask_dtype)
            tok_sem = sem_targets(pvm)

            a_ids = torch.cat([ids, idm], 0)
            a_lbl = torch.cat([lbl, lbm], 0)
            a_am = torch.cat([am, amm], 0)

            with accelerator.accumulate(model, hasvr_proj):
                _, lt, _, lm, all_h = model(
                    input_ids=a_ids, input_embeddings=None, attention_mask=a_am, labels=a_lbl,
                    batch_size_t2i=bst, batch_size_lm=0, batch_size_mmu=bsm,
                    max_seq_length=pc.max_seq_length,
                    return_hidden_states=True, return_all_hidden_states=True)

                lh, gl = compute_hasvr_loss(proj_uw, rqt_sem, vision_tokenizer,
                    all_h, tok_sem, layer_indices, depth_ranges, hasvr_log_var,
                    bst, img_start, img_end, vq_spatial, sem_spatial)

                loss = config.training.t2i_coeff*lt + config.training.mmu_coeff*lm + config.training.asvr_coeff*lh
                accelerator.backward(loss)
                if accelerator.num_processes > 1 and accelerator.sync_gradients:
                    for p in list(rqt_sem.parameters()) + [hasvr_log_var]:
                        if p.grad is not None:
                            dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
                grad_norm = None
                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(
                        list(model.parameters())+list(hasvr_proj.parameters())+list(rqt_sem.parameters())+[hasvr_log_var],
                        config.training.max_grad_norm)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                btm.update(time.time()-end); end=time.time()
                if (global_step+1) % config.experiment.log_every == 0:
                    at=accelerator.gather(lt.repeat(config.training.batch_size_t2i)).mean()
                    am_=accelerator.gather(lm.repeat(config.training.batch_size_mmu)).mean()
                    ah=accelerator.gather(lh.repeat(config.training.batch_size_mmu)).mean()
                    sps=config.training.gradient_accumulation_steps*total_batch_size_per_gpu/btm.val
                    logs={"step_loss_t2i":at.item(),"step_loss_mmu":am_.item(),"step_loss_hasvr":ah.item(),
                          "lr":lr_scheduler.get_last_lr()[0],"samples/sec/gpu":sps,"total_loss":loss.item()}
                    for gn in GROUP_NAMES:
                        g=gl.get(gn,torch.tensor(0.0)); logs[f"loss_hasvr_{gn}"]=g.item() if torch.is_tensor(g) else g
                    if grad_norm is not None:
                        logs["grad_norm"] = grad_norm.item()
                    for i in range(total_depths):
                        logs[f"hasvr_log_var/{i}"]=hasvr_log_var[i].item()
                    accelerator.log(logs, step=global_step+1)
                    gs=" ".join(f"{gn}={gl.get(gn,torch.tensor(0.0)).item():.4f}" for gn in GROUP_NAMES)
                    logger.info(f"Step:{global_step+1} T2I:{at.item():.4f} MMU:{am_.item():.4f} HASVR:{ah.item():.4f} [{gs}] {sps:.1f}/s/gpu LR:{lr_scheduler.get_last_lr()[0]:.2e}")
                    btm.reset(); dtm.reset()
                if (global_step+1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step+1, rqt_sem, hasvr_proj, hasvr_log_var)
                global_step += 1
            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step, rqt_sem, hasvr_proj, hasvr_log_var)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(config.experiment.output_dir, safe_serialization=False)
    accelerator.end_training()


if __name__ == "__main__":
    main()
