# coding=utf-8
"""
Show-o pure SFT training script (no HASVR).

Two losses only:
  loss_t2i — masked token prediction (image generation)
  loss_mmu — next-token prediction (multimodal understanding)
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import json, logging, math, shutil, time
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import wandb
import torch
from torch.optim import AdamW
from lightning.pytorch.utilities import CombinedLoader

from transformers import AutoTokenizer
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, set_seed

from training.data import Text2ImageDataset
from models import Showo, MAGVITv2, get_mask_chedule
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


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    raise ValueError(f"model_type {model_type} not supported.")


def save_checkpoint(model, config, accelerator, global_step):
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
        json.dump({"global_step": global_step}, (save_path / "metadata.json").open("w+"))
        logger.info(f"Saved state to {save_path}")


@torch.no_grad()
def generate_images(model, vq_model, uni_prompting, accelerator, config, global_step, mask_schedule):
    logger.info("Generating images...")
    model.eval()

    with open(config.dataset.params.validation_prompts_file, "r") as f:
        validation_prompts = f.read().splitlines()

    if hasattr(model, 'module'):
        mask_dtype = model.module.showo.model.embed_tokens.weight.dtype
    else:
        mask_dtype = model.showo.model.embed_tokens.weight.dtype

    mask_token_id = config.model.showo.vocab_size - 1
    image_tokens = torch.ones((len(validation_prompts), config.model.showo.num_vq_tokens),
                              dtype=torch.long, device=accelerator.device) * mask_token_id
    input_ids, _ = uni_prompting((validation_prompts, image_tokens), 't2i_gen')

    if config.training.guidance_scale > 0:
        uncond_input_ids, _ = uni_prompting(([''] * len(validation_prompts), image_tokens), 't2i_gen')
        attention_mask = create_attention_mask_predict_next(
            torch.cat([input_ids, uncond_input_ids], dim=0),
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True).to(mask_dtype)
    else:
        attention_mask = create_attention_mask_predict_next(
            input_ids,
            pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
            soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
            eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
            rm_pad_in_image=True).to(mask_dtype)
        uncond_input_ids = None

    weight_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(accelerator.mixed_precision, torch.float32)
    with torch.autocast("cuda", dtype=weight_dtype, enabled=accelerator.mixed_precision != "no"):
        gen_token_ids = accelerator.unwrap_model(model).t2i_generate(
            input_ids=input_ids, uncond_input_ids=uncond_input_ids,
            attention_mask=attention_mask, guidance_scale=config.training.guidance_scale,
            temperature=config.training.get("generation_temperature", 1.0),
            timesteps=config.training.generation_timesteps, noise_schedule=mask_schedule,
            config=config)

    gen_token_ids = torch.clamp(gen_token_ids, max=accelerator.unwrap_model(model).config.codebook_size - 1, min=0)
    images = vq_model.decode_code(gen_token_ids)
    model.train()

    images = torch.clamp((images + 1.0) / 2.0, min=0.0, max=1.0) * 255.0
    images = images.permute(0, 2, 3, 1).cpu().numpy().astype(np.uint8)
    pil_images = [Image.fromarray(img) for img in images]
    wandb_images = [wandb.Image(img, caption=validation_prompts[i]) for i, img in enumerate(pil_images)]
    wandb.log({"Generated images": wandb_images}, step=global_step)


def main():
    config = get_config()
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

    # --- Optimizer ---
    oc = config.optimizer.params
    no_decay = ["bias", "layer_norm.weight", "mlm_ln.weight", "embeddings.weight"]
    optimizer = AdamW([
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and not any(nd in n for nd in no_decay)],
         "weight_decay": oc.weight_decay},
        {"params": [p for n, p in model.named_parameters() if p.requires_grad and any(nd in n for nd in no_decay)],
         "weight_decay": 0.0},
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
            accelerator.print(f"Resumed from step {global_step}")

    # --- Prepare ---
    model, optimizer, lr_scheduler = accelerator.prepare(model, optimizer, lr_scheduler)
    vq_model.to(accelerator.device)
    mask_dtype = (model.module if hasattr(model, 'module') else model).showo.model.embed_tokens.weight.dtype

    num_vq_tokens = config.model.showo.num_vq_tokens

    @torch.no_grad()
    def prep_t2i(pv, texts):
        it = vq_model.get_code(pv) + len(uni_prompting.text_tokenizer)
        ids, lbl, lw, mp = mask_or_random_replace_tokens(it, mask_id, config, mask_schedule=mask_schedule, is_train=True)
        ids, _, lbl = uni_prompting((texts, ids, lbl), 't2i')
        return ids, lbl, mp

    btm, dtm = AverageMeter(), AverageMeter()
    end = time.time()

    for epoch in range(first_epoch, n_epochs):
        model.train()

        for batch, batch_idx, _ in combined_dl:
            bst = batch["t2i_flow"]["images"].shape[0]
            bsm = batch["mmu_flow"]["images"].shape[0]
            dtm.update(time.time() - end)

            # --- T2I ---
            pv = batch["t2i_flow"]["images"].to(accelerator.device, non_blocking=True)
            ids, lbl, mp = prep_t2i(pv, batch["t2i_flow"]["input_ids"])
            am = create_attention_mask_predict_next(ids,
                pad_id=int(uni_prompting.sptids_dict['<|pad|>']),
                soi_id=int(uni_prompting.sptids_dict['<|soi|>']),
                eoi_id=int(uni_prompting.sptids_dict['<|eoi|>']),
                rm_pad_in_image=True, return_inverse_mask=True).to(mask_dtype)

            # --- MMU ---
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

            a_ids = torch.cat([ids, idm], 0)
            a_lbl = torch.cat([lbl, lbm], 0)
            a_am = torch.cat([am, amm], 0)

            with accelerator.accumulate(model):
                _, lt, _, lm = model(
                    input_ids=a_ids, input_embeddings=None, attention_mask=a_am, labels=a_lbl,
                    batch_size_t2i=bst, batch_size_lm=0, batch_size_mmu=bsm,
                    max_seq_length=pc.max_seq_length)

                loss = config.training.t2i_coeff * lt + config.training.mmu_coeff * lm
                accelerator.backward(loss)
                grad_norm = None
                if config.training.max_grad_norm is not None and accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.training.max_grad_norm)
                optimizer.step(); lr_scheduler.step(); optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                btm.update(time.time()-end); end=time.time()
                if (global_step+1) % config.experiment.log_every == 0:
                    at = accelerator.gather(lt.repeat(config.training.batch_size_t2i)).mean()
                    am_ = accelerator.gather(lm.repeat(config.training.batch_size_mmu)).mean()
                    sps = config.training.gradient_accumulation_steps * total_batch_size_per_gpu / btm.val
                    logs = {"step_loss_t2i": at.item(), "step_loss_mmu": am_.item(),
                            "lr": lr_scheduler.get_last_lr()[0], "samples/sec/gpu": sps,
                            "total_loss": loss.item()}
                    if grad_norm is not None:
                        logs["grad_norm"] = grad_norm.item()
                    accelerator.log(logs, step=global_step+1)
                    logger.info(
                        f"Step:{global_step+1} Loss_t2i:{at.item():.4f} Loss_mmu:{am_.item():.4f} "
                        f"{sps:.1f}/s/gpu LR:{lr_scheduler.get_last_lr()[0]:.2e}")
                    btm.reset(); dtm.reset()

                if (global_step+1) % config.experiment.save_every == 0:
                    save_checkpoint(model, config, accelerator, global_step+1)

                if (global_step+1) % config.experiment.generate_every == 0 and accelerator.is_main_process:
                    generate_images(model, vq_model, uni_prompting, accelerator, config, global_step+1, mask_schedule)

                global_step += 1
            if global_step >= config.training.max_train_steps:
                break

    accelerator.wait_for_everyone()
    save_checkpoint(model, config, accelerator, global_step)
    if accelerator.is_main_process:
        accelerator.unwrap_model(model).save_pretrained(config.experiment.output_dir, safe_serialization=False)
    accelerator.end_training()


if __name__ == "__main__":
    main()
