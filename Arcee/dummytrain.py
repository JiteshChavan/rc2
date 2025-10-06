use_ema_for_eval = False
use_ema_for_samples = False
# utils
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
import os
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm


import torch
import torch.nn as nn
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# DDP
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# VAE
from diffusers.models import AutoencoderKL

from create_model import create_model
from datasets_prep import get_dataset
from Arcee_Models import interpolate_pos_embed

# dataset scaffolding
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import save_image

# Flow matching
from transport2 import Sampler, create_transport # create_flow is defined in __init__.py for flowMatching

# for easy imports from eval toolbox
eval_import_path = (Path(__file__).parent.parent / "eval_toolbox").resolve().as_posix()
sys.path.append(eval_import_path)
# import dnnlib from evaltoolbox for convenience classes like EasyDict
import dnnlib
# for FID evaluations
from pytorch_fid import metric_main, metric_utils

import wandb
os.environ['WANDB_API_KEY'] = '2f92f218fe46708930c460c6f57055ac6ce1361c'

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format="[\033[34m%(asctime)s\033[0m] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")],
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(tuple(x // 2 for x in pil_image.size), resample=Image.BOX)

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC)

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size])


def adjust_learning_rate(optimizer, epoch, args):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < args.warmup_epochs:
        lr = args.lr * epoch / args.warmup_epochs
    else:
        lr = args.min_lr + (args.lr - args.min_lr) * 0.5 * (
            1.0 + math.cos(math.pi * (epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs))
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


#################################################################################
#                                  Training Loop                                #
#################################################################################


def main(args):
    """
    Trains a new DiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    # Setup DDP:
    dist.init_process_group("nccl")
    # if "SLURM_PROCID" in os.environ:
    #     rank = int(os.environ["SLURM_PROCID"])
    #     gpu = rank % torch.cuda.device_count()
    #     world_size = int(os.environ["WORLD_SIZE"], 1)
    # else:
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    assert args.global_batch_size % world_size == 0, "Batch size must be divisible by world size."
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # Setup an experiment folder:
    experiment_index = args.exp
    experiment_dir = f"{args.results_dir}/{experiment_index}"  # Create an experiment folder
    checkpoint_dir = f"{experiment_dir}/checkpoints"  # Stores saved model checkpoints
    sample_dir = f"{experiment_dir}/samples"
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)  # Make results folder (holds all experiment subfolders)
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

        wandb.init(
            project="Arcee",
            config=vars(args),
            name=experiment_index,
        )
    else:
        logger = create_logger(None)

    # Create model:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    model = create_model(args)  # mamba_models[args.model]()
    
    # Note that parameter initialization is done within the DiT constructor
    ema = deepcopy(model).to(device)  # Create an EMA of the model for use after training
    model = DDP(model.to(device), device_ids=[rank], find_unused_parameters=False)
    if rank == 0:
        wandb.watch(model.module if hasattr(model, "module") else model, log="all")

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps,
        path_args={
            "diffusion_form": args.diffusion_form,
            "use_blurring": args.use_blurring,
            "blur_sigma_max": args.blur_sigma_max,
            "blur_upscale": args.blur_upscale,
        },
        t_sample_mode=args.t_sample_mode,
    )  # default: velocity;
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Model :{args.model} Parameter count : {sum(p.numel() for p in model.parameters()):,}")

    # Setup optimizer (we used default Adam betas=(0.9, 0.999) and a constant learning rate of 1e-4 in our paper):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs, eta_min=1e-6, verbose=True)

    update_ema(ema, model.module, decay=0)  # Ensure EMA is initialized with synced weights

    if args.model_ckpt and os.path.exists(args.model_ckpt):
        # NOTE: fine tune branch
        checkpoint = torch.load(args.model_ckpt, map_location=torch.device(f"cuda:{device}"))
        
        init_epoch = 0
        init_step = 0

        state_dict = model.module.state_dict()
        for i, k in enumerate(["x_embedder.proj.weight", "final_layer.linear.weight", "final_layer.linear.bias"]):
            # NOTE: fixed for DiM-L/4 to DiM-L/2
            if k in checkpoint["model"] and checkpoint["model"][k].shape != state_dict[k].shape:
                if i == 0:
                    K1, K2 = state_dict[k].shape[2:]
                    checkpoint["model"][k] = checkpoint["model"][k][:, :, :K1, :K2]  # state_dict[k]
                    checkpoint["ema"][k] = checkpoint["ema"][k][:, :, :K1, :K2]  # state_dict[k]
                else:
                    Cin = state_dict[k].size(0)
                    checkpoint["model"][k] = checkpoint["model"][k][:Cin]
                    checkpoint["ema"][k] = checkpoint["ema"][k][:Cin]

        # interpolate position embedding
        interpolate_pos_embed(model.module, checkpoint["model"])
        interpolate_pos_embed(ema, checkpoint["ema"])

        msg = model.module.load_state_dict(checkpoint["model"], strict=True)
        print(msg)

        ema.load_state_dict(checkpoint["ema"])
        # optimizer not required for refining pretrained weights
        # opt.load_state_dict(checkpoint["opt"])
        for g in opt.param_groups:
            g["lr"] = args.lr

        logger.info("=> loaded checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    elif args.resume and os.path.exists(os.path.join(checkpoint_dir, "content.pth")):
        checkpoint_file = os.path.join(checkpoint_dir, "content.pth")
        checkpoint = torch.load(checkpoint_file, map_location=torch.device(f"cuda:{device}"))
        init_epoch = checkpoint["epoch"]
        init_step = checkpoint["step"] + 1
        model.module.load_state_dict(checkpoint["model"])
        opt.load_state_dict(checkpoint["opt"])
        ema.load_state_dict(checkpoint["ema"])

        for g in opt.param_groups:
            g["lr"] = args.lr

        logger.info("=> resume checkpoint (epoch {})".format(checkpoint["epoch"]))
        del checkpoint
    else:
        init_epoch = 0
        init_step = 0
    requires_grad(ema, False)

    dataset = get_dataset(args)
    
    from torch.utils.data import Subset
    dataset = Subset(dataset, indices=range(1000))
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, seed=args.global_seed)
    loader = DataLoader(
        dataset,
        batch_size=int(args.global_batch_size // world_size),
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    logger.info(f"Dataset contains {len(dataset):,} images ({args.datadir})")

    # Prepare models for training:
    model.train()  # important! This enables embedding dropout for classifier-free guidance
    ema.eval()  # EMA model should always be in eval mode

    # Variables for monitoring/logging purposes:
    log_steps = 0
    running_loss = 0
    running_grad_norm = 0.0
    start_time = time()

    assert args.num_classes >= 1
    use_label = args.num_classes > 1
    use_cfg = args.cfg_scale > 1.0 and use_label
    # sample noise and label
    sample_bs = args.sample_bs
    zs = torch.randn(sample_bs, 4, latent_size, latent_size, device=device)
    if use_label:
        ys = torch.randint(args.num_classes - 1, size =(sample_bs, ), device = device, dtype=torch.long) # sample label [0, num_classes-1) index num_classes-1 is null label
        if use_cfg:
            zs = torch.cat([zs, zs], dim=0)
            y_null = torch.tensor([args.num_classes-1] * sample_bs, device=device, dtype=torch.long)    # as seen here
            ys = torch.cat([ys, y_null], dim=0)
            sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
            if use_ema_for_samples:
                model_fn = ema.forward_with_cfg
            else:
                model_fn = model.module.forward_with_cfg
        else:
            sample_model_kwargs = dict(y=ys)
            if use_ema_for_samples:
                model_fn = ema.forward
            else:
                model_fn = model.module.forward
    else:
        ys = None
        sample_model_kwargs = dict(y=ys)
        if use_ema_for_samples:
            model_fn = ema.forward
        else:
            model_fn = model.module.forward

    use_latent = True if "latent" in args.dataset else False
    logger.info(f"Training for {args.train_steps} steps...")


    total_steps = args.train_steps
    current_step = init_step
    epoch = init_epoch
    effective_batch_size = args.global_batch_size
    steps_per_epoch = math.ceil(len(dataset) / effective_batch_size)
    sampler.set_epoch(epoch)
    data_iter = iter(loader)
    def next_batch():
        """Get next batch; if dataloader is exhausted, reshuffle and bump epoch."""
        nonlocal data_iter, epoch
        try:
            return next(data_iter)
        except StopIteration:
            epoch += 1
            sampler.set_epoch(epoch)  # shuffle for the next pass
            if pbar is not None and dist.get_rank() == 0:
                pbar.set_description(f"epoch: {epoch}")
            logger.info(f"Beginning epoch {epoch}...")
            data_iter = iter(loader)
            return next(data_iter)
    # progress bar on rank 0
    pbar = None
    if dist.get_rank() == 0:
        pbar = tqdm(total=total_steps - init_step, ncols=100, disable=True)
        pbar.set_description(f"epoch: {epoch}")

    for current_step in range(init_step, total_steps):
        x, y = next_batch()
        # adjust_learning_rate(opt, i / len(loader) + epoch, args)
        x = x.to(device)
        y = None if not use_label else y.to(device)
        if not use_latent:
            with torch.no_grad():
                # Map input images to latent space + normalize latents:
                x = vae.encode(x).latent_dist.sample().mul_(0.18215)
        model_kwargs = dict(y=y)
        before_forward = torch.cuda.memory_allocated(device)
        loss_dict = transport.training_losses(model, x, model_kwargs)
        loss = loss_dict["loss"].mean()
        after_forward = torch.cuda.memory_allocated(device)
        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        opt.step()
        after_backward = torch.cuda.memory_allocated(device)
        update_ema(ema, model.module)
        

        # Log loss values:
        running_loss += loss.item()
        running_grad_norm += grad_norm.item()
        log_steps += 1
        
        # control switches for logging
        will_log_samples = (rank == 0) and (current_step % args.plot_every == 0)
        will_log_fid = (current_step > 0) and (current_step % args.eval_every == 0) and (args.eval_refdir is not None) and Path(args.eval_refdir).exists()

        if current_step % args.log_every == 0:
            # Measure training speed:
            torch.cuda.synchronize()
            end_time = time()
            steps_per_sec = log_steps / (end_time - start_time)
            # Reduce loss history over all processes:
            avg_loss = torch.tensor(running_loss / log_steps, device=device)
            dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
            avg_loss = avg_loss.item() / world_size

            # Reduce grad norm history over all processes:
            avg_grad_norm = torch.tensor(running_grad_norm / log_steps, device=device)
            dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
            avg_grad_norm = avg_grad_norm.item() / world_size

            # TODO: wandb
            if rank == 0:
                wandb.log(
                    {
                        "train_loss": avg_loss,
                        "train_steps_per_sec": steps_per_sec,
                        "gpu_mem/before_forward_gb": before_forward/1e9,
                        "gpu_mem/after_forward_gb":  after_forward/1e9,
                        "gpu_mem/after_backward_gb": after_backward/1e9,
                        "grad/avg_norm": avg_grad_norm,
                    },
                    step=current_step,   # drives x-axis
                    commit = not (will_log_samples or will_log_fid)        
                )
                logger.info(
                    f"(step={current_step:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Train Steps/Sec: {steps_per_sec:.2f}, "
                    f"GPU Mem before forward: {before_forward/10**9:.2f}Gb, "
                    f"GPU Mem after forward: {after_forward/10**9:.2f}Gb, "
                    f"GPU Mem after backward: {after_backward/10**9:.2f}Gb "
                    f"Avg Grad Norm: {avg_grad_norm:.4f}"
                )
            
           
            # Reset monitoring variables:
            running_loss = 0
            running_grad_norm = 0
            log_steps = 0
            start_time = time()

        # if not args.no_lr_decay:
        #     scheduler.step()

        if rank == 0:
            # latest checkpoint
            if current_step % args.save_content_every == 0 and current_step!=0:
                logger.info("Saving content.")
                content = {
                    "epoch": epoch,
                    "step": current_step,
                    "args": args,
                    "model": model.module.state_dict(),
                    "opt": opt.state_dict(),
                    "ema": ema.state_dict(),
                }
                torch.save(content, os.path.join(checkpoint_dir, "content.pth"))

            # Save Arcee checkpoint:
            if current_step % args.ckpt_every == 0 and current_step > 0:
                checkpoint = {
                    "epoch": epoch,
                    "step" : current_step,
                    "model": model.module.state_dict(),
                    "ema": ema.state_dict(),
                    "opt": opt.state_dict(),
                    "args": args,
                }
                checkpoint_path = f"{checkpoint_dir}/epoch{epoch:07d}_step{current_step}.pt"
                torch.save(checkpoint, checkpoint_path)
                logger.info(f"Saved checkpoint to {checkpoint_path}")
            # dist.barrier()

        if rank == 0 and current_step % args.plot_every == 0:
            logger.info(f"Generating base model samples...")
            model.eval()
            with torch.no_grad():
                #zs = torch.randn(sample_bs, 4, latent_size, latent_size, device=device)
                sample_fn = transport_sampler.sample_ode()  # default to ode sampling
                samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
                #samples_ema = sample_fn(zs, ema_fn, **sample_model_kwargs)[-1]
                
                if use_cfg:  # remove null samples
                    samples, _ = samples.chunk(2, dim=0)
                    #samples_ema, _ = samples_ema.chunk(2, dim=0)
                #samples = torch.cat((samples,samples_ema), dim=0)
                samples = vae.decode(samples / 0.18215).sample

            # Save and display images:
            save_image(samples, f"{sample_dir}/image_{current_step:07d}.jpg", nrow=4, normalize=True, value_range=(-1, 1))
            wandb.log({"samples": wandb.Image(f"{sample_dir}/image_{current_step:07d}.jpg")}, step=current_step, commit= not will_log_fid)
            del samples
            model.train()
        # dist.barrier()

        if current_step % args.eval_every == 0 and current_step > 0:
            ref_dir = Path(args.eval_refdir)
            if ref_dir.exists():
                n = args.eval_bs
                global_batch_size = n * world_size
                total_samples = int(math.ceil(args.eval_nsamples / global_batch_size) * global_batch_size)
                samples_needed_this_gpu = int(total_samples // world_size)
                iterations = int(samples_needed_this_gpu // n)
                eval_pbar = tqdm(range(iterations), disable=(rank != 0))
                total = 0
                p = Path(experiment_dir) / f"fid{args.eval_nsamples}_epoch{epoch}_step{current_step}"
                # if p.exists() and rank == 0:
                #     shutil.rmtree(p.as_posix())
                p.mkdir(exist_ok=True, parents=True)
                model.eval()

                use_cfg_eval = use_label and (args.cfg_scale > 1.0)
                for _ in eval_pbar:
                    # Sample inputs:
                    z = torch.randn(n, 4, latent_size, latent_size, device=device)
                    if use_label:
                        y = torch.randint(args.num_classes-1, size=(n,), dtype=torch.long, device=device) # (, ] sampling
                        if use_cfg_eval:
                            z = torch.cat([z, z], dim=0)
                            y_null = torch.tensor([args.num_classes-1] * n, dtype=torch.long, device=device)
                            y = torch.cat([y, y_null], dim=0)
                            sample_model_kwargs = dict(y=y, cfg_scale=args.cfg_scale)
                            if use_ema_for_eval:
                                model_eval_fn = ema.forward_with_cfg
                            else:
                                model_eval_fn = model.module.forward_with_cfg
                        else:
                            sample_model_kwargs = dict(y=y) # no cfg scale
                            if use_ema_for_eval:
                                model_eval_fn = ema.forward
                            else:
                                model_eval_fn = model.module.forward
                    else:
                        y = None
                        sample_model_kwargs = dict(y=y)
                        if use_ema_for_eval:
                            model_eval_fn = ema.forward
                        else:
                            model_eval_fn = model.module.forward
                    

                    # Sample images:
                    with torch.no_grad():
                        sample_fn = transport_sampler.sample_ode()  # default to ode sampling
                        samples = sample_fn(z, model_eval_fn, **sample_model_kwargs)[-1]

                    if use_cfg_eval:
                        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples

                    samples = vae.decode(samples / 0.18215).sample
                    samples = (
                        torch.clamp(127.5 * samples + 128.0, 0, 255)
                        .permute(0, 2, 3, 1)
                        .to("cpu", dtype=torch.uint8)
                        .numpy()
                    )

                    # Save samples to disk as individual .png files
                    for i, sample in enumerate(samples):
                        index = i * world_size + rank + total
                        if index >= args.eval_nsamples:
                            break
                        pp = p / f"{index:06d}.jpg"
                        Image.fromarray(sample).save(pp.as_posix())
                    total += global_batch_size
                del z, y, sample_fn, samples  
                torch.cuda.empty_cache()
                gc.collect()
                model.train()
                eval_args = dnnlib.EasyDict()
                eval_args.dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=ref_dir.as_posix(),
                    xflip=True,
                )
                eval_args.gen_dataset_kwargs = dnnlib.EasyDict(
                    class_name="training.dataset.ImageFolderDataset",
                    path=p.resolve().as_posix(),
                    xflip=True,
                )
                progress = metric_utils.ProgressMonitor(verbose=True)
                if rank == 0:
                    print("Calculating FID...")
                    assert metric_main.is_valid_metric(args.eval_metric), f"Unknown metric {args.eval_metric}. Valid: {metric_main.list_valid_metrics()}"
                result_dict = metric_main.calc_metric(
                    metric=args.eval_metric,
                    dataset_kwargs=eval_args.dataset_kwargs,
                    num_gpus=world_size,
                    rank=rank,
                    device=device,
                    progress=progress,
                    gen_dataset_kwargs=eval_args.gen_dataset_kwargs,
                    cache=True,
                )
                if rank == 0:
                    
                    eval_metric_string = args.eval_metric
                    assert eval_metric_string in result_dict["results"], f"Expected key {eval_metric_string}, got {list(result_dict['results'].keys())}"
                    wandb.log({eval_metric_string: result_dict["results"][eval_metric_string]}, step=current_step, commit=True)

                    metric_main.report_metric(result_dict, run_dir=experiment_dir, snapshot_pkl=p.as_posix())
                del result_dict
                gc.collect()
                torch.cuda.empty_cache()
            else:
                print(f"Reference directory {ref_dir} does not exist, skip eval")
            dist.barrier()
        if pbar is not None:
            pbar.update(1)
        

    model.eval()  # important! This disables randomized embedding dropout
    # do any sampling/FID calculation/etc. with ema (or model) in eval mode ...
    if rank == 0:
        logger.info(f"Generating base model samples...")
        with torch.no_grad():
            #zs = torch.randn(sample_bs, 4, latent_size, latent_size, device=device)
            sample_fn = transport_sampler.sample_ode()  # default to ode sampling
            samples = sample_fn(zs, model_fn, **sample_model_kwargs)[-1]
            #samples_ema = sample_fn(zs, ema_fn, **sample_model_kwargs)[-1]
            
            if use_cfg:  # remove null samples
                samples, _ = samples.chunk(2, dim=0)
                #samples_ema, _ = samples_ema.chunk(2, dim=0)
            #samples = torch.cat((samples,samples_ema), dim=0)
            samples = vae.decode(samples / 0.18215).sample

        # Save and display images:
        save_image(samples, f"{sample_dir}/image_FINAL_SAMPLE.jpg", nrow=4, normalize=True, value_range=(-1, 1))
        wandb.log({"samples": wandb.Image(f"{sample_dir}/image_FINAL_SAMPLE.jpg")}, step=total_steps+500, commit= True)
        del samples
        model.train()
    
    logger.info("Done!")
    if pbar is not None:
        pbar.close()
    cleanup()


def none_or_str(value):
    if value == "None":
        return None
    return value

if __name__ == "__main__":
    # Default args here will train the model with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")

    parser.add_argument("--model", type=str, default="Arcee-XS/2")
    parser.add_argument("--ssm-dstate", type=int, default=16, help="dstate for each d_inner")
    parser.add_argument(
        "--scan-type",
        type=str,
        default="none",
        choices=["none", "Arcee_1", "Arcee_8", "Zigma_1", "Zigma_8"],
    )
    parser.add_argument("--block-type", type=str, default="normal", choices=["normal", "combined"])


    # NOTE: Functionality
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument("--use-attn-every-k-layers",type=int,default=-1,)
    
    

    parser.add_argument("--image-size", type=int, choices=[256, 512, 1024], default=256)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--num-classes", type=int, default=1)
    parser.add_argument("--label-dropout", type=float, default=-1)
    
    
    
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    
    # Logging
    # log every step
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--plot-every", type=int, default=5) # epoch # NOTE: work it
    parser.add_argument("--save-content-every", type=int, default=400)
    parser.add_argument("--ckpt-every", type=int, default=25)
    
    # RESUME TRAINING
    parser.add_argument("--model-ckpt", type=str, default="")
    parser.add_argument("--resume", action="store_true")

        
    

    
    # NOTE: training dynamics
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-steps", type=int, default=50000)
    parser.add_argument("--no-lr-decay", action="store_true", default=False) 
    parser.add_argument("--min-lr",type=float,default=1e-6,)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs",type=int,default=5,)
    parser.add_argument("--max-grad-norm",type=float,default=2.0,)
    

    # EVAL
    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-metric", type=str, default="fid10k_full")
    group.add_argument("--sample-bs", type=int, default=4) # NOTE: sampling (plotting batch size only done on rank 0)
    group.add_argument("--eval-nsamples", type=int, default=10000)
    group.add_argument("--eval-bs", type=int, default=4) # NOTE: eval batch size for fid calc (per GPU)
    group.add_argument("--eval-every", type=int, default=9999)
    group.add_argument("--eval-refdir", type=str, default=None)
    parser.add_argument("--global-batch-size", type=int, default=2)
    

    # Flow Matching 
    group = parser.add_argument_group("Transport arguments")
    group.add_argument("--path-type", type=str, default="Linear", choices=["Linear", "GVP", "VP"])
    group.add_argument("--prediction", type=str, default="velocity", choices=["velocity", "score", "noise"])
    group.add_argument("--loss-weight", type=none_or_str, default=None, choices=[None, "velocity", "likelihood"])
    group.add_argument("--sample-eps", type=float)
    group.add_argument("--train-eps", type=float)
    group.add_argument(
        "--diffusion-form",
        type=str,
        default="none",
        choices=["none", "constant", "SBDM", "sigma", "linear", "decreasing", "increasing-decreasing", "log"],
        help="form of diffusion coefficient in the SDE",
    )
    group.add_argument("--t-sample-mode", type=str, default="uniform")
    group.add_argument("--use-blurring", action="store_true")
    group.add_argument("--blur-sigma-max", type=int, default=3)
    group.add_argument("--blur-upscale", type=int, default=4)



    # NOTE: MOE
    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--gated-linear-unit", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2"], default="top1")
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)

    args = parser.parse_args()
    main(args)



    


