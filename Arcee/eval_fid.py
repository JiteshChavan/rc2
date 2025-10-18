import shutil
import random
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

import wandb

import re
_STEP_RE = re.compile(r'step(\d+)\.pt$')


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

def cleanup():
    """
    End DDP.
    """
    dist.destroy_process_group()

def list_checkpoints(directory):
    with os.scandir(directory) as dir:
        ckpts = [ckpt.path for ckpt in dir if ckpt.is_file() and ckpt.name.endswith(".pt")]
    return sorted(ckpts, key=get_step_from_ckpt_path)

def get_step_from_ckpt_path(ckpt_path):
    name = os.path.basename(ckpt_path)
    m = _STEP_RE.search(name)
    if not m:
        raise ValueError(f"Could not parse step from: {name}")
    return int(m.group(1))


    
def main(args):
    assert torch.cuda.is_available(), "Eval currently requires at least one GPU."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    world_size = dist.get_world_size()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    # setup directory for eval
    exp_path = f"results/{args.exp}" # target for fid eval
    assert os.path.exists(exp_path), f"target experiment directory doesnt exist : {exp_path} is invalid"
    checkpoint_directory = f"{exp_path}/checkpoints"
    checkpoints = list_checkpoints(checkpoint_directory)
    dist.barrier()


    eval_index = args.exp # directory to save eval_results
    eval_dir = f"{args.results_dir}/{eval_index}"
    sample_dir = f"{args.results_dir}/samples"
    
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        os.makedirs(sample_dir, exist_ok=True)

        logger = create_logger(eval_dir)
        logger.info(f"Eval directory created at {eval_dir}")
        logger.info(f"Found {len(checkpoints)} checkpoints.")
        for ckpt in checkpoints:
            logger.info(ckpt)

        mode = "disabled"
        if args.use_wandb:
            mode = "online"
        wandb.init(project="Arcee", entity="red-blue-violet", config=vars(args), name=eval_index, mode=mode)
    else:
        logger = create_logger(None)
    dist.barrier()
    

    # create model
    assert args.image_size % 8 == 0, f"Image size must be divisible by 8 (for the VAE encoder)"
    latent_size = args.image_size // 8
    model = create_model(args) # Models[args.model](args)

    # will this work on multi gpu single node ?
    model = model.to(device)
    model.eval()

    flow = create_transport(
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
    )

    flow_sampler = Sampler(flow)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)

    logger.info(f"Model : {args.model}, parameter count : {sum(p.numel() for p in model.parameters()):,}")

    ref_dir = args.eval_refdir
    eval_bs = args.eval_bs
    eval_nsamples = args.eval_nsamples
    latent_res = args.image_size // 8
    use_label = args.num_classes > 1
    use_cfg = use_label and (args.cfg_scale > 1.0)
    num_classes = args.num_classes
    cfg_scale = args.cfg_scale
    for ckpt_file in checkpoints:

        dist.barrier()
        logger.info(f"{ckpt_file}: Starting {args.model_type} eval ...")
        
        step = get_step_from_ckpt_path(ckpt_file)
        ckpt = torch.load(ckpt_file, map_location=torch.device(f"cuda:{device}"), weights_only=False)
        model.module.load_state_dict(ckpt[args.model_type], strict=True)
        del ckpt
        torch.cuda.synchronize()
        logger.info(f"Checkpoint {step} loaded successfully")




        # maybe try to eval real stats just once that would save time?
        assert os.path.exists(ref_dir)
        global_batch_size = eval_bs * world_size
        total_samples = int (math.ceil(eval_nsamples / global_batch_size) * global_batch_size)
        samples_needed_this_gpu = int(total_samples // world_size)
        iterations = int (samples_needed_this_gpu // eval_bs)
        eval_pbar = tqdm(range(iterations, disable=(rank != 0)))
        total = 0
        p = Path(eval_dir) / f"fid{eval_nsamples}_step{step}"
    
        dist.barrier()
        if rank == 0:
            if p.exists():
                shutil.rmtree(p.as_posix())
            p.mkdir(exist_ok=True, parents=True)
        dist.barrier()
    
        for _ in eval_pbar:
            # x0
            z = torch.randn(eval_bs, 4, latent_res, latent_res, device=device)
            if use_label:
                y = torch.randint(num_classes-1, size=(eval_bs), dtype=torch.long, device=device) # sample [,) num classes = 2 sample 0 and 1 0 is null class
                if use_cfg:
                    z = torch.cat([z, z], dim=0)
                    y_null = torch.tensor([num_classes - 1] * eval_bs, dtype=torch.long, device=device)
                    y = torch.cat([y, y_null], dim=0)
                    sample_model_kwargs = dict(y=y, cfg_scale=cfg_scale)
                    model_fn = model.forward_with_cfg
                else:
                    sample_model_kwargs = dict(y=y) #no cfg scale
                    model_fn = model.forward
            else:
                y = None
                sample_model_kwargs = dict(y=y)
                model_fn = model.forward
            
            # inference
            with torch.no_grad():
                sample_fn = flow_sampler.sample_ode() # vectorfield ODE sampling (continuity equation)
                samples = sample_fn(z, model_fn, **sample_model_kwargs)[-1]
            
            if use_cfg:
                samples, null_samples = samples.chunk(2, dim=0) # discarrd null samples
            
            del null_samples, z, y
            samples = vae.decode(samples / 0.18215).sample
            samples = (
                torch.clamp(127.5 * samples + 128.0, 0, 255)
                .permute(0, 2, 3, 1)
                .to("cpu", dtype=torch.uint8)
                .numpy()
            )

            for i, sample in enumerate(samples):
                index = i * world_size + rank + total
                if index >= eval_nsamples:
                    break
                image_path = p / f"{index:06d}.png"
                Image.fromarray(sample).save(image_path.as_posix())
            total += global_batch_size
            del samples
        torch.cuda.empty_cache()
        gc.collect()

     

    cleanup()
    
        



def none_or_str(value):
    if value == "None":
        return None
    return value

if __name__ == "__main__":
    # Default args here will train the model with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, help="name of dataset")
    parser.add_argument("--exp", type=str, required=True, help="runs fid eval on all checkpoints from this experiment")
    parser.add_argument("--results-dir", type=str, default="eval_results")

    parser.add_argument("--model-type", type=str, required=True, choices=["base", "ema"])

    parser.add_argument("--model", type=str, default="Arcee-XS/2")
    parser.add_argument("--ssm-dstate", type=int, default=16, help="dstate for each d_inner")
    parser.add_argument(
        "--scan-type",
        type=str,
        default="none",
        choices=["none", "Arcee_1", "Arcee_2", "Arcee_4", "Arcee_8", "Zigma_1", "Zigma_2", "Zigma_4", "Zigma_8"],
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

    
    # RESUME TRAINING
    parser.add_argument("--use-wandb", action="store_true")
        

    # EVAL
    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-metric", type=str, required=True)
    group.add_argument("--eval-nsamples", type=int, default=10000)
    group.add_argument("--eval-bs", type=int, default=4) # NOTE: eval batch size for fid calc (per GPU)
    group.add_argument("--eval-every", type=int, default=9999)
    group.add_argument("--eval-refdir", type=str, default=None)
    

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


    # MoE
    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--gated-linear-unit", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2"], default="top1")
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)



    args = parser.parse_args()
    main(args)




    


    


