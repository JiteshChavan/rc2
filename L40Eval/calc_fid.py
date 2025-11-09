import cleanfid
from cleanfid import fid

import os
import shutil
import random
import gc
import math
import sys
from pathlib import Path

import argparse
import logging
from collections import OrderedDict
from copy import deepcopy
from time import time
from tqdm import tqdm

# DDP
import torch
import numpy as np
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import wandb

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

def count_images(directory):
    count = sum(1 for _ in directory.glob("*.png"))
    return count

def main(args):
    """
    Evaluates FID for all checkpoints for given experiment.
    Computes stats for reference split once and deposits the npz file in the specified datadir.
    """
    assert torch.cuda.is_available(), "Eval currently requires at least one GPU."
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    device_str = f"cuda:{device}"
    world_size = dist.get_world_size()
    seed = args.global_seed * world_size + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={world_size}.")

    np.random.seed(seed)
    random.seed(seed)

    eval_index = os.path.basename(os.path.normpath(args.exp))
    eval_calc_dir = f"{args.eval_results_dir}/{eval_index}_metric_calcs"

    if rank == 0:
        assert 1 <= len(args.eval_metrics) <= 2, f"Invalid eval_metrics specified, the script only supports FID, KID eval"

        os.makedirs(args.eval_results_dir, exist_ok=True)
        os.makedirs(eval_calc_dir, exist_ok=True)
    

        logger = create_logger(eval_calc_dir)
        logger.info(f"Eval calc directory created at {eval_calc_dir}")
        
        mode = "disabled"
        if args.use_wandb:
            mode = "online"
        wandb.init(project="Arcee", entity="red-blue-violet", config=vars(args), name=f"fid50k_{eval_index}", mode=mode)
    else:
        logger = create_logger(None)
    dist.barrier()

    # setup directory for eval
    generated_samples_path = Path(args.eval_results_dir) / eval_index / f"Samples_{args.eval_nsamples}_step{args.ckpt_step}"
    assert os.path.exists(generated_samples_path), f"target directory doesnt exist : {generated_samples_path} is invalid"
    gen_image_count = count_images(generated_samples_path)
    logger.info(f"found {gen_image_count} images in specified target {generated_samples_path}")
    assert gen_image_count == args.eval_nsamples, f"required samples = {args.eval_nsamples} does not equate to generated_samples {gen_image_count} found in folder {generated_samples_path}"
    dist.barrier()

    stats_name = args.dataset # celeba_256 eg.
    if rank == 0:
        # store real stats in the same directory as data, one single time
        assert os.path.exists(args.datadir)


        # compute real stats, if they dont exist in CleanFID's cache
        if not fid.test_stats_exists(stats_name, mode="clean", metric="FID"):
            logger.info (f"Calculating real_stats for {stats_name}...")
            fid.make_custom_stats(
                stats_name,
                args.eval_refdir,
                mode="clean",
                num_workers=args.num_workers,
                device=torch.device(device_str),
                verbose=True,
                batch_size=args.fid_batch_size,
            )
        else:
            logger.info (f"real_stats for {stats_name} already exist")


       
        assert fid.test_stats_exists(stats_name, mode="clean", metric="FID")
        assert fid.test_stats_exists(stats_name, mode="clean", metric="KID")
       
    
    dist.barrier()

    if rank == 0:
        for i, metric in enumerate(args.eval_metrics):
            metric_key = f"{metric}{args.eval_nsamples // 1000}K"
            logger.info (f"Calculating {metric_key} for {eval_index} checkpoint_{args.ckpt_step}.....")
            assert fid.test_stats_exists(stats_name, mode="clean", metric=metric), f"specified real stats : {stats_name} are invalid"

            if metric == "FID":
                score = fid.compute_fid(
                    generated_samples_path.as_posix(),
                    dataset_name=stats_name,            # the REAL stats name you built earlier
                    dataset_split="custom",
                    dataset_res=args.image_size,
                    mode="clean",
                    num_workers=args.num_workers,
                    device=torch.device(device_str),
                    verbose=True,
                    batch_size=args.fid_batch_size,
                )
            elif metric == "KID":
                score = fid.compute_kid(
                    generated_samples_path.as_posix(),
                    dataset_name=stats_name,
                    dataset_split="custom",
                    dataset_res=args.image_size,
                    mode="clean",
                    num_workers=args.num_workers,
                    device=torch.device(device_str),
                    verbose=True,
                    batch_size=args.kid_batch_size,
                    
                )
            wandb.log({f"{metric_key}" : score}, step=args.ckpt_step, commit= (i == len(args.eval_metrics) - 1))
            logger.info(f"{metric_key} : {score}")
    dist.barrier()


    cleanup()
    



def none_or_str(value):
    if value == "None":
        return None
    return value


if __name__ == "__main__":
    # Default args here will train the model with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-step", required=True, type=int, help="checkpoint step specs eg:50000")
    parser.add_argument("--dataset", required=True, help="name of dataset")
    parser.add_argument("--image-size", type=int, required=True, help="256")
    parser.add_argument("--datadir", required=True, help="Path to dataset")
    parser.add_argument("--exp", type=str, required=True, help="runs fid eval on all checkpoints from this experiment")
    parser.add_argument("--eval-results-dir", type=str, default="eval_results")   
        
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)

    # WANDB
    parser.add_argument("--use-wandb", action="store_true")
        
    # EVAL
    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-refdir", type=str, required=True)
    group.add_argument("--eval-metrics", type=str, nargs="+", required=True, choices=["FID", "KID"], help="space separated metrics --eval-metrics FID KID")
    group.add_argument("--eval-nsamples", type=int, default=10000)
    group.add_argument("--eval-bs", type=int, default=4) # NOTE: eval batch size for fid calc (per GPU)
    group.add_argument("--fid-batch-size", type=int, default=32) # NOTE: batch size for fid calc through the feature extractor inceptionV3 model
    group.add_argument("--kid-batch-size", type=int, default=32)

    args = parser.parse_args()
    main(args)



