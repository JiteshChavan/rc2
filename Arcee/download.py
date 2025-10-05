# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Functions for downloading pre-trained DiT models
"""
import os

import torch
from torchvision.datasets.utils import download_url
from huggingface_hub import hf_hub_download


def find_model(model_name, ema=False):
    """
    Finds a pre-trained DiT model, downloading it if necessary. Alternatively, loads a model from a local path.
    """
    print(f"ema:{ema}")
    if not os.path.isfile(model_name): # Find/download our pre-trained DiT checkpoints
        return download_model(model_name)
    else:  # Load a custom DiT checkpoint:
        assert os.path.isfile(model_name), f"Could not find DiT checkpoint at {model_name}"
        checkpoint = torch.load(model_name, map_location=lambda storage, loc: storage, weights_only=False)
        if "model" in checkpoint:
            if ema == False:
                checkpoint = checkpoint["model"]
            elif ema == True:
                checkpoint = checkpoint["ema"]
        return checkpoint
    # TODO: change for sampling ema later (just messing around for overfitting)
        if "ema" in checkpoint:  # supports checkpoints from train.py
            checkpoint = checkpoint["ema"]
        return checkpoint


def download_model(model_name):
    """
    Downloads a pre-trained DiT model from the web.
    """
    local_path = hf_hub_download("haopt/dimsum-L2-imagenet256", "pytorch_model.bin", cache_dir="./")
    model = torch.load(local_path, map_location=lambda storage, loc: storage)
    return model