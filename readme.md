### setup ffhq1024 datase
- cd data_prep/ffhq1024
- bash download.sh
- python convert.py
- there should be a folder "real_samples" for fid eval reference and train.lmdb in the directory after the scripts are successful/



### Create env.
- install cuda toolkit 12.8 ( try conda install -c "nvidia/label/cuda-12.8.0" cuda-toolkit)
- pip install -r req.txt
- remove the default torch installation as:
- pip uninstall -y torch torchvision torchaudio torchtext xformers triton torchtriton pytorch-triton
## Install the torch 2.8 cu128
- pip3 install torch torchvision
## Cd causal_conv1d
- pip install -e . --no-build-isolation -vvv
## Cd mamba
- pip install -e . --no-build-isolation -vvv



### Install Causal conv -> mamba_ssm -> requirements.lock.txt
### Install both causal conv and mamba as pip install -e . --no-build-isolation -vvv
### environment should look something like this 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2025 NVIDIA Corporation
Built on Wed_Jan_15_19:20:09_PST_2025
Cuda compilation tools, release 12.8, V12.8.61
Build cuda_12.8.r12.8/compiler.35404655_0
Torch: 2.8.0+cu128 CUDA build: 12.8 GPU available: True


on running following script
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
nvcc --version
conda activate arcee
python -c "import torch; print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
cd /mnt/e/Research/Fleurdelys

(exports might not be needed)