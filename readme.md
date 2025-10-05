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