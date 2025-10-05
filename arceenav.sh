export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:$LD_LIBRARY_PATH
nvcc --version
conda activate arcee
python -c "import torch; print('Torch:', torch.__version__, 'CUDA build:', torch.version.cuda, 'GPU available:', torch.cuda.is_available())"
cd /mnt/e/Research/Fleurdelys
