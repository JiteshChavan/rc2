export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
export CUDA_VISIBLE_DEVICES=0


EXP="eval_dummy"
NUM_GPUS=1
EVAL_BS=96

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/eval_fid.py --exp $EXP --dataset celeba_256 --eval-refdir ../data_prep/celeba256/real_samples \
  --model-type ema \
  --eval-metric fid-10k\
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model Arcee-XS/2 \
  --scan-type Arcee_8 \
  --ssm-dstate 256 \
  --eval-bs $EVAL_BS \
  --learnable-pe \
  --path-type GVP \
  --eval-nsamples 10000 \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0 \
  #--use-wandb \
  #--resume \
