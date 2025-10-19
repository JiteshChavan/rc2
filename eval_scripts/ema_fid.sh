export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
export CUDA_VISIBLE_DEVICES=0


EXP="eval_dummy"
NUM_GPUS=1
EVAL_BS=2
FID_BS=40
EVAL_SAMPLES=40

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/eval_fid.py --exp ../scripts/results/${EXP} --dataset celeba_256 --datadir ../data/celeba256/ --eval-refdir ../data/celeba256/real_samples \
  --model-type ema \
  --eval-metrics KID FID\
  --model Arcee-XS/2 \
  --scan-type Arcee_8 \
  --eval-nsamples $EVAL_SAMPLES \
  --block-type normal \
  --image-size 256 \
  --num-classes 1 \
  --ssm-dstate 256 \
  --eval-bs $EVAL_BS \
  --fid-batch-size $FID_BS\
  --kid-batch-size $FID_BS\
  --learnable-pe \
  --path-type GVP \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0 \
  --use-wandb
  
