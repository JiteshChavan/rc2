export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

WANDB="offline"


EXP="Zigma8-B2-ffhq1024"
NODES=2
NUM_GPUS=8
BATCH_SIZE=8
GPUS_PER_NODE=$((NUM_GPUS / NODES))
GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))


FID_BS=64 # for inceptionV3 feature extraction model forward pass, can be much higher than backbone

EVAL_SAMPLES=10000


torchrun --nnodes=$NODES --nproc_per_node=$GPUS_PER_NODE ../Arcee/eval_fid.py --exp ../ffhq_scripts/results/${EXP}  --datadir ../data_prep/ffhq1024/ --dataset ffhq_1024 --eval-refdir ../data_prep/ffhq1024/real_samples \
  --scan-type Zigma_8 \
  --model Zigma-B/2 \
  --model-type ema \
  --eval-metrics KID FID\
  --eval-nsamples $EVAL_SAMPLES \
  --block-type normal \
  --image-size 1024 \
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
  --use-wandb $WANDB