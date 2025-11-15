export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'

WANDB="offline"



# REMEMBER TO EXPORT RUNID FROM wandb url to resume
#export WANDB_RUN_ID="PASTE_ID_HERE"
#export WANDB_RESUME="must"
EXP="Zigma4-B2-ffhq1024"
BATCH_SIZE=32
EVAL_BS=32

# important num_gpus should be nodes * gpu_per_node = total gpus
NUM_GPUS=8
NODES=2
GPUS_PER_NODE=$((NUM_GPUS / NODES))

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --nnodes=$NODES --nproc_per_node=$GPUS_PER_NODE ../Arcee/train.py --exp $EXP  --datadir ../data_prep/ffhq1024/ --dataset ffhq_1024 --eval-refdir ../data_prep/ffhq1024/real_samples \
  --image-size 1024 \
  --num-classes 1 \
  --block-type normal \
  --model Zigma-B/2 \
  --scan-type Zigma_4 \
  --ssm-dstate 256 \
  --train-steps 50050 \
  --plot-every 1000 \
  --ckpt-every 10000 \
  --log-every 1 \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --sample-bs $BATCH_SIZE \
  --eval-bs $EVAL_BS \
  --lr 3e-4 \
  --learnable-pe \
  --path-type GVP \
  --eval-nsamples 10000 \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0 \
  --save-content-every 5000 \
  --eval-every 500000000 \
  --use-wandb $WANDB\
  #--resume
  