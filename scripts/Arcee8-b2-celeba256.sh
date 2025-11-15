export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
# REMEMBER TO EXPORT RUNID FROM wandb url to resume
#export WANDB_RUN_ID="PASTE_ID_HERE"
#export WANDB_RESUME="must"
EXP="Arcee8-B-2-celeba256"

export WANDB_RUN_ID="jbebiziw"
export WANDB_RESUME="must"


NUM_GPUS=2
BATCH_SIZE=96
EVAL_BS=40

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/train.py --exp $EXP --datadir ../data_prep/celeba256/ --dataset celeba_256 --eval-refdir ../data_prep/celeba256/real_samples \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model Arcee-B/2 \
  --scan-type Arcee_8 \
  --ssm-dstate 256 \
  --train-steps 60050 \
  --eval-every 5000 \
  --plot-every 500 \
  --ckpt-every 10000 \
  --log-every 5 \
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
  --save-content-every 10000 \
  --use-wandb "online"\
  --resume \
