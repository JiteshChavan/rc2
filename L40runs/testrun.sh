export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
# REMEMBER TO EXPORT RUNID FROM wandb url to resume
#export WANDB_RUN_ID="PASTE_ID_HERE"
#export WANDB_RESUME="must"
EXP="ArceeVisionMamba-B2-celeba256"
MODEL="ArceeVim-B/2"
SCAN_TYPE="V2RC"
WANDB="online"



export CUDA_VISIBLE_DEVICES=0
NUM_GPUS=1
BATCH_SIZE=4
GRAD_ACC_STEPS=3
EVAL_BS=3


GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/train_grad_acc.py --exp $EXP  --datadir ../data/celeba256/ --dataset celeba_256 --eval-refdir ../data/celeba256/real_samples \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model $MODEL \
  --scan-type $SCAN_TYPE \
  --ssm-dstate 256 \
  --train-steps 50050 \
  --plot-every 500 \
  --ckpt-every 10000 \
  --log-every 5 \
  --grad-accum-steps $GRAD_ACC_STEPS\
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
  --eval-every 10000 \
  --use-wandb $WANDB
  #--resume

  