export CUDA_VISIBLE_DEVICES=4

NUM_GPUS=1
BATCH_SIZE=96
EVAL_BS=40

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/train.py --exp Arcee-1-B-2-celeba256 --datadir ../data_prep/celeba/ --dataset celeba_256 --eval-refdir ../data_prep/celeba/real_samples \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model Arcee-B/2 \
  --scan-type Arcee_1 \
  --ssm-dstate 256 \
  --train-steps 60000 \
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
  --save-content-every 5000 \
  --use-wandb