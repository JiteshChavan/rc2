
EXP="ArceeVisionMamba-B2-celeba256"
MODEL="ArceeVim-B/2"
SCAN_TYPE="V2RC"

export CUDA_VISIBLE_DEVICES=0,1,2,3
NUM_GPUS=4
EVAL_BS=8 # per GPU
FID_BS=128 # for inceptionV3 feature extraction model forward pass, can be much higher than backbone
EVAL_SAMPLES=50000


torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/fid50k.py --exp ../L40runs/results/${EXP}  --datadir ../data_prep/celeba256/ --dataset celeba_256 --eval-refdir ../data_prep/celeba256/real_samples \
  --model-type ema \
  --eval-metrics KID FID\
  --model $MODEL \
  --scan-type $SCAN_TYPE \
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
  
