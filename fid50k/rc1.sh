export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
export CUDA_VISIBLE_DEVICES=0,1


EXP="Arcee-1-B-2-celeba256"
NUM_GPUS=2
EVAL_BS=40 # per GPU
FID_BS=256 # for inceptionV3 feature extraction model forward pass, can be much higher than backbone
EVAL_SAMPLES=50000


torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/fid50k.py --exp ../scripts/results/${EXP}  --datadir ../data_prep/celeba256/ --dataset celeba_256 --eval-refdir ../data_prep/celeba256/real_samples \
  --model-type ema \
  --eval-metrics KID FID\
  --model Arcee-B/2 \
  --scan-type Arcee_1 \
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
  
