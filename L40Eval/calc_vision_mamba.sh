export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'

EXP="VisionMamba-B2-celeba256"
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1


FID_BS=128 # for inceptionV3 feature extraction model forward pass, can be much higher than backbone
EVAL_SAMPLES=50000


torchrun --standalone --nproc_per_node=$NUM_GPUS calc_fid.py --exp ../L40runs/results/${EXP}  --datadir ../data_prep/celeba256/ --dataset celeba_256 --eval-refdir ../data_prep/celeba256/real_samples \
  --eval-metrics KID FID\
  --eval-nsamples $EVAL_SAMPLES \
  --ckpt-step 50000\
  --image-size 256 \
  --fid-batch-size $FID_BS\
  --kid-batch-size $FID_BS\
  --use-wandb 