export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
export CUDA_VISIBLE_DEVICES=6,7


# to start a brand new run 
# rm -rf run_state/$EXP
# rm -rf results/"$EXP"/
# comment out --resume flag
EXP="Zigma-8-B2-ffhq1024"
STATE_DIR="run_state/${EXP}"
RUN_ID_FILE="${STATE_DIR}/wandb_run_id.txt"
mkdir -p "${STATE_DIR}"
# Resume if we already have a saved run
if [[ -f "${RUN_ID_FILE}" ]]; then
  export WANDB_RUN_ID="$(
  <"$RUN_ID_FILE"
  )"
  export WANDB_RESUME="must"
else
  export WANDB_RUN_ID="$(
python - <<'PY'
from wandb.util import generate_id
print(generate_id())
PY
)"
  echo "$WANDB_RUN_ID" > "${RUN_ID_FILE}"
  export WANDB_RESUME="allow"
fi

NUM_GPUS=2
BATCH_SIZE=96
EVAL_BS=32

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/train.py --exp $EXP --datadir ../data_prep/ffhq1024/ --dataset ffhq_1024 --eval-refdir ../data_prep/ffhq1024/real_samples \
  --image-size 1024 \
  --num-classes 1 \
  --block-type normal \
  --model Zigma-B/2 \
  --scan-type Zigma_8 \
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
  --use-wandb \
  #--resume \