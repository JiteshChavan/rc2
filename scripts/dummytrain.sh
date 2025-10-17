export WANDB_API_KEY='2f92f218fe46708930c460c6f57055ac6ce1361c'
export CUDA_VISIBLE_DEVICES=0

# to start a brand new run 
# rm -rf run_state/$EXP
# rm -rf results/"$EXP"/
# comment out --resume flag
EXP="eval_dummy"
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

NUM_GPUS=1
BATCH_SIZE=8
EVAL_BS=4

GLOBAL_BATCH_SIZE=$((BATCH_SIZE * NUM_GPUS))

torchrun --standalone --nproc_per_node=$NUM_GPUS ../Arcee/train.py --exp $EXP --datadir ../data/celeba256/lmdb_new --dataset celeba_256 --eval-refdir ../data/celeba256/real_samples \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model Arcee-XS/2 \
  --scan-type Arcee_8 \
  --ssm-dstate 256 \
  --train-steps 60000 \
  --eval-every 5000 \
  --plot-every 500 \
  --ckpt-every 10 \
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
  --save-content-every 5 \
  #--use-wandb \
  #--resume \
