MODEL=Arcee-B/2
SCAN_TYPE=Arcee_1
GLOBAL_BATCH_SIZE=4


python ../../Arcee/gen.py ODE \
  --ckpt step50000.pt \
  --inference-dir ./qual \
  --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model $MODEL \
  --scan-type $SCAN_TYPE \
  --ssm-dstate 256 \
  --global-batch-size $GLOBAL_BATCH_SIZE \
  --learnable-pe \
  --path-type GVP \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0 \
  --ema \
  --grid-rows 4 \
  --grid-cols 8 