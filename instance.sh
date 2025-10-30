SCAN_TYPE="V2RC"
MODEL="ArceeVim-B/2"

python Arcee/instance.py \
    --exp instantiation \
    --datadir . \
      --image-size 256 \
  --num-classes 1 \
  --block-type normal \
  --model $MODEL \
  --scan-type $SCAN_TYPE\
  --ssm-dstate 256 \
  --global-batch-size 8 \
  --learnable-pe \
  --rms-norm \
  --fused-add-norm \
  --drop-path 0.0