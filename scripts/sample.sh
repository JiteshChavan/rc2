MODEL=Arcee_B/2
SCAN_TYPE=Arcee_1


python ../Arcee/Sample2.py ODE --ckpt results/time_Arcee8/checkpoints/content.pth \
    --inference-dir inferences \
    --scan-type $SCAN_TYPE\
    --model $MODEL\
    --ssm-dstate 256 \
    --block-type normal \
    --rms-norm \
    --fused-add-norm \
    --image-size 256 \
    --num-sampling-steps 50\
    --global-batch-size 1\
    --path-type GVP \
    --learnable-pe \
    --measure-time \
    #--compute-nfe \
    #--ema \