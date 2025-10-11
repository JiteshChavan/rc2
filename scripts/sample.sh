python ../Arcee/Sample2.py ODE --ckpt results/time_Arcee8/checkpoints/content.pth \
    --inference-dir inferences \
    --scan-type Arcee_8\
    --model Arcee-B/2\
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