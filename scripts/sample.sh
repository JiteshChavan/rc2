python ../Arcee/Sample2.py ODE --ckpt results/3kstepsArcee_11k/checkpoints/content.pth \
    --inference-dir inferences \
    --model Arcee-XS/2 \
    --ssm-dstate 64 \
    --scan-type Arcee_1 \
    --block-type normal \
    --rms-norm \
    --fused-add-norm \
    --image-size 256 \
    --num-sampling-steps 50\
    --global-batch-size 16\
    --path-type GVP \
    --learnable-pe \
    #--ema \
    #--compute-nfe \
    #--measure-time \