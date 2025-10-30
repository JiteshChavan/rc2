

from create_model import create_model
import argparse
import torch

def main(args):
    model = create_model(args).to('cuda')
    model.train()  # Ensure requires_grad is respected
    #with torch.no_grad():
    #    print("unzero final layer")
    #    model.final_layer.linear.weight.normal_(0, 0.02)
    #    model.final_layer.linear.bias.zero_()
#
    #    with torch.no_grad():
    #        for blk in model.blocks:
    #            lin = blk.adaLN_modulation[-1]          # Linear(..., 3*dim) or 6*dim
    #            parts = 3 if getattr(blk, "no_ffn", True) else 6
    #            dim = lin.out_features // parts
    #            lin.bias[dim:2*dim].fill_(1.0)          # gate_ssm = 1.0 warm-start
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\ttotal number of parameters for {args.model} is {total_params}")

    # Dummy input
    h0 = torch.randn(
    1,
    model.blocks[0].mixer.d_inner,
    model.blocks[0].mixer.d_state,
    device=model.blocks[0].mixer.A_log.device,
    dtype=model.blocks[0].mixer.A_log.dtype,
    requires_grad=True,
).requires_grad_()
    
    h0 = None

    x0 = torch.randn(1, 4, args.image_size // 8, args.image_size // 8).to('cuda').requires_grad_()
    t = torch.ones(1).to('cuda')

    # Forward pass
    print
    output = model(x0, t, initial_state=h0)
    #output = model ("hello:x0", "hello:t", initial_state="initial_state")
    loss = output.mean()
    loss.backward()

    print(f"shapes match: {x0.shape == output.shape}")
    print("success!")
    if h0 is not None:
        assert h0.grad is not None
        print (f"h0 grad stats: norm: {h0.grad.norm()}, mean: {h0.grad.mean()}, variance: {h0.grad.var()}")

    # Report unused parameters
    print("\nUnused parameters (did not receive gradients):")
    unused = []
    for name, param in model.named_parameters():
        if param.grad is None:
            print(f" - {name} | shape: {tuple(param.shape)}")
            unused.append(name)

    if not unused:
        print("All parameters participated in the backward pass!")



def none_or_str(value):
    if value == "None":
        return None
    return value

if __name__ == "__main__":
    # Default args here will train the model with the hyperparameters we used in our paper (except training iters).
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="cifar10", help="name of dataset")
    parser.add_argument("--exp", type=str, required=True)
    parser.add_argument("--datadir", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    
    parser.add_argument("--ssm-dstate", type=int, default=16, help="dstate for each d_inner")
    parser.add_argument("--model", type=str, default="Arcee-XS/2")

    parser.add_argument("--image-size", type=int, choices=[64, 256, 512, 1024], default=256)
    parser.add_argument("--num-in-channels", type=int, default=4)
    parser.add_argument("--label-dropout", type=float, default=-1)
    parser.add_argument("--num-classes", type=int, default=0)
    parser.add_argument("--learn-sigma", action="store_true")
    parser.add_argument("--rms-norm", action="store_true")
    parser.add_argument("--fused-add-norm", action="store_true")
    parser.add_argument(
        "--scan-type",
        type=str,
        default="none",
        choices=["none", "Arcee_1", "Arcee_2", "Arcee_4", "Arcee_8", "Zigma_1", "Zigma_2", "Zigma_4", "Zigma_8", "V2", "V2RC"],
    )

    # MOE Arguments
    group = parser.add_argument_group("MoE arguments")
    group.add_argument("--num-moe-experts", type=int, default=8)
    group.add_argument("--gated-linear-unit", action="store_true")
    group.add_argument("--routing-mode", type=str, choices=["sinkhorn", "top1", "top2", "sinkhorn_top2"], default="top1")
    group.add_argument("--is-moe", action="store_true")
    group.add_argument("--mamba-moe-layers", type=none_or_str, nargs="*", default=None)


    parser.add_argument("--pe-type", type=str, default="ape", choices=["ape", "cpe", "rope"])
    parser.add_argument("--block-type", type=str, default="normal", choices=["normal", "combined"])
    parser.add_argument("--scanning-continuity", action="store_true") # always false
    parser.add_argument("--learnable-pe", action="store_true")
    parser.add_argument("--drop-path", type=float, default=0.0)
    parser.add_argument("--use-final-norm", action="store_true")
    parser.add_argument("--use-attn-every-k-layers",type=int,default=-1,)
    parser.add_argument("--not-use-gated-mlp", action="store_true")
    # parser.add_argument("--skip", action="store_true")
    


    # general
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=1400)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=25)
    parser.add_argument("--save-content-every", type=int, default=5)
    parser.add_argument("--plot-every", type=int, default=5)
    parser.add_argument("--model-ckpt", type=str, default="")
    parser.add_argument("--resume", action="store_true")
    



    parser.add_argument("--no-lr-decay", action="store_true", default=False)
    parser.add_argument("--min-lr",type=float,default=1e-6,)
    parser.add_argument("--max-lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5,)
    parser.add_argument("--max-grad-norm", type=float, default=2.0,)

    group = parser.add_argument_group("Eval")
    group.add_argument("--eval-every", type=int, default=100)
    group.add_argument("--eval-refdir", type=str, default=None)
    group.add_argument("--n-eval-samples", type=int, default=1000)
    group.add_argument("--eval-batch-size", type=int, default=4)
    group.add_argument("--eval-cfg-scale", type=float, default=1.0)
    #NOTE: Gradient Signal ablation flags
    parser.add_argument("--concat-context-across-time", action="store_true")
    parser.add_argument("--detach-context-across-time", action="store_true")

    


    args = parser.parse_args()
    main(args)





