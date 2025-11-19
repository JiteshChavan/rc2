import torch
import torch.nn as nn

from mamba_ssm.modules.mamba_simple import ArceeMamba


depth = 4
d_model = 64
scan_type = "arcee_1"

input = torch.randn(2, 256, 64).requires_grad_().to(device='cuda')

blocks = nn.ModuleList(
    [ArceeMamba(d_model=d_model, d_state=256, scan_type=scan_type) for i in range(depth)]
).to(device='cuda')

# Cross-block recurrent state chain
h0 = None # first block uses 0 initial state by default
for i in range(depth):
    out_z, last_state = blocks[i](input, initial_state=h0, return_last_state=True)
    h0 = last_state
    input = out_z

# Dummy loss to verify gradients flow through all parameters
loss = out_z.mean() + last_state.mean()
loss.backward()

print("\nUnused parameters (did not receive gradients):")
unused = []
for name, param in blocks.named_parameters():
    if param.grad is None:
        print(f" - {name} | shape: {tuple(param.shape)}")
        unused.append(name)

if not unused:
    print("All parameters participated in the backward pass!")


