# Copyright (c) 2023, Tri Dao, Albert Gu.

# TODO: IMPORTANT LOCK REINIT OF SSM params in backbone!!! use isinstance

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import arcee_mamba_inner_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton_ops.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton_ops.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from .statemod import StateModulators

class ArceeMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        last_state_modulation=False,
        shared_dstate_collapse=True,
        n_state_mods=0,
        lock_permutations=True,
        scan_type="none",
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        assert scan_type in ["arcee_1", "arcee_2", "arcee_4", "arcee_8"], f"invalid scan type : {scan_type}"
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # d_inner input, d_inner z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        
        self.last_state_modulation = last_state_modulation
        if last_state_modulation:
            assert n_state_mods == 2 or n_state_mods == 3
            self.state_modulators = StateModulators(dstate=d_state, d_inner=self.d_inner, shared_dstate_collapse=shared_dstate_collapse, n_mods=n_state_mods)
            

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj.RESIDUAL_ADDITION = True

        #self.last_state_norm = nn.LayerNorm(self.d_state)

        self.lock_permutations = lock_permutations
        if not lock_permutations:
            self.register_buffer("zigzag_paths", kwargs.get("zigzag_paths", None))
            self.register_buffer("zigzag_paths_reverse", kwargs.get("zigzag_paths_reverse", None))
        else:
            self.register_buffer("zigzag_paths", None)
            self.register_buffer("zigzag_paths_reverse", None)

    # hidden_states if forward activations here (think x)
    def forward(self, hidden_states, initial_state=None, return_last_state=False):
        """
        hidden_states: (B, L, D)
        last_state : (B, d_inner, d_state)
        Returns: (B, L, D) and last state (which can be None if return_last_state if False)
        """

        batch, seqlen, dim = hidden_states.shape


        conv_state, ssm_state = None, None
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1") # broadcast bias along channels

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if initial_state is not None:
            assert initial_state.shape == (batch, self.d_inner, self.d_state)
            initial_state = initial_state.to(dtype=A.dtype)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        assert self.use_fast_path and causal_conv1d_fn is not None

        if not self.lock_permutations and self.zigzag_paths is not None:
            #### rearrange
            path_index = self.layer_idx % self.zigzag_paths.shape[0]
            _perm = self.zigzag_paths[path_index]
            # xz = xz[:, :, _perm].contiguous()  # [B,D,L]
            xz = torch.gather(xz, 2, _perm[None, None, :].expand_as(xz))  # [B,D,L]
        
        # we moved the out_proj linear outside the inner funciton
        out = arcee_mamba_inner_fn(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            B = None,  # input-dependent B
            C = None,  # input-dependent C
            D = self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=return_last_state,
            h0=initial_state,   # h0 = global representation from previous block
            layer_idx=self.layer_idx,
        )
        
        if return_last_state:
            out_z, last_state = out # (B, L, d_inner), (B, d_inner, dstate)
        else:
            out_z = out
            last_state = None

        # TODO: remove redundant asserts
        if initial_state is not None:
            assert initial_state.shape[0] == hidden_states.shape[0]
            assert initial_state.shape[1] == self.d_inner
            assert initial_state.shape[2] == self.d_state
        if return_last_state:
            assert last_state.shape[0] == hidden_states.shape[0]
            assert last_state.shape[1] == self.d_inner
            assert last_state.shape[2] == self.d_state
        assert out_z.shape[-1] == self.d_inner

        # optional FiLM modulation
        if self.last_state_modulation and return_last_state:
            mods = self.state_modulators(last_state)
            if len(mods) == 2:
                scale, shift = mods # 2x(B, d_inner)
                out_z = modulate(F.layer_norm(out_z, (self.d_inner,)), shift, scale)
            else:
                scale, gate, shift = mods # 3x(B, d_inner)
                out_z = out_z + gate.unsqueeze(1) * modulate(F.layer_norm(out_z, (self.d_inner,)), shift, scale)
        
        # project back to backbone d_model
        out_z = self.out_proj(out_z) # (B, L, d_inner) -> (B, L, d_model)

        if not self.lock_permutations and self.zigzag_paths_reverse is not None:
            _perm_rev = self.zigzag_paths_reverse[path_index]
            # out = out[:, _perm_rev, :].contiguous()  # out is [B,L,D]
            out_z = torch.gather(out_z, 1, _perm_rev[None, :, None].expand_as(out_z))  # out is [B,L,D]
        
        #if last_state is not None:
            #last_state = self.last_state_norm(last_state)
            #last_state = F.layer_norm(last_state, (self.d_state,))

        return out_z, last_state

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)




# ------------------------------------------------------------------------------------------------------------------------------------------------
# Modules for baseline expt setups
# ------------------------------------------------------------------------------------------------------------------------------------------------
class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        scan_type="none",
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        assert scan_type in ["zigma_1", "zigma_2", "zigma_4", "zigma_8", "v2"], f"invalid scan type : {scan_type}"
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.scan_type = scan_type

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        assert (scan_type in ["zigma_1", "zigma_2", "zigma_4", "zigma_8"]
                or scan_type == "v2"), f"Invalid baseline scan type, {scan_type}"
        
        if scan_type == "v2":
            A_b = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
            self.A_b_log = nn.Parameter(A_b_log)
            self.A_b_log._no_weight_decay = True

            #############################

            self.conv1d_b = nn.Conv1d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                groups=self.d_inner,
                padding=d_conv - 1,
                **factory_kwargs,
            )

            self.x_proj_b = nn.Linear(
                self.d_inner,
                self.dt_rank + self.d_state * 2,
                bias=False,
                **factory_kwargs,
            )
            self.dt_proj_b = nn.Linear(
                self.dt_rank, self.d_inner, bias=True, **factory_kwargs
            )

            self.D_b = nn.Parameter(
                torch.ones(self.d_inner, device=device)
            )  # Keep in fp32
            self.D_b._no_weight_decay = True
        
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj.RESIDUAL_ADDITION = True

        self.register_buffer("zigzag_paths", kwargs.get("zigzag_paths", None))
        self.register_buffer("zigzag_paths_reverse", kwargs.get("zigzag_paths_reverse", None))
        if self.scan_type == "v2":
            assert self.zigzag_paths == None and self.zigzag_paths_reverse == None, f"Non none zigzag path buffers for scan type : {self.scan_type}"

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        ) # (B, 2*d_inner, l)
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1")

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        assert self.use_fast_path, f"Fused mamba kernel unavailable"

        if (
            self.scan_type.startswith("zigma")
        ):
            #### rearrange
            path_index = self.layer_idx % self.zigzag_paths.shape[0]
            _perm = self.zigzag_paths[path_index]
            # xz = xz[:, :, _perm].contiguous()  # [B,D,L]
            xz = torch.gather(xz, 2, _perm[None, None, :].expand_as(xz))  # [B,D,L]
        
            # Vanilla mamba_inner_fn no recurrent differentiable chain
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_z = self.out_proj(out)

            _perm_rev = self.zigzag_paths_reverse[path_index]
            # out = out[:, _perm_rev, :].contiguous()  # out is [B,L,D]
            out_z = torch.gather(out_z, 1, _perm_rev[None, :, None].expand_as(out_z))  # out is [B,L,D]
        
        elif self.scan_type == "v2":
            # Vanilla mamba_inner_fn no recurrent differentiable chain
            # without out_proj returns (B, L, d_inner)
            out = mamba_inner_fn(   # (B, L, d_inner)
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None, # input-dependent B
                None, # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )

            A_b = -torch.exp(self.A_b_log.float())
            out_b = mamba_inner_fn (    # (B, L, d_inner)
                xz.flip(
                    [-1] # xz(B, d_inner, L)
                ), # Flipping the xz is the same as flipping the x, as x will be processed by selective scan, while z will only go to a MLP layer.
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            out_z = self.out_proj(out + out_b.flip(dims=[1])) # (B, L, d_inner ) + (B, L, d_inner) <flipped along tokens>

        return out_z
    

class ArceeVisionMamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        lock_permutations=True,
        scan_type="none",
        **kwargs
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        assert scan_type == "v2rc", f"invalid scan type : {scan_type}"
        super().__init__()

        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # d_inner input, d_inner z
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True


        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        #############################

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.x_proj_b = nn.Linear(
            self.d_inner,
            self.dt_rank + self.d_state * 2,
            bias=False,
            **factory_kwargs,
        )
        self.dt_proj_b = nn.Linear(
            self.dt_rank, self.d_inner, bias=True, **factory_kwargs
        )

        self.D_b = nn.Parameter(
            torch.ones(self.d_inner, device=device)
        )  # Keep in fp32
        self.D_b._no_weight_decay = True
            

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.out_proj.RESIDUAL_ADDITION = True

        #self.last_state_norm = nn.LayerNorm(self.d_state)

        self.lock_permutations = lock_permutations
        if not lock_permutations:
            self.register_buffer("zigzag_paths", kwargs.get("zigzag_paths", None))
            self.register_buffer("zigzag_paths_reverse", kwargs.get("zigzag_paths_reverse", None))
        else:
            self.register_buffer("zigzag_paths", None)
            self.register_buffer("zigzag_paths_reverse", None)

    # hidden_states if forward activations here (think x)
    def forward(self, hidden_states, initial_state=None, initial_state_b=None, return_last_state=False):
        """
        hidden_states: (B, L, D)
        last_state : (B, d_inner, d_state)
        Returns: (B, L, D) and last state (which can be None if return_last_state if False)
        """

        batch, seqlen, dim = hidden_states.shape


        conv_state, ssm_state = None, None
        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj.weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj.bias is not None:
            xz = xz + rearrange(self.in_proj.bias.to(dtype=xz.dtype), "d -> d 1") # broadcast bias along channels

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        if initial_state is not None:
            initial_state = initial_state.to(dtype=A.dtype)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        assert self.use_fast_path and causal_conv1d_fn is not None
        
        # we moved the out_proj linear outside the inner funciton
        out = arcee_mamba_inner_fn(
            xz,
            self.conv1d.weight,
            self.conv1d.bias,
            self.x_proj.weight,
            self.dt_proj.weight,
            A,
            B = None,  # input-dependent B
            C = None,  # input-dependent C
            D = self.D.float(),
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=return_last_state,
            h0=initial_state,   # h0 = global representation from previous block
            layer_idx=self.layer_idx,
        )

        A_b = -torch.exp(self.A_b_log.float())
        out_b = arcee_mamba_inner_fn (
            xz.flip(
                [-1] # xz(B, d_inner, L)
            ), # Flipping the xz is the same as flipping the x, as x will be processed by selective scan, while z will only go to a MLP layer.
            self.conv1d_b.weight,
            self.conv1d_b.bias,
            self.x_proj_b.weight,
            self.dt_proj_b.weight,
            A_b,
            B = None,
            C = None,
            D = self.D_b.float(),
            delta_bias=self.dt_proj_b.bias.float(),
            delta_softplus=True,
            return_last_state=return_last_state,
            h0=initial_state_b,
            layer_idx=self.layer_idx,
        )
        
        if return_last_state:
            out, last_state = out # (B, L, d_inner), (B, d_inner, dstate)
            out_b, last_state_b = out_b
        else:
            out = out
            out_b = out_b
            last_state = None
            last_state_b = None

        out_z = self.out_proj(out + out_b.flip(dims=[1])) # (B, L, d_inner ) + (B, L, d_inner) <flipped along tokens>

        return out_z, last_state, last_state_b

