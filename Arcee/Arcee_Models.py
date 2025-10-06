import math
from functools import partial
from typing import Optional

from huggingface_hub import PyTorchModelHubMixin


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mamba_ssm.modules.mamba_simple import ArceeMamba, Mamba
from pe.cpe import AdaInPosCNN
from pe.my_rotary import apply_rotary, get_2d_sincos_rotary_embed
from timm.models.vision_transformer import Attention, Mlp, PatchEmbed
from torch import Tensor


try:
    from mamba_ssm.ops.triton_ops.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None

from CrossAttentionFusion import CrossAttentionFusion
#from CrossFusionMamba import CrossFusionMamba
from einops import rearrange
from mlp import GatedMLP
from scanning_orders import SCAN_ZOO, local_reverse, local_scan, reverse_permut_np
from switch_mlp import SwitchMLP

from wavelet_layer import DWT_2D, IDWT_2D

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# inspiredfrom
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    embed_dim : int dimensions of pos embed vector representation
    grid_size : int, height and width of the grid
    
    Returns:
    pos_embed : [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (/w or w/o cls_token)
    """

    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h) # width first
    grid = np.stack(grid, axis=0) # stack the two elements in the list in a single list 0 indexes into abscissae nad 1 into ordinates

    # rearrange the grid so that 0 and 1 explicitly index into abscissae and ordinates respectively
    grid = grid.reshape([2, 1, grid_size, grid_size])
    # get pos embedding from the constructed grid
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed # ([grid_size*grid_size, embed_dim]) includes prefix of extra tokens and cls tokens if specified

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0, f"embed_dim is required to be divisible by 2,\nhalf of the dimensions represent abscissae\nthe other half ordinates"

    emb_h = get_1d_sincos_pos_embed_from_grid(grid[0], embed_dim // 2) # (H*W, D/2) representation of ordinates
    emb_w = get_1d_sincos_pos_embed_from_grid(grid[1], embed_dim // 2) # (H*W, D/2) representation of abscissae

    emb = np.concatenate([emb_h, emb_w], axis=1) #(H*W, D) concatenate the representations along channels
    return emb

def get_1d_sincos_pos_embed_from_grid(pos_grid, embed_dim):
    """
    Takes a grid that specifies ordinates or abscissae of a co-ordinate grid, returns (H*W, embed_dim) representation
    of the grid.
    Args:
        pos_grid: np array specifies ordinates or abscissae of a co-ordinate grid.
        embed_dim: number of dimensions for vector representation of each position
    """

    assert embed_dim % 2 == 0, f"embed_dim:{embed_dim} must be divisible by 2\nhalf dimensions for sine components\nthe other half for cosine"

    # frequencies linearly spaced in log scale, will be exponentially spaced in normal scale
    log_freq = np.arange(embed_dim // 2, dtype=np.float64) # (0 through D//2 -1 )
    log_freq = log_freq / (embed_dim // 2) # normalize to be between 0 and 1 so that we dont have numerical instability while exponentiating
    freq = 1 / 1000.0 ** log_freq # (D/2)

    pos = pos_grid.reshape(-1) # (T,) (flatten H,W)
    # we want to infuse the frequencies with position
    # two ways
    #pos_modulated_freq = pos.reshape(-1, 1)  * freq # (T, 1) * (D/2) -> (T, D/2)
    # or
    pos_modulated_freq = np.einsum("T,D->TD", pos, freq) # (T) @ (D/2) -> (T, D/2) outer product

    sin_embd = np.sin(pos_modulated_freq)
    cos_embd = np.cos(pos_modulated_freq)

    emb = np.concatenate([sin_embd, cos_embd], axis=1) # (T, D)
    return emb

# --------------------------------------------------------------------------------
# Interpolate position embeddings for high-resolution
# Regerences:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------------------------------
def interpolate_pos_embed(model, checkpoint_model):
    if "pos_embed" in checkpoint_model:
        pos_embed_checkpoint = checkpoint_model["pos_embed"]
        embedding_size = pos_embed_checkpoint.shape[-1] # channels
        num_patches = model.x_embedder.num_patches # patches in new resolution
        num_extra_tokens = (model.pos_embed.shape[-2] - num_patches) # same number of extra tokens, only resolution is different
        # original height (== width) from the checkpoint
        orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens)**0.5)
        # new height (== width) for the new resolution
        new_size = int(num_patches**0.5) # num patches are derived from the model.x_embedder so that T corresponds to the new resolutions, also it doesnt coutn extra tokens
        # class_token and dist_token are kept unchanged
        if orig_size != new_size:
            print("Interpolating positional embedding from %dx%d to %dx%d" % (orig_size, orig_size, new_size, new_size))
            extra_tokens = pos_embed_checkpoint[:,:num_extra_tokens] # pos_embed is usually (1, T, C) for ease of broadcasting x = x + pos_embed
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2) # (B, C, H, W)
            pos_tokens = F.interpolate(
                pos_tokens, size=(new_size, new_size), mode="bicubic", align_corners=False
            ) # (B, C, H, W)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2) #(B, T, C)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1) # concatenate extra tokens along T not along B hence dim = 1
            checkpoint_model["pos_embed"] = new_pos_embed


# ----------------------------------------------------------------------------------------------
# Embedding layers for Timesteps and Class Labels
# ----------------------------------------------------------------------------------------------

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        Args:
            t : Float, a 1-D tensor of B indices, one per batch element.
            dim : the dimension of the output vector representation.
            max_period: controls the minimum frequency of the embeddings.
            retuns (B, dim=C) tensor representation corresponding to scalar timesteps t (B,) 
        """

        half = dim // 2
        # interpolate linearly between 0 and -log(max_period) [0, log(f_min)] then exponentiate (decay 1 -> f_min)
        # gives us linearly spaced frequencies in logspace
        # exponentiation results in exponential decay between [1, 1/max_period] i.e [1, f_min]
        # (C/2)
        freqs = torch.exp(-math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half).to(device=t.device)
        # t (B,)
        # outer product to get (B, C/2)
        # three ways
        #args = torch.einsum("B,C->BC", t, freqs) # outer product < (B), (C/2)> -> (B, C/2)
        #args = t.unsqueeze(1) * freqs #(B, 1) * (C/2) -> (B, C/2)
        args = t[:, None].float() * freqs # (B, 1) * (C/2)
        
        embedding = torch.cat((torch.sin(args), torch.cos(args)), dim=-1)

        if dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding
    
    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance
    """

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0.0 # 1
        self.in_channels = num_classes + 1 if use_cfg else num_classes # 1001 or 1000
        self.embedding_table = nn.Embedding(self.in_channels, hidden_size)
        self.num_classes = num_classes # 1000
        self.dropout_prob = dropout_prob
    
    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """

        if force_drop_ids is None:
            # drop labels where p < drop prob
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob # setup boolean tensor via elementwise comparison
        else:
            drop_ids = force_drop_ids == 1 # elementwise comparison with 1, drop labels where force_drop_ids is 1
        
        # for each index in labels, set labels to be num_classes where drop_ids is True
        labels = torch.where(drop_ids, self.num_classes, labels) # 1000 or labels (table[1000] = null label)

        return labels
    
    def forward (self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings
    
    def get_in_channels(self):
        """Returns in_channels or number of classes in the embedding table matrix"""
        return self.in_channels # 1001 if dropout_prob > 0.0 else 1000

class FinalLayer (nn.Module):
    """
    Final layer of the backbone
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
    
    def forward(self, x, y):

        scale, shift = self.adaLN_modulation(y).chunk(2, dim=-1) # 2x (B, C)
        x = modulate(self.norm_final(x), shift, scale) # x = x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1) -> (B, T, C)
        x = self.linear(x) # (B, T, C) - > (B, T, patch_size * patch_size * out_channels)
        return x

class LastStateWeaver(nn.Module):
    """Ties in the final last_state into the loss for a better gradient path (lightweight for nwo)"""

    def __init__(self, hidden_size, ssm_dinner, ssm_dstate):
        super().__init__()
        self.d_state_collapse = nn.Parameter(torch.zeros(ssm_dstate))
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(ssm_dinner, 3*hidden_size))

    def forward(self, x, last_state):
        collapsed_state = torch.einsum("b d s , s -> b d", last_state, self.d_state_collapse)
        scale, gate, shift = self.adaLN_modulation(collapsed_state).chunk(3, dim=-1)
        return x + gate.unsqueeze(1) * (modulate(x, shift, scale))


class Block(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            y_dim=None,
            no_ffn=True, # we turn off FFNs for our baseline experiment
    ):
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        y_dim = y_dim if y_dim is not None else dim

        self.no_ffn = no_ffn
        self.mixer = mixer_cls(dim)
        assert isinstance(self.mixer, (ArceeMamba, Mamba)), f"Invalid, model only supports ArceeMamba and Mamba modules"

        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm only supported for LayerNorm or RMSNorm"
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 3 * dim if self.no_ffn else 6 * dim, bias=True))

        if not self.no_ffn:
            self.norm_2 = norm_cls(dim)
            mlp_hidden_dim = int (4 * dim)
            approx_gelu = lambda:nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            x: Tensor,
            residual: Optional[Tensor] = None,
            initial_state: Optional[Tensor] = None, # Will be None for zigma baseline
            return_last_state: Optional[bool] = False, # False for zigma baseline
            y : Optional[Tensor] = None,
    ):
        if not self.fused_add_norm:
            if residual is None:
                residual = x
            else:
                residual = residual + self.drop_path(x)
            
            x = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            
            if residual is None:
                x, residual = fused_add_norm_fn (
                    x,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, #        None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps
                )
            else:
                x, residual = fused_add_norm_fn (
                    self.drop_path(x),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, #        Not NOne
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )

        # now we have a residual and a normalized x, modulate it, feed it into mixer, gate it
        # if theres an mlp, normalize this output, modualte it, feed it into mlp, gate it
        # branch if its an arcee mixer or not

        last_state = None
        if self.no_ffn:
            scale_ssm, gate_ssm, shift_ssm = self.adaLN_modulation(y).chunk(3, dim=-1) # (B, 3C) -> 3x (B,C)
            
            x_modulated = modulate(x, shift_ssm, scale_ssm)
            if isinstance(self.mixer, ArceeMamba):
                out_z, last_state = self.mixer (x_modulated, initial_state=initial_state, return_last_state=return_last_state)
            else:
                out_z = self.mixer(x_modulated)
            x = x + gate_ssm.unsqueeze(1) * out_z
        else:
            scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1)

            # now we have normalized x and a residual, modulate the normalized x, feed it into mixer
            # gate the output of the mixer relative to normalized x
            x_modulated = modulate(x, shift_ssm, scale_ssm)
            if isinstance(self.mixer, ArceeMamba):
                out_z, last_state = self.mixer(x_modulated, initial_state=initial_state, return_last_state=return_last_state)
            else:
                out_z = self.mixer(x_modulated)
            x = x + gate_ssm.unsqueeze(1) * out_z

            # now the same story for mlp part except now we dont have a normalized stream of x so we first normalize it, then modulate it
            x = x + gate_mlp.unsqueeze(1) * self.mlp (
                modulate(self.norm_2(x), shift_mlp, scale_mlp)
            )
        
        return x, residual, last_state
        



class FlBlockRaw(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            norm_cls_ffn=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            y_dim=None,
            no_ffn=True,
    ):
        """
        
        A block around SSM mixer block with residual and LayerNorm/RMSNorm.
        Takes two inputs hidden_states and residua. Residual being from the preivous block.

        Still the standard prenorm block with slightly different structure, solely for performance reasons
        as we can fuse addition, back into residual path, and norm.

        Residual has to be specified except for very first block.

        We add hidden_states from mixer of previous block, into residual pathway and then normalize to branch residual and mixer.
        """

        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        y_dim = dim if y_dim is None else y_dim
        
        self.no_ffn = no_ffn

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm only supported for RMSNorm or LayerNorm"
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 3 * dim if self.no_ffn else 6 * dim, bias=True))

        if not self.no_ffn:
            self.norm_2 = norm_cls_ffn(dim)

            mlp_hidden_dim = int (4 * dim)
            approx_gelu = lambda:nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            hidden_states: Tensor,
            state_space_representation: Optional[Tensor]=None,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params=None,
    ):
        """
        Pass the input signal (B, T, C) through the spatial SSM block.

        Args:
            hidden_states: input signal (B, T, C) to the block
            residual: residual from the previous block. hidden_states = Mixer(Norm(residual))
            We fuse add (residual, hidden_states) and then norm before branching residual and mixer

            y: class label embedding, guidance embedding
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual,
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # Now I have residual (<- To be returned (B, T, C)) and hidden_states after norm, to be processed by mixer
        T = hidden_states.shape[1]
        h = w = int (np.sqrt(T))
        if self.transpose:
            hidden_states = rearrange(hidden_states, "b (h w) c -> b (w h) c", h=h, w=w)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w) # (B, T, C)
        
        if self.reverse:
            hidden_states = hidden_states.flip(1) # flip along tokens

        if not self.no_ffn:
            scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) #(B, 6C) -> 6x(B,C)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer (
                modulate(hidden_states, shift_ssm, scale_ssm), cond_emb=state_space_representation, inference_params=inference_params
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
            )
        else:
            scale_ssm, gate_ssm, shift_ssm = self.adaLN_modulation(y).chunk(3, dim=-1) #(B, 3C) -> 3x(B, C)

            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), cond_emb=state_space_representation, inference_params=inference_params
            )
        
        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w)
        
        if self.transpose:
            hidden_states = rearrange (hidden_states, "b (w h) c -> b (h w) c", h=h, w=w)
        

        return hidden_states, residual

class WaveFlBlock(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            norm_cls_ffn=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            skip=False,
            no_ffn=True,
            y_dim=None, # label embedding size
            window_scan=True,
            num_wavelet_lv=2,
    ):
        """
        Simple block wrapping a mixer with LayerNorm/RMSNorm and a residual connection.

        Just as before still pre norm residual block with slightly different structure than standard pre norm block.
        Solely for performance reasons, as we can fuse addition, back into residual pathway, and normalization.
        Residual is required unless its the first block.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.no_ffn = no_ffn
        self.window_scan = window_scan
        self.num_wavelet_lv = num_wavelet_lv
        y_dim = dim if y_dim is None else y_dim

        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)

        self.dwt = DWT_2D(wave="haar")
        #self.idwt = IDWT_2D(wave="haar")

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance (self.norm, (RMSNorm, nn.LayerNorm)), f"fused_add_norm is only supported for LayerNorm and RMSNorm"
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(y_dim, 6 * dim if not self.no_ffn else 3 * dim, bias=True))

        if not self.no_ffn:
            self.norm_2 = norm_cls_ffn(dim)

            mlp_hidden_dim = int (4 * dim)
            gelu_approx = lambda:nn.GELU(approximate="tanh")
            self.mlp = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=gelu_approx, drop=0)
    
    # if output of DWT (B, T, C) is transformed to (B, C, H, W)
    # its a tiled representation of different frequency components
    # first row being all LLs and last row being all HHs because the column major indexing into concatenation trick
    # DWT(0 index) is always LL
    def _dwt_fast(self, x):
        # implementation supports only two consecutive DWT transformations
        T = x.size(1) # x(B, T, C)
        h = w = int(np.sqrt(T))
        x = rearrange(x, "b (h w) c -> b c h w", h=h)
        subbands = self.dwt(x) # xll, xlh, xhl, xhh where each has shape of [B, 4C, h/2, W/2]
        scale = 2**self.num_wavelet_lv
        patch_size = scale # receptive patch size DWT
        if self.num_wavelet_lv > 1:
            out = (self.dwt(subbands) / scale).chunk(patch_size * patch_size, dim=1) # (B, 16C, h/4, w/4) -> 16x(B, C, h/4, w/4)
            indices = []
            for i in range(patch_size * patch_size):
                # normally: value = row * 4 + column
                # here we want to have all LL components first, -> column wise vertical scan
                # same indices, values transpose
                # hence value = column(i) * 4 + row
                # indices [0, 4, 8, 12, 1, 5...]
                indices.append(i % 4 * patch_size + i // 4)
            out = torch.cat([out[i] for i in indices], dim=1) # 16x(B, C, h/4, w/4) -> (B, 16C, h/4, w/4) but ordered from LL to HH along dim=1
        else:
            out = subbands / scale
        
        return rearrange(out, "b (c p1 p2) h w -> b (h p1 w p2) c", p1=patch_size, p2=patch_size) # (B, 16C, h/4, w/4) -> (B, HxW, C)
    
    # We refrain from using IDWT
    def _idwt_fast(self, x):
        scale = 2**self.num_wavelet_lv
        patch_size = scale
        lowest_size = int(np.sqrt(x.size(1))) // patch_size
        subbands = rearrange(
            x * scale, "b (h p1 w p2) c -> b (c p1 p2) h w", p1=patch_size, p2=patch_size, h=lowest_size
        ).chunk(patch_size * patch_size, dim=1)
        if self.num_wavelet_lv > 1:
            indices = []
            for i in range(patch_size * patch_size):
                indices.append(i % 4 * patch_size + i // 4)
            subbands = torch.cat([subbands[i] for i in indices], dim=1)
            out = self.idwt(subbands)
            out = self.idwt(out)
        else:
            out = self.idwt(torch.cat(subbands, dim=1))
        return rearrange(out, "b c h w -> b (h w) c")  # [b, c, h, w]
    

    def forward(
            self,
            hidden_states: Tensor,
            state_space_representation: Optional[Tensor] = None,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params = None,
    ):
        r"""Pass the input signal (B, T, C) through the block
        
        Args:
        hidden_states: (B, T, C) input signal to the block (required)
        residual: residual from previous block, hidden_states = Mixer(LN(residual)) in previous block
        we fuse add and then norm before branching residual and mixer
        y: tensor (B, C) label embeddings. 
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)

            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))

            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states,
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states),
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # not none here
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        # now I have residual <- supposed to be retured and hidden_states = norm(residual) <- supposed to go into mixer
        # shape (B, T, C)
        # transpose, scanning continuity, token reversal (for directional bias)
        
        # DWT block, processes frequeny domain representation of the signal
        # hidden_states <- signal (B, T, C) implicit (B, C, H, W)

        T = hidden_states.shape[1]
        h = w = int (np.sqrt(T)) # Original image size
        hidden_states = self._dwt_fast(hidden_states).contiguous() # (B, T, C) implicitly flattened in LL x4 LH x4 ... order
        patch_size = int(2**self.num_wavelet_lv)
        
        if self.window_scan:
            # perform a non overlaping window scan over each subband / freq component
            column_first = True if self.transpose else False # (column first scans LL LH HL HH of same component (LL0) tiled in the first column)
            # Internally Constructs tiled representation of frequency components, the same resolution as original signal (B, C, H, W) from (B, T, C)
            # Performs row_wise or column_wise scans in non overlapping windows.
            # each non overlapping window corresponds to a frequency component (subband) first being LL(LL(x)) = LL0
            hidden_states = local_scan(hidden_states, w=w//patch_size, H=h, W=w, column_first=column_first).contiguous()
        else:
            if self.transpose:
                hidden_states = rearrange(hidden_states, "b (h w) c -> b (w h) c", h=h, w=w)
        
        if self.scanning_continuity:
            # does not integrate with window scan over non overlapping windows (each corresponding to a subband (h=w=H/patchsize))
            # we'll set this pos False
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h-> b (w h) c", h=h, w=w) # (B, T, C)
        
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        

        if not self.no_ffn:
            # there is ffn
            scale_ssm, gate_ssm, shift_ssm, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1) #(B,C)->(B,6C)->6x(B,C)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), cond_emb=state_space_representation, inference_params=inference_params
            )
            hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
                modulate(self.norm_2(hidden_states), shift_mlp, scale_mlp)
            )
        else:
            scale_ssm, gate_ssm, shift_ssm = self.adaLN_modulation(y).chunk(3, dim=-1) # (B,C)->(B,3C)->3x(B,C)
            hidden_states = hidden_states + gate_ssm.unsqueeze(1) * self.mixer(
                modulate(hidden_states, shift_ssm, scale_ssm), cond_emb=state_space_representation, inference_params=inference_params
            )
        
        # transform back
        if self.reverse:
            hidden_states = hidden_states.flip(1)
        
        if self.scanning_continuity:
            hidden_states = rearrange(hidden_states.clone(), "b (w h) c -> b c w h", h=h, w=w)
            hidden_states[:, :, 1::2] = hidden_states[:, :, 1::2].flip(-1)
            hidden_states = rearrange(hidden_states, "b c w h -> b (w h) c", h=h, w=w)
        
        if self.window_scan:
            hidden_states = local_reverse(hidden_states, w=w // patch_size, H=h, W=w, column_first=column_first)
        else:
            if self.transpose:
                hidden_states = rearrange (hidden_states, "b (w h) c -> b (h w) c", h=h, w=w)
        
        # NOTE: we refrain from IDWT for reasons explained in the paper
        return hidden_states, residual
    

class MoEBlock(nn.Module):
    def __init__(self, dim, mixer_cls, norm_cls=nn.LayerNorm, fused_add_norm=False, residual_in_fp32=False):
        
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.mixer = mixer_cls(dim)
        self.norm = norm_cls(dim)
        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm only supports LayerNorm and RMSNorm"
        
    def forward(self, hidden_states:Tensor, residual: Optional[Tensor]=None, inference_params=None):
        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            hidden_states, residual = fused_add_norm_fn(
                hidden_states,
                self.norm.weight,
                self.norm.bias,
                residual=residual,
                prenorm=True,
                residual_in_fp32=self.residual_in_fp32,
                eps=self.norm.eps,
            )
        hidden_states = self.mixer(hidden_states)
        return hidden_states, residual
    
    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        return self.mixer.allocate_inference_cache(batch_size, max_seqlen, dtype=None, **kwargs)


class FlBlockCombined(nn.Module):
    def __init__(
            self,
            dim,
            mixer_cls,
            norm_cls=nn.LayerNorm,
            fused_add_norm=False,
            residual_in_fp32=False,
            drop_path=0.0,
            reverse=False,
            transpose=False,
            scanning_continuity=False,
            
            no_spatial_ffn=True,
            no_freq_ffn=True,
            no_ssr_ffn=True,
            no_adaLN_ssr=True,
            cross_fusion_mamba=True,
            fusion_depth=2,
            fusion_ssm_expansion_factor=2,
            norm_cross_mamba=False,
            concat_context_across_time=False,
            detach_context_across_time=False,

    ):
        """
        Block wrapping a parallel spatial and frequency ssm blocks with residual and hidden states from previous block.
        Still the standard prenorm block, just slightly different structure so we can fuse addition, back into residual pathway,
        and then normalization to branch towards the mixer.
        Purely for performance reasons as we can fuse add->norm
        Residual has to be specified unless its the very first block.
        """
        super().__init__()
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm
        self.reverse = reverse
        self.transpose = transpose
        self.scanning_continuity = scanning_continuity
        self.no_ssr_ffn = no_ssr_ffn
        self.cross_fusion_mamba = cross_fusion_mamba

        self.norm = norm_cls(dim)

        # TODO: Ablate here, each branch having access to only half the channels and then concatenation
        self.spatial_mamba = FlBlockRaw(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            norm_cls_ffn=norm_cls,
            drop_path=0.0,
            fused_add_norm=False, # for indentity norm
            residual_in_fp32=residual_in_fp32,
            reverse=reverse,
            transpose=transpose,
            scanning_continuity=scanning_continuity,
            y_dim=dim,
            no_ffn=no_spatial_ffn,
        )
        # TODO: Ablate they dont setup an MLP in freq mamba block maybe because relying on IDWT
        # By default FreqMamba does non overlapping window scan within freq subband
        self.freq_mamba = WaveFlBlock(
            dim // 2,
            mixer_cls,
            norm_cls=nn.Identity,
            norm_cls_ffn=norm_cls,
            drop_path=0.0,
            fused_add_norm=False, # for indentity norm
            residual_in_fp32=residual_in_fp32,
            reverse=False,
            transpose=reverse, # tranpose, # disable if only left to right scanning is used
            scanning_continuity=scanning_continuity,
            no_ffn=no_freq_ffn,
            y_dim=dim,
            num_wavelet_lv=2,
        )
        # TODO: Ablate
        if cross_fusion_mamba:
            self.proj = CrossFusionMamba(dim, depth=fusion_depth, ssm_expansion_factor=fusion_ssm_expansion_factor, enable_norm=norm_cross_mamba, concat_context_across_time=concat_context_across_time, detach_context_across_time=detach_context_across_time)
        else:
            print("CREATING ATTENTION!")
            self.proj = CrossAttentionFusion(dim, n_head=8, qkv_bias=True, swap_k=False)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        if self.fused_add_norm:
            assert RMSNorm is not None, f"RMSNorm import failed"
            assert isinstance(self.norm, (nn.LayerNorm, RMSNorm)), f"fused_add_norm is only supported for layer norm and rms norm"
        
        self.norm_2 = norm_cls(dim)

        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3*dim, bias=True))
        mlp_hidden_dim = int(dim * 4)
        approx_gelu = lambda:nn.GELU(approximate="tanh")
        self.mlp = GatedMLP (fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # NOTE: process recurrent state representation of the signal or not
        if not no_ssr_ffn:
            self.norm_ssr = norm_cls(dim)
            mlp_hidden_dim = int (4 * dim)
            approx_gelu = lambda:nn.GELU(approximate="tanh")
                        
            self.adaLN_modulation_ssr = nn.Sequential(nn.SiLU(), nn.Linear(dim, 3*dim, bias=True)) if not no_adaLN_ssr else None

            
            self.mlp_ssr = GatedMLP(fan_in=dim, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
    
    def forward(
            self,
            hidden_states: Tensor,
            state_space_representation: Tensor,
            residual: Optional[Tensor] = None,
            y: Optional[Tensor] = None,
            inference_params=None,
    ):
        
        """
            Pass the input signal (B, T, C) through the FlBlock (spatial, freq ssm, ssm fusion)
            
            Args:
                hidden_states: input sequence to the FlBlock (required)
                residual: residual from previous block. hidden_states = Mixer(LN(residual))
                state_space_representation: representation of the signal from previous block (B, C)

            We fuse add (residual + hidden_states) then norm, before branching residual mixer
        """
        if not self.fused_add_norm:
            if residual is None:
                residual = hidden_states
            else:
                residual = residual + self.drop_path(hidden_states)
            
            hidden_states = self.norm(residual) if isinstance(self.norm, nn.Identity) else self.norm(residual.to(dtype=self.norm.weight.dtype))
            if self.residual_in_fp32:
                residual = residual.to(torch.float32)
        else:
            fused_add_norm_fn = rms_norm_fn if isinstance(self.norm, RMSNorm) else layer_norm_fn
            if residual is None:
                hidden_states, residual = fused_add_norm_fn(
                    hidden_states, # no drop path since residual is None (very first block)
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
            else:
                hidden_states, residual = fused_add_norm_fn(
                    self.drop_path(hidden_states), # drop path since residual is present
                    self.norm.weight,
                    self.norm.bias,
                    residual=residual, # Not None
                    prenorm=True,
                    residual_in_fp32=self.residual_in_fp32,
                    eps=self.norm.eps,
                )
        
        # process the recurrent state representation from previous block OR Not
        # shape (B, C)
        if not self.no_ssr_ffn:
            if self.adaLN_modulation_ssr is not None:
                scale_ssr, gate_ssr, shift_ssr = self.adaLN_modulation_ssr(y).chunk(3, dim=-1) # (B, 3C) -> 3x(B, C)
                state_space_representation = state_space_representation + gate_ssr * self.mlp_ssr(
                        self.norm_ssr(state_space_representation)*(1 + scale_ssr) + shift_ssr
                    )
                    
            else:
                state_space_representation = state_space_representation + self.mlp_ssr(self.norm_ssr(state_space_representation))
        
        # Now we have hidden_states (B, T, C) = LN(residual), residual and provessed SSR
        x1, x2 = hidden_states.chunk(2, dim=-1) #2x(B, T, C/2)
        
        if state_space_representation is None:
            state_space_representation = y
        
        x1, _ = self.spatial_mamba(x1, state_space_representation, None, y, inference_params) # 2x(B, T, C/2)
        x2, _ = self.freq_mamba(x2, state_space_representation, None, y, inference_params) # 2x(B, T, C/2)
        
        # TODO: Ablation, residual connection over fusion layer
        if isinstance(self.proj, CrossFusionMamba):       
            x = self.proj(x1, x2) # (B, T, C)
        elif isinstance(self.proj, CrossAttentionFusion):
            x = self.proj(x1, x2)
        else:
            x = torch.cat((x1, x2), dim=-1)
            x = self.proj(x) # (B, T, C)
        
        hidden_states = hidden_states + x
        # FFN
        scale_mp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(3, dim=-1) #3x(B, C)

        hidden_states = hidden_states + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm_2(hidden_states), shift_mlp, scale_mp)
        )

        # IMP NOTE: remove asserts later
        #state_space_representation = hidden_states[:, -1, :]
        #assert len(state_space_representation.shape) == 2
        return hidden_states, state_space_representation, residual


class DiTBlock (nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.nomr2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda:nn.GELU(approximate="tanh")
        
        self.mlp = GatedMLP (fan_in=hidden_size, fan_h=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, y=None, **kwargs):
        scale_msa, gate_msa, shift_msa, scale_mlp, gate_mlp, shift_mlp = self.adaLN_modulation(y).chunk(6, dim=-1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.nomr2(x), shift_mlp, scale_mlp))
        return x

def create_block(
        d_model,
        norm_eps=1e-5,
        drop_path=0.0,
        rms_norm=False,
        residual_in_fp32=True,
        fused_add_norm=False,
        layer_idx=None,
        device=None,
        dtype=None,
        add_bias_linear=False,
        gated_linear_unit=True,
        routing_mode: str = "sinkhorn", # 'sinkhorn', 'top1', 'top2', 'sinkhorn_top2'
        num_moe_experts: int = 8,
        mamba_moe_layers: list = None,
        is_moe = False,
        block_type="linear",
        reverse = False,
        transpose = False,
        scanning_continuity = False,
        skip = False,
        
        ssm_cfg=None, # switch for "Arcee" vs "Zigma"
        scan_type="arcee_1",
        ssm_dstate=16,
        block_kwargs = {},
        lock_permutations=True,
):
    """
    Creates a block with specified mixer every even layer, and MoE ffn every odd layer
    """
    assert ssm_cfg in ["Arcee", "Zigma", "NotArcee"]
    assert scan_type in ["arcee_1", "zigma_1", "arcee_8", "zigma_8"]
    factory_kwargs = {"device": device, "dtype": dtype}
    norm_cls = partial (nn.LayerNorm if not rms_norm else RMSNorm, eps=norm_eps, **factory_kwargs)
    
    if layer_idx % 2 == 0 or not is_moe:
        if ssm_cfg == "Arcee":
            mixer_cls = partial(
                ArceeMamba,
                d_state=ssm_dstate,
                layer_idx=layer_idx,
                lock_permutations=lock_permutations,
                **block_kwargs,
                **factory_kwargs,
            )
        else:
            mixer_cls = partial (Mamba, d_state=ssm_dstate, layer_idx=layer_idx, **block_kwargs, **factory_kwargs) # block_kwargs contain scan_type

        if block_type == "normal":
            block = Block(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                drop_path=drop_path,
            )
        elif block_type == "combined":
            # NOTE: This is what we instrument
            block = FlBlockCombined(
                d_model,
                mixer_cls,
                norm_cls=norm_cls,
                fused_add_norm=fused_add_norm,
                residual_in_fp32=residual_in_fp32,
                reverse=reverse,
                transpose=transpose,
                # NOTE: always false for our experiments
                scanning_continuity=scanning_continuity,
            )
    else:
        # every odd layer
        mixer_cls = partial(
            SwitchMLP,
            layer_idx=layer_idx,
            add_bias_linear=add_bias_linear,
            gated_linear_unit=gated_linear_unit,
            routing_mode=routing_mode,
            num_moe_experts=num_moe_experts,
            mamba_moe_layers=mamba_moe_layers,
        )

        block = MoEBlock(
            d_model,
            mixer_cls=mixer_cls,
            norm_cls=norm_cls,
            fused_add_norm=fused_add_norm,
            residual_in_fp32=residual_in_fp32,
        )
    block.layer_idx = layer_idx
    return block

class Arcee(nn.Module, PyTorchModelHubMixin):
    def __init__(
            self,
            ssm_cfg=None, # Arcee or Zigma baseline
            ssm_dstate=16, # dstate

            img_resolution=32, # spatial resolution in vae latent space default: 256 // 8 = 32, orig res // 8
            patch_size=2,
            in_channels=4, # in channels of input, 3 (RGB) or 4 latent space of stabilityai vae
            hidden_size=1024,
            depth=16,
            label_dropout=0.1, # dropout prob for randomly dropping class labels for classifier free guidance (CFG)
            num_classes=1000, # classes in imagenet
            learn_sigma=False, # we donot predict beta_t, interpolation factor of pinit in transport from (pinit to pdata)
            rms_norm=False,
            residual_in_fp32=True,
            fused_add_norm=False, # disable add norm fused kernel while prototyping, harder to trace
            scan_type="none", 
            block_type="normal", 

            initializer_cfg=None,
            num_moe_experts=8,
            mamba_moe_layers=None, 
            add_bias_linear=False, # Dont bother its for MoE
            gated_linear_unit=True, # Dont bother its for MoE
            routing_mode="top1",
            # TODO: verify use of MoE in backbone setup, otherwise weight init would break from init(block.adaln[-1]) moe doesnt have adaln
            is_moe=False,
            pe_type="ape", # absolute positional encoding as default
            # TODO: make sure is disabled in waveFLBLock (with DWT)
            scanning_continuity=False,
            learnable_pe=False,
            skip=False,
            drop_path=0.0,
            use_final_norm=False,
            use_attn_every_k_layers=-1,
            use_independent_attn=False, 
    ):
        super().__init__()
        self.depth = depth
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels # output two feature maps if we're predicting both vectorfield and pinit coefficient beta_t in flow construction
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.initializer_cfg = initializer_cfg
        self.fused_add_norm = fused_add_norm
        self.residual_in_fp32 = residual_in_fp32
        self.add_before = False
        self.use_attn_every_k_layers = use_attn_every_k_layers
        self.ssm_cfg = ssm_cfg

        self.use_independent_attn = use_independent_attn
        
        if self.use_independent_attn:
            num_transformer_blocks = (self.depth // use_attn_every_k_layers) - 1 # -1 to rectify for not starting with txfrmr
            self.depth = self.depth - num_transformer_blocks
        
        #APE
        self.pe_type = pe_type
        self.block_type = block_type

        self.x_embedder = PatchEmbed(img_resolution, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        # caption embedder internally handles dropout for cfg training
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, label_dropout)
        num_patches = self.x_embedder.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=learnable_pe)

        if self.pe_type == "rope":
            self.emb_sin, self.emb_cos = get_2d_sincos_rotary_embed(hidden_size, int(num_patches**0.5))
            self.emb_sin = torch.from_numpy(self.emb_sin).to(dtype=torch.float32)
            self.emb_cos = torch.from_numpy(self.emb_cos).to(dtype=torch.float32)
        elif self.pe_type == "cpe":
            self.pos_cnn = AdaInPosCNN(hidden_size, hidden_size)
        
        
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.depth)] # stochastic depth decay rule

        inter_dpr = [0.0] + dpr
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        grid_size = int(math.sqrt(num_patches))

        def gen_paths (N, scan_type):
            path_type = scan_type.split("_")[0]
            num_paths = int(scan_type.split("_")[1])
            if path_type == "arcee":
                path_type = "zigma" # Same base scan (zigzag)

            path_gen_fn = SCAN_ZOO[path_type]
            zz_paths = path_gen_fn(N)[:num_paths]

            zz_paths_rev = [reverse_permut_np(x) for x in zz_paths] 
            
            zz_paths = torch.stack([torch.from_numpy(x) for x in zz_paths], dim=0).long()
            zz_paths_rev = torch.stack([torch.from_numpy(x) for x in zz_paths_rev], dim=0).long()

            assert len (zz_paths) == len(zz_paths_rev), f"{len(zz_paths)} != {len(zz_paths_rev)}"

            block_kwargs = {}
            block_kwargs["zigzag_paths"] = zz_paths
            block_kwargs["zigzag_paths_reverse"] = zz_paths_rev
            block_kwargs["scan_type"] = scan_type
            return block_kwargs
        
        self.lock_permutations = False
        self.single_path = None
        scan_type = scan_type.lower()
        self.scan_type = scan_type
        if scan_type.startswith("arcee") or scan_type.startswith("zigma"):
                block_kwargs = gen_paths(grid_size, scan_type)
                if scan_type == "arcee_1":
                    self.lock_permutations = True
                    self.single_path = {}
                    self.register_buffer("locked_permutation_path", block_kwargs["zigzag_paths"])
                    self.register_buffer("locked_permutation_path_r", block_kwargs["zigzag_paths_reverse"])
                    # TODO: remove redundant asserts
                    assert self.locked_permutation_path.shape[0] == 1
                    assert self.locked_permutation_path.shape[1] == int (grid_size * grid_size)
                    assert self.locked_permutation_path_r.shape[0] == 1
                    assert self.locked_permutation_path_r.shape[1] == int (grid_size * grid_size)
                    
        else:
            block_kwargs = {}
        print(f"\n\tRegistered scan_type {scan_type}")
        print (f"\tPermutations locked:{self.lock_permutations}")
        
        
        self.blocks = nn.ModuleList(
            [
                create_block(
                    hidden_size,

                    norm_eps=1e-5,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    layer_idx=i,
                    drop_path=inter_dpr[i],
                    num_moe_experts=num_moe_experts,
                    mamba_moe_layers=mamba_moe_layers,
                    add_bias_linear=add_bias_linear,
                    gated_linear_unit=gated_linear_unit,
                    routing_mode=routing_mode,
                    is_moe=is_moe,
                    block_type=block_type, # (normal, combined)
                    reverse=False,
                    transpose=False,
                    scanning_continuity=scanning_continuity,

                    ssm_cfg=ssm_cfg, # define mixer class (Arcee, Zigma)
                    ssm_dstate = ssm_dstate,
                    scan_type=scan_type, # (arcee_1, arcee_8, zigma_8)
                    block_kwargs=block_kwargs,
                    lock_permutations=self.lock_permutations,
                )
                for i in range(self.depth)
            ]
        )

        if self.use_attn_every_k_layers > 0:
            print ("CREATING ATTENTION LAYERS!!!!!!!!")
            if self.use_independent_attn:
                self.attn_block = nn.ModuleList(
                    [DiTBlock(hidden_size, 16) for i in range(num_transformer_blocks)]
                )
            else:
                self.attn_block = DiTBlock(hidden_size, 16)
        
        if use_final_norm:
            self.norm_f = nn.LayerNorm(hidden_size, eps=1e-5) if not rms_norm else RMSNorm(hidden_size, eps=1e-5)
        else:
            self.norm_f = None
        
        
        if ssm_cfg == "Arcee":
            self.last_state_weaver = LastStateWeaver(hidden_size, self.blocks[0].mixer.d_inner, ssm_dstate)

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def count_residuals(self):
        count = 0
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if hasattr(module, "RESIDUAL_ADDITION") and module.RESIDUAL_ADDITION:
                    count += 1
        print (f"\n\tTOTAL RESIDUAL ADDITIONS : {count}")
        return count
    
    def initialize_weights(self):
        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view(w.shape[0], -1)) # same as w.flatten(1,-1)

        # Initialize label embedding table:
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # zero-out adaLN modulation layers in blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        
        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.ssm_cfg == "Arcee":
            nn.init.constant_(self.last_state_weaver.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(self.last_state_weaver.adaLN_modulation[-1].bias, 0)

        # Mamba init
        n_residuals = self.count_residuals()
        self.apply(
            partial(
                _init_weights,
                n_residuals=n_residuals,
                **(self.initializer_cfg if self.initializer_cfg is not None else {}),
            )
        )
    
    def forward(self, x, t, y=None, initial_state=None):
        """
        Forward pass of the ARCEE backbone

        x : (B, C, H, W) tensor of spatial inputs (images or latent representations of images) (flattened internally into B,T,C in the RC blocks)
        t: (B,)
        y: (B,)
        """

        if t is None:
            # for computing Gflops, t=None semantics sound better for meaningless forward pass
            # we never evaluate vectorfield ut_theta outside t in [0, 1]
            t = torch.randint(0, 1000, (x.shape[0],), device=x.device)
        
        if y is None:
            y = torch.ones(x.size(0), dtype=torch.long, device=x.device) * (self.y_embedder.get_in_channels() - 1)
            # y = tensor of index into last row in the label embedder, null label
        
        t = self.t_embedder(t) # (B, C)
        y = self.y_embedder(y, self.training) # (B, C)
        c = t + y # Modulate time with class label (B, C)
        # add positional information
        if self.pe_type == "ape":
            x = self.x_embedder(x) + self.pos_embed # (B, T, C) where T = H x W / patch_size**2
            # NOTE: we have T = 256 for 256x256 images with patch size = 2 in latent space of stability-ai vae
        elif self.pe_type == "rope":
            self.emb_cos = self.emb_cos.to(x.device)
            self.emb_sin = self.emb_sin.to(x.device)
            x = apply_rotary(self.x_embedder(x), self.emb_sin, self.emb_cos)
        elif self.pe_type == "cpe":
            x = self.x_embedder(x)
            h = w = int(self.x_embedder.num_patches**0.5) # root(T)
            x = self.pos_cnn(x, c, H=h, W=w)
        else:
            raise("Unsupported PE")
        

        if self.lock_permutations:
            # assert that len(zzpaths)==1 and permute once
            # TODO: Remove redundant asserts
            assert self.single_path is not None and self.locked_permutation_path.shape[0] == 1 and self.locked_permutation_path_r.shape[0] == 1
            _perm = self.locked_permutation_path[0]
            x = torch.gather(x, 1, _perm[None, :, None].expand_as(x)) # x (B, T, C)

        # freq_residual = None
        
        residual = None
        debug_h0 = None
        
        assert initial_state is None
        for idx, block in enumerate(self.blocks):
            x, residual, last_state = block (x, residual, initial_state=initial_state, return_last_state=True if self.ssm_cfg == "Arcee" else False, y=c)

            if self.ssm_cfg == "Arcee":
                assert last_state is not None
            else:
                assert last_state is None
                    
            if initial_state is None:
                initial_state = last_state
            else:
                initial_state = initial_state + last_state


            if self.use_attn_every_k_layers > 0 and (idx + 1) % self.use_attn_every_k_layers == 0:
                if self.use_independent_attn:
                    attn_idx = int((idx+1) // self.use_attn_every_k_layers - 1)
                    x = self.attn_block[attn_idx](x, c)
                else:
                    x = self.attn_block(x, c)
        
        
        if self.lock_permutations:
            _rperm = self.locked_permutation_path_r[0]
            x = torch.gather(x, 1, _rperm[None, :, None].expand_as(x)) # x (B, T, C)
        
        if self.norm_f is not None:
            if not self.fused_add_norm:
                if residual is None:
                    residual = x
                else:
                    residual = residual + self.drop_path(x)
                x = self.norm_f(residual.to(dtype=self.norm_f.weight.dtype))
            else:
                # set prenorm = False here since we don't need the residual, norm_f is post add norm / post norm
                fused_add_norm_fn = rms_norm_fn if isinstance(self.norm_f, RMSNorm) else layer_norm_fn
                x = fused_add_norm_fn(
                    self.drop_path(x),
                    self.norm_f.weight,
                    self.norm_f.bias,
                    residual=residual,
                    prenorm=False,
                    residual_in_fp32=residual,
                    eps=self.norm_f.eps,
                )
        if self.ssm_cfg == "Arcee":
            x = self.last_state_weaver(x, last_state)

        x = self.final_layer(x, c) # (B, T, patch_size*patch_size * outchannels) outchannels = 4 for vae latent space
        x = self.unpatchify(x) # (B, out_channels, H, W)
        return x
    
    def forward_with_cfg(self, x, t, y=None, initial_state=None, cfg_scale=1.0, **kwargs):
        """
            forward pass of ARCEE, batches the unconditional forward pass for classifier-free guidance.
        """

        half = x[:len(x) // 2] 
        combined = torch.cat((half, half), dim=0) # (2B, C, H, W)
        model_out = self.forward(combined, t, y, initial_state=initial_state) # (2B, C, H, W)
        # For exact reproducibility reasons, we apply classifier-free guidance on only
        # three channels by default. The standard approach to cfg applies it to all channels.
        # this can be done by uncommenting the following line and commenting-out the line following that.
        vector_field, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        #eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_vf, uncond_vf = torch.split(vector_field, len(vector_field) // 2, dim=0) # 2x(B, C, H, W)
        net_vector_field = uncond_vf + cfg_scale * (cond_vf - uncond_vf)
        vector_field = torch.cat([net_vector_field, net_vector_field], dim=0)
        return torch.cat([vector_field, rest], dim=1)
    

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum("nhwpqc->nchpwq", x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs


# GPT-2 init, same as Andrej's initialization. constant scaled down variance for each block weights, not depthwise init
# https://github.com/huggingface/transformers/blob/c28d04e9e252a1a099944e325685f14d242ecdcd/src/transformers/models/gpt2/modeling_gpt2.py#L454



def _init_weights(
    module,
    n_residuals=None,
    #n_layer,
    initializer_range=0.02,  # Now only used for embedding layer.
    rescale_prenorm_residual=True,
    #n_residuals_per_layer=1,  # Change to 2 if we have MLP
):
    if isinstance(module, nn.Linear):
        if module.bias is not None:
            # don't override the HIPPO/S4 inspired initialization
            # TODO: once cross check weight init with mamba repo
            if not hasattr(module.bias, "_no_reinit"):
                nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        # Reinitialize selected weights subject to the OpenAI GPT-2 Paper Scheme:
        #   > A modified initialization which accounts for the accumulation on the residual path with model depth. Scale
        #   > the weights of residual layers at initialization by a factor of 1/√N where N is the # of residual layers.
        #   >   -- GPT-2 :: https://openai.com/blog/better-language-models/
        #
        # Reference (Megatron-LM): https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/model/gpt_model.py
        if isinstance(module, nn.Linear):
            if hasattr(module, "RESIDUAL_ADDITION") and module.RESIDUAL_ADDITION:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5))
                with torch.no_grad():
                    module.weight /= math.sqrt(n_residuals)

                

        



        

def drop_path (x, drop_prob: float = 0.0, training: bool = False, scale_by_keep: bool = True):
    """
    Randomly kill all activations for some examples and scale up survivors to retain same expected value, so we dont have to upscale
    during inference when dropout is disabled

    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)

    This is the same as the DropConnect implementation for EfficientNet, however, the original name is misleading
    as 'Drop Connect' is a different form of droupout in a separate paper.
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956
    This implementation opts to changing the layer and the argument names to 'drop path' and 'drop_prob' rather than
    DropConnect and survival_rate respectively
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1) # (B, T, C) -> (B, 1, 1) so that works with diff dim tensors, not just 2D convnets
    # new tensor same device and dtype as x
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob) # A tensor of shape (B, [1]*(x.ndim-1)) containing 0/1 with keep_prob; broad castable
    
    # if we dont scale_by_keep, during training mask * x scales output down by keep_prob on average
    # to rectify we have to multiply by keep_prob during inference
    # with scaling (inverted dropout) mask = mask / keep_prob, it automatically rectifies by scaling surviving units up by keep_prob
    # so the expected value stays the same during train and test time
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob) 
    return x * random_tensor



class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample/ example level dropout (when applied in main path of residual blocks)
    Kills all activations across all tokens/channels for some random examples in a batch
    """

    def __init__(self, drop_prob: float=0.0, scale_by_keep:bool=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep
    
    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)
    
    def extra_repr(self):
        return f"drop_prob={round(self.drop_prob,3):0.3f}"


def Arcee_XS_2(**kwargs):
    return Arcee(
        depth = 24,
        hidden_size = 448,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_XS_2(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 448,
        patch_size = 2,
        initializer_cfg=None,
        ssm_cfg="Zigma",
        **kwargs,
    )

def Arcee_XS_1(**kwargs):
    return Arcee(
        depth = 24,
        hidden_size = 368,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_XS_1(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 368,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs, 
    )

def Arcee_B_2(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 768,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_B_2(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 768,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )



def Arcee_B_1(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 768,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_B_1(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 768,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )


def Arcee_XB_2(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 1024,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_XB_2(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 1024,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )


def Arcee_XB_1(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 1024,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_XB_1(**kwargs):
    return Arcee (
        depth = 24,
        hidden_size = 1024,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )

def Arcee_L_1(**kwargs):
    return Arcee(
        depth = 48,
        hidden_size = 1024,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_L_1(**kwargs):
    return Arcee(
        depth = 48,
        hidden_size = 1024,
        patch_size = 1,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )

def Arcee_L_2(**kwargs):
    return Arcee(
        depth = 48,
        hidden_size = 1024,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Arcee",
        **kwargs,
    )

def Zigma_L_2(**kwargs):
    return Arcee(
        depth = 48,
        hidden_size = 1024,
        patch_size = 2,
        initializer_cfg = None,
        ssm_cfg = "Zigma",
        **kwargs,
    )



Models = {
    "Arcee-XS/2" : Arcee_XS_2,
    "Arcee-XS/1" : Arcee_XS_1,
    "Arcee-B/2"  : Arcee_B_2,
    "Arcee-B/1"  : Arcee_B_1,
    "Arcee-XB/2" : Arcee_XB_2,
    "Arcee-XB/1" : Arcee_XB_1,
    "Arcee-L/2"  : Arcee_L_2,
    "Arcee-L/1"  : Arcee_L_1,

    #Zigma models
    "Zigma-XS/2" : Zigma_XS_2,
    "Zigma-XS/1" : Zigma_XS_1,
    "Zigma-B/2"  : Zigma_B_2,
    "Zigma-B/1"  : Zigma_B_1,
    "Zigma-XB/2" : Zigma_XB_2,
    "Zigma-XB/1" : Zigma_XB_1,
    "Zigma-L/2"  : Zigma_L_2,
    "Zigma-L/1"  : Zigma_L_1,
}