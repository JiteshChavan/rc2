import torch
import torch.nn as nn


from mamba_ssm.modules.mamba_simple import ArceeCondMamba, CondMamba

class CrossFusionMamba(nn.Module):
    def __init__(
            self,
            d_model,
            depth,
            ssm_expansion_factor=None,
            enable_norm=False,
            concat_context_across_time=False,
            detach_context_across_time=False,
    ):
        super().__init__()
        self.d_model = d_model
        self.depth = depth
        self.ssm_expansion_factor = ssm_expansion_factor

        self.space_ssm_stream = nn.ModuleList(
            [
                ArceeCondMamba(
                    d_model=d_model//2,
                    d_cond = d_model // 2,
                    scan_type="none",
                    expand = ssm_expansion_factor,
                    concat_context_across_time=concat_context_across_time,
                    detach_context_across_time=detach_context_across_time,
                )
                for _ in range (depth)
            ] 
        )

        self.freq_ssm_stream = nn.ModuleList(
            [
                ArceeCondMamba(
                    d_model=d_model // 2,
                    d_cond = d_model // 2,
                    scan_type="none",
                    expand= ssm_expansion_factor,
                    concat_context_across_time=concat_context_across_time,
                    detach_context_across_time=detach_context_across_time,
                )
                for _ in range(depth)   
            ]
        )
        
        
        self.norm_1 = nn.LayerNorm(d_model // 2) if enable_norm else nn.Identity()
        self.norm_2 = nn.LayerNorm(d_model // 2) if enable_norm else nn.Identity()
        self.proj = nn.Linear(d_model, d_model)
        self.proj.NANO_GPT_SCALE_INIT = True

    def forward(self, space, freq, inference_params=None):
        
        
        space = self.norm_1(space)
        freq = self.norm_2(freq)

        ssr_space = space[:, -1, :]
        ssr_freq = freq[:, -1, :]
        for i in range(self.depth):
            space =  self.space_ssm_stream[i](space, cond_emb=ssr_freq, inference_params=inference_params)
            freq =  self.freq_ssm_stream[i](freq, cond_emb=ssr_space, inference_params=inference_params)
            
            ssr_space = space[:, -1, :]
            ssr_freq = freq[:, -1, :]
            assert len(ssr_space.shape) == 2
            assert len(ssr_freq.shape) == 2
        
        x = torch.cat((space, freq), dim=-1)
        assert x.shape[-1] == self.d_model

        x = self.proj(x)

        return x
