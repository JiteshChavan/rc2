

import torch
import torch.nn as nn


class StateModulators(nn.Module):
    """
        Args : 
        last_state (b, d_inner, dstate)
        shared_dstate_collapse : 
            True -> Same collapse dstate -> k for all d_inner channels (cheaper)
            False -> per channel collapse (d_inner x dstate -> k) for each d_inner channels (more expressive)
        out: scale, gate, shift or scale/shift
    """
    def __init__(self, dstate, d_inner, shared_dstate_collapse=True, n_mods=2):
        super().__init__()
        self.dstate = dstate
        self.d_inner = d_inner
        self.shared = shared_dstate_collapse
        self.n_mods = n_mods
        if self.shared:
            self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(dstate, n_mods, bias=True))
            nn.init.zeros_(self.adaLN_modulation[-1].weight)
            nn.init.zeros_(self.adaLN_modulation[-1].bias)
            self.adaLN_modulation[-1]._no_reinit = True
        else:
            # torch.einsum ('bds, dsk -> bdk') + 'dk'
            # module.apply ignores nn.Parameters only traverses submodules so we dont need to have a no reinit flag here
            self.act = nn.SiLU()
            self.modulation_weight = nn.Parameter(torch.zeros(self.d_inner, self.dstate, n_mods))
            self.modulation_bias = nn.Parameter(torch.zeros(self.d_inner, n_mods))
    
    def forward(self, last_state):
        if self.shared:
            mods = self.adaLN_modulation(last_state) # (B, d_inner, dstate)
        else:
            last_state = self.act(last_state) # (B, d_inner, dstate)
            mods = torch.einsum('b d s, d s k -> b d k', last_state, self.modulation_weight) + self.modulation_bias
        

        out = mods.unbind(-1)
        if len(out) == 2:
            scale, shift = out
            return scale, shift
        elif len(out) == 3:
            scale, gate, shift = out
            return scale, gate, shift
        else:
            raise NotImplementedError("Modulator(last_state) only supports scale,gate,shift (3Mods) OR scale,shift (2Mods)")

