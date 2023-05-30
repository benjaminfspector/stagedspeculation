# Also largely stolen from Tri.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .mha import MHA
from .mlp import MLP

class Block(nn.Module):

    def __init__(self, dim, nheads, hidden_features=None, norm_cls=nn.LayerNorm, layer_idx=None, prenorm=True, mha_kwargs={}):
        """
        For prenorm=True, we have:
        LN -> MHA -> Add -> LN -> MLP -> Add.
        For prenorm=False, we have:
        MHA -> Add -> LN -> MLP -> Add -> LN.
        return_residual: whether each of the sub-layers (mixer and mlp) will return the residual.
        This is for performance reason: for post-norm architecture, returning the input allows us
        to fuse the backward of nn.Linear with the residual connection.
        """
        super().__init__()
        self.prenorm = prenorm
        self.mha = MHA(dim, nheads, layer_idx=layer_idx, causal=True, **mha_kwargs)
        self.norm1 = norm_cls(dim, dtype=torch.bfloat16)
        self.mlp = MLP(dim, hidden_features)
        self.norm2 = norm_cls(dim, dtype=torch.bfloat16)

    def forward(self, hidden_states: Tensor):
        """Pass the input through the encoder layer.
        Args:
            hidden_states: the sequence to the encoder layer (required).
            residual: if postnorm, residual=None, If prenorm, hidden_states = Attn/MLP(LN(residual))
            mixer_subset: for cross-attention only. If not None, will take a subset of x
                before applying the query projection. Useful for e.g., ViT where we only care
                about the CLS token in the last layer.
        """
        if self.prenorm:
            x = self.norm1(hidden_states)
            x = self.mha(x)
            hidden_states += x
            x = self.norm2(hidden_states)
            x = self.mlp(x)
            hidden_states += x
            return hidden_states
        else:
            x = self.mha(hidden_states)
            hidden_states += x
            x = self.norm1(hidden_states)
            x = self.mlp(x)
            hidden_states += x
            return self.norm2(hidden_states)
    
    def decode(self, hidden_states: Tensor, causal_mask: Tensor):
        if self.prenorm:
            x = self.norm1(hidden_states)
            x = self.mha.decode(x, causal_mask)
            hidden_states += x
            x = self.norm2(hidden_states)
            x = self.mlp(x)
            hidden_states += x
            return hidden_states
        else:
            x = self.mha.decode(hidden_states, causal_mask)
            hidden_states += x
            x = self.norm1(hidden_states)
            x = self.mlp(x)
            hidden_states += x
            return self.norm2(hidden_states)
        
    def update_kv(self, i):
        self.mha.update_kv(i)

    def truncate_kv(self, pos):
        self.mha.truncate_kv(pos)