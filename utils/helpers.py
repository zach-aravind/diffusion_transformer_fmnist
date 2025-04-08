import torch
import torch.nn as nn
import math
from einops import rearrange

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# AdaLN-Zero Helper Function (Modulation)
def modulate(x, shift, scale):
    if x.ndim == 3:  # Sequence input B, N, D
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    # Add dim for broadcasting: (B, D) -> (B, 1, D) or keep as (B, D) if x is (B, D)
    return x * (1 + scale) + shift

def unnormalize_to_zero_to_one(t):
    return (t.clamp(-1, 1) + 1) * 0.5


# Sinusoidal Position Embedding (for Timesteps)
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        if self.dim % 2 == 1:
            embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings