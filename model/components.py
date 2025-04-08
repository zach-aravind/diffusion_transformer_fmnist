import torch
import torch.nn as nn
from ..utils.helpers import pair, modulate

# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = pair(img_size)
        patch_size = pair(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, f"Image dimensions ({H}x{W}) must be divisible by patch size ({ph}x{pw})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

# DiT Block
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim), nn.GELU(), nn.Linear(mlp_hidden_dim, hidden_size),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):  # c is the combined time+label embedding
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_params.chunk(6, dim=1)
        x_norm1_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(x_norm1_modulated, x_norm1_modulated, x_norm1_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_output
        x_norm2_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x

# Final Layer for DiT
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):  # c is the combined time+label embedding
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x