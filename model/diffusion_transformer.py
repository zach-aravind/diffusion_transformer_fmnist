import torch
import torch.nn as nn
from .components import PatchEmbed, DiTBlock, FinalLayer
from ..utils.helpers import SinusoidalPosEmb

class DiffusionTransformer(nn.Module):
    def __init__(
        self,
        img_size=32,
        patch_size=4,
        in_channels=1,
        latent_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        num_classes=10,
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # Index for the null / unconditional token for CFG
        self.null_label_idx = num_classes

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=latent_dim
        )
        self.num_patches = self.patch_embed.num_patches

        # 2. Positional Embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, latent_dim))

        # 3. Timestep Embedding
        mlp_hidden_dim_time = int(latent_dim * mlp_ratio)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, mlp_hidden_dim_time),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim_time, latent_dim),
        )

        # Class Label Embedding (size num_classes + 1 for null token)
        self.label_embed = nn.Embedding(self.num_classes + 1, latent_dim)

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(latent_dim, patch_size, self.out_channels)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Positional embedding
        nn.init.normal_(self.pos_embed, std=0.02)
        # Timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)
        # Initialize label embedding
        nn.init.normal_(self.label_embed.weight, std=0.02)
        # Initialize DiT blocks
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight)
            nn.init.zeros_(block.attn.out_proj.bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)
            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.zeros_(block.mlp[0].bias)
            nn.init.xavier_uniform_(block.mlp[2].weight)
            nn.init.zeros_(block.mlp[2].bias)
            nn.init.constant_(block.adaLN_modulation[1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[1].bias, 0)
        # Initialize final layer
        nn.init.constant_(self.final_layer.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x):
        B, N, PP_C = x.shape
        P = self.patch_size
        C_out = self.out_channels
        H_patch = W_patch = int(N**0.5)
        assert H_patch * W_patch == N, "Num patches must form square grid."
        H, W = H_patch * P, W_patch * P
        x = x.reshape(B, H_patch, W_patch, P, P, C_out)
        x = torch.einsum('bhwpqc->bchpwq', x)
        imgs = x.reshape(B, C_out, H, W)
        return imgs

    def forward(self, x, t, y):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        t_emb = self.time_embed(t.float())  # (B, D)
        y_emb = self.label_embed(y)         # (B, D) - Embed labels (could be true or null)
        combined_emb = t_emb + y_emb        # (B, D) - Combine embeddings

        for block in self.blocks:
            x = block(x, combined_emb)      # Pass combined embedding

        x = self.final_layer(x, combined_emb)  # Pass combined embedding
        x = self.unpatchify(x)
        return x