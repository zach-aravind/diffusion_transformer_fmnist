import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
import os
# import matplotlib.pyplot as plt # Matplotlib wasn't used in the final script structure, can be added if needed for plotting loss etc.

# --- Configuration ---
IMG_SIZE = 32  # Resize Fashion MNIST 28x28 to 32x32 for easier patching
PATCH_SIZE = 4  # Patch size for ViT embedding
IMG_CHANNELS = 1 # Fashion MNIST is grayscale
LATENT_DIM = 256 # Dimension of the transformer embedding space
NUM_HEADS = 4   # Number of attention heads
TRANSFORMER_DEPTH = 6 # Number of transformer blocks
MLP_RATIO = 4.0 # Ratio for MLP hidden dimension in blocks
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 50 # Increase for better results
DIFFUSION_TIMESTEPS = 1000 # Number of diffusion steps
MODEL_SAVE_PATH = "dit_fashion_mnist.pth"
SAMPLE_SAVE_DIR = "samples_fashion_mnist"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create sample directory
os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)

# --- Helper Functions ---

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

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
        # Ensure embedding dim is correct if original dim was odd
        if self.dim % 2 == 1:
             embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

# Patch Embedding
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
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
        # Allow for dynamic image size, checking compatibility with patch size
        ph, pw = self.patch_size
        assert H % ph == 0 and W % pw == 0, \
            f"Input image size ({H}x{W}) must be divisible by patch size ({ph}x{pw})."
        x = self.proj(x).flatten(2).transpose(1, 2) # B, C, H, W -> B, embed_dim, H', W' -> B, embed_dim, N -> B, N, embed_dim
        return x

# AdaLN-Zero Helper Function (Modulation)
def modulate(x, shift, scale):
    # x: (B, N, D) or (B, D)
    # shift, scale: (B, D)
    # Add dim if x is sequence: (B, D) -> (B, 1, D) for broadcasting
    if x.ndim == 3:
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    return x * (1 + scale) + shift

# --- DiT Block ---
# Based on https://github.com/facebookresearch/DiT/blob/main/models.py
class DiTBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True) # Use torch MHA
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, hidden_size),
        )
        # AdaLN-Zero projection for scale/shift parameters
        # Need 6 * hidden_size output for: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size)
        )

    def forward(self, x, c):
        # x: (B, N, D) - sequence
        # c: (B, D)    - conditioning (time embedding)

        # Project conditioning vector c to get modulation parameters
        # Each parameter set has shape (B, D)
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_params.chunk(6, dim=1)

        # --- Attention Block ---
        # Modulate Norm1 -> (B, N, D) * (1 + (B, 1, D)) + (B, 1, D) = (B, N, D)
        x_norm1_modulated = modulate(self.norm1(x), shift_msa, scale_msa)

        # Attention (using torch.nn.MultiheadAttention)
        # Note: PyTorch MHA expects (Batch, Seq, Dim) if batch_first=True
        attn_output, _ = self.attn(x_norm1_modulated, x_norm1_modulated, x_norm1_modulated)

        # Apply gate and add residual connection: x + gate * attn_output
        # gate has shape (B, D), needs unsqueezing for broadcasting: (B, 1, D)
        x = x + gate_msa.unsqueeze(1) * attn_output

        # --- MLP Block ---
        # Modulate Norm2 -> (B, N, D) * (1 + (B, 1, D)) + (B, 1, D) = (B, N, D)
        x_norm2_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)

        # MLP
        mlp_output = self.mlp(x_norm2_modulated)

        # Apply gate and add residual connection: x + gate * mlp_output
        # gate has shape (B, D), needs unsqueezing for broadcasting: (B, 1, D)
        x = x + gate_mlp.unsqueeze(1) * mlp_output

        return x

# --- Final Layer for DiT ---
# Also uses AdaLN-Zero before output projection
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        # Need 2 * hidden_size for shift, scale
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c):
        # x: (B, N, D)
        # c: (B, D)
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1) # (B, D), (B, D)
        # Modulate -> (B, N, D) * (1 + (B, 1, D)) + (B, 1, D) = (B, N, D)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x) # (B, N, P*P*C_out)
        return x


# --- Diffusion Transformer Model ---
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
        learn_sigma=False # Not implemented here, predicts epsilon (noise) only
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim # Make latent_dim accessible

        # 1. Patch Embedding
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_channels, embed_dim=latent_dim
        )
        # Need num_patches after PatchEmbed is initialized
        self.num_patches = self.patch_embed.num_patches

        # 2. Positional Embedding (Learnable)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, latent_dim))

        # 3. Timestep Embedding
        # *** FIX: Ensure MLP dimensions are integers ***
        mlp_hidden_dim = int(latent_dim * mlp_ratio)
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(latent_dim),
            nn.Linear(latent_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, latent_dim),
        )

        # 4. Transformer Blocks
        self.blocks = nn.ModuleList([
            DiTBlock(latent_dim, num_heads, mlp_ratio=mlp_ratio) for _ in range(depth)
        ])

        # 5. Final Layer
        self.final_layer = FinalLayer(latent_dim, patch_size, self.out_channels)

        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize positional embedding and AdaLN modulation layers
        nn.init.normal_(self.pos_embed, std=0.02)

        # Initialize timestep embedding MLP
        nn.init.normal_(self.time_embed[1].weight, std=0.02)
        nn.init.normal_(self.time_embed[3].weight, std=0.02)

        # Initialize DiT blocks: Linear weights and AdaLN bias/weights
        for block in self.blocks:
            nn.init.xavier_uniform_(block.attn.in_proj_weight) # MHA weights
            nn.init.zeros_(block.attn.out_proj.bias)
            nn.init.xavier_uniform_(block.attn.out_proj.weight)

            nn.init.xavier_uniform_(block.mlp[0].weight)
            nn.init.zeros_(block.mlp[0].bias)
            nn.init.xavier_uniform_(block.mlp[2].weight)
            nn.init.zeros_(block.mlp[2].bias)

            # AdaLN modulation layers (SiLU -> Linear) - Zero init for the Linear layer
            nn.init.constant_(block.adaLN_modulation[1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[1].bias, 0)

        # Initialize final layer: AdaLN bias/weights and final projection
        nn.init.constant_(self.final_layer.adaLN_modulation[1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)


    def unpatchify(self, x):
        """
        x: (B, N, P*P*C_out)
        imgs: (B, C_out, H, W)
        """
        B, N, PP_C = x.shape
        P = self.patch_size
        C_out = self.out_channels
        assert N == self.num_patches, "Input sequence length N doesn't match num_patches"
        # Calculate H, W based on num_patches and patch_size
        H_patch = W_patch = int(N**0.5) # Number of patches along height/width
        assert H_patch * W_patch == N, "Number of patches must form a square grid"
        H = H_patch * P
        W = W_patch * P

        # Reshape and rearrange
        # (B, N, P*P*C_out) -> (B, H_patch*W_patch, P*P*C_out)
        x = x.reshape(B, H_patch, W_patch, P, P, C_out) # (B, H_patch, W_patch, P, P, C_out)
        x = torch.einsum('bhwpqc->bchpwq', x) # (B, C_out, H_patch, P, W_patch, P)
        imgs = x.reshape(B, C_out, H, W) # (B, C_out, H_patch*P, W_patch*P) -> (B, C_out, H, W)
        return imgs

    def forward(self, x, t):
        """
        x: (B, C, H, W) - Input image
        t: (B,) - Timesteps (integer)
        """
        B, C, H, W = x.shape

        # 1. Embeddings
        x = self.patch_embed(x)            # (B, N, D)
        x = x + self.pos_embed             # (B, N, D) - Add positional embedding

        # Timestep embedding expects Long type
        t_emb = self.time_embed(t.float()) # (B, D)    - Use t.float() if Sinusoidal expects float

        # 2. Apply Transformer blocks with AdaLN conditioning
        for block in self.blocks:
            x = block(x, t_emb)                # (B, N, D)

        # 3. Final layer and unpatchify
        x = self.final_layer(x, t_emb)         # (B, N, P*P*C_out)
        x = self.unpatchify(x)             # (B, C_out, H, W)

        return x # Predicted noise (or noise + variance if learn_sigma=True)


# --- Diffusion Utilities ---

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# Utility function (needed by q_sample)
def extract(a, t, x_shape):
    """
    Extracts values from tensor 'a' at indices specified by 't',
    and reshapes the result to broadcast correctly with tensor 'x_shape'.

    Args:
        a (torch.Tensor): Tensor to extract values from (usually shape [T,]).
        t (torch.Tensor): Timestep indices (usually shape [B,]).
        x_shape (torch.Size): The shape of the tensor x (e.g., image shape [B, C, H, W])
                               to which the extracted values will be applied.

    Returns:
        torch.Tensor: Extracted values reshaped for broadcasting (e.g., [B, 1, 1, 1]).
    """
    batch_size = t.shape[0]
    # Use gather to get values from 'a' corresponding to 't' indices
    out = a.gather(-1, t)
    # Reshape 'out' so it can broadcast with 'x' (shape [B, C, H, W])
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Precompute diffusion variables ---
T = DIFFUSION_TIMESTEPS
betas = linear_beta_schedule(timesteps=T) # Shape: (T,)
# Note: Ensure these are on the correct device later if needed, or move model/data to CPU if necessary
alphas = 1. - betas # Shape: (T,)
alphas_cumprod = torch.cumprod(alphas, axis=0) # Shape: (T,)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0) # Shape: (T,)

# Calculations for forward diffusion q(x_t | x_0) -> Eq 4 in DDPM paper
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod) # Shape: (T,)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod) # Shape: (T,)

# Calculations for posterior q(x_{t-1} | x_t, x_0) -> Eq 7 in DDPM paper
# Used in p_sample (sampling step)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod) # Shape: (T,)
# Clamp variance to avoid numerical instability (especially at t=0 where variance is 0)
# posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20)) # Not directly used if predicting noise (epsilon)



# --- Forward Process (Adding Noise) --- Updated Function ---
def q_sample(x_start, t, sqrt_alphas_cumprod_dev, sqrt_one_minus_alphas_cumprod_dev, noise=None):
    """
    Implements the forward diffusion process: adds noise to an image x_start
    for a given timestep t according to the closed-form formula. (Corrected Signature)

    Args:
        x_start (torch.Tensor): The initial clean image(s) (B, C, H, W).
        t (torch.Tensor): The timestep(s) (B,) as integer indices.
        sqrt_alphas_cumprod_dev (torch.Tensor): Precomputed sqrt(alpha_bar) schedule values
                                                on the target device (Shape: [T,]).
        sqrt_one_minus_alphas_cumprod_dev (torch.Tensor): Precomputed sqrt(1 - alpha_bar) schedule
                                                          values on the target device (Shape: [T,]).
        noise (torch.Tensor, optional): Pre-sampled noise tensor of the same shape
                                        as x_start. If None, standard Gaussian noise
                                        is sampled inside the function. Defaults to None.

    Returns:
        torch.Tensor: The noisy image(s) at timestep t (B, C, H, W).
    """
    # If noise is not provided externally, sample it
    if noise is None:
        noise = torch.randn_like(x_start)

    # Extract the precomputed coefficients for the given timesteps t
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod_dev, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod_dev, t, x_start.shape
    )

    # Apply the closed-form formula: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    return noisy_image

# --- Loss Calculation --- Updated Function ---
def p_losses(denoise_model, x_start, t, sqrt_alphas_cumprod_dev, sqrt_one_minus_alphas_cumprod_dev, loss_type="l1"):
    """
    Calculates the loss for the diffusion model. (Corrected Signature)

    Args:
        denoise_model (nn.Module): The diffusion model being trained.
        x_start (torch.Tensor): The initial clean image(s) (B, C, H, W).
        t (torch.Tensor): The timestep(s) (B,) as integer indices.
        sqrt_alphas_cumprod_dev (torch.Tensor): Precomputed sqrt(alpha_bar) schedule values
                                                on the target device (Shape: [T,]).
        sqrt_one_minus_alphas_cumprod_dev (torch.Tensor): Precomputed sqrt(1 - alpha_bar) schedule
                                                          values on the target device (Shape: [T,]).
        loss_type (str, optional): Type of loss function ('l1', 'l2', 'huber').
                                   Defaults to "l1".

    Returns:
        torch.Tensor: The calculated loss (scalar tensor).
    """
    # Sample noise from standard Gaussian distribution
    noise = torch.randn_like(x_start)

    # Create noisy image x_t using the corrected q_sample function
    # Note: q_sample itself now expects noise=None at the end if using default behavior
    x_noisy = q_sample(x_start=x_start, t=t,
                       sqrt_alphas_cumprod_dev=sqrt_alphas_cumprod_dev,
                       sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                       noise=noise) # Pass the sampled noise explicitly

    # Get the model's prediction for the noise added to x_noisy at timestep t
    predicted_noise = denoise_model(x_noisy, t)

    # Calculate the loss between the actual noise and the predicted noise
    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise) # Also known as Smooth L1 Loss
    else:
        raise NotImplementedError(f"Loss type '{loss_type}' not implemented.")

    return loss


# --- Sampling (Reverse Process) ---
@torch.no_grad()
def p_sample(model, x, t, t_index, betas_dev, sqrt_one_minus_alphas_cumprod_dev, sqrt_recip_alphas_dev, posterior_variance_dev):
    # Extract precomputed schedule values for the current timestep t
    betas_t = extract(betas_dev, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(
        sqrt_one_minus_alphas_cumprod_dev, t, x.shape
    )
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas_dev, t, x.shape)

    # Equation 11 in the DDPM paper: Calculate mean based on model output (predicted noise)
    # model(x, t) predicts the noise epsilon
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        # The final step, no noise is added
        return model_mean
    else:
        # Add noise using the calculated posterior variance
        posterior_variance_t = extract(posterior_variance_dev, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4: x_{t-1} = mean + sqrt(posterior_variance) * noise
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# Algorithm 2 (Sampling loop)
@torch.no_grad()
def p_sample_loop(model, shape, timesteps, betas_dev, sqrt_one_minus_alphas_cumprod_dev, sqrt_recip_alphas_dev, posterior_variance_dev):
    device = next(model.parameters()).device
    b = shape[0]
    # Start from pure noise (x_T)
    img = torch.randn(shape, device=device)
    imgs = []

    # Iterate backwards from T-1 down to 0
    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling loop time step', total=timesteps):
        # Current timestep tensor (all elements are i)
        t = torch.full((b,), i, device=device, dtype=torch.long)
        img = p_sample(model, img, t, i,
                       betas_dev, sqrt_one_minus_alphas_cumprod_dev, sqrt_recip_alphas_dev, posterior_variance_dev)
        # Optionally save intermediate steps
        # if i % 50 == 0: # Save every 50 steps for visualization
        #     imgs.append(img.cpu())
    imgs.append(img.cpu()) # Append the final image
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1):
    # Get device-specific tensors for sampling
    betas_dev = betas.to(DEVICE)
    sqrt_one_minus_alphas_cumprod_dev = sqrt_one_minus_alphas_cumprod.to(DEVICE)
    sqrt_recip_alphas_dev = torch.sqrt(1.0 / alphas).to(DEVICE)
    posterior_variance_dev = posterior_variance.to(DEVICE)

    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size), timesteps=T,
                         betas_dev=betas_dev,
                         sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                         sqrt_recip_alphas_dev=sqrt_recip_alphas_dev,
                         posterior_variance_dev=posterior_variance_dev)


# --- Data Preparation ---
# Define transformations: Resize and normalize to [-1, 1]
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),                # Converts PIL image or numpy array to [C, H, W] tensor in [0, 1] range
    transforms.Lambda(lambda t: (t * 2) - 1) # Scale tensor values from [0, 1] to [-1, 1]
])

# Inverse transform for visualization
def unnormalize_to_zero_to_one(t):
    # Takes a tensor in [-1, 1] and scales it back to [0, 1]
    return (t.clamp(-1, 1) + 1) * 0.5

dataset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


# --- Model & Optimizer ---
model = DiffusionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IMG_CHANNELS,
    latent_dim=LATENT_DIM,
    depth=TRANSFORMER_DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Move precomputed schedule tensors to the correct device ONCE
sqrt_alphas_cumprod_dev = sqrt_alphas_cumprod.to(DEVICE)
sqrt_one_minus_alphas_cumprod_dev = sqrt_one_minus_alphas_cumprod.to(DEVICE)


# --- Training Loop ---
print(f"Starting training on {DEVICE}...")
print(f"Image Size: {IMG_SIZE}, Patch Size: {PATCH_SIZE}, Latent Dim: {LATENT_DIM}, Heads: {NUM_HEADS}, Depth: {TRANSFORMER_DEPTH}")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    model.train() # Ensure model is in training mode
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        optimizer.zero_grad()

        batch_size = batch[0].shape[0]
        # Move batch to device; batch[0] contains images, batch[1] labels (ignored)
        batch_images = batch[0].to(DEVICE)

        # Algorithm 1 line 3: sample t uniformally for every example in the batch
        # t should be torch.long for indexing
        t = torch.randint(0, T, (batch_size,), device=DEVICE).long()

        # Calculate loss using device tensors for schedule
        loss = p_losses(model, batch_images, t, loss_type="huber",
                        sqrt_alphas_cumprod_dev=sqrt_alphas_cumprod_dev,
                        sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev)

        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    # --- Sampling and Saving ---
    if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1: # Sample every 5 epochs and at the end
        print(f"Sampling images at epoch {epoch+1}...")
        model.eval() # Set model to evaluation mode for sampling
        generated_images_sequence = sample(model, image_size=IMG_SIZE, batch_size=64, channels=IMG_CHANNELS)
        # Get the final denoised images (last element in the sequence returned by p_sample_loop)
        final_sampled_images = generated_images_sequence[-1]
        # Unnormalize images back to [0, 1] for saving
        final_sampled_images = unnormalize_to_zero_to_one(final_sampled_images)

        save_image(final_sampled_images, os.path.join(SAMPLE_SAVE_DIR, f"sample_epoch_{epoch+1}.png"), nrow=8)
        print(f"Saved sample images to {SAMPLE_SAVE_DIR}/sample_epoch_{epoch+1}.png")
        # Set model back to training mode - crucial if training continues!
        # Already handled by model.train() at the start of the next epoch loop


    # --- Save Model Checkpoint ---
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1: # Save every 10 epochs and at the end
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved model checkpoint to {MODEL_SAVE_PATH}")


print("Training finished.")
# Final save (optional, could rely on the last save within the loop)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Final model saved to {MODEL_SAVE_PATH}")