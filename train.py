import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid

import math
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from tqdm.auto import tqdm
import os
import random # For CFG probability

# --- Configuration ---
IMG_SIZE = 32
PATCH_SIZE = 4
IMG_CHANNELS = 1
LATENT_DIM = 256
NUM_HEADS = 4
TRANSFORMER_DEPTH = 6
MLP_RATIO = 4.0
NUM_CLASSES = 10 # Fashion MNIST has 10 classes
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 75 # Conditional models might need more epochs
DIFFUSION_TIMESTEPS = 1000
CFG_PROB = 0.15 # Probability of using null label during training (10-20% is common)
DEFAULT_GUIDANCE_SCALE = 5.0 # Classifier-Free Guidance scale for sampling
MODEL_SAVE_PATH = "dit_fmnist_conditional_cfg.pth"
SAMPLE_SAVE_DIR = "samples_fmnist_conditional_cfg"
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

# Sinusoidal Position Embedding (for Timesteps) - Unchanged
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

# Patch Embedding - Unchanged
def pair(t):
    return t if isinstance(t, tuple) else (t, t)

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

# AdaLN-Zero Helper Function (Modulation) - Unchanged
def modulate(x, shift, scale):
    if x.ndim == 3: # Sequence input B, N, D
        shift = shift.unsqueeze(1)
        scale = scale.unsqueeze(1)
    # Add dim for broadcasting: (B, D) -> (B, 1, D) or keep as (B, D) if x is (B, D)
    return x * (1 + scale) + shift

# --- DiT Block --- Unchanged internal logic, takes combined embedding 'c'
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

    def forward(self, x, c): # c is the combined time+label embedding
        modulation_params = self.adaLN_modulation(c)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = modulation_params.chunk(6, dim=1)
        x_norm1_modulated = modulate(self.norm1(x), shift_msa, scale_msa)
        attn_output, _ = self.attn(x_norm1_modulated, x_norm1_modulated, x_norm1_modulated)
        x = x + gate_msa.unsqueeze(1) * attn_output
        x_norm2_modulated = modulate(self.norm2(x), shift_mlp, scale_mlp)
        mlp_output = self.mlp(x_norm2_modulated)
        x = x + gate_mlp.unsqueeze(1) * mlp_output
        return x

# --- Final Layer for DiT --- Unchanged internal logic, takes combined embedding 'c'
class FinalLayer(nn.Module):
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size)
        )

    def forward(self, x, c): # c is the combined time+label embedding
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


# --- Conditional Diffusion Transformer Model ---
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
        num_classes=10,  # <<< Added: Number of classes
        learn_sigma=False
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        # <<< Added: Index for the null / unconditional token for CFG
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

        # <<< Modified: Class Label Embedding (size num_classes + 1 for null token) >>>
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
        # <<< Initialize label embedding >>>
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

    # <<< Modified: Takes label 'y' as input >>>
    def forward(self, x, t, y):
        B, C, H, W = x.shape
        x = self.patch_embed(x)
        x = x + self.pos_embed
        t_emb = self.time_embed(t.float()) # (B, D)
        y_emb = self.label_embed(y)        # (B, D) - Embed labels (could be true or null)
        combined_emb = t_emb + y_emb       # (B, D) - Combine embeddings

        for block in self.blocks:
            x = block(x, combined_emb)     # Pass combined embedding

        x = self.final_layer(x, combined_emb) # Pass combined embedding
        x = self.unpatchify(x)
        return x


# --- Diffusion Utilities ---

def linear_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

# Corrected extract function
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

# --- Precompute diffusion variables --- (Same as before)
T = DIFFUSION_TIMESTEPS
betas = linear_beta_schedule(timesteps=T)
alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# --- Forward Process (Adding Noise) - Corrected Signature ---
def q_sample(x_start, t, sqrt_alphas_cumprod_dev, sqrt_one_minus_alphas_cumprod_dev, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod_dev, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod_dev, t, x_start.shape)
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

# --- Loss Calculation - Modified for Conditioning ---
def p_losses(denoise_model, x_start, t, y, # <<< Takes labels 'y'
             sqrt_alphas_cumprod_dev, sqrt_one_minus_alphas_cumprod_dev, loss_type="l1"):
    noise = torch.randn_like(x_start)
    x_noisy = q_sample(x_start=x_start, t=t,
                       sqrt_alphas_cumprod_dev=sqrt_alphas_cumprod_dev,
                       sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                       noise=noise)
    # Pass labels 'y' (which might be true or null due to CFG) to the model
    predicted_noise = denoise_model(x_noisy, t, y)

    if loss_type == 'l1':
        loss = F.l1_loss(noise, predicted_noise)
    elif loss_type == 'l2':
        loss = F.mse_loss(noise, predicted_noise)
    elif loss_type == "huber":
        loss = F.smooth_l1_loss(noise, predicted_noise)
    else:
        raise NotImplementedError()
    return loss

# --- Sampling (Reverse Process) - Modified for CFG ---
@torch.no_grad()
def p_sample(model, x, t, y, t_index, guidance_scale, # <<< Takes labels 'y' and guidance_scale 'w'
             betas_dev, sqrt_one_minus_alphas_cumprod_dev,
             sqrt_recip_alphas_dev, posterior_variance_dev):

    betas_t = extract(betas_dev, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod_dev, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_recip_alphas_dev, t, x.shape)

    # --- Classifier-Free Guidance Prediction ---
    # Get the conditional prediction (with the desired class label y)
    pred_noise_cond = model(x, t, y)

    # Get the unconditional prediction (by passing the null label index)
    # Create a tensor of null labels with the same batch size as y
    batch_size = y.shape[0]
    null_labels = torch.full((batch_size,), model.null_label_idx, device=y.device, dtype=torch.long)
    pred_noise_uncond = model(x, t, null_labels)

    # Combine predictions using the guidance scale:
    # final_pred = uncond + w * (cond - uncond)
    final_predicted_noise = pred_noise_uncond + guidance_scale * (pred_noise_cond - pred_noise_uncond)
    # --- End CFG ---

    # Calculate model mean using the CFG-combined noise prediction
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * final_predicted_noise / sqrt_one_minus_alphas_cumprod_t
    )

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance_dev, t, x.shape)
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

# --- Sampling Loop - Modified for CFG ---
@torch.no_grad()
def p_sample_loop(model, shape, target_labels, guidance_scale, # <<< Takes labels and guidance_scale
                  timesteps, betas_dev, sqrt_one_minus_alphas_cumprod_dev,
                  sqrt_recip_alphas_dev, posterior_variance_dev):
    device = next(model.parameters()).device
    b = shape[0]
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, timesteps)), desc='Sampling loop time step', total=timesteps):
        t = torch.full((b,), i, device=device, dtype=torch.long)
        # Pass target_labels and guidance_scale to p_sample
        img = p_sample(model, img, t, target_labels, i, guidance_scale,
                       betas_dev, sqrt_one_minus_alphas_cumprod_dev,
                       sqrt_recip_alphas_dev, posterior_variance_dev)
        # Optional: save intermediate steps
        # if i % 50 == 0: imgs.append(img.cpu())
    imgs.append(img.cpu()) # Append final image
    return imgs

# --- Top-level Sampling Function - Modified for CFG ---
@torch.no_grad()
def sample(model, image_size, target_labels, guidance_scale=DEFAULT_GUIDANCE_SCALE, batch_size=16, channels=1):
    # Ensure target_labels is a tensor on the correct device
    device = next(model.parameters()).device
    if not isinstance(target_labels, torch.Tensor):
         if isinstance(target_labels, int): # Single label for all samples
              target_labels = torch.tensor([target_labels] * batch_size, device=device, dtype=torch.long)
         else: # List or tuple of labels
              assert len(target_labels) == batch_size, "Length of target_labels list must match batch_size"
              target_labels = torch.tensor(target_labels, device=device, dtype=torch.long)
    else: # Already a tensor
         target_labels = target_labels.to(device=device, dtype=torch.long)
         assert target_labels.shape[0] == batch_size, "Shape of target_labels tensor must match batch_size"

    # Get device-specific tensors for schedule
    betas_dev = betas.to(device)
    sqrt_one_minus_alphas_cumprod_dev = sqrt_one_minus_alphas_cumprod.to(device)
    sqrt_recip_alphas_dev = torch.sqrt(1.0 / alphas).to(device)
    posterior_variance_dev = posterior_variance.to(device)

    img_sequence = p_sample_loop(model, shape=(batch_size, channels, image_size, image_size),
                                 target_labels=target_labels,
                                 guidance_scale=guidance_scale,
                                 timesteps=T,
                                 betas_dev=betas_dev,
                                 sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                                 sqrt_recip_alphas_dev=sqrt_recip_alphas_dev,
                                 posterior_variance_dev=posterior_variance_dev)
    return img_sequence # Return sequence, last element is the final image


# --- Data Preparation --- (Same transform, DataLoader loads labels now)
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda t: (t * 2) - 1)
])

def unnormalize_to_zero_to_one(t):
    return (t.clamp(-1, 1) + 1) * 0.5

dataset = datasets.FashionMNIST('.', train=True, download=True, transform=transform)
# pin_memory=True speeds up host-to-GPU transfers if using CUDA
dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)


# --- Model & Optimizer ---
# <<< Instantiate model with num_classes >>>
model = DiffusionTransformer(
    img_size=IMG_SIZE,
    patch_size=PATCH_SIZE,
    in_channels=IMG_CHANNELS,
    latent_dim=LATENT_DIM,
    depth=TRANSFORMER_DEPTH,
    num_heads=NUM_HEADS,
    mlp_ratio=MLP_RATIO,
    num_classes=NUM_CLASSES # Pass num_classes
).to(DEVICE)

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# Move schedule tensors to device once
sqrt_alphas_cumprod_dev = sqrt_alphas_cumprod.to(DEVICE)
sqrt_one_minus_alphas_cumprod_dev = sqrt_one_minus_alphas_cumprod.to(DEVICE)

# --- Training Loop - Modified for CFG ---
print(f"Starting training on {DEVICE} with CFG (prob={CFG_PROB})...")
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

for epoch in range(EPOCHS):
    epoch_loss = 0.0
    model.train()
    for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        optimizer.zero_grad()

        batch_size = batch[0].shape[0]
        images = batch[0].to(DEVICE)
        labels = batch[1].to(DEVICE) # <<< Get labels

        t = torch.randint(0, T, (batch_size,), device=DEVICE).long()

        # --- Classifier-Free Guidance Training ---
        final_labels = labels
        # With probability CFG_PROB, set labels to the null index for unconditional training
        if random.random() < CFG_PROB:
            null_label_idx = model.null_label_idx # Get null index from model
            final_labels = torch.full_like(labels, null_label_idx)
        # --- End CFG Training ---

        # Pass the potentially modified labels (final_labels) to p_losses
        loss = p_losses(model, images, t, final_labels,
                        sqrt_alphas_cumprod_dev=sqrt_alphas_cumprod_dev,
                        sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                        loss_type="huber")

        loss.backward()
        # Optional: Gradient Clipping
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        epoch_loss += loss.item()

    avg_epoch_loss = epoch_loss / len(dataloader)
    print(f"Epoch {epoch+1} Average Loss: {avg_epoch_loss:.4f}")

    # --- Sampling and Saving (Modified for Conditional) ---
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1: # Sample every 10 epochs
        print(f"Sampling images at epoch {epoch+1}...")
        model.eval()

        # Generate N samples for each class (0-9)
        samples_per_class = 8
        num_classes_to_sample = NUM_CLASSES
        total_samples = samples_per_class * num_classes_to_sample

        # Create target labels: [0, 0, ..., 1, 1, ..., 9, 9, ...]
        target_labels_list = []
        for i in range(num_classes_to_sample):
            target_labels_list.extend([i] * samples_per_class)
        target_labels_tensor = torch.tensor(target_labels_list, device=DEVICE, dtype=torch.long)

        generated_images_sequence = sample(model,
                                           image_size=IMG_SIZE,
                                           target_labels=target_labels_tensor,
                                           guidance_scale=DEFAULT_GUIDANCE_SCALE, # Use guidance
                                           batch_size=total_samples,
                                           channels=IMG_CHANNELS)

        final_sampled_images = generated_images_sequence[-1] # Get final images
        final_sampled_images = unnormalize_to_zero_to_one(final_sampled_images)

        save_path = os.path.join(SAMPLE_SAVE_DIR, f"sample_epoch_{epoch+1}_cfg_w{DEFAULT_GUIDANCE_SCALE}.png")
        # nrow = samples_per_class so each row shows samples from one class
        save_image(final_sampled_images, save_path, nrow=samples_per_class)
        print(f"Saved conditional samples to {save_path}")
        # model.train() # Set back at start of loop

    # --- Save Model Checkpoint ---
    if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1: # Save every 10 epochs
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Saved model checkpoint to {MODEL_SAVE_PATH}")

print("Training finished.")