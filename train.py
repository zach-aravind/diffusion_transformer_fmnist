import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image, make_grid
import os
import random
from tqdm.auto import tqdm

# Import from our reorganized modules
from config import *
from model import DiffusionTransformer
from diffusion import get_diffusion_parameters, q_sample, p_losses, sample
from data import get_fashion_mnist_dataset, get_dataloader
from utils import unnormalize_to_zero_to_one

# Create sample directory
os.makedirs(SAMPLE_SAVE_DIR, exist_ok=True)

def main():
    # Get diffusion parameters
    diffusion_params = get_diffusion_parameters(DIFFUSION_TIMESTEPS)
    
    # Prepare dataset and dataloader
    dataset = get_fashion_mnist_dataset(IMG_SIZE)
    dataloader = get_dataloader(dataset, BATCH_SIZE)
    
    # Initialize model
    model = DiffusionTransformer(
        img_size=IMG_SIZE,
        patch_size=PATCH_SIZE,
        in_channels=IMG_CHANNELS,
        latent_dim=LATENT_DIM,
        depth=TRANSFORMER_DEPTH,
        num_heads=NUM_HEADS,
        mlp_ratio=MLP_RATIO,
        num_classes=NUM_CLASSES
    ).to(DEVICE)
    
    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    
    # Move diffusion parameters to device
    sqrt_alphas_cumprod_dev = diffusion_params['sqrt_alphas_cumprod'].to(DEVICE)
    sqrt_one_minus_alphas_cumprod_dev = diffusion_params['sqrt_one_minus_alphas_cumprod'].to(DEVICE)
    
    # Training loop
    print(f"Starting training on {DEVICE} with CFG (prob={CFG_PROB})...")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    for epoch in range(EPOCHS):
        epoch_loss = 0.0
        model.train()
        for step, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")):
            optimizer.zero_grad()
            
            batch_size = batch[0].shape[0]
            images = batch[0].to(DEVICE)
            labels = batch[1].to(DEVICE)  # Get labels
            
            t = torch.randint(0, DIFFUSION_TIMESTEPS, (batch_size,), device=DEVICE).long()
            
            # Classifier-Free Guidance Training
            final_labels = labels
            # With probability CFG_PROB, set labels to the null index for unconditional training
            if random.random() < CFG_PROB:
                null_label_idx = model.null_label_idx  # Get null index from model
                final_labels = torch.full_like(labels, null_label_idx)
            
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
        
        # Sampling and Saving
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:  # Sample every 10 epochs
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
            
            generated_images_sequence = sample(
                model,
                image_size=IMG_SIZE,
                target_labels=target_labels_tensor,
                guidance_scale=DEFAULT_GUIDANCE_SCALE,
                batch_size=total_samples,
                channels=IMG_CHANNELS,
                timesteps=DIFFUSION_TIMESTEPS,
                diffusion_params=diffusion_params
            )
            
            final_sampled_images = generated_images_sequence[-1]  # Get final images
            final_sampled_images = unnormalize_to_zero_to_one(final_sampled_images)
            
            save_path = os.path.join(SAMPLE_SAVE_DIR, f"sample_epoch_{epoch+1}_cfg_w{DEFAULT_GUIDANCE_SCALE}.png")
            # nrow = samples_per_class so each row shows samples from one class
            save_image(final_sampled_images, save_path, nrow=samples_per_class)
            print(f"Saved conditional samples to {save_path}")
        
        # Save Model Checkpoint
        if (epoch + 1) % 10 == 0 or epoch == EPOCHS - 1:  # Save every 10 epochs
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Saved model checkpoint to {MODEL_SAVE_PATH}")
    
    print("Training finished.")

if __name__ == "__main__":
    main()