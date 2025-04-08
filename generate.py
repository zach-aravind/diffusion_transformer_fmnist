import torch
import os
from torchvision.utils import save_image
import argparse

from config import *
from model import DiffusionTransformer
from diffusion import get_diffusion_parameters, sample
from utils import unnormalize_to_zero_to_one

def main(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get diffusion parameters
    diffusion_params = get_diffusion_parameters(DIFFUSION_TIMESTEPS)
    
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
    
    # Load trained model weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE))
    model.eval()
    
    # Setup generation parameters
    samples_per_class = args.samples_per_class
    if args.class_idx is not None:
        # Generate images for a specific class
        num_classes_to_sample = 1
        classes_to_sample = [args.class_idx]
    else:
        # Generate images for all classes
        num_classes_to_sample = NUM_CLASSES
        classes_to_sample = list(range(NUM_CLASSES))
    
    total_samples = samples_per_class * num_classes_to_sample
    
    # Create target labels
    target_labels_list = []
    for i in classes_to_sample:
        target_labels_list.extend([i] * samples_per_class)
    target_labels_tensor = torch.tensor(target_labels_list, device=DEVICE, dtype=torch.long)
    
    print(f"Generating {total_samples} images with guidance scale {args.guidance_scale}...")
    
    # Generate images
    generated_images_sequence = sample(
        model,
        image_size=IMG_SIZE,
        target_labels=target_labels_tensor,
        guidance_scale=args.guidance_scale,
        batch_size=total_samples,
        channels=IMG_CHANNELS,
        timesteps=DIFFUSION_TIMESTEPS,
        diffusion_params=diffusion_params
    )
    
    # Process and save images
    final_sampled_images = generated_images_sequence[-1]  # Get final images
    final_sampled_images = unnormalize_to_zero_to_one(final_sampled_images)
    
    # Save all images in a grid
    class_name = f"class_{args.class_idx}" if args.class_idx is not None else "all_classes"
    grid_save_path = os.path.join(args.output_dir, f"samples_{class_name}_cfg_w{args.guidance_scale}.png")
    save_image(final_sampled_images, grid_save_path, nrow=samples_per_class)
    print(f"Saved grid of samples to {grid_save_path}")
    
    # Optionally save individual images
    if args.save_individuals:
        print("Saving individual images...")
        for idx, img in enumerate(final_sampled_images):
            class_idx = target_labels_list[idx]
            img_save_path = os.path.join(args.output_dir, f"sample_class_{class_idx}_idx_{idx}.png")
            save_image(img, img_save_path)
    
    print("Generation complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Fashion MNIST images with trained DiT model")
    parser.add_argument("--model_path", type=str, default=MODEL_SAVE_PATH, 
                        help="Path to the trained model weights")
    parser.add_argument("--output_dir", type=str, default=SAMPLE_SAVE_DIR,
                        help="Directory to save generated images")
    parser.add_argument("--guidance_scale", type=float, default=DEFAULT_GUIDANCE_SCALE,
                        help="Classifier-free guidance scale")
    parser.add_argument("--samples_per_class", type=int, default=8,
                        help="Number of samples to generate per class")
    parser.add_argument("--class_idx", type=int, default=None,
                        help="Class index to generate (0-9, None for all classes)")
    parser.add_argument("--save_individuals", action="store_true",
                        help="Save individual images in addition to the grid")
    
    args = parser.parse_args()
    main(args)