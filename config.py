import torch

# --- Configuration ---
IMG_SIZE = 32
PATCH_SIZE = 4
IMG_CHANNELS = 1
LATENT_DIM = 256
NUM_HEADS = 4
TRANSFORMER_DEPTH = 6
MLP_RATIO = 4.0
NUM_CLASSES = 10  # Fashion MNIST has 10 classes
BATCH_SIZE = 128
LEARNING_RATE = 1e-4
EPOCHS = 75  # Conditional models might need more epochs
DIFFUSION_TIMESTEPS = 1000
CFG_PROB = 0.15  # Probability of using null label during training (10-20% is common)
DEFAULT_GUIDANCE_SCALE = 5.0  # Classifier-Free Guidance scale for sampling
MODEL_SAVE_PATH = "dit_fmnist_conditional_cfg.pth"
SAMPLE_SAVE_DIR = "samples"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"