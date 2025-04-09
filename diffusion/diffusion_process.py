import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from diffusion.scheduler import extract

# Forward Process (Adding Noise)
def q_sample(x_start, t, sqrt_alphas_cumprod_dev, sqrt_one_minus_alphas_cumprod_dev, noise=None):
    if noise is None:
        noise = torch.randn_like(x_start)
    sqrt_alphas_cumprod_t = extract(sqrt_alphas_cumprod_dev, t, x_start.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphas_cumprod_dev, t, x_start.shape)
    noisy_image = sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    return noisy_image

# Loss Calculation
def p_losses(denoise_model, x_start, t, y, 
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

# Sampling (Reverse Process) - Modified for CFG
@torch.no_grad()
def p_sample(model, x, t, y, t_index, guidance_scale,
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

# Sampling Loop
@torch.no_grad()
def p_sample_loop(model, shape, target_labels, guidance_scale,
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
    imgs.append(img.cpu())  # Append final image
    return imgs

# Top-level Sampling Function
@torch.no_grad()
def sample(model, image_size, target_labels, guidance_scale, batch_size=16, channels=1, timesteps=1000,
           diffusion_params=None):
    # Ensure target_labels is a tensor on the correct device
    device = next(model.parameters()).device
    if not isinstance(target_labels, torch.Tensor):
        if isinstance(target_labels, int):  # Single label for all samples
            target_labels = torch.tensor([target_labels] * batch_size, device=device, dtype=torch.long)
        else:  # List or tuple of labels
            assert len(target_labels) == batch_size, "Length of target_labels list must match batch_size"
            target_labels = torch.tensor(target_labels, device=device, dtype=torch.long)
    else:  # Already a tensor
        target_labels = target_labels.to(device=device, dtype=torch.long)
        assert target_labels.shape[0] == batch_size, "Shape of target_labels tensor must match batch_size"

    # Get device-specific tensors for schedule
    betas_dev = diffusion_params['betas'].to(device)
    sqrt_one_minus_alphas_cumprod_dev = diffusion_params['sqrt_one_minus_alphas_cumprod'].to(device)
    sqrt_recip_alphas_dev = torch.sqrt(1.0 / diffusion_params['alphas']).to(device)
    posterior_variance_dev = diffusion_params['posterior_variance'].to(device)

    img_sequence = p_sample_loop(model, shape=(batch_size, channels, image_size, image_size),
                                 target_labels=target_labels,
                                 guidance_scale=guidance_scale,
                                 timesteps=timesteps,
                                 betas_dev=betas_dev,
                                 sqrt_one_minus_alphas_cumprod_dev=sqrt_one_minus_alphas_cumprod_dev,
                                 sqrt_recip_alphas_dev=sqrt_recip_alphas_dev,
                                 posterior_variance_dev=posterior_variance_dev)
    return img_sequence  # Return sequence, last element is the final image