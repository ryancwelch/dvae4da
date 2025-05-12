import torch
import torch.nn.functional as F
import numpy as np
from typing import Union, Tuple


def add_gaussian_noise(images: torch.Tensor, 
                      noise_factor: float = 0.2, 
                      clip_min: float = 0.0, 
                      clip_max: float = 1.0) -> torch.Tensor:
    """
    Add Gaussian noise to a batch of images.
    """
    device = images.device
    noise = torch.randn_like(images) * noise_factor
    noisy_images = images + noise
    return torch.clamp(noisy_images, clip_min, clip_max)


def add_salt_and_pepper_noise(images: torch.Tensor, 
                             noise_factor: float = 0.1,
                             salt_vs_pepper: float = 0.5,
                             clip_min: float = 0.0, 
                             clip_max: float = 1.0) -> torch.Tensor:
    """
    Add salt and pepper noise to a batch of images.
    """
    device = images.device
    noisy_images = images.clone()
    
    mask = torch.rand_like(images) < noise_factor
    
    salt_mask = mask & (torch.rand_like(images) < salt_vs_pepper)
    pepper_mask = mask & ~salt_mask
    
    noisy_images[salt_mask] = clip_max
    noisy_images[pepper_mask] = clip_min
    
    return noisy_images


def add_structured_noise(images: torch.Tensor,
                        noise_type: str = 'block',
                        block_size: int = 8,
                        noise_factor: float = 0.2,
                        clip_min: float = 0.0,
                        clip_max: float = 1.0) -> torch.Tensor:
    """
    Add structured noise like blocks or lines to images.
    """
    device = images.device
    batch_size, channels, height, width = images.shape
    noisy_images = images.clone()
    
    if noise_type == 'block':
        for i in range(batch_size):
            num_blocks = int((height * width) / (block_size * block_size) * noise_factor)
            for _ in range(num_blocks):
                h_start = np.random.randint(0, height - block_size)
                w_start = np.random.randint(0, width - block_size)
                
                noise_value = np.random.uniform(clip_min, clip_max)
                
                noisy_images[i, :, h_start:h_start+block_size, w_start:w_start+block_size] = noise_value
                
    elif noise_type == 'line_h':
        for i in range(batch_size):
            num_lines = int(height / block_size * noise_factor)
            for _ in range(num_lines):
                h_start = np.random.randint(0, height - block_size)
                
                noise_value = np.random.uniform(clip_min, clip_max)
                
                noisy_images[i, :, h_start:h_start+block_size, :] = noise_value
                
    elif noise_type == 'line_v':
        for i in range(batch_size):
            num_lines = int(width / block_size * noise_factor)
            for _ in range(num_lines):
                w_start = np.random.randint(0, width - block_size)
                
                noise_value = np.random.uniform(clip_min, clip_max)
                
                noisy_images[i, :, :, w_start:w_start+block_size] = noise_value
    
    return noisy_images


def add_blur_noise(images: torch.Tensor,
                  noise_factor: float = 0.2,
                  clip_min: float = 0.0,
                  clip_max: float = 1.0) -> torch.Tensor:
    """
    Add blur noise to a batch of images.
    The noise_factor controls the strength of the blur by interpolating between the original and blurred image.
    """
    device = images.device
    batch_size, channels, height, width = images.shape
    noisy_images = images.clone()

    kernel_size = 5  # You can adjust this size
    # Create a random blur kernel for each channel
    kernel = torch.randn(channels, 1, kernel_size, kernel_size, device=device)
    kernel = kernel / kernel.view(channels, -1).sum(dim=1, keepdim=True).view(channels, 1, 1, 1)

    for i in range(batch_size):
        # F.conv2d expects input shape [N, C, H, W], kernel shape [C, 1, k, k], groups=C
        blurred = F.conv2d(
            noisy_images[i].unsqueeze(0),  # [1, C, H, W]
            kernel,
            padding=kernel_size // 2,
            groups=channels
        )
        # Interpolate between original and blurred image using noise_factor
        mixed = (1 - noise_factor) * noisy_images[i] + noise_factor * blurred.squeeze(0)
        noisy_images[i] = torch.clamp(mixed, clip_min, clip_max)

    return noisy_images
    


def add_noise(images: torch.Tensor,
             noise_type: str = 'gaussian',
             noise_params: dict = None) -> torch.Tensor:
    """
    Add noise to a batch of images based on specified noise type.
    """
    if noise_params is None:
        noise_params = {}
    
    if noise_type == 'gaussian':
        return add_gaussian_noise(
            images,
            noise_factor=noise_params.get('noise_factor', 0.2),
            clip_min=noise_params.get('clip_min', 0.0),
            clip_max=noise_params.get('clip_max', 1.0)
        )
    elif noise_type == 'salt_and_pepper':
        return add_salt_and_pepper_noise(
            images,
            noise_factor=noise_params.get('noise_factor', 0.1),
            salt_vs_pepper=noise_params.get('salt_vs_pepper', 0.5),
            clip_min=noise_params.get('clip_min', 0.0),
            clip_max=noise_params.get('clip_max', 1.0)
        )
    elif noise_type in ['block', 'line_h', 'line_v']:
        return add_structured_noise(
            images,
            noise_type=noise_type,
            block_size=noise_params.get('block_size', 8),
            noise_factor=noise_params.get('noise_factor', 0.2),
            clip_min=noise_params.get('clip_min', 0.0),
            clip_max=noise_params.get('clip_max', 1.0)
        )
    elif noise_type == 'blur':
        return add_blur_noise(
            images,
            noise_factor=noise_params.get('noise_factor', 0.2),
            clip_min=noise_params.get('clip_min', 0.0),
            clip_max=noise_params.get('clip_max', 1.0)
        )
    else:
        raise ValueError(f"Unsupported noise type: {noise_type}") 