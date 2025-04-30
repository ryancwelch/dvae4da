#!/usr/bin/env python
import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.datasets import get_cifar10_dataset, NoisyDataset

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create figure to display images
plt.figure(figsize=(15, 8))

# Noise factors to display
noise_factors = [0.0, 0.1, 0.3, 0.5, 0.7, 0.9]

# Get clean CIFAR-10 dataset
cifar_dataset = get_cifar10_dataset(
    root="./data",
    train=False,
    noise_type=None,  # No noise initially
    download=True
)

# Get a few sample images
sample_indices = [3, 12, 25, 42]  # Different class examples
samples = []

for idx in sample_indices:
    img, label = cifar_dataset[idx]
    samples.append((img, label))

# Set up the subplot grid
num_rows = len(samples) * 2  # 2 noise types
num_cols = len(noise_factors)

# Class names for CIFAR-10
class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Plot images with different noise types and factors
for sample_idx, (clean_img, label) in enumerate(samples):
    # Row for Gaussian noise
    row_idx_gaussian = sample_idx * 2
    
    for col_idx, noise_factor in enumerate(noise_factors):
        plt.subplot(num_rows, num_cols, row_idx_gaussian * num_cols + col_idx + 1)
        
        if noise_factor == 0.0:
            # Show clean image
            img_to_show = clean_img
            plt.title(f"{class_names[label]}\nClean")
        else:
            # Create single-image dataset with Gaussian noise
            single_img_dataset = [(clean_img, label)]
            noisy_dataset = NoisyDataset(
                base_dataset=single_img_dataset,
                noise_type='gaussian',
                noise_params={'noise_factor': noise_factor, 'clip_min': 0.0, 'clip_max': 1.0},
                add_noise_online=True,
                return_pairs=True
            )
            noisy_img, _, _ = noisy_dataset[0]
            img_to_show = noisy_img
            plt.title(f"Gaussian\n{noise_factor:.1f}")
        
        # Convert to numpy and transpose for plotting (C,H,W) -> (H,W,C)
        img_np = img_to_show.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.axis('off')
    
    # Row for Salt & Pepper noise
    row_idx_sp = sample_idx * 2 + 1
    
    for col_idx, noise_factor in enumerate(noise_factors):
        plt.subplot(num_rows, num_cols, row_idx_sp * num_cols + col_idx + 1)
        
        if noise_factor == 0.0:
            # Show clean image again
            img_to_show = clean_img
            plt.title(f"Clean")
        else:
            # Create single-image dataset with Salt & Pepper noise
            single_img_dataset = [(clean_img, label)]
            noisy_dataset = NoisyDataset(
                base_dataset=single_img_dataset,
                noise_type='salt_and_pepper',
                noise_params={'noise_factor': noise_factor, 'clip_min': 0.0, 'clip_max': 1.0, 'salt_vs_pepper': 0.5},
                add_noise_online=True,
                return_pairs=True
            )
            noisy_img, _, _ = noisy_dataset[0]
            img_to_show = noisy_img
            plt.title(f"Salt & Pepper\n{noise_factor:.1f}")
        
        # Convert to numpy and transpose for plotting (C,H,W) -> (H,W,C)
        img_np = img_to_show.permute(1, 2, 0).numpy()
        plt.imshow(img_np)
        plt.axis('off')

plt.suptitle("CIFAR-10 Images with Different Noise Levels", fontsize=16)
plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("cifar10_noise_examples.png", dpi=300)
plt.show()

print("Visualization saved to cifar10_noise_examples.png") 