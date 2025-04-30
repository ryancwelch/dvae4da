#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add the parent directory to the path so we can import our modules
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

# Direct imports from our model files
from models.vae import VAE
from models.dvae import DVAE

# Configuration
data_dir = os.path.join(os.path.dirname(parent_dir), 'data')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Simple noise function to avoid dependencies
def add_gaussian_noise(images, std=0.1):
    return images + torch.randn_like(images) * std

def test_vae_shapes():
    """Test VAE model for shape compatibility"""
    
    # Data loading with transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    # Load MNIST dataset
    mnist_data = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    
    # Subsample the dataset for quick testing
    subset_indices = list(range(1000))  # Use 1000 samples
    mnist_subset = Subset(mnist_data, subset_indices)
    
    # Create DataLoader
    batch_size = 64
    train_loader = DataLoader(mnist_subset, batch_size=batch_size, shuffle=True)
    
    # Print data shapes for debugging
    for images, labels in train_loader:
        print(f"Input batch shape: {images.shape}")
        break
    
    # Initialize models with correct parameters
    img_channels = 1  # MNIST has 1 channel (grayscale)
    img_size = 28  # MNIST images are 28x28
    hidden_dims = [32, 64, 128]  # Use smaller architecture for 28x28 images
    latent_dim = 20
    
    print("Testing VAE model...")
    vae = VAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim
    )
    
    # Test forward pass
    images = images.to(device)
    vae = vae.to(device)
    
    # Debug forward pass
    try:
        recon_batch, mu, logvar = vae(images)
        print(f"Input shape: {images.shape}")
        print(f"Reconstruction shape: {recon_batch.shape}")
        print(f"mu shape: {mu.shape}")
        print(f"logvar shape: {logvar.shape}")
        print("Forward pass successful!")
    except Exception as e:
        print(f"Forward pass error: {e}")
        
        # Additional debugging
        if "size mismatch" in str(e):
            print("\nPotential shape mismatch in model. Checking model dimensions:")
            print(f"Expected input shape: [{batch_size}, {img_channels}, {img_size}, {img_size}]")
            print(f"Actual input shape: {images.shape}")
    
    # Test DVAE as well
    print("\nTesting DVAE model...")
    try:
        dvae = DVAE(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim
        )
        dvae = dvae.to(device)
        
        # Add noise to test images
        noise_stddev = 0.3
        noisy_images = add_gaussian_noise(images, noise_stddev)
        
        # Try forward pass
        recon_batch, mu, logvar = dvae(noisy_images)
        print(f"Noisy input shape: {noisy_images.shape}")
        print(f"DVAE output shape: {recon_batch.shape}")
        print("DVAE forward pass successful!")
    except Exception as e:
        print(f"DVAE forward pass error: {e}")

def suggest_fix():
    """Print suggested fixes based on common shape issues"""
    print("\nSuggested fixes for shape issues:")
    print("1. VAE and DVAE expect inputs in [B, C, H, W] format (batch, channels, height, width)")
    print("2. Make sure your input tensors have the right shape before passing to the models")
    print("3. Check that the VAE/DVAE parameters match the image dimensions")
    print("4. The encoder/decoder architectures are designed for specific image sizes, ensure compatibility")
    print("5. If using custom hidden_dims, make sure they're properly configured for both encoder and decoder")

if __name__ == "__main__":
    test_vae_shapes()
    suggest_fix()
