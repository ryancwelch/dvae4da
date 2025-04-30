import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, create_data_loaders, subsample_dataset
from src.models import get_dvae_model
from src.utils.training import Trainer, compute_psnr
from src.utils.visualization import (
    visualize_noise_examples,
    visualize_reconstructions, 
    visualize_generated_samples,
    visualize_latent_space
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a DVAE on MNIST with noise")
    
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--subsample", type=int, default=0, help="Number of training samples to use (0 for all)")
    
    parser.add_argument("--noise-type", type=str, default="gaussian", 
                       choices=["gaussian", "salt_and_pepper", "block", "line_h", "line_v"],
                       help="Type of noise to add")
    parser.add_argument("--noise-factor", type=float, default=0.2, 
                       help="Noise factor (std for gaussian, probability for salt_and_pepper, etc.)")
    
    parser.add_argument("--img-size", type=int, default=28, help="Size of the input images")
    parser.add_argument("--latent-dim", type=int, default=16, help="Dimension of the latent space")
    parser.add_argument("--hidden-dims", type=str, default="32,64,128,256", 
                       help="Dimensions of hidden layers (comma-separated)")
    parser.add_argument("--kl-weight", type=float, default=1.0, 
                       help="Weight for the KL divergence term in the loss")
    
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    parser.add_argument("--early-stopping", action="store_true", help="Whether to use early stopping")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping")
    parser.add_argument("--save-interval", type=int, default=5, help="Interval for saving model checkpoints")
    parser.add_argument("--save-dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--experiment-name", type=str, default=None, help="Name of the experiment")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to train on (cuda or cpu)")
    
    return parser.parse_args()


def main():
    """Main function to train a DVAE on MNIST with noise."""
    args = parse_args()
    
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    device = torch.device(args.device)
    
    noise_params = {
        'noise_factor': args.noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    if args.noise_type == 'salt_and_pepper':
        noise_params['salt_vs_pepper'] = 0.5
    elif args.noise_type in ['block', 'line_h', 'line_v']:
        noise_params['block_size'] = 4
    
    train_dataset = get_mnist_dataset(
        root=args.data_dir,
        train=True,
        noise_type=args.noise_type,
        noise_params=noise_params,
        download=True
    )
    test_dataset = get_mnist_dataset(
        root=args.data_dir,
        train=False,
        noise_type=args.noise_type,
        noise_params=noise_params,
        download=True
    )
    
    if args.subsample > 0:
        train_dataset = subsample_dataset(train_dataset, args.subsample, stratified=True)
        print(f"Subsampled training dataset to {len(train_dataset)} examples")
    
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        batch_size=args.batch_size,
        val_split=0.1,
        shuffle=True,
        num_workers=args.num_workers
    )
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=args.batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    model = get_dvae_model(
        img_channels=1,
        img_size=args.img_size,
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight
    )
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=device,
        save_dir=args.save_dir,
        experiment_name=args.experiment_name
    )
    
    print(f"Training DVAE with latent dimension {args.latent_dim} on MNIST with {args.noise_type} noise")
    print(f"Noise factor: {args.noise_factor}")
    print(f"Model architecture: {hidden_dims}")
    print(f"Training on {len(train_loader.dataset)} examples")
    print(f"Validating on {len(val_loader.dataset)} examples")
    
    trainer.train(
        num_epochs=args.epochs,
        save_interval=args.save_interval,
        early_stopping=args.early_stopping,
        patience=args.patience,
        save_best_only=True
    )
    
    print("Generating samples...")
    samples = trainer.generate_samples(n_samples=16)
    
    fig = visualize_generated_samples(
        samples,
        num_examples=16,
        grid_size=(4, 4),
        title="DVAE Generated Samples"
    )
    
    fig.savefig(os.path.join(trainer.sample_save_dir, "generated_samples.png"))
    plt.close(fig)
    
    print("Visualizing noise examples and reconstructions...")
    batch = next(iter(test_loader))
    noisy_imgs, clean_imgs, _ = batch
    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)
    
    with torch.no_grad():
        recon_imgs, _, _ = model(noisy_imgs)
    
    noisy_psnr = compute_psnr(clean_imgs, noisy_imgs)
    recon_psnr = compute_psnr(clean_imgs, recon_imgs)
    print(f"Noisy PSNR: {noisy_psnr:.2f} dB")
    print(f"Reconstructed PSNR: {recon_psnr:.2f} dB")
    print(f"PSNR improvement: {recon_psnr - noisy_psnr:.2f} dB")
    
    fig = visualize_noise_examples(
        clean_imgs[:8],
        noisy_imgs[:8],
        recon_imgs[:8],
        num_examples=8,
        title=f"DVAE Denoising: {args.noise_type.replace('_', ' ')} noise (factor: {args.noise_factor})"
    )
    
    fig.savefig(os.path.join(trainer.sample_save_dir, "noise_examples.png"))
    plt.close(fig)
    
    print("Visualizing latent space...")
    latent_codes, labels = trainer.encode_dataset(test_loader)
    
    fig = visualize_latent_space(
        latent_codes,
        labels,
        method='tsne',
        title="DVAE Latent Space (t-SNE)"
    )
    
    fig.savefig(os.path.join(trainer.sample_save_dir, "latent_space.png"))
    plt.close(fig)
    
    print(f"Training complete. Results saved in {trainer.save_dir}/{trainer.experiment_name}")


if __name__ == "__main__":
    main() 