import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import get_vae_model
from src.utils.training import Trainer
from src.data import subsample_dataset, create_data_loaders


def get_cifar10_dataset(root: str, train: bool = True, download: bool = True):
    """
    Get CIFAR-10 dataset.
    
    Args:
        root: Root directory to store dataset
        train: Whether to get training set
        download: Whether to download dataset if not present
        
    Returns:
        CIFAR-10 dataset
    """
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    dataset = datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    return dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a VAE model on CIFAR-10 dataset")
    
    # Dataset parameters
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--subsample", type=int, default=None, 
                        help="Number of training samples to use (default: use all)")
    
    # Model parameters
    parser.add_argument("--hidden-dims", type=str, default="32,64,128,256", 
                        help="Hidden dimensions (comma-separated)")
    parser.add_argument("--latent-dim", type=int, default=128, help="Latent dimension size")
    parser.add_argument("--kl-weight", type=float, default=0.001, help="Weight for KL divergence term")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to train on (cuda or cpu)")
    
    # Output parameters
    parser.add_argument("--save-dir", type=str, default="results/vae_cifar10", 
                        help="Directory to save model and results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    print("Loading CIFAR-10 dataset...")
    train_dataset = get_cifar10_dataset(
        root=args.data_dir,
        train=True,
        download=True
    )
    test_dataset = get_cifar10_dataset(
        root=args.data_dir,
        train=False,
        download=True
    )
    
    if args.subsample is not None:
        train_dataset = subsample_dataset(
            train_dataset, 
            args.subsample,
            stratified=True
        )
        print(f"Subsampled training dataset to {len(train_dataset)} examples")
    
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        batch_size=args.batch_size,
        val_split=0.1,
        shuffle=True,
        num_workers=4
    )
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=args.batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=4
    )
    
    # Parse hidden dimensions
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    # Create model
    print("Creating VAE model...")
    model = get_vae_model(
        img_channels=3,  # CIFAR-10 has 3 color channels
        img_size=32,     # CIFAR-10 images are 32x32
        hidden_dims=hidden_dims,
        latent_dim=args.latent_dim,
        kl_weight=args.kl_weight
    )
    model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=args.lr,
        device=device,
        model_save_path=os.path.join(args.save_dir, "vae_cifar10_model.pt")
    )
    
    # Train model
    print("Training VAE model...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs
    )
    
    # Save loss plots
    trainer.plot_losses()
    plt.savefig(os.path.join(args.save_dir, "vae_cifar10_losses.png"))
    
    # Generate and save sample reconstructions
    print("Generating sample reconstructions...")
    trainer.generate_reconstructions(test_loader, num_samples=10, 
                                    save_path=os.path.join(args.save_dir, "vae_cifar10_reconstructions.png"))
    
    # Generate random samples
    print("Generating random samples...")
    trainer.generate_samples(num_samples=100, grid_size=(10, 10), 
                            save_path=os.path.join(args.save_dir, "vae_cifar10_samples.png"))
    
    print(f"Training complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 