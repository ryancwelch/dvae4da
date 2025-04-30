import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
import sys
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, get_cifar10_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model
from src.utils.training import Trainer, compute_psnr
from src.utils.visualization import visualize_noise_examples, visualize_latent_space


def parse_args():
    parser = argparse.ArgumentParser(description="Compare DVAE performance with different noise types")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--subsample", type=int, default=0, 
                        help="Number of training samples to use (0 for all)")
    
    # Noise parameters
    parser.add_argument("--noise-types", type=str, default="gaussian,salt_and_pepper,blur,block,line_h",
                       help="Comma-separated list of noise types to evaluate")
    parser.add_argument("--noise-factors", type=str, default="0.1,0.3,0.5",
                        help="Comma-separated list of noise factors to evaluate")
    
    # Model parameters
    parser.add_argument("--hidden-dims", type=str, default="32,64,128", 
                        help="Hidden dimensions (comma-separated)")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent dimension size")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="Weight for KL divergence term")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Experiment parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="results/noise_experiment", 
                        help="Directory to save results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    
    return parser.parse_args()


def get_dataset(dataset_name: str,
                data_dir: str,
                train: bool,
                noise_type: str,
                noise_factor: float,
                return_pairs: bool = True,
                download: bool = True):
    """Get the specified dataset with noise applied."""
    
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    if noise_type == 'salt_and_pepper':
        noise_params['salt_vs_pepper'] = 0.5
    elif noise_type in ['block', 'line_h', 'line_v']:
        noise_params['block_size'] = 4
    
    if dataset_name == "mnist":
        dataset = get_mnist_dataset(
            root=data_dir,
            train=train,
            noise_type=noise_type,
            noise_params=noise_params,
            download=download
        )
    elif dataset_name == "cifar10":
        dataset = get_cifar10_dataset(
            root=data_dir,
            train=train,
            noise_type=noise_type,
            noise_params=noise_params,
            download=download
        )
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    # Set return_pairs based on parameter
    dataset.return_pairs = return_pairs
    
    return dataset


def train_and_evaluate(model_type: str,
                      dataset_name: str,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader,
                      noise_type: str,
                      noise_factor: float,
                      hidden_dims: List[int],
                      latent_dim: int,
                      kl_weight: float,
                      epochs: int,
                      lr: float,
                      device: torch.device,
                      save_dir: str):
    """Train and evaluate a model with specific noise settings."""
    
    # Determine dataset properties
    sample_batch = next(iter(train_loader))
    if len(sample_batch) == 3:  # Noisy dataset returns (noisy_img, clean_img, label)
        noisy_img, clean_img, _ = sample_batch
        img_channels = noisy_img.shape[1]
        img_size = noisy_img.shape[2]
    else:  # Standard dataset returns (img, label)
        img, _ = sample_batch
        img_channels = img.shape[1]
        img_size = img.shape[2]
    
    print(f"Training {model_type.upper()} on {dataset_name} with image size {img_size}x{img_size}, {img_channels} channels")
    
    # Adjust architecture based on dataset
    if dataset_name == "cifar10" and not hidden_dims:
        # Use a larger architecture for CIFAR-10 if not specified
        hidden_dims = [32, 64, 128, 256]
        latent_dim = max(latent_dim, 128)  # Ensure sufficient latent space
        print(f"Using adjusted architecture for CIFAR-10: hidden_dims={hidden_dims}, latent_dim={latent_dim}")
    
    # Create model
    if model_type == "vae":
        model = get_vae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    else:  # dvae
        model = get_dvae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    
    model.to(device)
    
    # Create experiment name
    experiment_name = f"{model_type}_{dataset_name}_{noise_type}_factor{noise_factor}"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=lr,
        device=device,
        save_dir=save_dir,
        experiment_name=experiment_name
    )
    
    # Train model
    print(f"Training {model_type.upper()} with {noise_type} noise (factor: {noise_factor})...")
    trainer.train(
        num_epochs=epochs,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # Evaluate on test set
    print("Evaluating on test set...")
    
    # Get a batch of test images for evaluation
    batch = next(iter(test_loader))
    if len(batch) == 3:  # Noisy dataset returns (noisy_img, clean_img, label)
        noisy_imgs, clean_imgs, labels = batch
    else:  # Standard dataset returns (img, label)
        clean_imgs, labels = batch
        # Create noisy images for standard dataset
        noise_params = {'noise_factor': noise_factor, 'clip_min': 0.0, 'clip_max': 1.0}
        if noise_type == 'salt_and_pepper':
            noise_params['salt_vs_pepper'] = 0.5
        elif noise_type in ['block', 'line_h', 'line_v']:
            noise_params['block_size'] = 4
        
        from src.utils.noise import add_noise
        noisy_imgs = add_noise(clean_imgs, noise_type=noise_type, noise_params=noise_params)
    
    noisy_imgs = noisy_imgs.to(device)[:16]  # Use first 16 images
    clean_imgs = clean_imgs.to(device)[:16]
    
    # Generate reconstructions
    with torch.no_grad():
        if model_type == "vae":
            recon_imgs, _, _ = model(noisy_imgs)
        else:  # dvae
            recon_imgs, _, _ = model(noisy_imgs, clean_imgs)
    
    # Move these tensors to CPU before visualization
    clean_imgs_cpu = clean_imgs.detach().cpu()
    noisy_imgs_cpu = noisy_imgs.detach().cpu()
    recon_imgs_cpu = recon_imgs.detach().cpu()
    
    # Debug information
    print(f"Clean images shape: {clean_imgs_cpu.shape}, min: {clean_imgs_cpu.min().item():.4f}, max: {clean_imgs_cpu.max().item():.4f}")
    print(f"Noisy images shape: {noisy_imgs_cpu.shape}, min: {noisy_imgs_cpu.min().item():.4f}, max: {noisy_imgs_cpu.max().item():.4f}")
    print(f"Recon images shape: {recon_imgs_cpu.shape}, min: {recon_imgs_cpu.min().item():.4f}, max: {recon_imgs_cpu.max().item():.4f}")
    
    # Compute PSNR
    noisy_psnr = compute_psnr(clean_imgs_cpu, noisy_imgs_cpu)
    recon_psnr = compute_psnr(clean_imgs_cpu, recon_imgs_cpu)
    psnr_improvement = recon_psnr - noisy_psnr
    
    print(f"Noisy PSNR: {noisy_psnr:.2f} dB")
    print(f"Reconstructed PSNR: {recon_psnr:.2f} dB")
    print(f"PSNR improvement: {psnr_improvement:.2f} dB")
    
    # Add extra normalization to ensure values are properly scaled for visualization
    def normalize_for_viz(img_tensor):
        img_min = img_tensor.min()
        img_max = img_tensor.max()
        if img_max > img_min:
            return (img_tensor - img_min) / (img_max - img_min)
        return img_tensor

    # Normalize the images for better visualization
    clean_imgs_viz = normalize_for_viz(clean_imgs_cpu)
    noisy_imgs_viz = normalize_for_viz(noisy_imgs_cpu)
    recon_imgs_viz = normalize_for_viz(recon_imgs_cpu)

    # Save sample images for debugging
    debug_dir = os.path.join(save_dir, experiment_name, "debug")
    os.makedirs(debug_dir, exist_ok=True)

    # Save first clean, noisy, and reconstructed image as PNG files
    for i in range(min(3, len(clean_imgs_cpu))):
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 3, 1)
        plt.imshow(clean_imgs_cpu[i, 0].numpy(), cmap='gray')
        plt.title(f"Clean {i}")
        plt.colorbar()
        
        plt.subplot(1, 3, 2)
        plt.imshow(noisy_imgs_cpu[i, 0].numpy(), cmap='gray')
        plt.title(f"Noisy {i}")
        plt.colorbar()
        
        plt.subplot(1, 3, 3)
        plt.imshow(recon_imgs_cpu[i, 0].numpy(), cmap='gray')
        plt.title(f"Recon {i}")
        plt.colorbar()
        
        plt.tight_layout()
        plt.savefig(os.path.join(debug_dir, f"sample_image_{i}.png"))
        plt.close()

    # Then use the normalized images for the main visualization
    fig = visualize_noise_examples(
        clean_imgs_viz[:8],
        noisy_imgs_viz[:8],
        recon_imgs_viz[:8],
        num_examples=8,
        title=f"{model_type.upper()} Denoising: {noise_type.replace('_', ' ')} noise (factor: {noise_factor})"
    )
    
    os.makedirs(os.path.join(save_dir, experiment_name), exist_ok=True)
    fig.savefig(os.path.join(save_dir, experiment_name, "noise_examples.png"))
    plt.close(fig)
    
    # Visualize latent space
    latent_codes, labels = trainer.encode_dataset(test_loader)
    
    fig = visualize_latent_space(
        latent_codes,
        labels,
        method='tsne',
        title=f"{model_type.upper()} Latent Space: {noise_type} noise (factor: {noise_factor})"
    )
    
    fig.savefig(os.path.join(save_dir, experiment_name, "latent_space.png"))
    plt.close(fig)
    
    # Return metrics (handling both tensor and float cases)
    return {
        'model_type': model_type,
        'noise_type': noise_type,
        'noise_factor': noise_factor,
        'noisy_psnr': noisy_psnr.item() if torch.is_tensor(noisy_psnr) else noisy_psnr,
        'recon_psnr': recon_psnr.item() if torch.is_tensor(recon_psnr) else recon_psnr,
        'psnr_improvement': psnr_improvement.item() if torch.is_tensor(psnr_improvement) else psnr_improvement
    }


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Parse parameters
    noise_types = args.noise_types.split(",")
    noise_factors = [float(f) for f in args.noise_factors.split(",")]
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    
    # Track all results
    all_results = []
    
    # Set device
    device = torch.device(args.device)
    
    for noise_type in noise_types:
        for noise_factor in noise_factors:
            print(f"\n{'='*80}")
            print(f"Experiment: {args.dataset} with {noise_type} noise (factor: {noise_factor})")
            print(f"{'='*80}\n")
            
            # VAE Experiment
            print("\nTraining and evaluating VAE...")
            
            # Get datasets for VAE (return_pairs=False for (img, label) format)
            vae_train_dataset = get_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                train=True,
                noise_type=noise_type,
                noise_factor=noise_factor,
                return_pairs=False,
                download=True
            )
            
            vae_test_dataset = get_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                train=False,
                noise_type=noise_type,
                noise_factor=noise_factor,
                return_pairs=False,
                download=True
            )
            
            # Subsample if requested
            if args.subsample > 0:
                vae_train_dataset = subsample_dataset(
                    vae_train_dataset, 
                    args.subsample,
                    stratified=True
                )
                print(f"Subsampled VAE training dataset to {len(vae_train_dataset)} examples")
            
            # Create data loaders for VAE
            vae_train_loader, vae_val_loader = create_data_loaders(
                vae_train_dataset,
                batch_size=args.batch_size,
                val_split=0.1,
                shuffle=True,
                num_workers=args.num_workers
            )
            
            vae_test_loader, _ = create_data_loaders(
                vae_test_dataset,
                batch_size=args.batch_size,
                val_split=0.0,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            # Train and evaluate VAE
            vae_results = train_and_evaluate(
                model_type="vae",
                dataset_name=args.dataset,
                train_loader=vae_train_loader,
                val_loader=vae_val_loader,
                test_loader=vae_test_loader,
                noise_type=noise_type,
                noise_factor=noise_factor,
                hidden_dims=hidden_dims,
                latent_dim=args.latent_dim,
                kl_weight=args.kl_weight,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                save_dir=args.save_dir
            )
            
            all_results.append(vae_results)
            
            # DVAE Experiment
            print("\nTraining and evaluating DVAE...")
            
            # Get datasets for DVAE (return_pairs=True for (noisy_img, clean_img, label) format)
            dvae_train_dataset = get_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                train=True,
                noise_type=noise_type,
                noise_factor=noise_factor,
                return_pairs=True,
                download=True
            )
            
            dvae_test_dataset = get_dataset(
                dataset_name=args.dataset,
                data_dir=args.data_dir,
                train=False,
                noise_type=noise_type,
                noise_factor=noise_factor,
                return_pairs=True,
                download=True
            )
            
            # Subsample if requested
            if args.subsample > 0:
                dvae_train_dataset = subsample_dataset(
                    dvae_train_dataset, 
                    args.subsample,
                    stratified=True
                )
                print(f"Subsampled DVAE training dataset to {len(dvae_train_dataset)} examples")
            
            # Create data loaders for DVAE
            dvae_train_loader, dvae_val_loader = create_data_loaders(
                dvae_train_dataset,
                batch_size=args.batch_size,
                val_split=0.1,
                shuffle=True,
                num_workers=args.num_workers
            )
            
            dvae_test_loader, _ = create_data_loaders(
                dvae_test_dataset,
                batch_size=args.batch_size,
                val_split=0.0,
                shuffle=False,
                num_workers=args.num_workers
            )
            
            # Train and evaluate DVAE
            dvae_results = train_and_evaluate(
                model_type="dvae",
                dataset_name=args.dataset,
                train_loader=dvae_train_loader,
                val_loader=dvae_val_loader,
                test_loader=dvae_test_loader,
                noise_type=noise_type,
                noise_factor=noise_factor,
                hidden_dims=hidden_dims,
                latent_dim=args.latent_dim,
                kl_weight=args.kl_weight,
                epochs=args.epochs,
                lr=args.lr,
                device=device,
                save_dir=args.save_dir
            )
            
            all_results.append(dvae_results)
    
    # Summarize results
    print("\nResults Summary:")
    print(f"{'Model Type':<10} {'Noise Type':<15} {'Noise Factor':<12} {'Noisy PSNR':<12} {'Recon PSNR':<12} {'Improvement':<12}")
    print("-" * 80)
    
    for result in all_results:
        print(f"{result['model_type']:<10} {result['noise_type']:<15} {result['noise_factor']:<12.2f} "
              f"{result['noisy_psnr']:<12.2f} {result['recon_psnr']:<12.2f} {result['psnr_improvement']:<12.2f}")
    
    # Create comparison bar chart
    plt.figure(figsize=(15, 10))
    
    noise_configs = [f"{r['noise_type']}_{r['noise_factor']:.1f}" for r in all_results[::2]]
    vae_improvements = [r['psnr_improvement'] for r in all_results[::2]]
    dvae_improvements = [r['psnr_improvement'] for r in all_results[1::2]]
    
    x = np.arange(len(noise_configs))
    width = 0.35
    
    plt.bar(x - width/2, vae_improvements, width, label='VAE')
    plt.bar(x + width/2, dvae_improvements, width, label='DVAE')
    
    plt.xlabel('Noise Configuration')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Denoising Performance Comparison')
    plt.xticks(x, noise_configs, rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(os.path.join(args.save_dir, "noise_comparison.png"))
    
    # Save numerical results
    with open(os.path.join(args.save_dir, "results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nExperiment complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 