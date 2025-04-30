import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import seaborn as sns
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, get_cifar10_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model
from src.utils.training import Trainer, compute_psnr
from src.utils.visualization import (
    visualize_reconstructions, 
    visualize_generated_samples,
    visualize_latent_space
)


def calculate_fid(real_features: np.ndarray, generated_features: np.ndarray) -> float:
    """
    Calculate Fréchet Inception Distance (FID) between two sets of features.
    
    This is a simplified implementation of FID using the feature vectors directly.
    
    Args:
        real_features: Features of real images [N, d]
        generated_features: Features of generated images [N, d]
        
    Returns:
        FID score (lower is better)
    """
    # Calculate mean and covariance of real features
    mu1 = np.mean(real_features, axis=0)
    sigma1 = np.cov(real_features, rowvar=False)
    
    # Calculate mean and covariance of generated features
    mu2 = np.mean(generated_features, axis=0)
    sigma2 = np.cov(generated_features, rowvar=False)
    
    # Calculate square root of product of covariances
    # (add small epsilon to avoid numerical issues)
    epsilon = 1e-6
    covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    # Calculate FID
    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2*covmean)
    
    return fid


def calculate_diversity(latent_vectors: np.ndarray) -> float:
    """
    Calculate diversity score of latent vectors.
    
    Using average pairwise L2 distance as a diversity metric.
    
    Args:
        latent_vectors: Latent vectors [N, d]
        
    Returns:
        Diversity score (higher is better)
    """
    from sklearn.metrics.pairwise import euclidean_distances
    
    # Sample a smaller set if there are too many vectors (for efficiency)
    if latent_vectors.shape[0] > 1000:
        indices = np.random.choice(latent_vectors.shape[0], 1000, replace=False)
        latent_vectors = latent_vectors[indices]
    
    # Calculate pairwise distances
    distances = euclidean_distances(latent_vectors)
    
    # Exclude self-distances (diagonal elements)
    mask = np.ones_like(distances, dtype=bool)
    np.fill_diagonal(mask, 0)
    
    # Calculate average distance
    diversity = np.mean(distances[mask])
    
    return diversity


def evaluate_classification_accuracy(latent_vectors: np.ndarray, labels: np.ndarray) -> Tuple[float, Dict]:
    """
    Evaluate classification accuracy using a simple KNN classifier on latent vectors.
    
    Args:
        latent_vectors: Latent vectors [N, d]
        labels: Ground truth labels [N]
        
    Returns:
        Tuple of (accuracy, classification report)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        latent_vectors, labels, test_size=0.3, random_state=42, stratify=labels
    )
    
    # Train a KNN classifier
    clf = KNeighborsClassifier(n_neighbors=5)
    clf.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return accuracy, report


def calculate_ssim(img1: torch.Tensor, img2: torch.Tensor) -> float:
    """
    Calculate Structural Similarity Index (SSIM) between two images.
    
    Args:
        img1: First image tensor [B, C, H, W]
        img2: Second image tensor [B, C, H, W]
        
    Returns:
        Average SSIM value
    """
    from skimage.metrics import structural_similarity as ssim
    import torch.nn.functional as F
    
    # Convert tensors to numpy arrays
    img1_np = img1.detach().cpu().numpy()
    img2_np = img2.detach().cpu().numpy()
    
    batch_size = img1_np.shape[0]
    ssim_values = []
    
    for i in range(batch_size):
        # For each image in the batch
        # SSIM operates on 2D images, so handle channels appropriately
        channels = img1_np.shape[1]
        
        if channels == 1:
            # Grayscale
            ssim_val = ssim(
                img1_np[i, 0], 
                img2_np[i, 0],
                data_range=1.0
            )
        else:
            # Multi-channel (calculate for each channel and average)
            ssim_val = np.mean([
                ssim(
                    img1_np[i, c], 
                    img2_np[i, c],
                    data_range=1.0
                )
                for c in range(channels)
            ])
        
        ssim_values.append(ssim_val)
    
    # Return average SSIM
    return np.mean(ssim_values)


def parse_args():
    parser = argparse.ArgumentParser(description="Quantitatively evaluate VAE and DVAE models")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    
    # Noise parameters
    parser.add_argument("--noise-type", type=str, default="gaussian",
                       choices=["gaussian", "salt_and_pepper", "blur", "block", "line_h", "line_v"],
                       help="Type of noise to add")
    parser.add_argument("--noise-factor", type=float, default=0.2, 
                        help="Noise factor (std for gaussian, probability for salt_and_pepper, etc.)")
    
    # Model parameters
    parser.add_argument("--hidden-dims", type=str, default="32,64,128", 
                        help="Hidden dimensions (comma-separated)")
    parser.add_argument("--latent-dims", type=str, default="16,32,64", 
                        help="Latent dimensions to compare (comma-separated)")
    parser.add_argument("--kl-weight", type=float, default=0.1, help="Weight for KL divergence term")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Evaluation parameters
    parser.add_argument("--num-samples", type=int, default=1000, 
                        help="Number of samples to generate for evaluation")
    
    # Experiment parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to train on (cuda or cpu)")
    parser.add_argument("--save-dir", type=str, default="results/quantitative_evaluation", 
                        help="Directory to save results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    
    return parser.parse_args()


def train_model(model_type: str,
               dataset_name: str,
               train_loader: torch.utils.data.DataLoader,
               val_loader: torch.utils.data.DataLoader,
               latent_dim: int,
               hidden_dims: List[int],
               kl_weight: float,
               epochs: int,
               lr: float,
               device: torch.device,
               save_dir: str,
               noise_type: str,
               noise_factor: float):
    """Train a model and return the trained model."""
    
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
    experiment_name = f"{model_type}_{dataset_name}_latent{latent_dim}"
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=lr,
        device=device,
        save_dir=save_dir,
        experiment_name=experiment_name,
        noise_type=noise_type,
        noise_params={'noise_factor': noise_factor}
    )
    
    # Train model
    print(f"Training {model_type.upper()} with latent dimension {latent_dim}...")
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=epochs,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    return model, trainer


def evaluate_model(model: nn.Module,
                  model_type: str,
                  trainer: Trainer,
                  test_loader: torch.utils.data.DataLoader,
                  device: torch.device,
                  num_samples: int = 1000):
    """Evaluate a model and return metrics."""
    
    model.eval()
    
    # Get a batch of test images
    batch = next(iter(test_loader))
    if len(batch) == 3:  # Noisy dataset returns (noisy_img, clean_img, label)
        noisy_imgs, clean_imgs, labels = batch
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
    else:  # Standard dataset returns (img, label)
        imgs, labels = batch
        clean_imgs = imgs.to(device)
        
        # Create noisy images manually
        from src.utils.noise import add_noise
        noise_params = {'noise_factor': trainer.noise_params.get('noise_factor', 0.2)}
        noisy_imgs = add_noise(clean_imgs, noise_type=trainer.noise_type, noise_params=noise_params)
    
    labels = labels.to(device)
    
    # Get reconstructions
    with torch.no_grad():
        if model_type == "vae":
            recon_imgs, mu, log_var = model(noisy_imgs)
        else:  # dvae
            recon_imgs, mu, log_var = model(noisy_imgs, clean_imgs)
    
    # Calculate reconstruction metrics
    psnr_val = compute_psnr(clean_imgs, recon_imgs).item()
    
    try:
        ssim_val = calculate_ssim(clean_imgs, recon_imgs)
    except:
        # If SSIM calculation fails, use a default value
        ssim_val = 0.0
        print("Warning: SSIM calculation failed. Setting to 0.0")
    
    # Get latent vectors for all test data
    latent_vectors, all_labels = trainer.encode_dataset(test_loader)
    
    # Calculate latent space metrics
    try:
        diversity = calculate_diversity(latent_vectors)
    except:
        diversity = 0.0
        print("Warning: Diversity calculation failed. Setting to 0.0")
    
    try:
        clf_accuracy, clf_report = evaluate_classification_accuracy(latent_vectors, all_labels)
    except:
        clf_accuracy = 0.0
        clf_report = {}
        print("Warning: Classification evaluation failed. Setting accuracy to 0.0")
    
    # Generate random samples
    with torch.no_grad():
        samples = model.sample(num_samples=min(100, num_samples), device=device)
    
    # Generate a visualization of latent space
    fig = visualize_latent_space(
        latent_vectors,
        all_labels,
        method='tsne',
        title=f"{model_type.upper()} Latent Space"
    )
    
    # Save the figure
    experiment_dir = os.path.join(trainer.save_dir, trainer.experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    fig.savefig(os.path.join(experiment_dir, "latent_space.png"))
    plt.close(fig)
    
    # Generate reconstructions visualization
    fig = visualize_reconstructions(
        clean_imgs[:8].cpu(),
        noisy_imgs[:8].cpu(),
        recon_imgs[:8].cpu(),
        title=f"{model_type.upper()} Reconstructions"
    )
    
    fig.savefig(os.path.join(experiment_dir, "reconstructions.png"))
    plt.close(fig)
    
    # Generate samples visualization
    fig = visualize_generated_samples(
        samples[:64].cpu(),
        grid_size=(8, 8),
        title=f"{model_type.upper()} Generated Samples"
    )
    
    fig.savefig(os.path.join(experiment_dir, "samples.png"))
    plt.close(fig)
    
    # Collect all metrics
    metrics = {
        'psnr': psnr_val,
        'ssim': ssim_val,
        'latent_diversity': diversity,
        'classification_accuracy': clf_accuracy,
        'classification_report': clf_report
    }
    
    return metrics


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Parse parameters
    hidden_dims = [int(dim) for dim in args.hidden_dims.split(",")]
    latent_dims = [int(dim) for dim in args.latent_dims.split(",")]
    
    # Set device
    device = torch.device(args.device)
    
    # Set up noise parameters
    noise_params = {
        'noise_factor': args.noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    if args.noise_type == 'salt_and_pepper':
        noise_params['salt_vs_pepper'] = 0.5
    elif args.noise_type in ['block', 'line_h', 'line_v']:
        noise_params['block_size'] = 4
    
    # Get datasets
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mnist":
        dataset_func = get_mnist_dataset
    else:  # cifar10
        dataset_func = get_cifar10_dataset
    
    train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=args.noise_type,
        noise_params=noise_params,
        download=True
    )
    
    test_dataset = dataset_func(
        root=args.data_dir,
        train=False,
        noise_type=args.noise_type,
        noise_params=noise_params,
        download=True
    )
    
    # Create data loaders
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
    
    # Dictionary to store all results
    all_results = {}
    
    # Train and evaluate models with different latent dimensions
    for latent_dim in latent_dims:
        print(f"\n{'='*80}")
        print(f"Evaluating with latent dimension: {latent_dim}")
        print(f"{'='*80}")
        
        # Train and evaluate VAE
        vae_model, vae_trainer = train_model(
            model_type="vae",
            dataset_name=args.dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kl_weight=args.kl_weight,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=args.save_dir,
            noise_type=args.noise_type,
            noise_factor=args.noise_factor
        )
        
        vae_metrics = evaluate_model(
            model=vae_model,
            model_type="vae",
            trainer=vae_trainer,
            test_loader=test_loader,
            device=device,
            num_samples=args.num_samples
        )
        
        # Train and evaluate DVAE
        dvae_model, dvae_trainer = train_model(
            model_type="dvae",
            dataset_name=args.dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            latent_dim=latent_dim,
            hidden_dims=hidden_dims,
            kl_weight=args.kl_weight,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=args.save_dir,
            noise_type=args.noise_type,
            noise_factor=args.noise_factor
        )
        
        dvae_metrics = evaluate_model(
            model=dvae_model,
            model_type="dvae",
            trainer=dvae_trainer,
            test_loader=test_loader,
            device=device,
            num_samples=args.num_samples
        )
        
        # Store results
        all_results[f"vae_latent{latent_dim}"] = vae_metrics
        all_results[f"dvae_latent{latent_dim}"] = dvae_metrics
        
        # Print current results
        print(f"\nResults for latent dimension {latent_dim}:")
        print(f"VAE PSNR: {vae_metrics['psnr']:.2f}, SSIM: {vae_metrics['ssim']:.4f}, Classification Accuracy: {vae_metrics['classification_accuracy']:.4f}")
        print(f"DVAE PSNR: {dvae_metrics['psnr']:.2f}, SSIM: {dvae_metrics['ssim']:.4f}, Classification Accuracy: {dvae_metrics['classification_accuracy']:.4f}")
    
    # Save all results
    with open(os.path.join(args.save_dir, "metrics.json"), 'w') as f:
        json.dump(all_results, f, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Create summary plots
    plt.figure(figsize=(12, 10))
    
    # PSNR comparison
    plt.subplot(2, 2, 1)
    vae_psnrs = [all_results[f"vae_latent{dim}"]["psnr"] for dim in latent_dims]
    dvae_psnrs = [all_results[f"dvae_latent{dim}"]["psnr"] for dim in latent_dims]
    
    plt.plot(latent_dims, vae_psnrs, 'o-', label='VAE')
    plt.plot(latent_dims, dvae_psnrs, 's-', label='DVAE')
    plt.xlabel('Latent Dimension')
    plt.ylabel('PSNR (dB)')
    plt.title('Reconstruction Quality (PSNR)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # SSIM comparison
    plt.subplot(2, 2, 2)
    vae_ssims = [all_results[f"vae_latent{dim}"]["ssim"] for dim in latent_dims]
    dvae_ssims = [all_results[f"dvae_latent{dim}"]["ssim"] for dim in latent_dims]
    
    plt.plot(latent_dims, vae_ssims, 'o-', label='VAE')
    plt.plot(latent_dims, dvae_ssims, 's-', label='DVAE')
    plt.xlabel('Latent Dimension')
    plt.ylabel('SSIM')
    plt.title('Structural Similarity (SSIM)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Classification accuracy comparison
    plt.subplot(2, 2, 3)
    vae_accs = [all_results[f"vae_latent{dim}"]["classification_accuracy"] for dim in latent_dims]
    dvae_accs = [all_results[f"dvae_latent{dim}"]["classification_accuracy"] for dim in latent_dims]
    
    plt.plot(latent_dims, vae_accs, 'o-', label='VAE')
    plt.plot(latent_dims, dvae_accs, 's-', label='DVAE')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Accuracy')
    plt.title('Classification Accuracy')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Latent diversity comparison
    plt.subplot(2, 2, 4)
    vae_divs = [all_results[f"vae_latent{dim}"]["latent_diversity"] for dim in latent_dims]
    dvae_divs = [all_results[f"dvae_latent{dim}"]["latent_diversity"] for dim in latent_dims]
    
    plt.plot(latent_dims, vae_divs, 'o-', label='VAE')
    plt.plot(latent_dims, dvae_divs, 's-', label='DVAE')
    plt.xlabel('Latent Dimension')
    plt.ylabel('Diversity')
    plt.title('Latent Space Diversity')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "metrics_comparison.png"))
    plt.close()
    
    print(f"\nEvaluation complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 