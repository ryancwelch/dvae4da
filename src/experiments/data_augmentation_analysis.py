import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import numpy as np
import os
import sys
import argparse
from typing import Dict, Tuple
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models.vae import VAE
from src.models.dvae import DVAE, ConditionalDVAE
from src.utils.training import Trainer
from src.data import get_mnist_dataset, get_cifar10_dataset, subsample_dataset, create_data_loaders
from src.experiments.data_augmentation_experiment import SimpleCNN, train_classifier, evaluate_classifier


def check_image_dimensions(loader: DataLoader, name: str):
    """Check and print the dimensions of images in a data loader."""
    for batch in loader:
        if len(batch) == 2:  # (image, label)
            images, _ = batch
            print(f"{name} dataset image dimensions: {images.shape}")
        elif len(batch) == 3:  # (noisy_img, clean_img, label)
            noisy_imgs, clean_imgs, _ = batch
            print(f"{name} dataset noisy image dimensions: {noisy_imgs.shape}")
            print(f"{name} dataset clean image dimensions: {clean_imgs.shape}")
        break


def ensure_consistent_dimensions(tensor_data: torch.Tensor, target_size: int = 28) -> torch.Tensor:
    """Ensure tensor data has the correct dimensions."""
    # Check if the tensor has the right dimensions
    if tensor_data.shape[2:] != (target_size, target_size):
        print(f"Resizing tensor from {tensor_data.shape[2:]} to {target_size}x{target_size}")
        # Resize using bilinear interpolation
        return torch.nn.functional.interpolate(
            tensor_data, 
            size=(target_size, target_size), 
            mode='bilinear', 
            align_corners=False
        )
    return tensor_data


def run_experiment(
    samples_per_class: int = 100,
    noise_factor: float = 0.5,
    noise_type: str = "gaussian",
    dataset_name: str = "mnist",
    num_epochs_vae: int = 20,
    num_epochs_clf: int = 5,
    save_dir: str = "results/data_aug_analysis",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 4,
    seed: int = 42,
    use_conditional: bool = False
) -> Dict:
    """
    Run the data augmentation experiment and return accuracy metrics.
    
    Args:
        samples_per_class: Number of training samples per class to use
        noise_factor: Noise factor to apply
        noise_type: Type of noise to apply
        dataset_name: Dataset to use (mnist or cifar10)
        num_epochs_vae: Number of epochs to train VAE/DVAE models
        num_epochs_clf: Number of epochs to train classifiers
        save_dir: Directory to save results
        device: Device to use
        batch_size: Batch size for training
        val_split: Validation split ratio
        num_workers: Number of workers for data loading
        seed: Random seed
        use_conditional: Whether to use conditional DVAE
        
    Returns:
        Dictionary containing accuracy metrics for each method
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    
    # Create experiment directory
    exp_dir = os.path.join(save_dir, f"{dataset_name}_samples{samples_per_class}_noise{noise_factor}")
    os.makedirs(exp_dir, exist_ok=True)
    
    # Set device
    device = torch.device(device)
    
    # Dataset settings
    if dataset_name == "mnist":
        dataset_func = get_mnist_dataset
        img_channels = 1
        img_size = 28
        num_classes = 10
    elif dataset_name == "cifar10":
        dataset_func = get_cifar10_dataset
        img_channels = 3
        img_size = 32
        num_classes = 10
    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    # Model architecture - adjusted based on dataset
    if dataset_name == "mnist":
        hidden_dims = [32, 64, 128]
        latent_dim = 32
    else:  # CIFAR-10 needs more capacity
        hidden_dims = [32, 64, 128, 256]
        latent_dim = 128
    
    kl_weight = 0.1
    
    # Noise parameters
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    print(f"Running experiment on {dataset_name} with {noise_type} noise (factor: {noise_factor})")
    print(f"Model architecture: hidden_dims={hidden_dims}, latent_dim={latent_dim}")
    
    # 1. Train VAE
    print(f"Training VAE on {samples_per_class} samples per class with {noise_type} noise (factor: {noise_factor})...")
    
    # Get VAE training dataset
    vae_train_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    vae_train_dataset.return_pairs = False
    
    # Subsample the dataset
    print(f"Subsampling the dataset to {samples_per_class} samples per class...")
    vae_train_dataset = subsample_dataset(
        vae_train_dataset,
        num_samples=samples_per_class * num_classes,
        stratified=True
    )
    print(f"VAE training dataset size after subsampling: {len(vae_train_dataset)}")
    
    # Create data loaders
    vae_train_loader, vae_val_loader = create_data_loaders(
        vae_train_dataset,
        batch_size=batch_size,
        val_split=val_split,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create and train VAE
    vae_model = VAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )
    vae_model.to(device)
    
    vae_trainer = Trainer(
        model=vae_model,
        train_loader=vae_train_loader,
        val_loader=vae_val_loader,
        device=device,
        save_dir=exp_dir,
        experiment_name="vae"
    )
    
    vae_trainer.train(
        num_epochs=num_epochs_vae,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # 2. Train DVAE
    print(f"Training DVAE on {samples_per_class} samples per class with {noise_type} noise (factor: {noise_factor})...")
    
    # Get DVAE training dataset
    dvae_train_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_train_dataset.return_pairs = True
    
    # Subsample the dataset
    print(f"Subsampling the dataset to {samples_per_class} samples per class...")
    dvae_train_dataset = subsample_dataset(
        dvae_train_dataset,
        num_samples=samples_per_class * num_classes,
        stratified=True
    )
    print(f"DVAE training dataset size after subsampling: {len(dvae_train_dataset)}")
    
    # Create data loaders
    dvae_train_loader, dvae_val_loader = create_data_loaders(
        dvae_train_dataset,
        batch_size=batch_size,
        val_split=val_split,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Create and train DVAE
    dvae_model = DVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=0.1  # Using lower KL weight
    )
    dvae_model.to(device)
    
    dvae_trainer = Trainer(
        model=dvae_model,
        train_loader=dvae_train_loader,
        val_loader=dvae_val_loader,
        lr=3e-4,  # Slightly lower learning rate for better convergence
        device=device,
        save_dir=exp_dir,
        experiment_name="dvae"
    )
    
    dvae_trainer.train(
        num_epochs=num_epochs_vae,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # 3. Train ConditionalDVAE if requested
    cdvae_model = None
    cdvae_recon_train_loader = None
    cdvae_recon_val_loader = None
    
    if use_conditional:
        print(f"Training ConditionalDVAE on {samples_per_class} samples per class with {noise_type} noise (factor: {noise_factor})...")
        
        # Dataset is the same as for DVAE (contains noisy_img, clean_img, label)
        # Create and train ConditionalDVAE
        cdvae_model = ConditionalDVAE(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_classes=num_classes,
            kl_weight=0.05  # Even lower KL weight for conditional model
        )
        
        cdvae_trainer = Trainer(
            model=cdvae_model,
            train_loader=dvae_train_loader,  # Same loader as DVAE
            val_loader=dvae_val_loader,      # Same loader as DVAE
            lr=2e-4,  # Even lower learning rate for better conditioning
            device=device,
            save_dir=exp_dir,
            experiment_name="cdvae"
        )
        
        cdvae_trainer.train(
            num_epochs=num_epochs_vae,
            early_stopping=True,
            patience=5,
            save_best_only=True
        )
    
    # 3. Setup test dataset
    print("Preparing test data for evaluation...")
    test_dataset = dataset_func(
        root="./data",
        train=False,
        noise_type=None,
        download=True
    )
    test_dataset.return_pairs = False
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=num_workers
    )
    
    # 4. Prepare datasets for classifier training
    
    # 4.1 Original noisy dataset
    print("Preparing original noisy dataset...")
    noisy_train_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    noisy_train_dataset.return_pairs = False
    
    # Subsample
    noisy_train_dataset = subsample_dataset(
        noisy_train_dataset,
        num_samples=samples_per_class * num_classes,
        stratified=True
    )
    print(f"Noisy classifier training dataset size: {len(noisy_train_dataset)}")
    
    noisy_train_loader, noisy_val_loader = create_data_loaders(
        noisy_train_dataset,
        batch_size=batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 4.2 VAE reconstructed dataset
    print("Creating VAE reconstructed dataset...")
    
    # Get the test set for reconstruction
    vae_recon_data = []
    vae_recon_labels = []
    
    vae_model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(noisy_train_loader, desc="VAE reconstructions"):
            batch_data = batch_data.to(device)
            recons, _, _ = vae_model(batch_data)
            
            # Ensure consistent dimensions
            recons = ensure_consistent_dimensions(recons)
            
            vae_recon_data.append(recons.cpu())
            vae_recon_labels.append(batch_labels)
    
    vae_recon_data = torch.cat(vae_recon_data, dim=0)
    vae_recon_labels = torch.cat(vae_recon_labels, dim=0)
    
    vae_recon_dataset = TensorDataset(vae_recon_data, vae_recon_labels)
    vae_recon_train_loader, vae_recon_val_loader = create_data_loaders(
        vae_recon_dataset,
        batch_size=batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 4.3 DVAE reconstructed dataset
    print("Creating DVAE reconstructed dataset...")
    dvae_recon_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_recon_dataset.return_pairs = True
    
    # Subsample
    dvae_recon_dataset = subsample_dataset(
        dvae_recon_dataset,
        num_samples=samples_per_class * num_classes,
        stratified=True
    )
    
    dvae_recon_loader = DataLoader(
        dvae_recon_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )
    
    # Reconstruct images using DVAE
    dvae_recon_data = []
    dvae_recon_labels = []
    
    dvae_model.eval()
    with torch.no_grad():
        for batch in tqdm(dvae_recon_loader, desc="DVAE reconstructions"):
            noisy_img, clean_img, labels = batch
            noisy_img = noisy_img.to(device)
            
            # DVAE forward only needs noisy_img for inference
            recons, _, _ = dvae_model(noisy_img)
            
            # Ensure consistent dimensions
            recons = ensure_consistent_dimensions(recons)
            
            dvae_recon_data.append(recons.cpu())
            dvae_recon_labels.append(labels)
    
    dvae_recon_data = torch.cat(dvae_recon_data, dim=0)
    dvae_recon_labels = torch.cat(dvae_recon_labels, dim=0)
    
    dvae_recon_dataset = TensorDataset(dvae_recon_data, dvae_recon_labels)
    dvae_recon_train_loader, dvae_recon_val_loader = create_data_loaders(
        dvae_recon_dataset,
        batch_size=batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=num_workers
    )
    
    # 4.4 ConditionalDVAE reconstructed dataset (if requested)
    cdvae_recon_data = None
    cdvae_recon_labels = None
    cdvae_recon_dataset = None
    cdvae_recon_train_loader = None
    cdvae_recon_val_loader = None
    
    if use_conditional and cdvae_model is not None:
        print("Creating ConditionalDVAE reconstructed dataset...")
        # Use the create_cdvae_dataset function to properly handle the reconstructions
        cdvae_model.to(device)
        cdvae_recon_dataset = create_cdvae_dataset(cdvae_model, dvae_recon_dataset, device)
        
        cdvae_recon_train_loader, cdvae_recon_val_loader = create_data_loaders(
            cdvae_recon_dataset,
            batch_size=batch_size,
            val_split=0.2,
            shuffle=True,
            num_workers=num_workers
        )
    
    # Train classifiers on each dataset
    
    # 5.1 Baseline (using noisy data)
    print("\nTraining classifier on noisy data...")
    noisy_classifier = SimpleCNN(img_channels, img_size, num_classes).to(device)
    noisy_results = train_classifier(
        model=noisy_classifier,
        train_loader=noisy_train_loader,
        val_loader=noisy_val_loader,
        device=device,
        num_epochs=num_epochs_clf
    )
    
    # Evaluate on test set
    noisy_test_results = evaluate_classifier(noisy_classifier, test_loader, device)
    noisy_results.update(noisy_test_results)
    
    # 5.2 VAE reconstructed data
    print("\nTraining classifier on VAE reconstructed data...")
    vae_classifier = SimpleCNN(img_channels, img_size, num_classes).to(device)
    vae_results = train_classifier(
        model=vae_classifier,
        train_loader=vae_recon_train_loader,
        val_loader=vae_recon_val_loader,
        device=device,
        num_epochs=num_epochs_clf
    )
    
    # Evaluate on test set
    vae_test_results = evaluate_classifier(vae_classifier, test_loader, device)
    vae_results.update(vae_test_results)
    
    # 5.3 DVAE reconstructed data
    print("\nTraining classifier on DVAE reconstructed data...")
    dvae_classifier = SimpleCNN(img_channels, img_size, num_classes).to(device)
    dvae_results = train_classifier(
        model=dvae_classifier,
        train_loader=dvae_recon_train_loader,
        val_loader=dvae_recon_val_loader,
        device=device,
        num_epochs=num_epochs_clf
    )
    
    # Evaluate on test set
    dvae_test_results = evaluate_classifier(dvae_classifier, test_loader, device)
    dvae_results.update(dvae_test_results)
    
    # 5.4 ConditionalDVAE reconstructed data (if requested)
    cdvae_results = None
    if use_conditional and cdvae_recon_train_loader is not None:
        print("\nTraining classifier on ConditionalDVAE reconstructed data...")
        cdvae_classifier = SimpleCNN(img_channels, img_size, num_classes).to(device)
        cdvae_results = train_classifier(
            model=cdvae_classifier,
            train_loader=cdvae_recon_train_loader,
            val_loader=cdvae_recon_val_loader,
            device=device,
            num_epochs=num_epochs_clf
        )
        
        # Evaluate on test set
        cdvae_test_results = evaluate_classifier(cdvae_classifier, test_loader, device)
        cdvae_results.update(cdvae_test_results)
    
    # Summarize results
    results = {
        'noisy': noisy_results,
        'vae': vae_results,
        'dvae': dvae_results,
        'models': {
            'vae': vae_model,
            'dvae': dvae_model
        }
    }
    
    if use_conditional and cdvae_results is not None:
        results['cdvae'] = cdvae_results
        if cdvae_model is not None:
            results['models']['cdvae'] = cdvae_model
    
    print("\nResults Summary:")
    print(f"Noisy data classifier accuracy: {noisy_results['accuracy']:.4f}")
    print(f"VAE reconstructed data classifier accuracy: {vae_results['accuracy']:.4f}")
    print(f"DVAE reconstructed data classifier accuracy: {dvae_results['accuracy']:.4f}")
    if use_conditional and cdvae_results is not None:
        print(f"ConditionalDVAE reconstructed data classifier accuracy: {cdvae_results['accuracy']:.4f}")
    
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Run data augmentation experiment with specific parameters")
    parser.add_argument("--samples-per-class", type=int, default=100, help="Number of samples per class")
    parser.add_argument("--noise-factor", type=float, default=0.5, help="Noise factor")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "salt_and_pepper"],
                       help="Type of noise to apply")
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist"],
                       help="Dataset to use")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    parser.add_argument("--save-dir", type=str, default="results/data_aug_analysis", 
                       help="Directory to save results")
    parser.add_argument("--vae-epochs", type=int, default=20, help="Number of epochs to train VAE/DVAE")
    parser.add_argument("--clf-epochs", type=int, default=5, help="Number of epochs to train classifiers")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--use-conditional", action="store_true", help="Use conditional DVAE")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    results = run_experiment(
        samples_per_class=args.samples_per_class,
        noise_factor=args.noise_factor,
        noise_type=args.noise_type,
        dataset_name=args.dataset,
        num_epochs_vae=args.vae_epochs,
        num_epochs_clf=args.clf_epochs,
        save_dir=args.save_dir,
        device=args.device,
        seed=args.seed,
        use_conditional=args.use_conditional
    )
    
    # Print final results
    print("\nFinal Results:")
    for method, metrics in results.items():
        print(f"{method.upper()} accuracy: {metrics['accuracy']:.4f}")

def create_cdvae_dataset(model: ConditionalDVAE, dataset: Dataset, device: torch.device) -> TensorDataset:
    """Create a dataset with ConditionalDVAE reconstructions."""
    
    # Create data loader for original dataset
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=2
    )
    
    all_reconstructions = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        with tqdm(dataloader, desc="ConditionalDVAE reconstructions") as pbar:
            for batch in pbar:
                if len(batch) == 3:  # (noisy_img, clean_img, label)
                    noisy_imgs, clean_imgs, labels = batch
                    noisy_imgs = noisy_imgs.to(device)
                    labels = labels.to(device)
                    
                    # Reconstruct using the conditionalDVAE
                    reconstructions, _, _ = model(noisy_imgs, labels)
                    
                    # Ensure reconstructions are 28x28
                    reconstructions = ensure_consistent_dimensions(reconstructions)
                    
                    all_reconstructions.append(reconstructions.cpu())
                    all_labels.append(labels.cpu())
                elif len(batch) == 2:  # (img, label)
                    imgs, labels = batch
                    imgs = imgs.to(device)
                    labels = labels.to(device)
                    
                    # Reconstruct using the conditionalDVAE
                    reconstructions, _, _ = model(imgs, labels)
                    
                    # Ensure reconstructions are 28x28
                    reconstructions = ensure_consistent_dimensions(reconstructions)
                    
                    all_reconstructions.append(reconstructions.cpu())
                    all_labels.append(labels.cpu())
    
    # Stack all the tensors
    reconstructions = torch.cat(all_reconstructions, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    # Create a new dataset
    recon_dataset = TensorDataset(reconstructions, labels)
    
    return recon_dataset 