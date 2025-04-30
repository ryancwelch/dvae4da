import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
from tqdm import tqdm
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, get_cifar10_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model
from src.models.vae import VAE
from src.models.dvae import DVAE
from src.utils.training import Trainer
from src.utils.visualization import visualize_generated_samples
from src.utils.noise import add_noise, add_gaussian_noise


# Define a simple CNN classifier
class SimpleCNN(nn.Module):
    def __init__(self, img_channels: int = 1, img_size: int = 28, num_classes: int = 10):
        super().__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Use adaptive pooling to get a fixed output size regardless of input dimensions
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))  # Fixed output size
        self.flat_dim = 128 * 3 * 3  # Fixed based on adaptive pool output
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
        print(f"SimpleCNN initialized with flat_dim: {self.flat_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Check input dimensions and resize if needed
        if x.dim() != 4:
            raise ValueError(f"Expected 4D input (B,C,H,W), got {x.dim()}D")
            
        # Run through convolutional layers
        x = self.features(x)
        
        # Use adaptive pooling to get fixed size regardless of input dimensions
        x = self.adaptive_pool(x)
        
        # Debug info for first batch
        if not hasattr(self, '_shape_printed'):
            print(f"Feature shape before classifier: {x.shape}")
            self._shape_printed = True
        
        # Classify
        x = self.classifier(x)
        return x


def train_classifier(model: nn.Module, 
                    train_loader: DataLoader, 
                    val_loader: Optional[DataLoader] = None, 
                    num_epochs: int = 10,
                    lr: float = 1e-3,
                    device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                    early_stopping: bool = True,
                    patience: int = 5) -> Dict:
    """Train a classifier and return training history."""
    
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    
    best_val_acc = 0.0
    no_improvement = 0
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        with tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                optimizer.zero_grad()
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == targets).sum().item()
                train_total += targets.size(0)
                
                pbar.set_postfix({
                    'loss': train_loss / train_total,
                    'acc': train_correct / train_total
                })
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                with tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") as pbar:
                    for inputs, targets in pbar:
                        inputs, targets = inputs.to(device), targets.to(device)
                        
                        outputs = model(inputs)
                        loss = criterion(outputs, targets)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_correct += (predicted == targets).sum().item()
                        val_total += targets.size(0)
                        
                        pbar.set_postfix({
                            'loss': val_loss / val_total,
                            'acc': val_correct / val_total
                        })
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                no_improvement = 0
            else:
                no_improvement += 1
                if early_stopping and no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                 f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        else:
            print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    return history


def evaluate_classifier(model: nn.Module, 
                       test_loader: DataLoader, 
                       device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')) -> Dict:
    """Evaluate a classifier and return performance metrics."""
    
    model.to(device)
    model.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    all_targets = []
    all_predictions = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        with tqdm(test_loader, desc="Evaluating") as pbar:
            for inputs, targets in pbar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                test_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                test_correct += (predicted == targets).sum().item()
                test_total += targets.size(0)
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(predicted.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': test_loss / test_total,
                    'acc': test_correct / test_total
                })
    
    test_loss = test_loss / test_total
    test_acc = test_correct / test_total
    
    report = classification_report(all_targets, all_predictions, output_dict=True)
    conf_matrix = confusion_matrix(all_targets, all_predictions)
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'report': report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets
    }


def generate_samples(model: nn.Module, 
                    num_samples: int, 
                    num_classes: int, 
                    device: torch.device,
                    apply_noise: bool = False,
                    noise_type: str = 'gaussian', 
                    noise_factor: float = 0.5) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic samples from a VAE/DVAE model."""
    
    # Prepare storage for samples and labels
    all_samples = []
    all_labels = []
    
    samples_per_class = num_samples // num_classes
    
    # Generate samples for each class
    for class_idx in range(num_classes):
        # Generate labels tensor
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)
        
        # Generate samples
        with torch.no_grad():
            # If model is conditional, use labels
            if hasattr(model, 'sample') and 'labels' in model.sample.__code__.co_varnames:
                samples = model.sample(samples_per_class, labels.to(device), device)
            else:
                # Standard VAE sample method
                samples = model.sample(samples_per_class, device)
        
        # Apply noise if requested
        if apply_noise:
            samples = add_noise(
                samples, 
                noise_type=noise_type,
                noise_params={'noise_factor': noise_factor}
            )
        
        all_samples.append(samples.cpu())
        all_labels.append(labels)
    
    # Concatenate all samples and labels
    samples = torch.cat(all_samples, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return samples, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate data augmentation with VAE/DVAE")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--samples-per-class", type=int, default=100, 
                        help="Number of training samples per class to use")
    
    # Models
    parser.add_argument("--vae-model", type=str, default=None, help="Path to trained VAE model")
    parser.add_argument("--dvae-model", type=str, default=None, help="Path to trained DVAE model")
    
    # Augmentation parameters
    parser.add_argument("--aug-samples-per-class", type=int, default=100, 
                        help="Number of augmentation samples per class")
    parser.add_argument("--apply-noise", action="store_true",
                        help="Apply noise to generated samples")
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        choices=["gaussian", "salt_and_pepper", "block", "line_h", "line_v"],
                        help="Type of noise to apply")
    parser.add_argument("--noise-factor", type=float, default=0.5,
                        help="Noise factor to apply (std for gaussian, probability for salt_and_pepper, etc.)")
    
    # Classifier parameters
    parser.add_argument("--clf-epochs", type=int, default=20, help="Number of epochs to train classifiers")
    parser.add_argument("--clf-lr", type=float, default=1e-3, help="Learning rate for classifier")
    parser.add_argument("--clf-batch-size", type=int, default=64, help="Batch size for classifiers")
    
    # Experiment parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--save-dir", type=str, default="results/data_augmentation", 
                        help="Directory to save results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # The purpose of this experiment is to demonstrate how VAE and DVAE models can improve
    # classification performance on small, noisy datasets. We use only a small subset of the
    # full dataset (100 samples per class) to simulate a real-world scenario where only limited
    # labeled data is available.
    
    # Get dataset
    if args.dataset == "mnist":
        dataset_func = get_mnist_dataset
        img_channels = 1
        img_size = 28
    else:  # cifar10
        dataset_func = get_cifar10_dataset
        img_channels = 3
        img_size = 32
    
    num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
    
    # Define the exact same architecture as in noise_experiment.py
    hidden_dims = [32, 64, 128]
    latent_dim = 32
    kl_weight = 0.1
    
    # Apply Gaussian noise with factor 0.5
    noise_type = "gaussian"
    noise_factor = 0.5
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    # First, train the VAE model similar to noise_experiment.py
    print(f"Training VAE model on {args.dataset} with {noise_type} noise (factor: {noise_factor})...")
    
    # Get VAE training dataset - return_pairs=False for (img, label) format
    vae_train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    vae_train_dataset.return_pairs = False
    
    # Subsample the dataset to use only samples_per_class samples per class
    print(f"Subsampling the dataset to {args.samples_per_class} samples per class...")
    vae_train_dataset = subsample_dataset(
        vae_train_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )
    print(f"VAE training dataset size after subsampling: {len(vae_train_dataset)}")
    
    # Create VAE data loaders with the same configuration as noise_experiment.py
    vae_train_loader, vae_val_loader = create_data_loaders(
        vae_train_dataset,
        batch_size=64,  # Same as noise_experiment
        val_split=0.1,  # Same as noise_experiment
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create VAE with same architecture
    vae_model = VAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )
    vae_model.to(device)
    
    # Train VAE using same settings
    vae_experiment_name = f"vae_{args.dataset}_{noise_type}_factor{noise_factor}"
    vae_trainer = Trainer(
        model=vae_model,
        train_loader=vae_train_loader,
        val_loader=vae_val_loader,
        lr=1e-3,  # Same as noise_experiment
        device=device,
        save_dir=args.save_dir,
        experiment_name=vae_experiment_name
    )
    
    vae_trainer.train(
        num_epochs=20,  # Reduced from 30 in noise_experiment for time
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # Now train the DVAE model
    print(f"Training DVAE model on {args.dataset} with {noise_type} noise (factor: {noise_factor})...")
    
    # Get DVAE training dataset - return_pairs=True for (noisy_img, clean_img, label) format
    dvae_train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_train_dataset.return_pairs = True
    
    # Subsample the dataset to use only samples_per_class samples per class
    print(f"Subsampling the dataset to {args.samples_per_class} samples per class...")
    dvae_train_dataset = subsample_dataset(
        dvae_train_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )
    print(f"DVAE training dataset size after subsampling: {len(dvae_train_dataset)}")
    
    # Create DVAE data loaders
    dvae_train_loader, dvae_val_loader = create_data_loaders(
        dvae_train_dataset,
        batch_size=64,  # Same as noise_experiment
        val_split=0.1,  # Same as noise_experiment
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Create DVAE with same architecture
    dvae_model = DVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )
    dvae_model.to(device)
    
    # Train DVAE using same settings
    dvae_experiment_name = f"dvae_{args.dataset}_{noise_type}_factor{noise_factor}"
    dvae_trainer = Trainer(
        model=dvae_model,
        train_loader=dvae_train_loader,
        val_loader=dvae_val_loader,
        lr=1e-3,  # Same as noise_experiment
        device=device,
        save_dir=args.save_dir,
        experiment_name=dvae_experiment_name
    )
    
    dvae_trainer.train(
        num_epochs=20,  # Reduced from 30 in noise_experiment for time
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # Now prepare datasets for classifier training
    
    # First, prepare noisy test data for final evaluation
    print("Preparing noisy test data for evaluation...")
    test_dataset = dataset_func(
        root=args.data_dir,
        train=False,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    test_dataset.return_pairs = False  # (noisy_img, label) format
    
    # Note: We use the full test set for evaluation to get a comprehensive 
    # assessment of model performance on unseen data
    print(f"Test dataset size: {len(test_dataset)}")
    
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # 1. Original noisy dataset
    print("Preparing original noisy dataset...")
    noisy_train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    noisy_train_dataset.return_pairs = False  # (noisy_img, label) format
    
    # Subsample the dataset to use only samples_per_class samples per class
    print(f"Subsampling the dataset to {args.samples_per_class} samples per class...")
    noisy_train_dataset = subsample_dataset(
        noisy_train_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )
    print(f"Noisy classifier training dataset size after subsampling: {len(noisy_train_dataset)}")
    
    # Create data loaders for classifier training
    noisy_train_loader, noisy_val_loader = create_data_loaders(
        noisy_train_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 2. VAE reconstructed dataset - first get all noisy images and reconstruct them
    print("Creating VAE reconstructed dataset...")
    
    # Create a dataloader for noisy training data
    vae_recon_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    vae_recon_dataset.return_pairs = False  # (noisy_img, label) format
    
    # Subsample the dataset to use only samples_per_class samples per class
    print(f"Subsampling the dataset to {args.samples_per_class} samples per class...")
    vae_recon_dataset = subsample_dataset(
        vae_recon_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )
    print(f"VAE reconstruction dataset size after subsampling: {len(vae_recon_dataset)}")
    
    vae_recon_loader = DataLoader(
        vae_recon_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Reconstruct all noisy images using VAE
    vae_recon_data = []
    vae_recon_labels = []
    
    vae_model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(vae_recon_loader, desc="VAE reconstructions"):
            batch_data = batch_data.to(device)
            
            # Get reconstructions
            recons, _, _ = vae_model(batch_data)
            
            # Store reconstructions and labels
            vae_recon_data.append(recons.cpu())
            vae_recon_labels.append(batch_labels)
    
    vae_recon_data = torch.cat(vae_recon_data, dim=0)
    vae_recon_labels = torch.cat(vae_recon_labels, dim=0)
    
    # Create VAE reconstructed dataset
    vae_recon_dataset = TensorDataset(vae_recon_data, vae_recon_labels)
    vae_recon_train_loader, vae_recon_val_loader = create_data_loaders(
        vae_recon_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # 3. DVAE reconstructed dataset
    print("Creating DVAE reconstructed dataset...")
    
    # Create a dataloader with paired data for DVAE
    dvae_recon_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_recon_dataset.return_pairs = True  # (noisy_img, clean_img, label) format
    
    # Subsample the dataset to use only samples_per_class samples per class
    print(f"Subsampling the dataset to {args.samples_per_class} samples per class...")
    dvae_recon_dataset = subsample_dataset(
        dvae_recon_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )
    print(f"DVAE reconstruction dataset size after subsampling: {len(dvae_recon_dataset)}")
    
    dvae_recon_loader = DataLoader(
        dvae_recon_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Reconstruct all noisy images using DVAE
    dvae_recon_data = []
    dvae_recon_labels = []
    
    dvae_model.eval()
    with torch.no_grad():
        for batch in tqdm(dvae_recon_loader, desc="DVAE reconstructions"):
            noisy_data, clean_data, batch_labels = batch
            noisy_data = noisy_data.to(device)
            
            # DVAE only needs noisy data for inference
            recons, _, _ = dvae_model(noisy_data)
            
            # Store reconstructions and labels
            dvae_recon_data.append(recons.cpu())
            dvae_recon_labels.append(batch_labels)
    
    dvae_recon_data = torch.cat(dvae_recon_data, dim=0)
    dvae_recon_labels = torch.cat(dvae_recon_labels, dim=0)
    
    # Create DVAE reconstructed dataset
    dvae_recon_dataset = TensorDataset(dvae_recon_data, dvae_recon_labels)
    dvae_recon_train_loader, dvae_recon_val_loader = create_data_loaders(
        dvae_recon_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    # Save sample images
    
    # Get a batch of noisy images
    noisy_batch = next(iter(noisy_train_loader))[0][:64].to(device)
    
    # Get VAE reconstructions
    vae_model.eval()
    with torch.no_grad():
        vae_recons, _, _ = vae_model(noisy_batch)
    
    # For DVAE, we need a batch with paired data
    dvae_batch = next(iter(dvae_recon_loader))
    noisy_dvae_batch, clean_dvae_batch = dvae_batch[0][:64].to(device), dvae_batch[1][:64].to(device)
    
    # Get DVAE reconstructions
    dvae_model.eval()
    with torch.no_grad():
        dvae_recons, _, _ = dvae_model(noisy_dvae_batch, clean_dvae_batch)
    
    # Visualize samples
    # Noisy originals
    noisy_fig = visualize_generated_samples(
        noisy_batch.cpu(),
        grid_size=(8, 8),
        title="Noisy Original Samples"
    )
    noisy_fig.savefig(os.path.join(args.save_dir, "noisy_samples.png"))
    plt.close(noisy_fig)
    
    # VAE reconstructions
    vae_fig = visualize_generated_samples(
        vae_recons.cpu(),
        grid_size=(8, 8),
        title="VAE Reconstructed Samples"
    )
    vae_fig.savefig(os.path.join(args.save_dir, "vae_reconstructed_samples.png"))
    plt.close(vae_fig)
    
    # DVAE reconstructions
    dvae_fig = visualize_generated_samples(
        dvae_recons.cpu(),
        grid_size=(8, 8),
        title="DVAE Reconstructed Samples"
    )
    dvae_fig.savefig(os.path.join(args.save_dir, "dvae_reconstructed_samples.png"))
    plt.close(dvae_fig)
    
    # Train separate classifiers on each dataset
    results = {}
    
    # 1. Train classifier on noisy data
    print("\nTraining classifier on original noisy data...")
    noisy_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    noisy_history = train_classifier(
        noisy_classifier,
        noisy_train_loader,
        noisy_val_loader,
        num_epochs=5,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )
    
    noisy_metrics = evaluate_classifier(noisy_classifier, test_loader, device)
    
    # 2. Train classifier on VAE reconstructed data
    print("\nTraining classifier on VAE reconstructed data...")
    vae_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    vae_history = train_classifier(
        vae_classifier,
        vae_recon_train_loader,
        vae_recon_val_loader,
        num_epochs=5,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )
    
    vae_metrics = evaluate_classifier(vae_classifier, test_loader, device)
    
    # 3. Train classifier on DVAE reconstructed data
    print("\nTraining classifier on DVAE reconstructed data...")
    dvae_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    dvae_history = train_classifier(
        dvae_classifier,
        dvae_recon_train_loader,
        dvae_recon_val_loader,
        num_epochs=5,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )
    
    dvae_metrics = evaluate_classifier(dvae_classifier, test_loader, device)
    
    # Collect results
    results = {
        'noisy': {
            'accuracy': noisy_metrics['accuracy'],
            'report': noisy_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in noisy_history.items()}
        },
        'vae_reconstructed': {
            'accuracy': vae_metrics['accuracy'],
            'report': vae_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in vae_history.items()}
        },
        'dvae_reconstructed': {
            'accuracy': dvae_metrics['accuracy'],
            'report': dvae_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in dvae_history.items()}
        }
    }
    
    # Print summary
    print("\nClassification Results:")
    print(f"Noisy data: Test Accuracy = {results['noisy']['accuracy']:.4f}")
    print(f"VAE reconstructed data: Test Accuracy = {results['vae_reconstructed']['accuracy']:.4f}")
    print(f"DVAE reconstructed data: Test Accuracy = {results['dvae_reconstructed']['accuracy']:.4f}")
    
    # Save results
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    
    # Create plots
    # 1. Training history
    plt.figure(figsize=(12, 10))
    
    # Training accuracy
    plt.subplot(2, 2, 1)
    plt.plot(results['noisy']['history']['train_acc'], label='Noisy')
    plt.plot(results['vae_reconstructed']['history']['train_acc'], label='VAE Reconstructed')
    plt.plot(results['dvae_reconstructed']['history']['train_acc'], label='DVAE Reconstructed')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation accuracy
    plt.subplot(2, 2, 2)
    plt.plot(results['noisy']['history']['val_acc'], label='Noisy')
    plt.plot(results['vae_reconstructed']['history']['val_acc'], label='VAE Reconstructed')
    plt.plot(results['dvae_reconstructed']['history']['val_acc'], label='DVAE Reconstructed')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Training loss
    plt.subplot(2, 2, 3)
    plt.plot(results['noisy']['history']['train_loss'], label='Noisy')
    plt.plot(results['vae_reconstructed']['history']['train_loss'], label='VAE Reconstructed')
    plt.plot(results['dvae_reconstructed']['history']['train_loss'], label='DVAE Reconstructed')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation loss
    plt.subplot(2, 2, 4)
    plt.plot(results['noisy']['history']['val_loss'], label='Noisy')
    plt.plot(results['vae_reconstructed']['history']['val_loss'], label='VAE Reconstructed')
    plt.plot(results['dvae_reconstructed']['history']['val_loss'], label='DVAE Reconstructed')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    plt.close()
    
    # 2. Confusion matrices
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(noisy_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Noisy Data\nAccuracy: {noisy_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(vae_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'VAE Reconstructed\nAccuracy: {vae_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(dvae_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'DVAE Reconstructed\nAccuracy: {dvae_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrices.png'))
    plt.close()
    
    # 3. Class-wise performance
    plt.figure(figsize=(12, 6))
    
    classes = list(range(num_classes))
    noisy_f1 = [noisy_metrics['report'][str(i)]['f1-score'] for i in classes]
    vae_f1 = [vae_metrics['report'][str(i)]['f1-score'] for i in classes]
    dvae_f1 = [dvae_metrics['report'][str(i)]['f1-score'] for i in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, noisy_f1, width, label='Noisy')
    plt.bar(x, vae_f1, width, label='VAE Reconstructed')
    plt.bar(x + width, dvae_f1, width, label='DVAE Reconstructed')
    
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('Class-wise F1-Score Comparison')
    plt.xticks(x, classes)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'class_wise_performance.png'))
    plt.close()
    
    print(f"\nExperiment complete! Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 