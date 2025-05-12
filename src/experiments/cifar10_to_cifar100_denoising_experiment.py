import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, Optional
import sys
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import json
import torch.nn.functional as F
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_cifar10_dataset, get_cifar100_dataset, create_data_loaders, get_stratified_indices
from src.models.vae import VAE
from src.models.dvae import DVAE
from src.utils.training import Trainer
from src.utils.visualization import visualize_generated_samples
from src.utils.noise import add_noise
from src.experiments.data_augmentation_experiment import train_classifier, evaluate_classifier

# --- Simple CNN Classifier (same as in your file) ---
class SimpleCNN(nn.Module):
    def __init__(self, img_channels: int = 3, img_size: int = 32, num_classes: int = 1000):
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
        self.adaptive_pool = nn.AdaptiveAvgPool2d((3, 3))
        self.flat_dim = 128 * 3 * 3
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        x = self.adaptive_pool(x)
        x = self.classifier(x)
        return x

# --- Main Experiment ---
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10-to-CIFAR100 Denoising Transfer Experiment")
    parser.add_argument("--cifar10-data-dir", type=str, default="./data/cifar10")
    parser.add_argument("--cifar100-data-dir", type=str, default="./data/cifar100")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--save-dir", type=str, default="results/cifar10_to_cifar100_denoising")
    parser.add_argument("--noise-type", type=str, default="gaussian")
    parser.add_argument("--noise-factor", type=float, default=0.5)
    parser.add_argument("--vae-epochs", type=int, default=20)
    parser.add_argument("--dvae-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)

    parser.add_argument("--clf-batch-size", type=int, default=64)
    parser.add_argument("--clf-epochs", type=int, default=10, help="Number of epochs to train classifiers")
    parser.add_argument("--clf-lr", type=float, default=1e-3, help="Learning rate for classifier")

    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--samples-per-class", type=int, default=100)
    parser.add_argument('--classes', nargs='+', type=int, help='List of class indices')
    parser.add_argument('--use-pretrained-model', type=bool, default=True)
    parser.add_argument('--subsample-indices-path', type=str, default=None,
                        help='Path to save/load subsample indices for reproducibility')
    
    return parser.parse_args()

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (optional, can slow down training)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def select_random_classes(dataset, num_classes=10, seed=42, selected_classes=None):
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    unique_labels = list(set(all_labels))
    if selected_classes is None:
        random.seed(seed)
        selected_classes = random.sample(unique_labels, num_classes)
    selected_indices = [i for i, label in enumerate(all_labels) if label in selected_classes]
    label_map = {old: new for new, old in enumerate(selected_classes)}
    images, labels = [], []
    for i in selected_indices:
        x, y = dataset[i]
        images.append(x)
        labels.append(label_map[y])
    images = torch.stack(images)
    labels = torch.tensor(labels)
    return torch.utils.data.TensorDataset(images, labels), selected_classes

def subsample_per_class(tensor_dataset, samples_per_class, num_classes=10, seed=42):
    # tensor_dataset: TensorDataset(images, labels)
    images, labels = tensor_dataset.tensors
    indices = []
    rng = np.random.default_rng(seed)
    for c in range(num_classes):
        class_indices = (labels == c).nonzero(as_tuple=True)[0]
        chosen = rng.choice(class_indices, size=samples_per_class, replace=False)
        indices.extend(chosen.tolist())
    images = images[indices]
    labels = labels[indices]
    return torch.utils.data.TensorDataset(images, labels)

def make_noised_dataset(tensor_dataset, noise_type, noise_factor):
    images, labels = tensor_dataset.tensors
    noised_images = []
    for x in images:
        x_noisy = add_noise(
            x.unsqueeze(0),
            noise_type=noise_type,
            noise_params={'noise_factor': noise_factor, 'clip_min': 0.0, 'clip_max': 1.0}
        ).squeeze(0)
        noised_images.append(x_noisy)
    noised_images = torch.stack(noised_images)
    return torch.utils.data.TensorDataset(noised_images, labels.clone())

def make_denoised_dataset(tensor_dataset, model, device="cpu", batch_size=64, dvae=False):
    """
    Denoise a dataset using a (D)VAE model.
    - tensor_dataset: a TensorDataset of (noised) images and labels
    - model: the trained VAE or DVAE
    - dvae: if True, use DVAE forward signature
    Returns: TensorDataset of denoised images and original labels
    """
    images, labels = tensor_dataset.tensors
    model.eval()
    denoised_images = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].to(device)
            if dvae:
                recons, *_ = model(batch, None)
            else:
                recons, *_ = model(batch)
            denoised_images.append(recons.detach().cpu())
    denoised_images = torch.cat(denoised_images, dim=0)
    return torch.utils.data.TensorDataset(denoised_images, labels.clone())

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    # Set a global seed for reproducibility
    seed = getattr(args, "seed", 42)  # Use args.seed if present, else default to 42
    set_seed(seed)

    # 1. Prepare CIFAR-10 datasets for VAE and DVAE training
    dataset_func = get_cifar10_dataset
    img_channels = 3
    img_size = 32
    
    num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
    
    # Define the exact same architecture as in noise_experiment.py
    hidden_dims = [32, 64, 128]
    latent_dim = 32
    kl_weight = 0.1
    
    # Apply Gaussian noise with factor 0.5
    noise_type = args.noise_type
    noise_factor = args.noise_factor
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    # First, train the VAE model similar to noise_experiment.py
    print(f"Training VAE model on cifar10 with {noise_type} noise (factor: {noise_factor})...")
    
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
    '''vae_train_dataset = subsample_dataset(
        vae_train_dataset,
        num_samples=args.samples_per_class * num_classes,
        stratified=True
    )'''
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
    if args.use_pretrained_model:
        vae_model.load_state_dict(torch.load(os.path.join(args.save_dir, f'vae_cifar10_{noise_type}_factor{noise_factor}/models/best.pt'))["model_state_dict"])
    else:
        # Train VAE using same settings
        vae_experiment_name = f"vae_cifar10_{noise_type}_factor{noise_factor}"
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
            num_epochs=args.vae_epochs,  # Reduced from 30 in noise_experiment for time
            early_stopping=True,
            patience=5,
            save_best_only=True
        )
    
    # Now train the DVAE model
    print(f"Training DVAE model on cifar10 with {noise_type} noise (factor: {noise_factor})...")
    
    # Get DVAE training dataset - return_pairs=True for (noisy_img, clean_img, label) format
    dvae_train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_train_dataset.return_pairs = True
    
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
    if args.use_pretrained_model:
        dvae_model.load_state_dict(torch.load(os.path.join(args.save_dir, f'dvae_cifar10_{noise_type}_factor{noise_factor}/models/best.pt'))["model_state_dict"])
    else:
        # Train DVAE using same settings
        dvae_experiment_name = f"dvae_cifar10_{noise_type}_factor{noise_factor}"
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
            num_epochs=args.dvae_epochs,  # Reduced from 30 in noise_experiment for time
            early_stopping=True,
            patience=5,
            save_best_only=True
        )


    ############################################################
    ############################################################
    #PART 2: TRAINING CLASSIFIERS ON CIFAR-100
    ############################################################
    ############################################################



    dataset_func = get_cifar100_dataset
    img_channels = 3
    img_size = 32
    
    num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
    
    # Define the exact same architecture as in noise_experiment.py
    hidden_dims = [32, 64, 128]
    latent_dim = 32
    kl_weight = 0.1

    remap_labels = True

    noise_factor = 0.0
    noise_params = {
        'noise_factor': 0.0,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    print(args.classes)


    # First, prepare noisy test data for final evaluation
    print("Preparing noisy test data for evaluation...")
    unnoised_test_dataset = dataset_func(
        root=args.data_dir,
        train=False,
        noise_type=None,
        noise_params=noise_params,
        select_classes=args.classes,
        remap_labels=remap_labels,
        download=True
    )
    unnoised_test_dataset.return_pairs = False  # (noisy_img, label) format
    
    # Note: We use the full test set for evaluation to get a comprehensive 
    # assessment of model performance on unseen data
    print(f"Test dataset size: {len(unnoised_test_dataset)}")
    
    unnoised_test_loader, _ = create_data_loaders(
        unnoised_test_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=args.num_workers
    )


    # 0. Original unnoised dataset
    print("Preparing original unnoised dataset...")
    unnoised_train_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=None,
        noise_params=noise_params,
        select_classes=args.classes,
        remap_labels=remap_labels,
        download=True,
        add_noise_online=False
    )
    unnoised_train_dataset.return_pairs = False  # (noisy_img, label) format

    indices = None
    if args.subsample_indices_path is not None and os.path.exists(args.subsample_indices_path):
        print(f"Loading subsample indices from {args.subsample_indices_path}")
        indices = torch.load(args.subsample_indices_path)
    else:
        print("Generating new stratified indices...")
        indices = get_stratified_indices(
            unnoised_train_dataset,
            num_samples=args.samples_per_class * len(args.classes),
            stratified=True,
            seed=args.seed
        )
        if args.subsample_indices_path is not None:
            torch.save(indices, args.subsample_indices_path)
            print(f"Saved subsample indices to {args.subsample_indices_path}")

    unnoised_train_dataset = Subset(unnoised_train_dataset, indices)
    print(f"Unnoised classifier training dataset size after subsampling: {len(unnoised_train_dataset)}")
    
    # Create data loaders for classifier training
    unnoised_train_loader, unnoised_val_loader = create_data_loaders(
        unnoised_train_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.2,
        shuffle=False,
        num_workers=args.num_workers
    )

    

    

    






    noise_type = args.noise_type
    noise_factor = args.noise_factor
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0, 
        'clip_max': 1.0
    }

    
    # 1. Original noisy dataset
    print("Preparing original noisy dataset...")
    noisy_train_dataset_original = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        select_classes=args.classes,
        remap_labels=remap_labels,
        download=True,
        add_noise_online=False
    )
    noisy_train_dataset_original.return_pairs = False  # (noisy_img, label) format
    
    # Subsample the dataset to use only samples_per_class samples per class
    noisy_train_dataset = Subset(noisy_train_dataset_original, indices)
    print(f"Noisy classifier training dataset size after subsampling: {len(noisy_train_dataset)}")
    
    # Create data loaders for classifier training
    noisy_train_loader, noisy_val_loader = create_data_loaders(
        noisy_train_dataset,
        batch_size=args.clf_batch_size,
        val_split=0.2,
        shuffle=False,
        num_workers=args.num_workers
    )
    













    # 2. VAE reconstructed dataset - first get all noisy images and reconstruct them
    print("Creating VAE reconstructed dataset...")
    
    # Create a dataloader for noisy training data
    '''vae_recon_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        select_classes=args.classes,
        remap_labels=remap_labels,
        download=True,
        add_noise_online=False
    )
    vae_recon_dataset.return_pairs = False  # (noisy_img, label) format'''
    vae_recon_dataset = Subset(noisy_train_dataset_original, indices)
    print(f"VAE reconstruction dataset size after subsampling: {len(vae_recon_dataset)}")
    
    vae_recon_loader = DataLoader(
        noisy_train_dataset,
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
        shuffle=False,
        num_workers=args.num_workers
    )


    


    # 3. DVAE reconstructed dataset
    print("Creating DVAE reconstructed dataset...")
    
    # Create a dataloader with paired data for DVAE
    '''dvae_recon_dataset = dataset_func(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        select_classes=args.classes,
        remap_labels=remap_labels,
        download=True,
        add_noise_online=False
    )
    dvae_recon_dataset.return_pairs = True  # (noisy_img, label) format'''
    noisy_train_dataset_original.return_pairs = True
    dvae_recon_dataset = Subset(noisy_train_dataset_original, indices)
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
        shuffle=False,
        num_workers=args.num_workers
    )
    


    # Train separate classifiers on each dataset
    results = {}

    # 0. Train classifier on unnoised data
    print("\nTraining classifier on original unnoised data...")
    unnoised_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    unnoised_history = train_classifier(
        unnoised_classifier,
        unnoised_train_loader,
        unnoised_val_loader,
        num_epochs=args.clf_epochs,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False,
        patience=5
    )
    
    unnoised_metrics = evaluate_classifier(unnoised_classifier, unnoised_test_loader, device)
    
    # 1. Train classifier on noisy data
    print("\nTraining classifier on original noisy data...")
    noisy_train_dataset_original.return_pairs = False
    noisy_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    noisy_history = train_classifier(
        noisy_classifier,
        noisy_train_loader,
        noisy_val_loader,
        num_epochs=args.clf_epochs,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False,
        patience=5
    )
    
    noisy_metrics = evaluate_classifier(noisy_classifier, unnoised_test_loader, device)
    
    # 2. Train classifier on VAE reconstructed data
    print("\nTraining classifier on VAE reconstructed data...")
    vae_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    vae_history = train_classifier(
        vae_classifier,
        vae_recon_train_loader,
        vae_recon_val_loader,
        num_epochs=args.clf_epochs,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False,
        patience=5
    )
    
    vae_metrics = evaluate_classifier(vae_classifier, unnoised_test_loader, device)
    
    # 3. Train classifier on DVAE reconstructed data
    print("\nTraining classifier on DVAE reconstructed data...")
    noisy_train_dataset_original.return_pairs = True
    dvae_classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes)
    dvae_history = train_classifier(
        dvae_classifier,
        dvae_recon_train_loader,
        dvae_recon_val_loader,
        num_epochs=args.clf_epochs,  # Just 5 epochs as requested
        lr=args.clf_lr,
        device=device,
        early_stopping=False,
        patience=5
    )
    
    dvae_metrics = evaluate_classifier(dvae_classifier, unnoised_test_loader, device)
    
    # Collect results
    results = {
        'unnoised': {
            'accuracy': unnoised_metrics['accuracy'],
            'report': unnoised_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in unnoised_history.items()}
        },
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
    print(f"Unnoised data: Test Accuracy = {results['unnoised']['accuracy']:.4f}")
    print(f"Noisy data: Test Accuracy = {results['noisy']['accuracy']:.4f}")
    print(f"VAE reconstructed data: Test Accuracy = {results['vae_reconstructed']['accuracy']:.4f}")
    print(f"DVAE reconstructed data: Test Accuracy = {results['dvae_reconstructed']['accuracy']:.4f}")


    # Save results
    # Collect results
    results = {
        'unnoised': {
            'accuracy': unnoised_metrics['accuracy'],
            'report': unnoised_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in unnoised_history.items()}
        },
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

    with open(os.path.join(args.save_dir, f'results_{noise_type}_factor{noise_factor}.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


if __name__ == "__main__":
    #main()
    #exit()
    


    # Define the noise types and levels you want to sweep over
    noise_types = ["gaussian", "salt_and_pepper", "blur"]
    noise_factors = [0.25,0.5,0.75,1.0]

    classes = list(random.sample(range(100), 10))

    # Parse the base args once (to get data dirs, save dir, etc.)
    base_parser = argparse.ArgumentParser()
    base_parser.add_argument("--cifar10-data-dir", type=str, default="./data/cifar10")
    base_parser.add_argument("--cifar100-data-dir", type=str, default="./data/cifar100")
    base_parser.add_argument("--save-dir", type=str, default="results/cifar10_to_cifar100_denoising")
    base_parser.add_argument("--vae-epochs", type=int, default=20)
    base_parser.add_argument("--dvae-epochs", type=int, default=20)
    base_parser.add_argument("--clf-epochs", type=int, default=10)
    base_parser.add_argument("--batch-size", type=int, default=64)
    base_parser.add_argument("--num-workers", type=int, default=0)
    base_parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    base_parser.add_argument("--vae-model", type=str, default=None)
    base_parser.add_argument("--dvae-model", type=str, default=None)
    base_parser.add_argument("--samples-per-class", type=int, default=100)
    base_parser.add_argument('--classes', nargs='+', type=int, help='List of class indices')
    base_parser.add_argument('--use-pretrained-model', type=bool, default=False)
    base_parser.add_argument('--subsample-indices-path', type=str, default="results/cifar10_to_cifar100_denoising/subsample_indices.pth",
                            help='Path to save/load subsample indices for reproducibility')
    # Only parse known args to avoid issues with unknowns
    base_args, _ = base_parser.parse_known_args()

    dataset_func = get_cifar100_dataset
    noise_params = {
        'noise_factor': 0,
        'clip_min': 0.0,
        'clip_max': 1.0
    }

    unnoised_train_dataset = dataset_func(
        root=base_args.cifar100_data_dir,
        train=True,
        noise_type="gaussian",
        noise_params=noise_params,
        select_classes=classes,
        remap_labels=True,
        download=True
    )
    unnoised_train_dataset.return_pairs = False  # (noisy_img, label) format

    indices = get_stratified_indices(unnoised_train_dataset, base_args.samples_per_class * len(classes), stratified=True, seed=42)
    torch.save(indices, base_args.subsample_indices_path)

    # Run the experiment for each combination
    for noise_type in noise_types:
        for noise_factor in noise_factors:
            # Build sys.argv for each run
            sys.argv = [
                sys.argv[0],
                "--cifar10-data-dir", base_args.cifar10_data_dir,
                "--cifar100-data-dir", base_args.cifar100_data_dir,
                "--save-dir", base_args.save_dir,
                "--vae-epochs", str(base_args.vae_epochs),
                "--dvae-epochs", str(base_args.dvae_epochs),
                "--clf-epochs", str(base_args.clf_epochs),
                "--batch-size", str(base_args.batch_size),
                "--num-workers", str(base_args.num_workers),
                "--device", base_args.device,
                "--noise-type", noise_type,
                "--noise-factor", str(noise_factor),
                "--samples-per-class", str(base_args.samples_per_class),
                "--classes", *[str(c) for c in classes],
                "--use-pretrained-model", False,
                "--subsample-indices-path", base_args.subsample_indices_path
            ]
            if base_args.vae_model is not None:
                sys.argv += ["--vae-model", base_args.vae_model]
            if base_args.dvae_model is not None:
                sys.argv += ["--dvae-model", base_args.dvae_model]
            print(f"\n\n==== Running experiment: noise_type={noise_type}, noise_factor={noise_factor} ====")
            main()

    
    # Plot the results
    for noise_type in noise_types:
        for noise_factor in noise_factors:
            break
            results = json.load(open(os.path.join(base_args.save_dir, f'results_{noise_type}_factor{noise_factor}.json')))
            # Training accuracy
            plt.subplot(2, 2, 1)
            plt.plot(results['unnoised']['history']['train_acc'], label='Clean')
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
            plt.plot(results['unnoised']['history']['val_acc'], label='Clean')
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
            plt.plot(results['unnoised']['history']['train_loss'], label='Clean')
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
            plt.plot(results['unnoised']['history']['val_loss'], label='Clean')
            plt.plot(results['noisy']['history']['val_loss'], label='Noisy')
            plt.plot(results['vae_reconstructed']['history']['val_loss'], label='VAE Reconstructed')
            plt.plot(results['dvae_reconstructed']['history']['val_loss'], label='DVAE Reconstructed')
            plt.title('Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_args.save_dir, 'training_history.png'))
            plt.close()
            
            # 2. Confusion matrices
            plt.figure(figsize=(15, 5))

            plt.subplot(1, 4, 1)
            sns.heatmap(results['noisy']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Clean Data\nAccuracy: {results['unnoised']['accuracy']:.4f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.subplot(1, 4, 2)
            sns.heatmap(results['noisy']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'Noisy Data\nAccuracy: {results['noisy']['accuracy']:.4f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.subplot(1, 4, 3)
            sns.heatmap(results['vae_reconstructed']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'VAE Reconstructed\nAccuracy: {results['vae_reconstructed']['accuracy']:.4f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.subplot(1, 4, 4)
            sns.heatmap(results['dvae_reconstructed']['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.title(f'DVAE Reconstructed\nAccuracy: {results['dvae_reconstructed']['accuracy']:.4f}')
            plt.xlabel('Predicted')
            plt.ylabel('True')
            
            plt.tight_layout()
            plt.savefig(os.path.join(base_args.save_dir, 'confusion_matrices.png'))
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
            plt.savefig(os.path.join(base_args.save_dir, 'class_wise_performance.png'))
            plt.close()
            
            print(f"\nExperiment complete! Results saved to {base_args.save_dir}")
    

    fig, axes = plt.subplots(1, len(noise_types), figsize=(6 * len(noise_types), 5), sharey=True)

    for idx, noise_type in enumerate(noise_types):
        dvae = []
        vae = []
        noisy = []
        clean = []

        for noise_factor in noise_factors:
            results = json.load(open(os.path.join(base_args.save_dir, f'results_{noise_type}_factor{noise_factor}.json')))
            dvae.append(results['dvae_reconstructed']['accuracy'])
            vae.append(results['vae_reconstructed']['accuracy'])
            noisy.append(results['noisy']['accuracy'])
            clean.append(results['unnoised']['accuracy'])

        model_names = ['Clean', 'Noisy', 'VAE', 'DVAE']
        all_accs = [clean, noisy, vae, dvae]
        bar_width = 0.2
        x = np.arange(len(noise_factors))

        ax = axes[idx] if len(noise_types) > 1 else axes
        for i, accs in enumerate(all_accs):
            ax.bar(x + i * bar_width, accs, width=bar_width, label=model_names[i])
        ax.set_xticks(x + 1.5 * bar_width)
        ax.set_xticklabels([str(nf) for nf in noise_factors])
        ax.set_xlabel("Noise Level")
        if idx == 0:
            ax.set_ylabel("Accuracy")
        ax.set_title(f"{noise_type.capitalize()} Noise")
        ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Place the legend at the top, inside the figure, and ensure it's not cut off
    handles, labels = axes[0].get_legend_handles_labels() if isinstance(axes, np.ndarray) else axes.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 0.99), fontsize='large', frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve a bit more space at the top for the legend
    plt.savefig(os.path.join(base_args.save_dir, "all_noise_types_accuracy_bars.png"), bbox_inches='tight')