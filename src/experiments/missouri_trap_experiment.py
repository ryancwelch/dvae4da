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
import torchvision
from collections import Counter
from PIL import ImageFile

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_cifar10_dataset, get_cifar100_dataset, create_data_loaders, get_stratified_indices, get_missouri_camera_traps_dataset, NoisyDataset
from src.models.vae import VAE
from src.models.dvae import DVAE
from src.utils.training import Trainer
from src.utils.visualization import visualize_generated_samples
from src.utils.noise import add_noise
from src.experiments.data_augmentation_experiment import train_classifier, evaluate_classifier, SimpleCNN

ImageFile.LOAD_TRUNCATED_IMAGES = True

def show_random_samples(dataset, num_samples=5, save_dir=None, class_indices=None, model_name=None):
    class_indices = [9,11,13,14,15]
    class_names = ["Birds", "White-Tailed Deer", "Red Deer", "Roe Deer", "Wild Boar"]
    # Determine how to get integer labels for each sample
    if hasattr(dataset, 'tensors'):
        # TensorDataset: labels are in dataset.tensors[1]
        labels = dataset.tensors[1].tolist()
        get_label = lambda i: labels[i]
    elif hasattr(dataset, 'dataset') and hasattr(dataset, 'indices'):
        # Subset: get integer label from the underlying dataset's samples
        class_to_idx = dataset.dataset.class_to_idx
        get_label = lambda i: class_to_idx[dataset.dataset.samples[dataset.indices[i]][1]]
    elif hasattr(dataset, 'samples') and hasattr(dataset, 'class_to_idx'):
        # Custom dataset: get integer label from samples
        class_to_idx = dataset.class_to_idx
        get_label = lambda i: class_to_idx[dataset.samples[i][1]]
    else:
        # Fallback: try to get label as dataset[i][1]
        get_label = lambda i: dataset[i][1]

    samples = []
    if class_indices is not None:
        for index in class_indices:
            all_indices = [i for i in range(len(dataset)) if get_label(i) == index]
            indices = random.sample(all_indices, 1)
            samples.append(dataset[indices[0]])
    else:
        indices = random.sample(range(len(dataset)), num_samples)
        samples = [dataset[i] for i in indices]

    images, labels = zip(*samples)
    # Ensure all images are tensors
    images = [img if torch.is_tensor(img) else torchvision.transforms.ToTensor()(img) for img in images]

    # Convert label indices to class names if possible
    #class_names = None
    if hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'class_to_idx'):
        idx_to_class = {v: k for k, v in dataset.dataset.class_to_idx.items()}
        #class_names = [idx_to_class[lbl] for lbl in labels]
    elif hasattr(dataset, 'class_to_idx'):
        idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
        #class_names = [idx_to_class[lbl] for lbl in labels]
    else:
        pass
        #class_names = [str(lbl) for lbl in labels]

    grid = torchvision.utils.make_grid(images, nrow=num_samples, padding=2)
    plt.figure(figsize=(num_samples * 2, 2))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    if model_name is not None:
        plt.title(f"{model_name} reconstructed random samples: {class_names}")
    else:
        plt.title(f"Random samples: {class_names})")
    if save_dir is not None:
        save_dir = os.path.join(save_dir, "random_samples")
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, "random_samples.png")
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved random sample image grid to {save_path}")
    plt.show()

def print_class_distribution(dataset, split_name):
    # dataset is a Subset, so dataset.dataset is the underlying MissouriCameraTrapsBBDataset
    # and dataset.indices are the indices in the full dataset
    base_dataset = dataset.dataset
    indices = dataset.indices
    # base_dataset.samples is a list of (rel_path, class_name)
    class_names = [base_dataset.samples[i][1] for i in indices]
    counter = Counter(class_names)
    total = len(class_names)
    print(f"{split_name} class distribution (percentages):")
    for cls in sorted(counter):
        pct = 100.0 * counter[cls] / total
        print(f"  {cls}: {pct:.2f}%")
    print()

def main(args):
    args.save_dir = os.path.join(args.save_dir, args.model_name)
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)

    # Load training dataset
    train_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=True,
        noise_type=args.noise_type,
        noise_params=None,  # You can add noise params if needed
        add_noise_online=args.add_noise_online,
        return_pairs=args.return_pairs,
        img_size=args.img_size
    )

    # Load validation dataset
    test_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=False,
        noise_type=args.noise_type,
        noise_params=None,
        add_noise_online=args.add_noise_online,
        return_pairs=args.return_pairs,
        img_size=args.img_size
    )

    # Example: create DataLoaders
    train_loader, val_loader = create_data_loaders(train_dataset, batch_size=args.batch_size, val_split=0.1, shuffle=True, num_workers=args.num_workers)
    test_loader,_ = create_data_loaders(test_dataset, batch_size=args.batch_size, val_split=0.0, shuffle=False, num_workers=args.num_workers)

    # Print dataset and batch info for verification
    print(f"Train dataset size: {len(train_loader)}")
    print(f"Validation dataset size: {len(val_loader)}")
    print(f"Test dataset size: {len(test_loader)}")

    # Show and save a few random samples from the training dataset
    show_random_samples(test_dataset, num_samples=5, save_dir=args.save_dir)

    # Print class distributions
    print_class_distribution(train_dataset, "Train")
    print_class_distribution(test_dataset, "Test")

    #Train a classifier on the training dataset
    print("\nTraining classifier on original noisy data...")
    '''classifier = SimpleCNN(img_channels=3, img_size=args.img_size, num_classes=20)
    history = train_classifier(
        classifier,
        train_loader,
        val_loader,
        num_epochs=args.clf_epochs,
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )
    
    metrics = evaluate_classifier(classifier, test_loader, device)'''




    img_channels = 3
    img_size = args.img_size
    
    num_classes = 20  # Both MNIST and CIFAR-10 have 10 classes
    
    # Define the exact same architecture as in noise_experiment.py
    hidden_dims = [32, 64, 128]
    latent_dim = 32
    kl_weight = 0.1

    # Create VAE model
    vae_model = VAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )

    vae_model.to(device)
    vae_model.load_state_dict(torch.load(os.path.join(args.model_dir, f'vae_cifar10_{args.model_name}/models/best.pt'))["model_state_dict"])

    # Create DVAE model
    dvae_model = DVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )
    
    dvae_model.to(device)
    dvae_model.load_state_dict(torch.load(os.path.join(args.model_dir, f'dvae_cifar10_{args.model_name}/models/best.pt'))["model_state_dict"])



    noise_type = "gaussian"
    noise_params = {
        'noise_factor': 0.0,
        'clip_min': 0.0,
        'clip_max': 1.0
    }


    
    
    
    
    
    
    # Create a dataloader for noisy training data
    vae_recon_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        add_noise_online=True,
        return_pairs=True,
        img_size=args.img_size
    )
    vae_recon_dataset.return_pairs = False  # (noisy_img, label) format
    print(f"VAE reconstruction dataset size after subsampling: {len(vae_recon_dataset)}")
    
    vae_recon_loader = DataLoader(
        vae_recon_dataset,
        batch_size=args.batch_size,
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
        val_split=0.1,
        shuffle=False,
        num_workers=args.num_workers
    )

    vae_test_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=False,
        noise_type=args.noise_type,
        noise_params=None,
        add_noise_online=args.add_noise_online,
        return_pairs=args.return_pairs,
        img_size=args.img_size
    )
    vae_test_loader,_ = create_data_loaders(vae_test_dataset, batch_size=args.batch_size, val_split=0.0, shuffle=False, num_workers=args.num_workers)

    vae_recon_data_test = []
    vae_recon_labels_test = []
    
    vae_model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(vae_test_loader, desc="VAE reconstructions"):
            batch_data = batch_data.to(device)
            
            # Get reconstructions
            recons, _, _ = vae_model(batch_data)
            
            # Store reconstructions and labels
            vae_recon_data_test.append(recons.cpu())
            vae_recon_labels_test.append(batch_labels)
    
    vae_recon_data_test = torch.cat(vae_recon_data_test, dim=0)
    vae_recon_labels_test = torch.cat(vae_recon_labels_test, dim=0)
    
    # Create VAE reconstructed dataset
    vae_recon_dataset_test = TensorDataset(vae_recon_data_test, vae_recon_labels_test)
    vae_recon_test_loader, _ = create_data_loaders(
        vae_recon_dataset_test,
        batch_size=args.clf_batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    show_random_samples(vae_recon_dataset_test, num_samples=5, save_dir=os.path.join(args.save_dir, "vae_reconstructed"), class_indices=[9,11,13,14,15], model_name="VAE " + args.model_name)

    # Train a classifier on the VAE reconstructed dataset
    print("\nTraining classifier on VAE reconstructed data...")
    vae_recon_classifier = SimpleCNN(img_channels=3, img_size=args.img_size, num_classes=20)
    vae_recon_history = train_classifier(
        vae_recon_classifier,
        vae_recon_train_loader,
        vae_recon_val_loader,
        num_epochs=args.clf_epochs,
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )

    vae_recon_metrics = evaluate_classifier(vae_recon_classifier, vae_recon_test_loader, device)

    
    
    
    
    
    
    
    print("Creating DVAE reconstructed dataset...")
    
    # Create a dataloader with paired data for DVAE
    dvae_recon_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        add_noise_online=True,
        return_pairs=True,
        img_size=args.img_size
    )
    dvae_recon_dataset.return_pairs = True  # (noisy_img, label) format
    print(f"DVAE reconstruction dataset size after subsampling: {len(dvae_recon_dataset)}")
    
    dvae_recon_loader = DataLoader(
        dvae_recon_dataset,
        batch_size=args.batch_size,
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
        val_split=0.1,
        shuffle=False,
        num_workers=args.num_workers
    )

    dvae_test_dataset = get_missouri_camera_traps_dataset(
        root=args.data_dir,
        train=False,
        noise_type=args.noise_type,
        noise_params=None,
        add_noise_online=args.add_noise_online,
        return_pairs=args.return_pairs,
        img_size=args.img_size
    )
    dvae_test_loader,_ = create_data_loaders(dvae_test_dataset, batch_size=args.batch_size, val_split=0.0, shuffle=False, num_workers=args.num_workers)

    dvae_recon_data_test = []
    dvae_recon_labels_test = []
    
    dvae_model.eval()
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(dvae_test_loader, desc="DVAE reconstructions"):
            batch_data = batch_data.to(device)
            
            # Get reconstructions
            recons, _, _ = dvae_model(batch_data)
            
            # Store reconstructions and labels
            dvae_recon_data_test.append(recons.cpu())
            dvae_recon_labels_test.append(batch_labels)
    
    dvae_recon_data_test = torch.cat(dvae_recon_data_test, dim=0)
    dvae_recon_labels_test = torch.cat(dvae_recon_labels_test, dim=0)
    
    # Create VAE reconstructed dataset
    dvae_recon_dataset_test = TensorDataset(dvae_recon_data_test, dvae_recon_labels_test)
    dvae_recon_test_loader, _ = create_data_loaders(
        dvae_recon_dataset_test,
        batch_size=args.clf_batch_size,
        val_split=0.0,
        shuffle=False,
        num_workers=args.num_workers
    )
    

    show_random_samples(dvae_recon_dataset_test, num_samples=5, save_dir=os.path.join(args.save_dir, "dvae_reconstructed"), class_indices=[9,11,13,14,15], model_name="DVAE " + args.model_name)

    # Train a classifier on the DVAE reconstructed dataset
    print("\nTraining classifier on DVAE reconstructed data...")
    dvae_recon_classifier = SimpleCNN(img_channels=3, img_size=args.img_size, num_classes=20)
    dvae_recon_history = train_classifier(
        dvae_recon_classifier,
        dvae_recon_train_loader,
        dvae_recon_val_loader,
        num_epochs=args.clf_epochs,
        lr=args.clf_lr,
        device=device,
        early_stopping=False
    )

    dvae_recon_metrics = evaluate_classifier(dvae_recon_classifier, dvae_recon_test_loader, device)

    metrics = {
        'accuracy': 0.4578,
        'report': None
    }
    history = {}

    # Save the results
    results = {
        'unnoised': {
            'accuracy': metrics['accuracy'],
            'report': metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in history.items()}
        },
        'vae_reconstructed': {
            'accuracy': vae_recon_metrics['accuracy'],
            'report': vae_recon_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in vae_recon_history.items()}
        },
        'dvae_reconstructed': {
            'accuracy': dvae_recon_metrics['accuracy'],
            'report': dvae_recon_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in dvae_recon_history.items()}
        }
    }
    
    #Unnoised data: Test Accuracy = 0.4578
    # Print summary
    print("\nClassification Results:")
    print(f"Unnoised data: Test Accuracy = {results['unnoised']['accuracy']:.4f}")
    print(f"VAE reconstructed data: Test Accuracy = {results['vae_reconstructed']['accuracy']:.4f}")
    print(f"DVAE reconstructed data: Test Accuracy = {results['dvae_reconstructed']['accuracy']:.4f}")

    with open(os.path.join(args.save_dir, f'results_missouri_camera_traps.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

def plot_results():
    # Define the noise factors you want to plot
    noise_factors = [0.25, 0.5, 0.75, 1.0]
    results_dir = "/home/user/dvae4da/src/results/missouri_camera_traps"
    noise_type = "blur"

    dvae = []
    vae = []
    clean = []

    for noise_factor in noise_factors:
        results_path = os.path.join(
            results_dir,
            f"{noise_type}_factor{noise_factor}",
            "results_missouri_camera_traps.json"
        )
        with open(results_path, "r") as f:
            results = json.load(f)
        dvae.append(results['dvae_reconstructed']['accuracy'])
        vae.append(results['vae_reconstructed']['accuracy'])
        clean.append(results['unnoised']['accuracy'])

    model_names = ['Raw', 'VAE Reconstructed', 'DVAE Reconstructed']
    all_accs = [clean, vae, dvae]
    bar_width = 0.2
    x = np.arange(len(noise_factors))

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, accs in enumerate(all_accs):
        ax.bar(x + i * bar_width, accs, width=bar_width, label=model_names[i])
    ax.set_xticks(x + bar_width)
    ax.set_xticklabels([str(nf) for nf in noise_factors])
    ax.set_xlabel("Noise Level")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Missouri Camera Traps Accuracy with {noise_type.capitalize()} Reconstruction Models")
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.legend(loc='upper center', ncol=3, bbox_to_anchor=(0.5, 1.15), fontsize='large', frameon=False)

    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Reserve space at the top for the legend
    save_path = os.path.join(results_dir, f"{noise_type}_accuracy_bars.png")
    plt.savefig(save_path, bbox_inches='tight')
    print(f"Saved plot to {save_path}")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Missouri Camera Traps Dataset Loader")
    parser.add_argument("--save_dir", type=str, default="./results/missouri_camera_traps", help="Path to save results")
    parser.add_argument("--data-dir", type=str, default="./data/missouri_camera_traps/images/Set1", help="Path to Missouri Camera Traps data directory")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of DataLoader workers")
    parser.add_argument("--noise-type", type=str, default=None, help="Type of noise to add (if any)")
    parser.add_argument("--add-noise-online", action="store_true", help="Add noise online (default: False)")
    parser.add_argument("--return-pairs", action="store_true", help="Return (noisy, clean) pairs (default: False)")
    parser.add_argument("--clf-lr", type=float, default=0.001, help="Learning rate for classifier")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the experiment on")
    parser.add_argument("--img-size", type=int, default=32, help="Image size")
    parser.add_argument("--clf-epochs", type=int, default=10, help="Number of epochs to train the classifier")
    parser.add_argument("--clf-batch-size", type=int, default=64, help="Batch size for classifier")
    parser.add_argument("--model-dir", type=str, default="./results/cifar10_to_cifar100_denoising", help="Path to model directory")
    parser.add_argument("--model-name", type=str, default="gaussian_factor1.0", help="Model name")
    parser.add_argument("--noise-factor", type=float, default=0.1, help="Noise factor")
    args = parser.parse_args()
    #main(args)
    plot_results()

    
    