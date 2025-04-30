import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, ConcatDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
import json
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_mnist_dataset, get_cifar10_dataset, subsample_dataset, create_data_loaders
from src.models import get_vae_model, get_dvae_model, get_conditional_dvae_model
from src.utils.training import Trainer
from src.experiments.data_augmentation_experiment import SimpleCNN, train_classifier, evaluate_classifier


def parse_args():
    parser = argparse.ArgumentParser(description="Run augmentation experiment with VAE/DVAE/Conditional DVAE")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                        help="Dataset to use")
    parser.add_argument("--noise-factor", type=float, default=1.0,
                        help="Noise factor to apply")
    parser.add_argument("--noise-type", type=str, default="gaussian", choices=["gaussian", "salt_and_pepper"],
                        help="Type of noise to apply")
    
    # Training parameters
    parser.add_argument("--vae-epochs", type=int, default=30,
                        help="Number of epochs to train VAE/DVAE models")
    parser.add_argument("--clf-epochs", type=int, default=20,
                        help="Number of epochs to train classifiers")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--noise-scale", type=float, default=0.7,
                        help="Scale of noise to add in latent space for synthetic samples")
    parser.add_argument("--num-synthetic", type=int, default=4,
                        help="Number of synthetic samples to generate per image")
    
    # Experiment parameters
    parser.add_argument("--dataset-sizes", type=int, nargs='+',
                        default=[10, 25, 50, 100, 200, 500],
                        help="Dataset sizes (samples per class) to test")
    parser.add_argument("--save-dir", type=str, default="results/augmentation_experiment",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use for training (cuda or cpu)")
    
    return parser.parse_args()


def train_models_on_full_dataset(dataset_name="mnist",
                                noise_factor=1.0, 
                                noise_type="gaussian", 
                                save_dir="results/big_augmentation_experiment",
                                device="cuda",
                                vae_epochs=30,
                                batch_size=64,
                                num_workers=4):
    """Train VAE, DVAE, and ConditionalDVAE on the full dataset."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Dataset-specific parameters
    if dataset_name == "mnist":
        img_channels = 1
        img_size = 28
        hidden_dims = [32, 64, 128]
        latent_dim = 32
    else:  # cifar10
        img_channels = 3
        img_size = 32
        hidden_dims = [32, 64, 128, 256]
        latent_dim = 128
    
    num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
    kl_weight = 0.1  # For standard models
    cond_kl_weight = 0.05  # Lower for conditional model
    
    # Noise parameters
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    # Get datasets
    print(f"Loading {dataset_name} datasets with {noise_type} noise (factor: {noise_factor})...")
    
    # Get appropriate dataset function
    dataset_func = get_mnist_dataset if dataset_name == "mnist" else get_cifar10_dataset
    
    # Standard VAE dataset (no pairs needed)
    vae_train_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    vae_train_dataset.return_pairs = False
    
    # DVAE dataset (need noisy/clean pairs)
    dvae_train_dataset = dataset_func(
        root="./data",
        train=True,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    dvae_train_dataset.return_pairs = True
    
    # Create data loaders
    vae_train_loader, vae_val_loader = create_data_loaders(
        vae_train_dataset,
        batch_size=batch_size,
        val_split=0.1,
        shuffle=True,
        num_workers=num_workers
    )
    
    dvae_train_loader, dvae_val_loader = create_data_loaders(
        dvae_train_dataset,
        batch_size=batch_size,
        val_split=0.1,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Train VAE
    print(f"Training VAE on full {dataset_name} dataset...")
    vae_model = get_vae_model(
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
        save_dir=save_dir,
        experiment_name=f"vae_full_{dataset_name}"
    )
    
    vae_trainer.train(
        num_epochs=vae_epochs,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # Train DVAE
    print(f"Training DVAE on full {dataset_name} dataset...")
    dvae_model = get_dvae_model(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )
    dvae_model.to(device)
    
    dvae_trainer = Trainer(
        model=dvae_model,
        train_loader=dvae_train_loader,
        val_loader=dvae_val_loader,
        device=device,
        save_dir=save_dir,
        experiment_name=f"dvae_full_{dataset_name}"
    )
    
    dvae_trainer.train(
        num_epochs=vae_epochs,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    # Train Conditional DVAE
    print(f"Training ConditionalDVAE on full {dataset_name} dataset...")
    cdvae_model = get_conditional_dvae_model(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=cond_kl_weight,
        num_classes=num_classes
    )
    cdvae_model.to(device)
    
    # Create dataset from DVAE pairs and add labels
    all_noisy_imgs = []
    all_clean_imgs = []
    all_labels = []
    
    for batch in dvae_train_loader:
        noisy_imgs, clean_imgs, labels = batch
        all_noisy_imgs.append(noisy_imgs)
        all_clean_imgs.append(clean_imgs)
        all_labels.append(labels)
    
    all_noisy_imgs = torch.cat(all_noisy_imgs)
    all_clean_imgs = torch.cat(all_clean_imgs)
    all_labels = torch.cat(all_labels)
    
    # Create TensorDataset
    cdvae_train_dataset = TensorDataset(all_noisy_imgs, all_clean_imgs, all_labels)
    
    # Create proper DataLoaders
    cdvae_train_loader, cdvae_val_loader = create_data_loaders(
        cdvae_train_dataset,
        batch_size=batch_size,
        val_split=0.1,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Remove the unused collate function and create optimizer for the cdvae model
    optimizer = optim.Adam(cdvae_model.parameters(), lr=3e-4)  # Lower learning rate for better training
    
    cdvae_trainer = Trainer(
        model=cdvae_model,
        train_loader=cdvae_train_loader,
        val_loader=cdvae_val_loader,
        optimizer=optimizer,  # Add optimizer
        device=device,
        save_dir=save_dir,
        experiment_name=f"cdvae_full_{dataset_name}"
    )
    
    cdvae_trainer.train(
        num_epochs=vae_epochs,
        early_stopping=True,
        patience=5,
        save_best_only=True
    )
    
    return {
        'vae': vae_model,
        'dvae': dvae_model,
        'cdvae': cdvae_model
    }


def create_augmented_datasets(models, dataset_sizes, dataset_name="mnist", noise_factor=1.0, noise_type="gaussian", 
                             device="cuda", save_dir="results/augmentation_experiment",
                             num_synthetic_per_image=4, noise_scale=0.7, batch_size=64, clf_epochs=20):
    """
    Create various augmented datasets for each dataset size.
    
    Args:
        models: Dict containing trained 'vae', 'dvae', and 'cdvae' models
        dataset_sizes: List of dataset sizes (samples per class) to test
        dataset_name: Dataset name (mnist or cifar10)
        noise_factor: Noise factor to apply
        noise_type: Type of noise to apply
        device: Computing device
        save_dir: Directory to save results
        num_synthetic_per_image: Number of synthetic samples to generate per real image
        noise_scale: Scale of noise to add in latent space
    """
    results = {}
    
    # Dataset-specific parameters
    if dataset_name == "mnist":
        img_channels = 1
        img_size = 28
    else:  # cifar10
        img_channels = 3
        img_size = 32
    
    num_classes = 10  # Both MNIST and CIFAR-10 have 10 classes
    
    # Noise parameters
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    # Get appropriate dataset function
    dataset_func = get_mnist_dataset if dataset_name == "mnist" else get_cifar10_dataset
    
    # Set models to evaluation mode
    for model_name, model in models.items():
        model.eval()
    
    # For each dataset size
    for samples_per_class in tqdm(dataset_sizes, desc="Processing different dataset sizes"):
        print(f"\nCreating datasets for {samples_per_class} samples per class...")
        size_dir = os.path.join(save_dir, f"size_{samples_per_class}")
        os.makedirs(size_dir, exist_ok=True)
        
        # Get limited noisy dataset
        noisy_dataset = dataset_func(
            root="./data",
            train=True,
            noise_type=noise_type,
            noise_params=noise_params,
            download=True
        )
        noisy_dataset.return_pairs = True
        
        # Subsample the dataset
        limited_dataset = subsample_dataset(
            noisy_dataset,
            num_samples=samples_per_class * num_classes,  # 10 classes
            stratified=True
        )
        
        # Create a dataloader to process the dataset
        dataloader = DataLoader(
            limited_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        # Initialize storage for reconstructed and synthetic images
        noisy_images = []
        noisy_labels = []
        vae_recon_images = []
        dvae_recon_images = []
        cdvae_recon_images = []
        vae_synthetic_images = []
        dvae_synthetic_images = []
        cdvae_synthetic_images = []
        
        # Process each batch
        with torch.no_grad():
            for noisy_imgs, clean_imgs, labels in tqdm(dataloader, desc=f"Processing images for size {samples_per_class}"):
                noisy_imgs = noisy_imgs.to(device)
                labels = labels.to(device)
                
                # Store noisy images and labels
                noisy_images.append(noisy_imgs.cpu())
                noisy_labels.append(labels.cpu())
                
                # VAE reconstructions
                vae_recons, _, _ = models['vae'](noisy_imgs)
                vae_recon_images.append(vae_recons.cpu())
                
                # DVAE reconstructions
                dvae_recons, _, _ = models['dvae'](noisy_imgs)
                dvae_recon_images.append(dvae_recons.cpu())
                
                # ConditionalDVAE reconstructions
                cdvae_recons, _, _ = models['cdvae'](noisy_imgs, labels)
                cdvae_recon_images.append(cdvae_recons.cpu())
                
                # Generate synthetic samples
                for i in range(len(noisy_imgs)):
                    img = noisy_imgs[i:i+1]
                    label = labels[i:i+1]
                    
                    # VAE synthetic samples
                    mean, logvar = models['vae'].encode(img)
                    for _ in range(num_synthetic_per_image):
                        # Add noise to the latent representation
                        noise = torch.randn_like(mean) * noise_scale
                        z = mean + noise
                        # Decode to get a synthetic image
                        vae_synthetic = models['vae'].decode(z)
                        vae_synthetic_images.append(vae_synthetic.cpu())
                        noisy_labels.append(label.cpu())  # Same label as original
                    
                    # DVAE synthetic samples
                    mean, logvar = models['dvae'].encode(img)
                    for _ in range(num_synthetic_per_image):
                        noise = torch.randn_like(mean) * noise_scale
                        z = mean + noise
                        dvae_synthetic = models['dvae'].decode(z)
                        dvae_synthetic_images.append(dvae_synthetic.cpu())
                        noisy_labels.append(label.cpu())
                    
                    # ConditionalDVAE synthetic samples
                    mean, logvar = models['cdvae'].encode(img)
                    for _ in range(num_synthetic_per_image):
                        noise = torch.randn_like(mean) * noise_scale
                        z = mean + noise
                        cdvae_synthetic = models['cdvae'].conditional_decode(z, label)
                        cdvae_synthetic_images.append(cdvae_synthetic.cpu())
                        noisy_labels.append(label.cpu())
        
        # Concatenate all images and labels
        noisy_images = torch.cat(noisy_images)
        noisy_labels = torch.cat(noisy_labels)
        vae_recon_images = torch.cat(vae_recon_images)
        dvae_recon_images = torch.cat(dvae_recon_images)
        cdvae_recon_images = torch.cat(cdvae_recon_images)
        
        # Handle the fact that synthetic images are generated one at a time
        vae_synthetic_images = torch.cat(vae_synthetic_images) if vae_synthetic_images else None
        dvae_synthetic_images = torch.cat(dvae_synthetic_images) if dvae_synthetic_images else None
        cdvae_synthetic_images = torch.cat(cdvae_synthetic_images) if cdvae_synthetic_images else None
        
        # Create the 7 datasets
        datasets = {
            '1_noisy': TensorDataset(noisy_images, noisy_labels[:len(noisy_images)]),
            '2_vae_recon': TensorDataset(vae_recon_images, noisy_labels[:len(vae_recon_images)]),
            '3_dvae_recon': TensorDataset(dvae_recon_images, noisy_labels[:len(dvae_recon_images)]),
            '4_cdvae_recon': TensorDataset(cdvae_recon_images, noisy_labels[:len(cdvae_recon_images)])
        }
        
        # Combine reconstructed + synthetic
        datasets['5_vae_recon_synthetic'] = ConcatDataset([
            datasets['2_vae_recon'],
            TensorDataset(vae_synthetic_images, noisy_labels[len(noisy_images):len(noisy_images)+len(vae_synthetic_images)])
        ])
        
        datasets['6_dvae_recon_synthetic'] = ConcatDataset([
            datasets['3_dvae_recon'],
            TensorDataset(dvae_synthetic_images, noisy_labels[len(noisy_images):len(noisy_images)+len(dvae_synthetic_images)])
        ])
        
        datasets['7_cdvae_recon_synthetic'] = ConcatDataset([
            datasets['4_cdvae_recon'],
            TensorDataset(cdvae_synthetic_images, noisy_labels[len(noisy_images):len(noisy_images)+len(cdvae_synthetic_images)])
        ])
        
        # Save visualization of datasets
        visualize_datasets(
            datasets=datasets,
            save_path=os.path.join(size_dir, 'dataset_samples.png')
        )
        
        # Train and evaluate classifiers on each dataset
        dataset_results = {}
        
        # Get the test dataset (clean, no noise)
        test_dataset = dataset_func(
            root="./data",
            train=False,
            download=True
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2
        )
        
        for name, dataset in datasets.items():
            print(f"Training classifier on dataset: {name}")
            
            # Create data loader
            train_loader, val_loader = create_data_loaders(
                dataset,
                batch_size=batch_size,
                val_split=0.1,
                shuffle=True,
                num_workers=2
            )
            
            # Create and train classifier
            classifier = SimpleCNN(img_channels=img_channels, img_size=img_size, num_classes=num_classes).to(device)
            
            history = train_classifier(
                model=classifier,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                num_epochs=clf_epochs,  # Train for a fixed number of epochs
                early_stopping=True,
                patience=3
            )
            
            # Evaluate on test set
            metrics = evaluate_classifier(
                model=classifier,
                test_loader=test_loader,
                device=device
            )
            
            dataset_results[name] = {
                'history': history,
                'test_metrics': metrics
            }
            
            print(f"Dataset {name} - Test Accuracy: {metrics['accuracy']:.4f}")
        
        # Save results for this dataset size
        with open(os.path.join(size_dir, 'results.json'), 'w') as f:
            json_results = {k: {
                'test_accuracy': v['test_metrics']['accuracy'],
                'test_loss': v['test_metrics']['loss']
            } for k, v in dataset_results.items()}
            json.dump(json_results, f, indent=4)
        
        results[samples_per_class] = dataset_results
    
    return results


def visualize_datasets(datasets, save_path, num_samples=10):
    """Visualize samples from each dataset."""
    plt.figure(figsize=(15, len(datasets) * 2))
    
    for i, (name, dataset) in enumerate(datasets.items()):
        # Get random samples
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        samples = [dataset[idx][0] for idx in indices]
        
        # Create a grid
        samples = torch.stack(samples)
        grid = torchvision.utils.make_grid(samples, nrow=5, normalize=True, padding=2)
        plt_img = np.transpose(grid.numpy(), (1, 2, 0))
        
        plt.subplot(len(datasets), 1, i+1)
        plt.imshow(plt_img)
        plt.title(name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def analyze_results(results, dataset_sizes, save_dir="results/augmentation_experiment"):
    """Analyze and visualize the results across dataset sizes."""
    # Extract accuracies for each dataset type
    accuracies = {
        '1_noisy': [],
        '2_vae_recon': [],
        '3_dvae_recon': [],
        '4_cdvae_recon': [],
        '5_vae_recon_synthetic': [],
        '6_dvae_recon_synthetic': [],
        '7_cdvae_recon_synthetic': []
    }
    
    for size in dataset_sizes:
        for dataset_name in accuracies.keys():
            accuracies[dataset_name].append(
                results[size][dataset_name]['test_metrics']['accuracy']
            )
    
    # Plot results
    plt.figure(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'd', '*', 'x', '+']
    colors = ['blue', 'green', 'red', 'purple', 'orange', 'brown', 'pink']
    
    for i, (name, acc_list) in enumerate(accuracies.items()):
        plt.plot(dataset_sizes, acc_list, marker=markers[i], color=colors[i], label=name)
    
    plt.xscale('log')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Samples per Class (log scale)')
    plt.ylabel('Test Accuracy')
    plt.title('Effect of Different Augmentation Strategies on Classification Accuracy')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, 'augmentation_comparison.png'), dpi=300)
    plt.savefig(os.path.join(save_dir, 'augmentation_comparison.pdf'))
    
    # Create a summary table with improvements over baseline
    print("\nSummary of Results:")
    print(f"{'Dataset Size':<15} {'Noisy':<10} {'VAE Recon':<10} {'DVAE Recon':<10} {'CDVAE Recon':<10} {'VAE+Synth':<10} {'DVAE+Synth':<10} {'CDVAE+Synth':<10}")
    print('-' * 100)
    
    for i, size in enumerate(dataset_sizes):
        row = [f"{size:<15}"]
        for name in accuracies.keys():
            row.append(f"{accuracies[name][i]:.4f}")
        print(' '.join(row))
    
    # Calculate improvements over noisy baseline
    print("\nImprovements over Noisy Baseline:")
    print(f"{'Dataset Size':<15} {'VAE Recon':<12} {'DVAE Recon':<12} {'CDVAE Recon':<12} {'VAE+Synth':<12} {'DVAE+Synth':<12} {'CDVAE+Synth':<12}")
    print('-' * 100)
    
    for i, size in enumerate(dataset_sizes):
        baseline = accuracies['1_noisy'][i]
        row = [f"{size:<15}"]
        
        for name in list(accuracies.keys())[1:]:  # Skip the baseline
            improvement = accuracies[name][i] - baseline
            row.append(f"{improvement:.4f} ({improvement*100:.1f}%)")
        
        print(' '.join(row))


def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Set parameters from arguments
    dataset_name = args.dataset
    noise_factor = args.noise_factor
    noise_type = args.noise_type
    dataset_sizes = args.dataset_sizes
    save_dir = os.path.join(args.save_dir, dataset_name)
    device = torch.device(args.device)
    
    # Step 1: Train models on the full dataset
    print(f"Training models on full {dataset_name.upper()} dataset with {noise_type} noise (factor: {noise_factor})...")
    models = train_models_on_full_dataset(
        dataset_name=dataset_name,
        noise_factor=noise_factor,
        noise_type=noise_type,
        save_dir=save_dir,
        device=device,
        vae_epochs=args.vae_epochs,
        batch_size=args.batch_size
    )
    
    # Step 2: Create augmented datasets and evaluate
    print("Creating and evaluating augmented datasets...")
    results = create_augmented_datasets(
        models=models,
        dataset_sizes=dataset_sizes,
        dataset_name=dataset_name,
        noise_factor=noise_factor,
        noise_type=noise_type,
        device=device,
        save_dir=save_dir,
        num_synthetic_per_image=args.num_synthetic,
        noise_scale=args.noise_scale,
        batch_size=args.batch_size,
        clf_epochs=args.clf_epochs
    )
    
    # Step 3: Analyze results
    print("Analyzing results...")
    analyze_results(
        results=results,
        dataset_sizes=dataset_sizes,
        save_dir=save_dir
    )
    
    print(f"Experiment complete! Results saved to {save_dir}")


if __name__ == "__main__":
    main()