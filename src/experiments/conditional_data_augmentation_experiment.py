import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, ConcatDataset, TensorDataset, Subset
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
from typing import Dict, List, Tuple, Optional, Union
import sys
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
from tqdm import tqdm
import torchvision
import json
import random

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model, get_conditional_vae_model, get_conditional_dvae_model
from src.utils.training import Trainer
from src.utils.visualization import visualize_generated_samples
from src.data.datasets import NoisyDataset


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
        
        # Calculate feature map size after convolutions
        feature_size = img_size // (2**3)
        if feature_size < 1:
            feature_size = 1
            print(f"Warning: Feature size too small, setting to 1")
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * feature_size * feature_size, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
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
    f1 = f1_score(all_targets, all_predictions, average='weighted')
    
    return {
        'loss': test_loss,
        'accuracy': test_acc,
        'f1_score': f1,
        'report': report,
        'confusion_matrix': conf_matrix,
        'predictions': all_predictions,
        'targets': all_targets
    } 


def train_conditional_model(model_type: str,
                        train_loader: DataLoader,
                        val_loader: Optional[DataLoader] = None,
                        img_channels: int = 1,
                        img_size: int = 28,
                        hidden_dims: List[int] = None,
                        latent_dim: int = 16,
                        num_classes: int = 10,
                        kl_weight: float = 1.0,
                        learning_rate: float = 1e-3,
                        num_epochs: int = 50,
                        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                        save_dir: str = 'results/conditional_models') -> nn.Module:
    """Train a conditional VAE or DVAE model."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    if model_type.lower() == 'vae':
        model = get_conditional_vae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_classes=num_classes,
            kl_weight=kl_weight
        )
        save_path = os.path.join(save_dir, f'conditional_vae_model.pt')
    else:  # dvae
        model = get_conditional_dvae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            num_classes=num_classes,
            kl_weight=kl_weight
        )
        save_path = os.path.join(save_dir, f'conditional_dvae_model.pt')
    
    model.to(device)
    
    # Create a custom trainer for conditional models
    class ConditionalTrainer(Trainer):
        def train_epoch(self) -> Dict[str, float]:
            """
            Train for one epoch.
            
            Returns:
                Dictionary with average losses for the epoch
            """
            self.model.train()
            epoch_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
            num_batches = len(self.train_loader)

            with tqdm(total=num_batches, desc=f'Epoch {self.current_epoch+1}', unit='batch') as pbar:
                for batch_idx, batch in enumerate(self.train_loader):
                    # Process batch - NoisyDataset returns (noisy_img, label) when return_pairs=False
                    inputs, labels = batch
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Get model (handle DataParallel)
                    if isinstance(self.model, torch.nn.DataParallel):
                        model = self.model.module
                    else:
                        model = self.model
                    
                    # Forward pass with conditional model
                    x_hat, mu, log_var = model(inputs, labels)
                    
                    # Compute loss
                    loss_dict = model.loss_function(x_hat, inputs, mu, log_var)
                    
                    # Backpropagation
                    self.optimizer.zero_grad()
                    loss_dict['loss'].backward()
                    self.optimizer.step()
                    
                    # Update epoch losses
                    epoch_losses['total'] += loss_dict['loss'].item()
                    epoch_losses['recon'] += loss_dict['recon_loss'].item()
                    epoch_losses['kl'] += loss_dict['kl_loss'].item()
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'loss': loss_dict['loss'].item() / len(inputs),
                        'recon': loss_dict['recon_loss'].item() / len(inputs),
                        'kl': loss_dict['kl_loss'].item() / len(inputs)
                    })
            
            # Compute average losses
            for k in epoch_losses:
                epoch_losses[k] /= len(self.train_loader.dataset)
            
            # Update tracking variables
            for k in epoch_losses:
                self.train_losses[k].append(epoch_losses[k])
            
            return epoch_losses
        
        def validate(self) -> Dict[str, float]:
            """
            Validate the model.
            
            Returns:
                Dictionary with average validation losses
            """
            self.model.eval()
            val_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
            
            with torch.no_grad():
                for batch in self.val_loader:
                    inputs, labels = batch  # Expecting (img, label) format
                    inputs = inputs.to(self.device)
                    labels = labels.to(self.device)
                    
                    # Get model (handle DataParallel)
                    if isinstance(self.model, torch.nn.DataParallel):
                        model = self.model.module
                    else:
                        model = self.model
                    
                    # Forward pass with conditional model
                    x_hat, mu, log_var = model(inputs, labels)
                    
                    # Compute loss
                    loss_dict = model.loss_function(x_hat, inputs, mu, log_var)
                    
                    val_losses['total'] += loss_dict['loss'].item()
                    val_losses['recon'] += loss_dict['recon_loss'].item()
                    val_losses['kl'] += loss_dict['kl_loss'].item()
            
            for k in val_losses:
                val_losses[k] /= len(self.val_loader.dataset)
            
            for k in val_losses:
                self.val_losses[k].append(val_losses[k])
            
            return val_losses
    
    trainer = ConditionalTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=learning_rate,
        device=device
    )
    
    print(f"Training {model_type.upper()}...")
    trainer.train(num_epochs=num_epochs)
    
    # Save model
    model_params = {
        'img_channels': img_channels,
        'img_size': img_size,
        'latent_dim': latent_dim,
        'kl_weight': kl_weight,
        'hidden_dims': hidden_dims,
        'num_classes': num_classes
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_params': model_params
    }, save_path)
    
    print(f"Model saved to {save_path}")
    return model


def generate_conditional_samples(model: nn.Module, 
                                samples_per_class: int, 
                                num_classes: int, 
                                device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate class-conditional samples from a trained conditional VAE/DVAE model.
    
    Args:
        model: Trained conditional VAE or DVAE model
        samples_per_class: Number of samples to generate per class
        num_classes: Number of classes (0-9 for MNIST)
        device: Device to generate samples on
        
    Returns:
        Tuple of (generated_samples, labels)
    """
    model.to(device)
    model.eval()
    
    total_samples = samples_per_class * num_classes
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for class_idx in range(num_classes):
            # Create class labels tensor
            labels = torch.full((samples_per_class,), class_idx, dtype=torch.long, device=device)
            
            # Generate samples for this class
            samples = model.sample(samples_per_class, labels, device)
            
            all_samples.append(samples)
            all_labels.append(labels)
    
    # Concatenate all samples and labels
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_samples, all_labels


def create_augmented_datasets(original_data: Tuple[torch.Tensor, torch.Tensor],
                             cond_vae_model: nn.Module,
                             cond_dvae_model: nn.Module,
                             samples_per_class: int,
                             num_classes: int,
                             device: torch.device) -> Dict[str, Dataset]:
    """
    Create different datasets for the experiment:
    1. Original noisy dataset
    2. Conditional VAE augmented dataset
    3. Conditional DVAE augmented dataset
    4. Original + Conditional VAE dataset
    5. Original + Conditional DVAE dataset
    
    Args:
        original_data: Tuple of (noisy_images, labels)
        cond_vae_model: Trained conditional VAE model
        cond_dvae_model: Trained conditional DVAE model
        samples_per_class: Number of samples to generate per class
        num_classes: Number of classes
        device: Device to generate samples on
        
    Returns:
        Dictionary containing the different datasets
    """
    noisy_images, labels = original_data
    
    # Create original dataset
    original_dataset = TensorDataset(noisy_images, labels)
    
    # Generate samples from conditional VAE
    print("Generating samples from conditional VAE...")
    vae_images, vae_labels = generate_conditional_samples(
        cond_vae_model, samples_per_class, num_classes, device
    )
    
    # Generate samples from conditional DVAE
    print("Generating samples from conditional DVAE...")
    dvae_images, dvae_labels = generate_conditional_samples(
        cond_dvae_model, samples_per_class, num_classes, device
    )
    
    # Create synthetic datasets
    vae_dataset = TensorDataset(vae_images, vae_labels)
    dvae_dataset = TensorDataset(dvae_images, dvae_labels)
    
    # Create combined datasets
    combined_vae_dataset = ConcatDataset([original_dataset, vae_dataset])
    combined_dvae_dataset = ConcatDataset([original_dataset, dvae_dataset])
    
    return {
        "original": original_dataset,
        "cond_vae": vae_dataset,
        "cond_dvae": dvae_dataset,
        "original_plus_cond_vae": combined_vae_dataset,
        "original_plus_cond_dvae": combined_dvae_dataset
    } 


def plot_results(results: Dict[str, Dict], save_dir: str):
    """Plot performance comparison of different datasets."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Extract metrics
    dataset_names = list(results.keys())
    accuracies = [results[name]['accuracy'] for name in dataset_names]
    f1_scores = [results[name]['f1_score'] for name in dataset_names]
    
    # Create figure
    plt.figure(figsize=(14, 7))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.bar(dataset_names, accuracies, color='skyblue')
    plt.xlabel('Dataset')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    # Plot F1 score
    plt.subplot(1, 2, 2)
    plt.bar(dataset_names, f1_scores, color='lightgreen')
    plt.xlabel('Dataset')
    plt.ylabel('F1 Score')
    plt.title('F1 Score Comparison')
    plt.ylim(0, 1.0)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'performance_comparison.png'))
    plt.close()
    
    # Save results as JSON
    with open(os.path.join(save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)


def visualize_dataset_samples(datasets: Dict[str, Dataset], save_dir: str, num_samples: int = 10):
    """Visualize samples from each dataset."""
    
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, len(datasets) * 2))
    
    for i, (name, dataset) in enumerate(datasets.items()):
        for j in range(num_samples):
            if j < len(dataset):
                # Get a sample
                if isinstance(dataset, TensorDataset):
                    image = dataset[j][0]
                else:  # ConcatDataset
                    # Try to get from first dataset
                    try:
                        image = dataset.datasets[0][j][0]
                    except:
                        image = dataset.datasets[1][j-len(dataset.datasets[0])][0]
                
                # Plot the sample
                plt.subplot(len(datasets), num_samples, i * num_samples + j + 1)
                plt.imshow(image.squeeze().cpu().numpy(), cmap='gray')
                plt.axis('off')
                
                if j == 0:
                    plt.title(name, fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_samples.png'))
    plt.close()


def parse_args():
    """Parse command line arguments."""
    
    parser = argparse.ArgumentParser(description='Conditional Data Augmentation Experiment')
    
    # Dataset parameters
    parser.add_argument('--noise-factor', type=float, default=0.3,
                       help='Noise factor for the noisy dataset (default: 0.3)')
    parser.add_argument('--noise-type', type=str, default='gaussian',
                       choices=['gaussian', 'salt_pepper', 'poisson'],
                       help='Type of noise to add (default: gaussian)')
    parser.add_argument('--samples-per-class', type=int, default=100,
                       help='Number of samples per class in the limited dataset (default: 100)')
    parser.add_argument('--generated-samples-per-class', type=int, default=100,
                       help='Number of generated samples per class for augmentation (default: 100)')
    
    # Model parameters
    parser.add_argument('--hidden-dims', type=str, default='32,64,128',
                       help='Comma-separated list of hidden dimensions (default: 32,64,128)')
    parser.add_argument('--latent-dim', type=int, default=32,
                       help='Dimension of the latent space (default: 32)')
    parser.add_argument('--kl-weight', type=float, default=1.0,
                       help='Weight for KL divergence term (default: 1.0)')
    
    # Training parameters
    parser.add_argument('--batch-size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--vae-epochs', type=int, default=20,
                       help='Number of epochs for VAE/DVAE training (default: 20)')
    parser.add_argument('--cnn-epochs', type=int, default=10,
                       help='Number of epochs for CNN training (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (default: 1e-3)')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    parser.add_argument('--save-dir', type=str, default='results/conditional_augmentation',
                       help='Directory to save results (default: results/conditional_augmentation)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                       help='Disable CUDA training')
    
    args = parser.parse_args()
    
    # Process hidden dims
    args.hidden_dims = [int(dim) for dim in args.hidden_dims.split(',')]
    
    return args


def main():
    """Main function."""
    
    args = parse_args()
    
    # Set seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check for CUDA
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load MNIST dataset with noise for model training
    print(f"Loading MNIST dataset with {args.noise_type} noise (factor: {args.noise_factor})...")
    
    # Create a standard MNIST dataset WITHOUT noise first
    mnist_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Manually create a small version with the specified number of samples per class
    print(f"Subsampling {args.samples_per_class} examples per class...")
    indices = []
    labels = mnist_dataset.targets
    for class_idx in range(10):  # MNIST has 10 classes
        class_indices = (labels == class_idx).nonzero().flatten()
        # Take the first samples_per_class indices
        if len(class_indices) > args.samples_per_class:
            class_indices = class_indices[:args.samples_per_class]
        indices.extend(class_indices.tolist())
    
    # Shuffle the indices
    random.shuffle(indices)
    
    # Create a small dataset
    small_mnist_dataset = Subset(mnist_dataset, indices)
    
    # Now create a noisy version with return_pairs=False
    noise_params = {'factor': args.noise_factor}
    limited_train_dataset = NoisyDataset(
        base_dataset=small_mnist_dataset,
        noise_type=args.noise_type,
        noise_params=noise_params,
        return_pairs=False  # Return (noisy_img, label) pairs
    )
    
    # For evaluation, use clean test dataset
    clean_test_dataset = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        limited_train_dataset,  # Using same data for train/val
        batch_size=args.batch_size,
        val_split=0.1,  # 10% for validation
        shuffle=True,
        num_workers=2
    )
    
    test_loader = DataLoader(
        clean_test_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=2
    )
    
    # Extract data for dataset creation - using a temporary data loader with no shuffling to preserve order
    extract_loader = DataLoader(
        limited_train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0  # Use 0 workers to avoid multiprocessing issues
    )
    
    all_images = []
    all_labels = []
    
    # Collect all images and labels from the dataset
    with torch.no_grad():
        for imgs, labels in extract_loader:
            all_images.append(imgs)
            all_labels.append(labels)
    
    noisy_images = torch.cat(all_images)
    labels = torch.cat(all_labels)
    
    # Step 1: Train conditional VAE and DVAE models
    print("Training conditional VAE model...")
    cond_vae_model = train_conditional_model(
        model_type='vae',
        train_loader=train_loader,
        img_channels=1,
        img_size=28,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_classes=10,  # MNIST has 10 classes
        kl_weight=args.kl_weight,
        learning_rate=args.lr,
        num_epochs=args.vae_epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    print("Training conditional DVAE model...")
    cond_dvae_model = train_conditional_model(
        model_type='dvae',
        train_loader=train_loader,
        img_channels=1,
        img_size=28,
        hidden_dims=args.hidden_dims,
        latent_dim=args.latent_dim,
        num_classes=10,  # MNIST has 10 classes
        kl_weight=args.kl_weight,
        learning_rate=args.lr,
        num_epochs=args.vae_epochs,
        device=device,
        save_dir=args.save_dir
    )
    
    # Step 2: Create different datasets
    print("Creating augmented datasets...")
    datasets = create_augmented_datasets(
        original_data=(noisy_images, labels),
        cond_vae_model=cond_vae_model,
        cond_dvae_model=cond_dvae_model,
        samples_per_class=args.generated_samples_per_class,
        num_classes=10,
        device=device
    )
    
    # Visualize samples from each dataset
    print("Visualizing dataset samples...")
    visualize_dataset_samples(
        datasets=datasets,
        save_dir=args.save_dir,
        num_samples=10
    )
    
    # Step 3: Train and evaluate CNN on each dataset
    results = {}
    
    for dataset_name, dataset in datasets.items():
        print(f"Training and evaluating on {dataset_name} dataset...")
        
        # Create data loader
        dataset_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        # Train CNN
        cnn_model = SimpleCNN(img_channels=1, img_size=28, num_classes=10)
        history = train_classifier(
            model=cnn_model,
            train_loader=dataset_loader,
            val_loader=None,  # No validation for simplicity
            num_epochs=args.cnn_epochs,
            lr=args.lr,
            device=device,
            early_stopping=False
        )
        
        # Evaluate on clean test data
        results[dataset_name] = evaluate_classifier(
            model=cnn_model,
            test_loader=test_loader,
            device=device
        )
        
        # Save training history
        with open(os.path.join(args.save_dir, f'{dataset_name}_history.json'), 'w') as f:
            json.dump(history, f, indent=4)
    
    # Step 4: Plot and save results
    print("Plotting results...")
    plot_results(
        results=results,
        save_dir=args.save_dir
    )
    
    print(f"Experiment completed. Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 