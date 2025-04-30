import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset
import matplotlib.pyplot as plt
import os
import sys
import argparse
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import get_mnist_dataset, create_data_loaders, subsample_dataset
from src.utils.noise import add_noise
from src.utils.visualization import visualize_generated_samples
from src.models.vae import VAE
from src.models.dvae import DVAE


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


def train_classifier(model, train_loader, val_loader, num_epochs=10, device='cuda'):
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
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
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == targets).sum().item()
                val_total += targets.size(0)
        
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            print(f"New best validation accuracy: {best_val_acc:.4f}")
    
    return best_val_acc


def evaluate_classifier(model, test_loader, device='cuda'):
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
    
    test_acc = test_correct / test_total
    return test_acc


def load_model(model_path, model_type, device='cuda'):
    """Load a trained VAE or DVAE model using the correct architecture."""
    
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters from the state dict
    model_params = checkpoint.get('model_params', {})
    img_channels = model_params.get('img_channels', 1)
    img_size = model_params.get('img_size', 28)
    latent_dim = model_params.get('latent_dim', 32)
    kl_weight = model_params.get('kl_weight', 0.1)
    
    # Get hidden dims from the state dict (needed for matching architecture)
    hidden_dims = model_params.get('hidden_dims', [32, 64, 128])
    
    print(f"Loading {model_type} model with parameters:")
    print(f"- img_channels: {img_channels}")
    print(f"- img_size: {img_size}")
    print(f"- latent_dim: {latent_dim}")
    print(f"- hidden_dims: {hidden_dims}")
    print(f"- kl_weight: {kl_weight}")
    
    # Create model with matching architecture
    if model_type.lower() == 'vae':
        model = VAE(
            img_channels=img_channels,
            img_size=img_size, 
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    else:  # dvae
        model = DVAE(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model


def generate_samples(model, num_samples, num_classes, device):
    """Generate samples from a trained VAE/DVAE model."""
    samples_per_class = num_samples // num_classes
    all_samples = []
    all_labels = []
    
    for class_idx in range(num_classes):
        print(f"Generating {samples_per_class} samples for class {class_idx}...")
        # Create labels tensor
        labels = torch.full((samples_per_class,), class_idx, dtype=torch.long)
        
        # Generate samples
        with torch.no_grad():
            # If the model can use labels, use them
            if hasattr(model, 'sample') and callable(getattr(model, 'sample')):
                if 'labels' in model.sample.__code__.co_varnames:
                    samples = model.sample(samples_per_class, labels.to(device), device)
                else:
                    samples = model.sample(samples_per_class, device)
            else:
                # Manual sampling from latent space
                z = torch.randn(samples_per_class, model.latent_dim).to(device)
                samples = model.decode(z)
        
        all_samples.append(samples.cpu())
        all_labels.append(labels)
    
    samples = torch.cat(all_samples, dim=0)
    labels = torch.cat(all_labels, dim=0)
    
    return samples, labels


def apply_noise_to_samples(samples, noise_type='gaussian', noise_factor=0.5):
    """Apply noise to generated samples."""
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    return add_noise(samples, noise_type=noise_type, noise_params=noise_params)


def main():
    parser = argparse.ArgumentParser(description="Generate samples from VAE/DVAE and train classifier")
    
    # Models
    parser.add_argument("--vae-model", type=str, required=True, help="Path to trained VAE model")
    parser.add_argument("--dvae-model", type=str, required=True, help="Path to trained DVAE model")
    
    # Data parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist"], help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory for datasets")
    parser.add_argument("--train-samples", type=int, default=1000, help="Number of training samples to use")
    
    # Noise parameters
    parser.add_argument("--noise-type", type=str, default="gaussian", help="Type of noise to apply")
    parser.add_argument("--noise-factor", type=float, default=0.5, help="Noise factor to apply")
    
    # Generation parameters
    parser.add_argument("--samples-per-class", type=int, default=100, help="Samples per class to generate")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train")
    
    # Other parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use")
    parser.add_argument("--save-dir", type=str, default="results/generate_samples", 
                       help="Directory to save results")
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    
    # Step 1: Load the trained models
    print(f"Loading VAE model from {args.vae_model}...")
    vae_model = load_model(args.vae_model, 'vae', device)
    
    print(f"Loading DVAE model from {args.dvae_model}...")
    dvae_model = load_model(args.dvae_model, 'dvae', device)
    
    # Step 2: Get MNIST dataset with noise
    print(f"Loading {args.dataset} dataset...")
    if args.dataset == "mnist":
        # Get MNIST dataset with noise
        noise_params = {
            'noise_factor': args.noise_factor,
            'clip_min': 0.0,
            'clip_max': 1.0
        }
        
        # Load train dataset
        train_dataset = get_mnist_dataset(
            root=args.data_dir,
            train=True,
            noise_type=args.noise_type,
            noise_params=noise_params,
            download=True
        )
        
        # Set return_pairs to False to get (img, label) format
        train_dataset.return_pairs = False
        
        # Subsample the training set
        train_dataset = subsample_dataset(
            train_dataset,
            args.train_samples,
            stratified=True
        )
        
        # Load test dataset
        test_dataset = get_mnist_dataset(
            root=args.data_dir,
            train=False,
            noise_type=args.noise_type,
            noise_params=noise_params,
            download=True
        )
        
        # Set return_pairs to False to get (img, label) format
        test_dataset.return_pairs = False
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True
    )
    
    test_loader, _ = create_data_loaders(
        test_dataset,
        batch_size=args.batch_size,
        val_split=0.0,
        shuffle=False
    )
    
    # Step 3: Generate synthetic samples from VAE
    print(f"Generating {args.samples_per_class} samples per class from VAE...")
    vae_samples, vae_labels = generate_samples(
        vae_model,
        args.samples_per_class * 10,  # 10 classes for MNIST
        10,
        device
    )
    
    # Apply noise to VAE samples
    print(f"Applying {args.noise_type} noise with factor {args.noise_factor} to VAE samples...")
    vae_samples_noisy = apply_noise_to_samples(vae_samples, args.noise_type, args.noise_factor)
    
    # Step 4: Generate synthetic samples from DVAE
    print(f"Generating {args.samples_per_class} samples per class from DVAE...")
    dvae_samples, dvae_labels = generate_samples(
        dvae_model,
        args.samples_per_class * 10,  # 10 classes for MNIST
        10,
        device
    )
    
    # Apply noise to DVAE samples
    print(f"Applying {args.noise_type} noise with factor {args.noise_factor} to DVAE samples...")
    dvae_samples_noisy = apply_noise_to_samples(dvae_samples, args.noise_type, args.noise_factor)
    
    # Step 5: Visualize generated samples
    print("Visualizing generated samples...")
    
    # Visualize VAE samples
    vae_fig = visualize_generated_samples(
        vae_samples[:64],
        grid_size=(8, 8),
        title=f"VAE Generated Samples ({args.noise_type}, factor {args.noise_factor})"
    )
    vae_fig.savefig(os.path.join(args.save_dir, "vae_samples.png"))
    
    # Visualize VAE noisy samples
    vae_noisy_fig = visualize_generated_samples(
        vae_samples_noisy[:64],
        grid_size=(8, 8),
        title=f"VAE Generated Samples with Noise ({args.noise_type}, factor {args.noise_factor})"
    )
    vae_noisy_fig.savefig(os.path.join(args.save_dir, "vae_samples_noisy.png"))
    
    # Visualize DVAE samples
    dvae_fig = visualize_generated_samples(
        dvae_samples[:64],
        grid_size=(8, 8),
        title=f"DVAE Generated Samples ({args.noise_type}, factor {args.noise_factor})"
    )
    dvae_fig.savefig(os.path.join(args.save_dir, "dvae_samples.png"))
    
    # Visualize DVAE noisy samples
    dvae_noisy_fig = visualize_generated_samples(
        dvae_samples_noisy[:64],
        grid_size=(8, 8),
        title=f"DVAE Generated Samples with Noise ({args.noise_type}, factor {args.noise_factor})"
    )
    dvae_noisy_fig.savefig(os.path.join(args.save_dir, "dvae_samples_noisy.png"))
    
    # Step 6: Create datasets for training classifiers
    
    # Original dataset loader (already created as train_loader, val_loader)
    
    # VAE augmented dataset
    vae_aug_dataset = TensorDataset(vae_samples_noisy, vae_labels)
    vae_aug_loader, vae_aug_val_loader = create_data_loaders(
        vae_aug_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True
    )
    
    # DVAE augmented dataset
    dvae_aug_dataset = TensorDataset(dvae_samples_noisy, dvae_labels)
    dvae_aug_loader, dvae_aug_val_loader = create_data_loaders(
        dvae_aug_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True
    )
    
    # Step 7: Train and evaluate classifiers
    results = {}
    
    # Train on original data
    print("\nTraining classifier on original data:")
    original_model = SimpleCNN(img_channels=1, img_size=28, num_classes=10)
    original_val_acc = train_classifier(original_model, train_loader, val_loader, args.epochs, device)
    original_test_acc = evaluate_classifier(original_model, test_loader, device)
    print(f"Original data - Test Accuracy: {original_test_acc:.4f}")
    results['original'] = {
        'val_acc': original_val_acc,
        'test_acc': original_test_acc
    }
    
    # Train on VAE augmented data
    print("\nTraining classifier on VAE augmented data:")
    vae_model = SimpleCNN(img_channels=1, img_size=28, num_classes=10)
    vae_val_acc = train_classifier(vae_model, vae_aug_loader, vae_aug_val_loader, args.epochs, device)
    vae_test_acc = evaluate_classifier(vae_model, test_loader, device)
    print(f"VAE augmented - Test Accuracy: {vae_test_acc:.4f}")
    results['vae'] = {
        'val_acc': vae_val_acc,
        'test_acc': vae_test_acc
    }
    
    # Train on DVAE augmented data
    print("\nTraining classifier on DVAE augmented data:")
    dvae_model = SimpleCNN(img_channels=1, img_size=28, num_classes=10)
    dvae_val_acc = train_classifier(dvae_model, dvae_aug_loader, dvae_aug_val_loader, args.epochs, device)
    dvae_test_acc = evaluate_classifier(dvae_model, test_loader, device)
    print(f"DVAE augmented - Test Accuracy: {dvae_test_acc:.4f}")
    results['dvae'] = {
        'val_acc': dvae_val_acc,
        'test_acc': dvae_test_acc
    }
    
    # Print summary
    print("\nResults Summary:")
    print(f"Original data: Test Accuracy = {results['original']['test_acc']:.4f}")
    print(f"VAE augmented: Test Accuracy = {results['vae']['test_acc']:.4f}")
    print(f"DVAE augmented: Test Accuracy = {results['dvae']['test_acc']:.4f}")
    
    print(f"\nExperiment complete! Results and samples saved to {args.save_dir}")


if __name__ == "__main__":
    main() 