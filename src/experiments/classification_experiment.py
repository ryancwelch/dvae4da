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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model
from src.utils.training import Trainer


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
        
        feature_size = img_size // (2**3)
        
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


def load_vae_model(model_path: str, device: torch.device) -> nn.Module:

    checkpoint = torch.load(model_path, map_location=device)
    
    model_class = checkpoint['model_class']
    model_params = checkpoint['model_params']
    
    if model_class == 'VAE':
        model = get_vae_model(
            img_channels=model_params['img_channels'],
            img_size=model_params['img_size'],
            latent_dim=model_params['latent_dim'],
            kl_weight=model_params['kl_weight']
        )
    elif model_class == 'DVAE':
        model = get_dvae_model(
            img_channels=model_params['img_channels'],
            img_size=model_params['img_size'],
            latent_dim=model_params['latent_dim'],
            kl_weight=model_params['kl_weight']
        )
    else:
        raise ValueError(f"Unsupported model class: {model_class}")
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model


def generate_synthetic_dataset(model: nn.Module, 
                              num_samples_per_class: int, 
                              num_classes: int,
                              device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic dataset using a generative model.
    
    Args:
        model: Generative model
        num_samples_per_class: Number of samples to generate per class
        num_classes: Number of classes
        device: Device to generate on
        
    Returns:
        Tuple of (images, labels)
    """
    model.eval()
    
    all_samples = []
    all_labels = []
    
    with torch.no_grad():
        for label in range(num_classes):
            labels = torch.full((num_samples_per_class,), label, dtype=torch.long)
            
            z = torch.randn(num_samples_per_class, model.latent_dim).to(device)
            samples = model.decode(z)
            
            all_samples.append(samples.cpu())
            all_labels.append(labels)
    
    all_samples = torch.cat(all_samples, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_samples, all_labels


def create_augmented_dataset(original_images: torch.Tensor, 
                           original_labels: torch.Tensor,
                           synthetic_images: torch.Tensor,
                           synthetic_labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create an augmented dataset by combining original and synthetic data.
    
    Args:
        original_images: Original images
        original_labels: Original labels
        synthetic_images: Synthetic images
        synthetic_labels: Synthetic labels
        
    Returns:
        Tuple of (augmented_images, augmented_labels)
    """
    augmented_images = torch.cat([original_images, synthetic_images], dim=0)
    augmented_labels = torch.cat([original_labels, synthetic_labels], dim=0)
    
    return augmented_images, augmented_labels


def parse_args():
    parser = argparse.ArgumentParser(description="Compare classification performance with VAE/DVAE generated samples")
    
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes in the dataset")
    parser.add_argument("--subsample", type=int, default=100, 
                       help="Number of training samples per class to use")
    
    parser.add_argument("--vae-model", type=str, required=True, help="Path to the trained VAE model")
    parser.add_argument("--dvae-model", type=str, required=True, help="Path to the trained DVAE model")
    parser.add_argument("--aug-samples", type=int, default=100, 
                       help="Number of augmentation samples per class to generate")
    
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train classifiers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the classifier")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and evaluation")
    
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to train on (cuda or cpu)")
    
    parser.add_argument("--save-dir", type=str, default="results/classification", 
                       help="Directory to save results")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    device = torch.device(args.device)
    
    print("Loading MNIST dataset...")
    train_dataset = get_mnist_dataset(
        root=args.data_dir,
        train=True,
        download=True
    )
    test_dataset = get_mnist_dataset(
        root=args.data_dir,
        train=False,
        download=True
    )
    
    train_dataset = subsample_dataset(
        train_dataset, 
        args.subsample * args.num_classes,  # Total number of samples
        stratified=True
    )
    print(f"Subsampled training dataset to {len(train_dataset)} examples "
         f"({args.subsample} per class)")
    
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
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
    
    original_images = []
    original_labels = []
    for images, labels in train_loader:
        original_images.append(images)
        original_labels.append(labels)
    original_images = torch.cat(original_images, dim=0)
    original_labels = torch.cat(original_labels, dim=0)
    
    print("Loading VAE and DVAE models...")
    vae_model = load_vae_model(args.vae_model, device)
    dvae_model = load_vae_model(args.dvae_model, device)
    
    print(f"Generating {args.aug_samples} synthetic samples per class with VAE...")
    vae_images, vae_labels = generate_synthetic_dataset(
        vae_model,
        args.aug_samples,
        args.num_classes,
        device
    )
    
    print(f"Generating {args.aug_samples} synthetic samples per class with DVAE...")
    dvae_images, dvae_labels = generate_synthetic_dataset(
        dvae_model,
        args.aug_samples,
        args.num_classes,
        device
    )
    
    original_dataset = TensorDataset(original_images, original_labels)
    original_train_loader, original_val_loader = create_data_loaders(
        original_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=4
    )
    
    vae_augmented_images, vae_augmented_labels = create_augmented_dataset(
        original_images, original_labels, vae_images, vae_labels
    )
    vae_augmented_dataset = TensorDataset(vae_augmented_images, vae_augmented_labels)
    vae_train_loader, vae_val_loader = create_data_loaders(
        vae_augmented_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=4
    )
    
    dvae_augmented_images, dvae_augmented_labels = create_augmented_dataset(
        original_images, original_labels, dvae_images, dvae_labels
    )
    dvae_augmented_dataset = TensorDataset(dvae_augmented_images, dvae_augmented_labels)
    dvae_train_loader, dvae_val_loader = create_data_loaders(
        dvae_augmented_dataset,
        batch_size=args.batch_size,
        val_split=0.2,
        shuffle=True,
        num_workers=4
    )
    
    print("Training classifiers...")
    
    print("Training classifier on original data only...")
    original_classifier = SimpleCNN(img_channels=1, img_size=28, num_classes=args.num_classes)
    original_history = train_classifier(
        original_classifier,
        original_train_loader,
        original_val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        early_stopping=True,
        patience=5
    )
    
    print("Training classifier on VAE augmented data...")
    vae_classifier = SimpleCNN(img_channels=1, img_size=28, num_classes=args.num_classes)
    vae_history = train_classifier(
        vae_classifier,
        vae_train_loader,
        vae_val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        early_stopping=True,
        patience=5
    )
    
    print("Training classifier on DVAE augmented data...")
    dvae_classifier = SimpleCNN(img_channels=1, img_size=28, num_classes=args.num_classes)
    dvae_history = train_classifier(
        dvae_classifier,
        dvae_train_loader,
        dvae_val_loader,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
        early_stopping=True,
        patience=5
    )
    
    print("Evaluating classifier trained on original data...")
    original_metrics = evaluate_classifier(original_classifier, test_loader, device)
    
    print("Evaluating classifier trained on VAE augmented data...")
    vae_metrics = evaluate_classifier(vae_classifier, test_loader, device)
    
    print("Evaluating classifier trained on DVAE augmented data...")
    dvae_metrics = evaluate_classifier(dvae_classifier, test_loader, device)
    
    print("\nClassification Results:")
    print(f"Original data only: Accuracy = {original_metrics['accuracy']:.4f}")
    print(f"VAE augmented data: Accuracy = {vae_metrics['accuracy']:.4f}")
    print(f"DVAE augmented data: Accuracy = {dvae_metrics['accuracy']:.4f}")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 2, 1)
    plt.plot(original_history['train_acc'], label='Original')
    plt.plot(vae_history['train_acc'], label='VAE Augmented')
    plt.plot(dvae_history['train_acc'], label='DVAE Augmented')
    plt.title('Training Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 2)
    plt.plot(original_history['val_acc'], label='Original')
    plt.plot(vae_history['val_acc'], label='VAE Augmented')
    plt.plot(dvae_history['val_acc'], label='DVAE Augmented')
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(2, 2, 3)
    plt.plot(original_history['train_loss'], label='Original')
    plt.plot(vae_history['train_loss'], label='VAE Augmented')
    plt.plot(dvae_history['train_loss'], label='DVAE Augmented')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(2, 2, 4)
    plt.plot(original_history['val_loss'], label='Original')
    plt.plot(vae_history['val_loss'], label='VAE Augmented')
    plt.plot(dvae_history['val_loss'], label='DVAE Augmented')
    plt.title('Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'training_history.png'))
    
    plt.figure(figsize=(18, 5))
    
    plt.subplot(1, 3, 1)
    sns.heatmap(original_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'Original Data\nAccuracy: {original_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 2)
    sns.heatmap(vae_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'VAE Augmented\nAccuracy: {vae_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.subplot(1, 3, 3)
    sns.heatmap(dvae_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
    plt.title(f'DVAE Augmented\nAccuracy: {dvae_metrics["accuracy"]:.4f}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'confusion_matrices.png'))
    
    plt.figure(figsize=(15, 10))
    
    classes = range(args.num_classes)
    original_f1 = [original_metrics['report'][str(i)]['f1-score'] for i in classes]
    vae_f1 = [vae_metrics['report'][str(i)]['f1-score'] for i in classes]
    dvae_f1 = [dvae_metrics['report'][str(i)]['f1-score'] for i in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    plt.bar(x - width, original_f1, width, label='Original')
    plt.bar(x, vae_f1, width, label='VAE Augmented')
    plt.bar(x + width, dvae_f1, width, label='DVAE Augmented')
    
    plt.xlabel('Class')
    plt.ylabel('F1-Score')
    plt.title('Class-wise F1-Score Comparison')
    plt.xticks(x, classes)
    plt.ylim(0, 1.1)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, 'class_wise_performance.png'))
    
    results = {
        'original': {
            'accuracy': original_metrics['accuracy'],
            'report': original_metrics['report'],
            'history': original_history
        },
        'vae_augmented': {
            'accuracy': vae_metrics['accuracy'],
            'report': vae_metrics['report'],
            'history': vae_history
        },
        'dvae_augmented': {
            'accuracy': dvae_metrics['accuracy'],
            'report': dvae_metrics['report'],
            'history': dvae_history
        }
    }
    
    import json
    
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return obj
    
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, default=convert_to_json_serializable, indent=4)
    
    print(f"Results saved to {args.save_dir}")


if __name__ == "__main__":
    main() 