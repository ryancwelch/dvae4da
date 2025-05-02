import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
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

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_cifar10_dataset, get_imagenet_dataset, create_data_loaders, subsample_dataset
from src.models.vae import VAE
from src.models.dvae import DVAE
from src.utils.training import Trainer
from src.utils.visualization import visualize_generated_samples
from src.utils.noise import add_noise

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

# --- Training and Evaluation Utilities (same as in your file) ---
def train_classifier(model, train_loader, val_loader=None, num_epochs=10, lr=1e-3, device=None, early_stopping=True, patience=5):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    no_improvement = 0
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
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
        train_loss /= train_total
        train_acc = train_correct / train_total
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        if val_loader is not None:
            model.eval()
            val_loss, val_correct, val_total = 0.0, 0, 0
            with torch.no_grad():
                for inputs, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == targets).sum().item()
                    val_total += targets.size(0)
            val_loss /= val_total
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

def evaluate_classifier(model, test_loader, device=None):
    device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    test_loss, test_correct, test_total = 0.0, 0, 0
    all_targets, all_predictions = [], []
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc="Evaluating"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            test_correct += (predicted == targets).sum().item()
            test_total += targets.size(0)
            all_targets.extend(targets.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
    test_loss /= test_total
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

# --- Main Experiment ---
def parse_args():
    parser = argparse.ArgumentParser(description="CIFAR10-to-ImageNet Denoising Transfer Experiment")
    parser.add_argument("--cifar10-data-dir", type=str, default="./data/cifar10")
    parser.add_argument("--imagenet-data-dir", type=str, default="./data/imagenet")
    parser.add_argument("--save-dir", type=str, default="results/cifar10_to_imagenet_denoising")
    parser.add_argument("--noise-type", type=str, default="gaussian")
    parser.add_argument("--noise-factor", type=float, default=0.5)
    parser.add_argument("--vae-epochs", type=int, default=20)
    parser.add_argument("--dvae-epochs", type=int, default=20)
    parser.add_argument("--clf-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imagenet-classes", type=int, default=100)  # For speed, use a subset
    parser.add_argument("--imagenet-samples-per-class", type=int, default=100)
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    # --- 1. Train VAE and DVAE on CIFAR-10 ---
    print("Preparing CIFAR-10 dataset...")
    cifar10_train = get_cifar10_dataset(
        root=args.cifar10_data_dir,
        train=True,
        noise_type=args.noise_type,
        noise_params={'noise_factor': args.noise_factor, 'clip_min': 0.0, 'clip_max': 1.0},
        download=True
    )
    cifar10_train.return_pairs = False
    cifar10_train_loader, cifar10_val_loader = create_data_loaders(
        cifar10_train, batch_size=args.batch_size, val_split=0.1, shuffle=True, num_workers=args.num_workers
    )
    hidden_dims = [32, 64, 128]
    latent_dim = 32
    kl_weight = 0.1
    vae = VAE(img_channels=3, img_size=32, hidden_dims=hidden_dims, latent_dim=latent_dim, kl_weight=kl_weight).to(device)
    vae_trainer = Trainer(
        model=vae,
        train_loader=cifar10_train_loader,
        val_loader=cifar10_val_loader,
        lr=1e-3,
        device=device,
        save_dir=args.save_dir,
        experiment_name="vae_cifar10"
    )
    print("Training VAE on CIFAR-10...")
    vae_trainer.train(num_epochs=args.vae_epochs, early_stopping=True, patience=5, save_best_only=True)
    # DVAE
    cifar10_train_pairs = get_cifar10_dataset(
        root=args.cifar10_data_dir,
        train=True,
        noise_type=args.noise_type,
        noise_params={'noise_factor': args.noise_factor, 'clip_min': 0.0, 'clip_max': 1.0},
        download=True
    )
    cifar10_train_pairs.return_pairs = True
    cifar10_train_pairs_loader, cifar10_val_pairs_loader = create_data_loaders(
        cifar10_train_pairs, batch_size=args.batch_size, val_split=0.1, shuffle=True, num_workers=args.num_workers
    )
    dvae = DVAE(img_channels=3, img_size=32, hidden_dims=hidden_dims, latent_dim=latent_dim, kl_weight=kl_weight).to(device)
    dvae_trainer = Trainer(
        model=dvae,
        train_loader=cifar10_train_pairs_loader,
        val_loader=cifar10_val_pairs_loader,
        lr=1e-3,
        device=device,
        save_dir=args.save_dir,
        experiment_name="dvae_cifar10"
    )
    print("Training DVAE on CIFAR-10...")
    dvae_trainer.train(num_epochs=args.dvae_epochs, early_stopping=True, patience=5, save_best_only=True)
    # --- 2. Prepare Noisy ImageNet Dataset ---
    print("Preparing noisy ImageNet dataset...")
    imagenet_train = get_imagenet_dataset(
        root=args.imagenet_data_dir,
        train=True,
        num_classes=args.imagenet_classes,
        samples_per_class=args.imagenet_samples_per_class,
        resize=32,  # Resize to 32x32 for compatibility
        download=True
    )
    imagenet_train.return_pairs = False
    # Add noise to all images
    noisy_imgs, noisy_labels = [], []
    for img, label in tqdm(imagenet_train, desc="Adding noise to ImageNet"):
        img_noisy = add_noise(
            img.unsqueeze(0),  # add_noise expects batch
            noise_type=args.noise_type,
            noise_params={'noise_factor': args.noise_factor, 'clip_min': 0.0, 'clip_max': 1.0}
        ).squeeze(0)
        noisy_imgs.append(img_noisy)
        noisy_labels.append(label)
    noisy_imgs = torch.stack(noisy_imgs)
    noisy_labels = torch.tensor(noisy_labels)
    noisy_imagenet_dataset = TensorDataset(noisy_imgs, noisy_labels)
    noisy_train_loader, noisy_val_loader = create_data_loaders(
        noisy_imagenet_dataset, batch_size=args.batch_size, val_split=0.2, shuffle=True, num_workers=args.num_workers
    )
    # --- 3. Train classifier on noisy ImageNet ---
    print("Training classifier on noisy ImageNet...")
    clf_noisy = SimpleCNN(img_channels=3, img_size=32, num_classes=args.imagenet_classes)
    noisy_history = train_classifier(
        clf_noisy, noisy_train_loader, noisy_val_loader, num_epochs=args.clf_epochs, lr=1e-3, device=device, early_stopping=False
    )
    # --- 4. Denoise ImageNet using CIFAR10-trained VAE and DVAE ---
    print("Denoising ImageNet with VAE...")
    vae.eval()
    vae_denoised_imgs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(noisy_imgs), args.batch_size), desc="VAE denoising"):
            batch = noisy_imgs[i:i+args.batch_size].to(device)
            recons, _, _ = vae(batch)
            vae_denoised_imgs.append(recons.cpu())
    vae_denoised_imgs = torch.cat(vae_denoised_imgs, dim=0)
    vae_denoised_dataset = TensorDataset(vae_denoised_imgs, noisy_labels)
    vae_denoised_train_loader, vae_denoised_val_loader = create_data_loaders(
        vae_denoised_dataset, batch_size=args.batch_size, val_split=0.2, shuffle=True, num_workers=args.num_workers
    )
    print("Denoising ImageNet with DVAE...")
    dvae.eval()
    dvae_denoised_imgs = []
    with torch.no_grad():
        for i in tqdm(range(0, len(noisy_imgs), args.batch_size), desc="DVAE denoising"):
            batch = noisy_imgs[i:i+args.batch_size].to(device)
            # DVAE expects (noisy, clean), but we only have noisy, so pass None for clean
            recons, _, _ = dvae(batch, None)
            dvae_denoised_imgs.append(recons.cpu())
    dvae_denoised_imgs = torch.cat(dvae_denoised_imgs, dim=0)
    dvae_denoised_dataset = TensorDataset(dvae_denoised_imgs, noisy_labels)
    dvae_denoised_train_loader, dvae_denoised_val_loader = create_data_loaders(
        dvae_denoised_dataset, batch_size=args.batch_size, val_split=0.2, shuffle=True, num_workers=args.num_workers
    )
    # --- 5. Train classifiers on denoised ImageNet datasets ---
    print("Training classifier on VAE-denoised ImageNet...")
    clf_vae = SimpleCNN(img_channels=3, img_size=32, num_classes=args.imagenet_classes)
    vae_history = train_classifier(
        clf_vae, vae_denoised_train_loader, vae_denoised_val_loader, num_epochs=args.clf_epochs, lr=1e-3, device=device, early_stopping=False
    )
    print("Training classifier on DVAE-denoised ImageNet...")
    clf_dvae = SimpleCNN(img_channels=3, img_size=32, num_classes=args.imagenet_classes)
    dvae_history = train_classifier(
        clf_dvae, dvae_denoised_train_loader, dvae_denoised_val_loader, num_epochs=args.clf_epochs, lr=1e-3, device=device, early_stopping=False
    )
    # --- 6. Evaluate and Compare ---
    print("Evaluating classifiers...")
    # For evaluation, use the validation set as test set
    noisy_metrics = evaluate_classifier(clf_noisy, noisy_val_loader, device)
    vae_metrics = evaluate_classifier(clf_vae, vae_denoised_val_loader, device)
    dvae_metrics = evaluate_classifier(clf_dvae, dvae_denoised_val_loader, device)
    results = {
        'noisy': {
            'accuracy': noisy_metrics['accuracy'],
            'report': noisy_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in noisy_history.items()}
        },
        'vae_denoised': {
            'accuracy': vae_metrics['accuracy'],
            'report': vae_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in vae_history.items()}
        },
        'dvae_denoised': {
            'accuracy': dvae_metrics['accuracy'],
            'report': dvae_metrics['report'],
            'history': {k: [float(v) for v in vs] for k, vs in dvae_history.items()}
        }
    }
    print("\nClassification Results:")
    print(f"Noisy ImageNet: Test Accuracy = {results['noisy']['accuracy']:.4f}")
    print(f"VAE Denoised ImageNet: Test Accuracy = {results['vae_denoised']['accuracy']:.4f}")
    print(f"DVAE Denoised ImageNet: Test Accuracy = {results['dvae_denoised']['accuracy']:.4f}")
    with open(os.path.join(args.save_dir, 'results.json'), 'w') as f:
        json.dump(results, f, indent=4, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
    # Optionally, add plots as in your original file
    print(f"\nExperiment complete! Results saved to {args.save_dir}")

if __name__ == "__main__":
    main() 