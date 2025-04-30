import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Union, Callable
import os
import time
from datetime import datetime
from ..models import VAE, DVAE, ConditionalDVAE


class Trainer:
    """
    Trainer class for VAE/DVAE models.
    """
    def __init__(self, 
                model: Union[VAE, DVAE, ConditionalDVAE],
                train_loader: DataLoader,
                val_loader: Optional[DataLoader] = None,
                optimizer: Optional[torch.optim.Optimizer] = None,
                lr: float = 1e-3,
                device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                save_dir: str = 'results',
                experiment_name: Optional[str] = None):
        """
        Initialize the trainer.
        
        Args:
            model: VAE or DVAE model to train
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            optimizer: Optional optimizer, if not provided, Adam will be used
            lr: Learning rate for the optimizer if not provided
            device: Device to train on
            save_dir: Directory to save results
            experiment_name: Name of the experiment
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=lr)
        else:
            self.optimizer = optimizer
        
        self.save_dir = save_dir
        if experiment_name is None:
            self.experiment_name = f"{model.__class__.__name__}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.experiment_name = experiment_name
        
        self.model_save_dir = os.path.join(save_dir, self.experiment_name, 'models')
        self.log_save_dir = os.path.join(save_dir, self.experiment_name, 'logs')
        self.sample_save_dir = os.path.join(save_dir, self.experiment_name, 'samples')
        
        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.log_save_dir, exist_ok=True)
        os.makedirs(self.sample_save_dir, exist_ok=True)
        
        self.train_losses = {'total': [], 'recon': [], 'kl': []}
        self.val_losses = {'total': [], 'recon': [], 'kl': []}
        self.best_val_loss = float('inf')
        self.current_epoch = 0
    
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
                # Handle different data formats (VAE vs DVAE)
                if isinstance(self.model, DVAE) and not isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        noisy_img, clean_img, _ = batch
                        noisy_img = noisy_img.to(self.device)
                        clean_img = clean_img.to(self.device)
                        
                        recon_batch, mu, log_var = self.model(noisy_img)
                        
                        loss_dict = self.model.loss_function(recon_batch, clean_img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                elif isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        noisy_img, clean_img, labels = batch
                        noisy_img = noisy_img.to(self.device)
                        clean_img = clean_img.to(self.device)
                        labels = labels.to(self.device)
                        
                        recon_batch, mu, log_var = self.model(noisy_img, labels)
                        
                        # Compute loss
                        loss_dict = self.model.loss_function(recon_batch, clean_img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                else:  # Standard VAE
                    if len(batch) == 2:  # (img, label)
                        img, _ = batch
                        img = img.to(self.device)
                        
                        # Forward pass
                        recon_batch, mu, log_var = self.model(img)
                        
                        # Compute loss
                        loss_dict = self.model.loss_function(recon_batch, img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (img, label)")
                
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
                    'loss': loss_dict['loss'].item() / len(batch[0]),
                    'recon': loss_dict['recon_loss'].item() / len(batch[0]),
                    'kl': loss_dict['kl_loss'].item() / len(batch[0])
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
                # Handle different data formats (VAE vs DVAE)
                if isinstance(self.model, DVAE) and not isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        noisy_img, clean_img, _ = batch
                        noisy_img = noisy_img.to(self.device)
                        clean_img = clean_img.to(self.device)
                        
                        recon_batch, mu, log_var = self.model(noisy_img)
                        
                        loss_dict = self.model.loss_function(recon_batch, clean_img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                elif isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        noisy_img, clean_img, labels = batch
                        noisy_img = noisy_img.to(self.device)
                        clean_img = clean_img.to(self.device)
                        labels = labels.to(self.device)
                        
                        recon_batch, mu, log_var = self.model(noisy_img, labels)
                        
                        loss_dict = self.model.loss_function(recon_batch, clean_img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                else:  # Standard VAE
                    if len(batch) == 2:  # (img, label)
                        img, _ = batch
                        img = img.to(self.device)
                        
                        recon_batch, mu, log_var = self.model(img)
                        
                        loss_dict = self.model.loss_function(recon_batch, img, mu, log_var)
                    else:
                        raise ValueError("Expected batch to contain (img, label)")
                
                val_losses['total'] += loss_dict['loss'].item()
                val_losses['recon'] += loss_dict['recon_loss'].item()
                val_losses['kl'] += loss_dict['kl_loss'].item()
        
        for k in val_losses:
            val_losses[k] /= len(self.val_loader.dataset)
        
        for k in val_losses:
            self.val_losses[k].append(val_losses[k])
        
        return val_losses
    
    def train(self, 
             num_epochs: int, 
             save_interval: int = 5,
             early_stopping: bool = True,
             patience: int = 10,
             save_best_only: bool = True) -> Dict[str, List[float]]:
        """
        Train the model for multiple epochs.
        
        Args:
            num_epochs: Number of epochs to train for
            save_interval: Interval for saving model checkpoints
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
            save_best_only: Whether to save only the best model
            
        Returns:
            Dictionary with training and validation losses
        """
        print(f"Training {self.model.__class__.__name__} for {num_epochs} epochs")
        print(f"Device: {self.device}")
        
        self.model.to(self.device)
        
        no_improvement = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_losses = self.train_epoch()
            
            if self.val_loader is not None:
                val_losses = self.validate()
                
                if val_losses['total'] < self.best_val_loss:
                    self.best_val_loss = val_losses['total']
                    no_improvement = 0
                    
                if save_best_only:
                    self.save_model('best')
                else:
                    no_improvement += 1
                
                if early_stopping and no_improvement >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
                
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses['total']:.4f}, "
                     f"Val Loss: {val_losses['total']:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_losses['total']:.4f}")
            
            if (epoch + 1) % save_interval == 0 and not save_best_only:
                self.save_model(f"epoch_{epoch+1}")
        
        self.save_model('final')
        
        self.plot_losses()
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
    
    def save_model(self, name: str) -> None:
        """
        Save the model.
        
        Args:
            name: Name to use for saving the model
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'current_epoch': self.current_epoch,
            'model_class': self.model.__class__.__name__,
            'model_params': {
                'img_channels': self.model.img_channels,
                'img_size': self.model.img_size,
                'latent_dim': self.model.latent_dim,
                'kl_weight': self.model.kl_weight
            }
        }
        
        if isinstance(self.model, ConditionalDVAE):
            checkpoint['model_params']['num_classes'] = self.model.num_classes
        
        torch.save(checkpoint, os.path.join(self.model_save_dir, f"{name}.pt"))
    
    def load_model(self, path: str) -> None:
        """
        Load a saved model.
        
        Args:
            path: Path to the saved model
        """
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint['train_losses']
        self.val_losses = checkpoint['val_losses']
        self.current_epoch = checkpoint['current_epoch']
    
    def plot_losses(self) -> None:
        """
        Plot and save the training and validation loss curves.
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        axes[0].plot(self.train_losses['total'], label='Train')
        if self.val_loader is not None:
            axes[0].plot(self.val_losses['total'], label='Validation')
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        
        axes[1].plot(self.train_losses['recon'], label='Train')
        if self.val_loader is not None:
            axes[1].plot(self.val_losses['recon'], label='Validation')
        axes[1].set_title('Reconstruction Loss')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        
        axes[2].plot(self.train_losses['kl'], label='Train')
        if self.val_loader is not None:
            axes[2].plot(self.val_losses['kl'], label='Validation')
        axes[2].set_title('KL Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_save_dir, 'loss_curves.png'))
        plt.close()
    
    def generate_samples(self, 
                        n_samples: int = 16, 
                        labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            labels: Optional class labels for conditional models
            
        Returns:
            Tensor of generated samples [n_samples, C, H, W]
        """
        self.model.eval()
        
        with torch.no_grad():
            if isinstance(self.model, ConditionalDVAE):
                if labels is None:
                    n_per_class = n_samples // self.model.num_classes
                    remainder = n_samples % self.model.num_classes
                    
                    labels_list = []
                    for i in range(self.model.num_classes):
                        count = n_per_class + (1 if i < remainder else 0)
                        labels_list.extend([i] * count)
                    
                    labels = torch.tensor(labels_list).to(self.device)
                
                # Generate samples with labels
                samples = self.model.sample(n_samples, labels, self.device)
            else:
                # Generate samples from standard VAE/DVAE
                samples = self.model.sample(n_samples, self.device)
        
        return samples
    
    def encode_dataset(self, dataloader: DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode a dataset to obtain latent representations and corresponding labels.
        
        Args:
            dataloader: DataLoader for the dataset
            
        Returns:
            Tuple of (latent_codes, labels)
        """
        self.model.eval()
        latent_codes = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding dataset"):
                if isinstance(self.model, DVAE) and not isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        _, clean_img, labels = batch
                        clean_img = clean_img.to(self.device)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                elif isinstance(self.model, ConditionalDVAE):
                    if len(batch) == 3:  # (noisy_img, clean_img, label)
                        _, clean_img, labels = batch
                        clean_img = clean_img.to(self.device)
                    else:
                        raise ValueError("Expected batch to contain (noisy_img, clean_img, label)")
                else:  # Standard VAE
                    if len(batch) == 2:  # (img, label)
                        img, labels = batch
                        clean_img = img.to(self.device)
                    else:
                        raise ValueError("Expected batch to contain (img, label)")
                
                mu, _ = self.model.encode(clean_img)
                
                latent_codes.append(mu.cpu())
                all_labels.append(labels)
        
        latent_codes = torch.cat(latent_codes, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        
        return latent_codes, all_labels


def compute_psnr(original: torch.Tensor, reconstructed: torch.Tensor) -> float:
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between original and reconstructed images.
    
    Args:
        original: Original images [B, C, H, W]
        reconstructed: Reconstructed images [B, C, H, W]
        
    Returns:
        PSNR value (higher is better)
    """
    mse = torch.mean((original - reconstructed) ** 2)
    if mse == 0:
        return float('inf')
    max_pixel = 1.0  # Assuming images are normalized to [0, 1]
    psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
    return psnr.item() 