import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable, Union
import os
import random
from ..utils.noise import add_noise


class NoisyDataset(Dataset):
    """
    A wrapper dataset that adds noise to images from a base dataset.
    """
    def __init__(self, 
                base_dataset: Dataset, 
                noise_type: str = 'gaussian',
                noise_params: Optional[Dict] = None,
                transform: Optional[Callable] = None,
                add_noise_online: bool = True,
                return_pairs: bool = True):

        self.base_dataset = base_dataset
        self.noise_type = noise_type
        self.noise_params = noise_params if noise_params is not None else {}
        self.transform = transform
        self.add_noise_online = add_noise_online
        self.return_pairs = return_pairs
        
        if not add_noise_online:
            self.noisy_data = []
            for i in range(len(base_dataset)):
                img, label = base_dataset[i]
                if isinstance(img, torch.Tensor):
                    noisy_img = add_noise(img.unsqueeze(0), 
                                        noise_type=noise_type, 
                                        noise_params=self.noise_params).squeeze(0)
                else:
                    transform = transforms.ToTensor()
                    img_tensor = transform(img)
                    noisy_img_tensor = add_noise(img_tensor.unsqueeze(0), 
                                              noise_type=noise_type, 
                                              noise_params=self.noise_params).squeeze(0)
                    noisy_img = transforms.ToPILImage()(noisy_img_tensor)
                self.noisy_data.append(noisy_img)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        clean_img, label = self.base_dataset[idx]
        
        if self.add_noise_online:
            if isinstance(clean_img, torch.Tensor):
                noisy_img = add_noise(clean_img.unsqueeze(0), 
                                     noise_type=self.noise_type, 
                                     noise_params=self.noise_params).squeeze(0)
            else:
                transform = transforms.ToTensor()
                img_tensor = transform(clean_img)
                noisy_img_tensor = add_noise(img_tensor.unsqueeze(0), 
                                           noise_type=self.noise_type, 
                                           noise_params=self.noise_params).squeeze(0)
                noisy_img = transforms.ToPILImage()(noisy_img_tensor)
        else:
            noisy_img = self.noisy_data[idx]
        
        if self.transform:
            if self.return_pairs:
                clean_img = self.transform(clean_img)
            noisy_img = self.transform(noisy_img)
        
        if self.return_pairs:
            return (noisy_img, clean_img, label)
        else:
            return (noisy_img, label)


def get_mnist_dataset(root: str = './data',
                     train: bool = True,
                     noise_type: Optional[str] = None,
                     noise_params: Optional[Dict] = None,
                     download: bool = True) -> Dataset:
    """
    Get the MNIST dataset with optional noise.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.MNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    if noise_type is not None:
        dataset = NoisyDataset(
            base_dataset=dataset,
            noise_type=noise_type,
            noise_params=noise_params,
            add_noise_online=True,
            return_pairs=True
        )
    
    return dataset


def get_fashion_mnist_dataset(root: str = './data',
                             train: bool = True,
                             noise_type: Optional[str] = None,
                             noise_params: Optional[Dict] = None,
                             download: bool = True) -> Dataset:
    """
    Get the Fashion-MNIST dataset with optional noise.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.FashionMNIST(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    if noise_type is not None:
        dataset = NoisyDataset(
            base_dataset=dataset,
            noise_type=noise_type,
            noise_params=noise_params,
            add_noise_online=True,
            return_pairs=True
        )
    
    return dataset


def get_cifar10_dataset(root: str = './data',
                       train: bool = True,
                       noise_type: Optional[str] = None,
                       noise_params: Optional[Dict] = None,
                       download: bool = True) -> Dataset:
    """
    Get the CIFAR-10 dataset with optional noise.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root=root,
        train=train,
        download=download,
        transform=transform
    )
    
    if noise_type is not None:
        dataset = NoisyDataset(
            base_dataset=dataset,
            noise_type=noise_type,
            noise_params=noise_params,
            add_noise_online=True,
            return_pairs=True
        )
    
    return dataset


def subsample_dataset(dataset: Dataset, 
                     num_samples: int, 
                     stratified: bool = True) -> Dataset:
    """
    Create a subsampled version of a dataset to simulate limited data.
    """
    if not stratified or not hasattr(dataset, 'targets'):
        indices = torch.randperm(len(dataset))[:num_samples]
        return Subset(dataset, indices)
    
    targets = torch.tensor(dataset.targets)
    classes = torch.unique(targets)
    n_classes = len(classes)
    
    samples_per_class = num_samples // n_classes
    
    selected_indices = []
    for c in classes:
        class_indices = torch.where(targets == c)[0]
        selected_indices.append(class_indices[:samples_per_class])
    
    indices = torch.cat(selected_indices)
    indices = indices[torch.randperm(len(indices))]
    
    return Subset(dataset, indices)


def create_data_loaders(dataset: Dataset, 
                       batch_size: int = 128,
                       val_split: float = 0.1,
                       shuffle: bool = True,
                       num_workers: int = 4) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Create training and validation data loaders from a dataset.
    """
    if val_split > 0:
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        
        return train_loader, val_loader
    else:
        train_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers
        )
        
        return train_loader, None 