import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, List, Optional, Callable, Union
import os
import random
from ..utils.noise import add_noise
from torchvision import datasets
from PIL import Image


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


def get_cifar100_dataset(root: str = './data',
                        train: bool = True,
                        noise_type: Optional[str] = None,
                        noise_params: Optional[Dict] = None,
                        download: bool = True,
                        select_classes: Optional[List[int]] = None,
                        remap_labels: bool = True,
                        add_noise_online: bool = True) -> Dataset:
    """
    Get the CIFAR-100 dataset with optional noise.
    Optionally select only a subset of classes and remap their labels to 0..N-1.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR100(
        root=root,
        train=train,
        download=download,
        transform=transform
    )

    if select_classes is not None:
        # Find indices for selected classes
        targets = np.array(dataset.targets)
        mask = np.isin(targets, select_classes)
        indices = np.where(mask)[0]
        images = []
        labels = []
        class_map = {orig: idx for idx, orig in enumerate(sorted(select_classes))}
        for i in indices:
            img, label = dataset[i]
            images.append(img)
            if remap_labels:
                labels.append(class_map[int(label)])
            else:
                labels.append(int(label))
        images = torch.stack([img if isinstance(img, torch.Tensor) else transforms.ToTensor()(img) for img in images])
        labels = torch.tensor(labels, dtype=torch.long)
        dataset = torch.utils.data.TensorDataset(images, labels)

    if noise_type is not None:
        dataset = NoisyDataset(
            base_dataset=dataset,
            noise_type=noise_type,
            noise_params=noise_params,
            add_noise_online=add_noise_online,
            return_pairs=True
        )
    
    return dataset

def subsample_dataset(dataset: Dataset, 
                     num_samples: int, 
                     stratified: bool = True,
                     seed: int = 42) -> Dataset:
    """
    Create a subsampled version of a dataset to simulate limited data.
    """
    if not stratified or not hasattr(dataset, 'targets'):
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:num_samples]
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
    indices = indices[torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed))]
    
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

def get_stratified_indices(dataset, num_samples, stratified=True, seed=42):
    """
    Return indices for a stratified (or random) subsample of a dataset.
    If stratified is True and dataset has 'targets', samples are drawn evenly from each class.
    """
    if not stratified or not hasattr(dataset, 'targets'):
        indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:num_samples]
        return indices

    targets = torch.tensor(dataset.targets)
    classes = torch.unique(targets)
    n_classes = len(classes)
    samples_per_class = num_samples // n_classes

    selected_indices = []
    rng = torch.Generator().manual_seed(seed)
    for c in classes:
        class_indices = torch.where(targets == c)[0]
        perm = class_indices[torch.randperm(len(class_indices), generator=rng)]
        selected_indices.append(perm[:samples_per_class])

    indices = torch.cat(selected_indices)
    indices = indices[torch.randperm(len(indices), generator=rng)]
    return indices 

class MissouriCameraTrapsBBDataset(Dataset):
    """
    Dataset for Missouri Camera Traps that crops images to the first bounding box (if present)
    and resizes to 32x32.
    """
    def __init__(self, root, labels_file, transform=None):
        self.root = root
        self.transform = transform
        self.samples = []
        self.bboxes = {}

        # Parse labels.txt for bounding boxes
        with open(labels_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                img_path = parts[0]
                n_boxes = int(parts[1])
                if n_boxes > 0:
                    # Only use the first bounding box
                    bbox = tuple(map(int, parts[2:6]))
                    self.bboxes[img_path] = bbox
                else:
                    self.bboxes[img_path] = None

        # Walk through all images in the root directory
        for class_name in os.listdir(root):
            class_dir = os.path.join(root, class_name)
            if not os.path.isdir(class_dir):
                continue
            for dirpath, _, filenames in os.walk(class_dir):
                for fname in filenames:
                    if fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                        rel_path = os.path.relpath(os.path.join(dirpath, fname), root)
                        self.samples.append((rel_path, class_name))

        # Build class_to_idx mapping
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(root))) if os.path.isdir(os.path.join(root, cls))}

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel_path, class_name = self.samples[idx]
        img_path = os.path.join(self.root, rel_path)
        image = Image.open(img_path).convert("RGB")

        bbox = self.bboxes.get(rel_path)
        if bbox is not None:
            image = image.crop(bbox)

        if self.transform:
            image = self.transform(image)

        label = self.class_to_idx[class_name]
        return image, label

def get_missouri_camera_traps_dataset(
    root: str = './data/missouri_camera_traps/images/Set1',
    train: bool = True,
    noise_type: Optional[str] = None,
    noise_params: Optional[Dict] = None,
    download: bool = False,  # for API compatibility, not used
    transform: Optional[transforms.Compose] = None,
    add_noise_online: bool = True,
    return_pairs: bool = True,
    split_ratio: float = 0.8,
    seed: int = 42,
    select_classes: Optional[list] = None,  # for API compatibility, not used
    remap_labels: bool = True,              # for API compatibility, not used
    img_size: int = 128,
) -> Dataset:
    """
    Get the Missouri Camera Traps dataset, cropping to bounding boxes and resizing to 32x32.
    Returns only the train or test split (80/20), with deterministic, class-balanced split (no shuffling).
    API is consistent with other get_*_dataset functions.
    """
    labels_file = os.path.join(root, "labels.txt")
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])

    dataset = MissouriCameraTrapsBBDataset(root, labels_file, transform=transform)

    # Stratified 80/20 split: even split across classes, deterministic order
    class_to_indices = {}
    for idx, (rel_path, class_name) in enumerate(dataset.samples):
        if class_name not in class_to_indices:
            class_to_indices[class_name] = []
        class_to_indices[class_name].append(idx)

    train_indices = []
    test_indices = []
    for class_name, indices in class_to_indices.items():
        n_total = len(indices)
        n_train = int(n_total * 0.8)
        # No shuffling, just take first 80% for train, last 20% for test
        train_indices.extend(indices[:n_train])
        test_indices.extend(indices[n_train:])

    from torch.utils.data import Subset

    if train:
        dataset = Subset(dataset, train_indices)
    else:
        dataset = Subset(dataset, test_indices)

    if noise_type is not None:
        dataset = NoisyDataset(
            base_dataset=dataset,
            noise_type=noise_type,
            noise_params=noise_params,
            add_noise_online=add_noise_online,
            return_pairs=return_pairs
        )

    return dataset 