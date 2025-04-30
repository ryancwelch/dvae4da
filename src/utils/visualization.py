import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from typing import List, Tuple, Optional, Union
from sklearn.manifold import TSNE
import io
from PIL import Image
import torchvision


def visualize_noise_examples(clean_images: torch.Tensor, 
                           noisy_images: torch.Tensor,
                           reconstructed_images: Optional[torch.Tensor] = None,
                           num_examples: int = 10,
                           title: str = "Noise Examples") -> Figure:
    """
    Visualize examples of clean, noisy, and optionally reconstructed images.
    
    Args:
        clean_images: Tensor of clean images [N, C, H, W]
        noisy_images: Tensor of noisy images [N, C, H, W]
        reconstructed_images: Optional tensor of reconstructed images [N, C, H, W]
        num_examples: Number of examples to show
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    n = min(num_examples, clean_images.shape[0])
    
    if reconstructed_images is not None:
        rows = 3
    else:
        rows = 2
    
    fig, axes = plt.subplots(rows, n, figsize=(n*2, rows*2))
    
    clean_images = clean_images.detach().cpu()
    noisy_images = noisy_images.detach().cpu()
    if reconstructed_images is not None:
        reconstructed_images = reconstructed_images.detach().cpu()
    
    for i in range(n):
        clean_img = clean_images[i]
        noisy_img = noisy_images[i]
        
        if clean_img.shape[0] == 1:
            clean_img = clean_img.squeeze(0)  # [H, W]
            noisy_img = noisy_img.squeeze(0)  # [H, W]
            cmap = 'gray'
        else:
            clean_img = clean_img.permute(1, 2, 0)  # [H, W, C]
            noisy_img = noisy_img.permute(1, 2, 0)  # [H, W, C]
            cmap = None
        
        if rows == 2 and n == 1:
            axes[0].imshow(clean_img, cmap=cmap)
            axes[0].set_title("Clean")
            axes[0].axis('off')
            
            axes[1].imshow(noisy_img, cmap=cmap)
            axes[1].set_title("Noisy")
            axes[1].axis('off')
        elif rows == 2:
            axes[0, i].imshow(clean_img, cmap=cmap)
            if i == 0:
                axes[0, i].set_ylabel("Clean")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(noisy_img, cmap=cmap)
            if i == 0:
                axes[1, i].set_ylabel("Noisy")
            axes[1, i].axis('off')
        
        if reconstructed_images is not None:
            recon_img = reconstructed_images[i]
            if recon_img.shape[0] == 1:
                recon_img = recon_img.squeeze(0)  # [H, W]
            else:
                recon_img = recon_img.permute(1, 2, 0)  # [H, W, C]
            
            if n == 1:
                axes[2].imshow(recon_img, cmap=cmap)
                axes[2].set_title("Reconstructed")
                axes[2].axis('off')
            else:
                axes[2, i].imshow(recon_img, cmap=cmap)
                if i == 0:
                    axes[2, i].set_ylabel("Reconstructed")
                axes[2, i].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def visualize_reconstructions(original_images: torch.Tensor,
                             reconstructed_images: torch.Tensor,
                             num_examples: int = 10,
                             title: str = "VAE Reconstructions") -> Figure:
    """
    Visualize original and reconstructed images side by side.
    
    Args:
        original_images: Tensor of original images [N, C, H, W]
        reconstructed_images: Tensor of reconstructed images [N, C, H, W]
        num_examples: Number of examples to show
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    n = min(num_examples, original_images.shape[0])
    
    fig, axes = plt.subplots(2, n, figsize=(n*2, 4))
    
    original_images = original_images.detach().cpu()
    reconstructed_images = reconstructed_images.detach().cpu()
    
    for i in range(n):
        orig_img = original_images[i]
        recon_img = reconstructed_images[i]
        
        if orig_img.shape[0] == 1:
            orig_img = orig_img.squeeze(0)  # [H, W]
            recon_img = recon_img.squeeze(0)  # [H, W]
            cmap = 'gray'
        else:
            orig_img = orig_img.permute(1, 2, 0)  # [H, W, C]
            recon_img = recon_img.permute(1, 2, 0)  # [H, W, C]
            cmap = None
        
        if n == 1:
            axes[0].imshow(orig_img, cmap=cmap)
            axes[0].set_title("Original")
            axes[0].axis('off')
            
            axes[1].imshow(recon_img, cmap=cmap)
            axes[1].set_title("Reconstructed")
            axes[1].axis('off')
        else:
            axes[0, i].imshow(orig_img, cmap=cmap)
            if i == 0:
                axes[0, i].set_ylabel("Original")
            axes[0, i].axis('off')
            
            axes[1, i].imshow(recon_img, cmap=cmap)
            if i == 0:
                axes[1, i].set_ylabel("Reconstructed")
            axes[1, i].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def visualize_generated_samples(generated_images: torch.Tensor,
                               num_examples: int = 25,
                               grid_size: Tuple[int, int] = (5, 5),
                               title: str = "Generated Samples") -> Figure:
    """
    Visualize generated samples from a VAE or DVAE.
    
    Args:
        generated_images: Tensor of generated images [N, C, H, W]
        num_examples: Number of examples to show (will be trimmed to grid_size[0] * grid_size[1])
        grid_size: Size of the grid (rows, cols)
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    n = min(num_examples, generated_images.shape[0], grid_size[0] * grid_size[1])
    
    rows, cols = grid_size
    fig, axes = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    
    generated_images = generated_images.detach().cpu()
    
    for i in range(rows):
        for j in range(cols):
            idx = i * cols + j
            if idx < n:
                img = generated_images[idx]
                
                if img.shape[0] == 1:
                    img = img.squeeze(0)  # [H, W]
                    cmap = 'gray'
                else:
                    img = img.permute(1, 2, 0)  # [H, W, C]
                    cmap = None
                
                if rows == 1 and cols == 1:
                    axes.imshow(img, cmap=cmap)
                    axes.axis('off')
                elif rows == 1:
                    axes[j].imshow(img, cmap=cmap)
                    axes[j].axis('off')
                elif cols == 1:
                    axes[i].imshow(img, cmap=cmap)
                    axes[i].axis('off')
                else:
                    axes[i, j].imshow(img, cmap=cmap)
                    axes[i, j].axis('off')
            else:
                if rows == 1 and cols == 1:
                    axes.axis('off')
                elif rows == 1:
                    axes[j].axis('off')
                elif cols == 1:
                    axes[i].axis('off')
                else:
                    axes[i, j].axis('off')
    
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def visualize_latent_space(latent_codes: np.ndarray,
                          labels: np.ndarray,
                          method: str = 'tsne',
                          perplexity: int = 30,
                          n_iter: int = 1000,
                          title: str = "Latent Space Visualization") -> Figure:
    """
    Visualize the latent space using dimensionality reduction.
    
    Args:
        latent_codes: Latent vectors [N, latent_dim]
        labels: Class labels for each point [N]
        method: Dimensionality reduction method ('tsne', 'pca', or 'umap')
        perplexity: Perplexity parameter for t-SNE
        n_iter: Number of iterations for t-SNE
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    if isinstance(latent_codes, torch.Tensor):
        latent_codes = latent_codes.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    
    # Apply dimensionality reduction if needed
    if latent_codes.shape[1] > 2:
        if method == 'tsne':
            # Apply t-SNE
            tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
            latent_2d = tsne.fit_transform(latent_codes)
        elif method == 'pca':
            # Apply PCA
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2, random_state=42)
            latent_2d = pca.fit_transform(latent_codes)
        elif method == 'umap':
            # Apply UMAP if available
            try:
                import umap
                reducer = umap.UMAP(random_state=42)
                latent_2d = reducer.fit_transform(latent_codes)
            except ImportError:
                # Fall back to t-SNE if UMAP not available
                print("UMAP not available. Falling back to t-SNE.")
                tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=n_iter, random_state=42)
                latent_2d = tsne.fit_transform(latent_codes)
        else:
            raise ValueError(f"Unsupported dimensionality reduction method: {method}")
    else:
        # Already 2D, no need for dimensionality reduction
        latent_2d = latent_codes
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Get unique labels and assign colors
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    # Plot points
    for i, label in enumerate(unique_labels):
        indices = labels == label
        ax.scatter(latent_2d[indices, 0], latent_2d[indices, 1], 
                  c=[colors[i]], label=f"Class {label}", alpha=0.7)
    
    # Add legend and labels
    ax.legend(loc='best')
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    
    # Set title
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def visualize_interpolation(start_image: torch.Tensor,
                           end_image: torch.Tensor,
                           interpolated_images: torch.Tensor,
                           title: str = "Latent Space Interpolation") -> Figure:
    """
    Visualize interpolation between two images in the latent space.
    
    Args:
        start_image: Start image tensor [C, H, W]
        end_image: End image tensor [C, H, W]
        interpolated_images: Tensor of interpolated images [N, C, H, W]
        title: Title for the figure
        
    Returns:
        Matplotlib figure
    """
    # Determine how many interpolation steps
    n_steps = interpolated_images.shape[0]
    
    # Create figure
    fig, axes = plt.subplots(1, n_steps + 2, figsize=((n_steps + 2) * 2, 2))
    
    # Move tensors to CPU
    start_image = start_image.detach().cpu()
    end_image = end_image.detach().cpu()
    interpolated_images = interpolated_images.detach().cpu()
    
    # Plot start image
    if start_image.shape[0] == 1:
        start_img = start_image.squeeze(0)  # [H, W]
        cmap = 'gray'
    else:
        start_img = start_image.permute(1, 2, 0)  # [H, W, C]
        cmap = None
    axes[0].imshow(start_img, cmap=cmap)
    axes[0].set_title("Start")
    axes[0].axis('off')
    
    # Plot interpolated images
    for i in range(n_steps):
        img = interpolated_images[i]
        if img.shape[0] == 1:
            img = img.squeeze(0)  # [H, W]
        else:
            img = img.permute(1, 2, 0)  # [H, W, C]
        axes[i + 1].imshow(img, cmap=cmap)
        axes[i + 1].set_title(f"{i+1}")
        axes[i + 1].axis('off')
    
    # Plot end image
    if end_image.shape[0] == 1:
        end_img = end_image.squeeze(0)  # [H, W]
    else:
        end_img = end_image.permute(1, 2, 0)  # [H, W, C]
    axes[-1].imshow(end_img, cmap=cmap)
    axes[-1].set_title("End")
    axes[-1].axis('off')
    
    # Set title
    fig.suptitle(title)
    plt.tight_layout()
    
    return fig


def fig_to_tensor(fig: Figure) -> torch.Tensor:
    """
    Convert a matplotlib figure to a PyTorch tensor.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=120, bbox_inches='tight')
    buf.seek(0)
    img = Image.open(buf)
    transform = torchvision.transforms.ToTensor()
    img_tensor = transform(img)
    
    return img_tensor


def make_grid(images: torch.Tensor, nrow: int = 8, padding: int = 2) -> torch.Tensor:
    """
    Make a grid of images.
    """
    return torchvision.utils.make_grid(images, nrow=nrow, padding=padding) 