import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
import sys
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils

# Add parent directory to system path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.experiments.data_augmentation_analysis import run_experiment
from src.data import get_mnist_dataset, get_cifar10_dataset


def visualize_reconstructions(models, dataset_name, noise_type, noise_factor, save_dir, device, num_samples=10):
    """
    Visualize original, noisy, and reconstructed images for both VAE and DVAE.
    
    Args:
        models: Dict containing 'vae' and 'dvae' models
        dataset_name: Dataset name ('mnist' or 'cifar10')
        noise_type: Type of noise applied
        noise_factor: Noise factor value
        save_dir: Directory to save visualizations
        device: Device to use for inference
        num_samples: Number of samples to visualize
    """
    # Get test dataset with noise
    noise_params = {
        'noise_factor': noise_factor,
        'clip_min': 0.0,
        'clip_max': 1.0
    }
    
    if noise_type == 'salt_and_pepper':
        noise_params['salt_vs_pepper'] = 0.5
    
    # Get appropriate dataset function
    dataset_func = get_mnist_dataset if dataset_name == 'mnist' else get_cifar10_dataset
    
    # Load test dataset with specified noise
    test_dataset = dataset_func(
        root='./data',
        train=False,
        noise_type=noise_type,
        noise_params=noise_params,
        download=True
    )
    test_dataset.return_pairs = True
    
    # Create a dataloader to get samples
    test_loader = DataLoader(test_dataset, batch_size=num_samples, shuffle=True)
    
    # Get a batch of samples
    noisy_imgs, clean_imgs, _ = next(iter(test_loader))
    noisy_imgs = noisy_imgs.to(device)
    clean_imgs = clean_imgs.to(device)
    
    # Get reconstructions from VAE and DVAE
    vae_model = models['vae'].to(device)
    dvae_model = models['dvae'].to(device)
    
    vae_model.eval()
    dvae_model.eval()
    
    with torch.no_grad():
        vae_recons, _, _ = vae_model(noisy_imgs)
        dvae_recons, _, _ = dvae_model(noisy_imgs)
    
    # Convert tensors to CPU for visualization
    clean_imgs = clean_imgs.cpu()
    noisy_imgs = noisy_imgs.cpu()
    vae_recons = vae_recons.cpu()
    dvae_recons = dvae_recons.cpu()
    
    # Create a grid of images
    nrow = num_samples if num_samples <= 8 else 8
    
    # Original clean images
    grid_clean = vutils.make_grid(clean_imgs, nrow=nrow, normalize=True)
    
    # Noisy images
    grid_noisy = vutils.make_grid(noisy_imgs, nrow=nrow, normalize=True)
    
    # VAE reconstructions
    grid_vae = vutils.make_grid(vae_recons, nrow=nrow, normalize=True)
    
    # DVAE reconstructions
    grid_dvae = vutils.make_grid(dvae_recons, nrow=nrow, normalize=True)
    
    # Create the visualization
    fig, axs = plt.subplots(4, 1, figsize=(12, 16))
    
    # Convert from tensor format (C,H,W) to matplotlib format (H,W,C)
    img_clean = np.transpose(grid_clean.numpy(), (1, 2, 0))
    img_noisy = np.transpose(grid_noisy.numpy(), (1, 2, 0))
    img_vae = np.transpose(grid_vae.numpy(), (1, 2, 0))
    img_dvae = np.transpose(grid_dvae.numpy(), (1, 2, 0))
    
    # Display images
    axs[0].imshow(img_clean)
    axs[0].set_title('Original Images')
    axs[0].axis('off')
    
    axs[1].imshow(img_noisy)
    axs[1].set_title(f'Noisy Images ({noise_type}, factor={noise_factor})')
    axs[1].axis('off')
    
    axs[2].imshow(img_vae)
    axs[2].set_title('VAE Reconstructions')
    axs[2].axis('off')
    
    axs[3].imshow(img_dvae)
    axs[3].set_title('DVAE Reconstructions')
    axs[3].axis('off')
    
    plt.tight_layout()
    
    # Create directory to save visualizations
    vis_dir = os.path.join(save_dir, 'visualizations')
    os.makedirs(vis_dir, exist_ok=True)
    
    # Save the visualization
    plt.savefig(os.path.join(vis_dir, f'reconstruction_samples_{dataset_name}_{noise_type}_{noise_factor}.png'))
    plt.close()
    
    print(f"Reconstruction visualizations saved to {vis_dir}")


def analyze_noise_levels(
    noise_factors: list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    samples_per_class: int = 500,
    noise_type: str = "gaussian",
    dataset_name: str = "cifar10",
    save_dir: str = "results/noise_level_analysis",
    device: str = "cuda",
    num_workers: int = 4,
    vae_epochs: int = 30,
    clf_epochs: int = 5,
    use_conditional: bool = True
):
    """
    Analyze how accuracy changes with noise level.
    
    Args:
        noise_factors: List of noise factors to test
        samples_per_class: Fixed number of samples per class to use
        noise_type: Type of noise to apply
        dataset_name: Dataset to use
        save_dir: Directory to save results
        device: Device to use for training
        num_workers: Number of workers for data loading
        vae_epochs: Number of epochs to train VAE/DVAE
        clf_epochs: Number of epochs to train classifiers
        use_conditional: Whether to use conditional DVAE
    """
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Store results
    results = {
        'noise_factors': noise_factors,
        'samples_per_class': samples_per_class,
        'noise_type': noise_type,
        'dataset': dataset_name,
        'noisy_accuracies': [],
        'vae_accuracies': [],
        'dvae_accuracies': []
    }
    
    # Add conditional DVAE results if requested
    if use_conditional:
        results['cdvae_accuracies'] = []
    
    # Run experiments for each noise level
    for noise_factor in tqdm(noise_factors, desc="Testing different noise levels"):
        print(f"\n\n{'='*80}\nRunning experiment with noise factor {noise_factor}\n{'='*80}\n")
        
        experiment_results = run_experiment(
            samples_per_class=samples_per_class,
            noise_factor=noise_factor,
            noise_type=noise_type,
            dataset_name=dataset_name,
            num_epochs_vae=vae_epochs,
            num_epochs_clf=clf_epochs,
            save_dir=os.path.join(save_dir, "individual_runs"),
            device=device,
            num_workers=num_workers,
            use_conditional=use_conditional
        )
        
        # Visualize reconstructed samples
        if 'models' in experiment_results:
            print("\nGenerating reconstruction visualizations...")
            visualize_reconstructions(
                models=experiment_results['models'],
                dataset_name=dataset_name,
                noise_type=noise_type,
                noise_factor=noise_factor,
                save_dir=os.path.join(save_dir, f"samples_factor{noise_factor}"),
                device=device
            )
        
        # Store accuracy results
        results['noisy_accuracies'].append(experiment_results['noisy']['accuracy'])
        results['vae_accuracies'].append(experiment_results['vae']['accuracy'])
        results['dvae_accuracies'].append(experiment_results['dvae']['accuracy'])
        
        # Add conditional DVAE results if available
        if use_conditional and 'cdvae' in experiment_results:
            results['cdvae_accuracies'].append(experiment_results['cdvae']['accuracy'])
    
    # Save consolidated results
    with open(os.path.join(save_dir, "noise_level_results.json"), 'w') as f:
        json.dump(results, f, indent=4)
    
    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(noise_factors, results['noisy_accuracies'], 'o-', label='Noisy Data')
    plt.plot(noise_factors, results['vae_accuracies'], 's-', label='VAE Reconstructed')
    plt.plot(noise_factors, results['dvae_accuracies'], '^-', label='DVAE Reconstructed')
    
    if use_conditional and 'cdvae_accuracies' in results:
        plt.plot(noise_factors, results['cdvae_accuracies'], 'd-', label='Conditional DVAE')
    
    plt.grid(True, alpha=0.3)
    plt.xlabel('Noise Factor')
    plt.ylabel('Test Accuracy')
    title = f'Effect of Noise Level on Classification Accuracy\n({samples_per_class} samples per class, {noise_type} noise, {dataset_name.upper()})'
    if use_conditional:
        title += " with Conditional DVAE"
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(os.path.join(save_dir, "noise_level_analysis.png"), dpi=300)
    plt.savefig(os.path.join(save_dir, "noise_level_analysis.pdf"))
    
    print(f"Results saved to {save_dir}")
    
    # Print summary
    print("\nSummary of Results:")
    if use_conditional:
        print(f"{'Noise Factor':<15} {'Noisy':<10} {'VAE':<10} {'DVAE':<10} {'CDVAE':<10}")
        print('-' * 55)
    else:
        print(f"{'Noise Factor':<15} {'Noisy':<10} {'VAE':<10} {'DVAE':<10}")
        print('-' * 45)
    
    for i, factor in enumerate(noise_factors):
        if use_conditional and 'cdvae_accuracies' in results:
            print(f"{factor:<15.1f} {results['noisy_accuracies'][i]:.4f}    {results['vae_accuracies'][i]:.4f}    "
                  f"{results['dvae_accuracies'][i]:.4f}    {results['cdvae_accuracies'][i]:.4f}")
        else:
            print(f"{factor:<15.1f} {results['noisy_accuracies'][i]:.4f}    {results['vae_accuracies'][i]:.4f}    "
                  f"{results['dvae_accuracies'][i]:.4f}")
    
    # Add analysis of when each method performs best
    best_method_by_noise = []
    for i, factor in enumerate(noise_factors):
        accuracies = [
            results['noisy_accuracies'][i],
            results['vae_accuracies'][i],
            results['dvae_accuracies'][i]
        ]
        
        method_names = ["Noisy", "VAE", "DVAE"]
        
        if use_conditional and 'cdvae_accuracies' in results:
            accuracies.append(results['cdvae_accuracies'][i])
            method_names.append("CDVAE")
            
        best_idx = np.argmax(accuracies)
        best_method_by_noise.append(method_names[best_idx])
    
    print("\nBest Performing Method by Noise Level:")
    for i, factor in enumerate(noise_factors):
        print(f"Noise factor {factor:.1f}: {best_method_by_noise[i]}")
    
    # Compute accuracy improvements over noisy baseline
    print("\nAccuracy Improvements over Noisy Baseline:")
    if use_conditional:
        print(f"{'Noise Factor':<15} {'VAE Improvement':<20} {'DVAE Improvement':<20} {'CDVAE Improvement':<20}")
        print('-' * 75)
    else:
        print(f"{'Noise Factor':<15} {'VAE Improvement':<20} {'DVAE Improvement':<20}")
        print('-' * 55)
    
    for i, factor in enumerate(noise_factors):
        vae_improvement = results['vae_accuracies'][i] - results['noisy_accuracies'][i]
        dvae_improvement = results['dvae_accuracies'][i] - results['noisy_accuracies'][i]
        
        if use_conditional and 'cdvae_accuracies' in results:
            cdvae_improvement = results['cdvae_accuracies'][i] - results['noisy_accuracies'][i]
            print(f"{factor:<15.1f} {vae_improvement:.4f} ({vae_improvement*100:.1f}%)    "
                  f"{dvae_improvement:.4f} ({dvae_improvement*100:.1f}%)    "
                  f"{cdvae_improvement:.4f} ({cdvae_improvement*100:.1f}%)")
        else:
            print(f"{factor:<15.1f} {vae_improvement:.4f} ({vae_improvement*100:.1f}%)    "
                  f"{dvae_improvement:.4f} ({dvae_improvement*100:.1f}%)")
    
    # Analyze trends between different methods
    if use_conditional and 'cdvae_accuracies' in results:
        # Compare DVAE vs VAE
        dvae_vs_vae_diff = np.array(results['dvae_accuracies']) - np.array(results['vae_accuracies'])
        # Compare CDVAE vs VAE
        cdvae_vs_vae_diff = np.array(results['cdvae_accuracies']) - np.array(results['vae_accuracies'])
        # Compare CDVAE vs DVAE
        cdvae_vs_dvae_diff = np.array(results['cdvae_accuracies']) - np.array(results['dvae_accuracies'])
        
        print("\nAnalysis of differences between methods:")
        print("At what noise levels is CDVAE better than DVAE?")
        for i, factor in enumerate(noise_factors):
            if cdvae_vs_dvae_diff[i] > 0:
                print(f"  Noise factor {factor:.1f}: CDVAE better by {cdvae_vs_dvae_diff[i]:.4f}")
        
        print("\nAt what noise levels is CDVAE better than VAE?")
        for i, factor in enumerate(noise_factors):
            if cdvae_vs_vae_diff[i] > 0:
                print(f"  Noise factor {factor:.1f}: CDVAE better by {cdvae_vs_vae_diff[i]:.4f}")
    else:
        # Regular DVAE vs VAE analysis
        vae_vs_dvae_diff = np.array(results['dvae_accuracies']) - np.array(results['vae_accuracies'])
        crossover_points = []
        
        for i in range(1, len(vae_vs_dvae_diff)):
            if (vae_vs_dvae_diff[i-1] < 0 and vae_vs_dvae_diff[i] >= 0) or (vae_vs_dvae_diff[i-1] <= 0 and vae_vs_dvae_diff[i] > 0):
                # Linear interpolation to find the exact crossover point
                x1, x2 = noise_factors[i-1], noise_factors[i]
                y1, y2 = vae_vs_dvae_diff[i-1], vae_vs_dvae_diff[i]
                if y1 != y2:  # Avoid division by zero
                    crossover = x1 + (x2 - x1) * (-y1) / (y2 - y1)
                    crossover_points.append(crossover)
        
        if crossover_points:
            print(f"\nDVAE starts outperforming VAE at noise factor approximately: {crossover_points[0]:.2f}")
        else:
            if np.all(vae_vs_dvae_diff < 0):
                print("\nVAE outperforms DVAE across all tested noise factors")
            elif np.all(vae_vs_dvae_diff > 0):
                print("\nDVAE outperforms VAE across all tested noise factors")
            else:
                print("\nNo clear pattern of when DVAE outperforms VAE")


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze accuracy vs noise level")
    parser.add_argument("--noise-factors", type=float, nargs='+', 
                        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help="Noise factors to test")
    parser.add_argument("--samples-per-class", type=int, default=100,
                        help="Fixed number of samples per class to use")
    parser.add_argument("--noise-type", type=str, default="gaussian",
                        choices=["gaussian", "salt_and_pepper"],
                        help="Type of noise to apply")
    parser.add_argument("--dataset", type=str, default="mnist",
                        choices=["mnist", "cifar10"],
                        help="Dataset to use")
    parser.add_argument("--save-dir", type=str, default="results/noise_level_analysis",
                        help="Directory to save results")
    parser.add_argument("--device", type=str, default="cuda",
                        help="Device to use for training")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--vae-epochs", type=int, default=20,
                        help="Number of epochs to train VAE/DVAE")
    parser.add_argument("--clf-epochs", type=int, default=5,
                        help="Number of epochs to train classifiers")
    parser.add_argument("--use-conditional", action="store_true",
                        help="Include conditional DVAE in the analysis")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    analyze_noise_levels(
        noise_factors=args.noise_factors,
        samples_per_class=args.samples_per_class,
        noise_type=args.noise_type,
        dataset_name=args.dataset,
        save_dir=args.save_dir,
        device=args.device,
        num_workers=args.num_workers,
        vae_epochs=args.vae_epochs,
        clf_epochs=args.clf_epochs,
        use_conditional=True
    ) 