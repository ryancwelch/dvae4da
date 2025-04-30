import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os
import argparse
import sys
from typing import Dict, List, Tuple, Optional, Union
from tqdm import tqdm
import json
import itertools

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.data import get_mnist_dataset, get_cifar10_dataset, create_data_loaders, subsample_dataset
from src.models import get_vae_model, get_dvae_model
from src.utils.training import Trainer, compute_psnr
from src.utils.visualization import visualize_noise_examples, visualize_latent_space


def parse_args():
    parser = argparse.ArgumentParser(description="Perform ablation studies on DVAE model")
    
    # Dataset parameters
    parser.add_argument("--dataset", type=str, default="mnist", choices=["mnist", "cifar10"],
                       help="Dataset to use")
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    
    # Noise parameters
    parser.add_argument("--noise-type", type=str, default="gaussian",
                       help="Type of noise to add (gaussian, salt_and_pepper, etc.)")
    parser.add_argument("--noise-factors", type=str, default="0.1,0.3,0.5", 
                        help="Comma-separated list of noise factors to test")
    
    # Model parameters for ablation
    parser.add_argument("--latent-dims", type=str, default="8,16,32,64", 
                        help="Comma-separated list of latent dimensions to test")
    parser.add_argument("--kl-weights", type=str, default="0.01,0.1,1.0", 
                        help="Comma-separated list of KL weights to test")
    parser.add_argument("--hidden-dims-variants", type=str, 
                        default="32-64-128,64-128-256,32-64-128-256", 
                        help="Comma-separated list of hidden dimension configurations (dash-separated)")
    
    # Training parameters
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    
    # Experiment parameters
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device to use")
    parser.add_argument("--save-dir", type=str, default="results/ablation_study", 
                        help="Directory to save results")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--max-experiments", type=int, default=None,
                       help="Maximum number of experiments to run (default: run all combinations)")
    parser.add_argument("--eval-only", action="store_true", 
                       help="Only evaluate existing models, don't train new ones")
    
    return parser.parse_args()


def train_and_evaluate(model_type: str,
                      dataset_name: str,
                      train_loader: torch.utils.data.DataLoader,
                      val_loader: torch.utils.data.DataLoader,
                      test_loader: torch.utils.data.DataLoader,
                      noise_type: str,
                      noise_factor: float,
                      latent_dim: int,
                      kl_weight: float,
                      hidden_dims: List[int],
                      epochs: int,
                      lr: float,
                      device: torch.device,
                      save_dir: str,
                      experiment_name: str,
                      eval_only: bool = False) -> Dict:
    """
    Train and evaluate a model with specific parameters.
    
    Returns:
        Dictionary of evaluation metrics
    """
    # Determine dataset properties
    sample_batch = next(iter(train_loader))
    if len(sample_batch) == 3:  # Noisy dataset returns (noisy_img, clean_img, label)
        noisy_img, clean_img, _ = sample_batch
        img_channels = noisy_img.shape[1]
        img_size = noisy_img.shape[2]
    else:  # Standard dataset returns (img, label)
        img, _ = sample_batch
        img_channels = img.shape[1]
        img_size = img.shape[2]
    
    model_save_path = os.path.join(save_dir, f"{experiment_name}_model.pt")
    
    # Check if model exists already for eval_only mode
    if eval_only and not os.path.exists(model_save_path):
        print(f"Model {model_save_path} not found, skipping in eval_only mode")
        return None
    
    # Create model
    if model_type == "vae":
        model = get_vae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    else:  # dvae
        model = get_dvae_model(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    
    model.to(device)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        learning_rate=lr,
        device=device,
        save_dir=save_dir,
        experiment_name=experiment_name,
        noise_type=noise_type,
        noise_params={'noise_factor': noise_factor}
    )
    
    # Train model if not in eval_only mode
    if not eval_only:
        print(f"Training {model_type.upper()} with latent_dim={latent_dim}, "
              f"kl_weight={kl_weight}, hidden_dims={hidden_dims}")
        
        trainer.train(
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=epochs,
            early_stopping=True,
            patience=5,
            save_best_only=True
        )
    else:
        # Load existing model
        checkpoint = torch.load(model_save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
    
    # Evaluate on test set
    print(f"Evaluating {experiment_name}...")
    
    # Get a batch of test images for visualization
    batch = next(iter(test_loader))
    if len(batch) == 3:  # Noisy dataset returns (noisy_img, clean_img, label)
        noisy_imgs, clean_imgs, labels = batch
    else:  # Standard dataset returns (img, label)
        clean_imgs, labels = batch
        
        # Create noisy images for evaluation
        from src.utils.noise import add_noise
        noise_params = {'noise_factor': noise_factor, 'clip_min': 0.0, 'clip_max': 1.0}
        noisy_imgs = add_noise(clean_imgs, noise_type=noise_type, noise_params=noise_params)
    
    noisy_imgs = noisy_imgs.to(device)[:16]  # Use first 16 images
    clean_imgs = clean_imgs.to(device)[:16]
    labels = labels.to(device)[:16]
    
    # Generate reconstructions
    with torch.no_grad():
        if model_type == "vae":
            recon_imgs, mu, log_var = model(noisy_imgs)
        else:  # dvae
            recon_imgs, mu, log_var = model(noisy_imgs, clean_imgs)
    
    # Compute PSNR
    noisy_psnr = compute_psnr(clean_imgs, noisy_imgs)
    recon_psnr = compute_psnr(clean_imgs, recon_imgs)
    psnr_improvement = recon_psnr - noisy_psnr
    
    # Compute final loss on test set
    total_loss = 0
    total_recon_loss = 0
    total_kl_loss = 0
    num_batches = 0
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:  # Noisy dataset
                noisy_x, clean_x, _ = batch
                noisy_x = noisy_x.to(device)
                clean_x = clean_x.to(device)
            else:  # Standard dataset
                x, _ = batch
                clean_x = x.to(device)
                noisy_x = add_noise(clean_x, noise_type=noise_type, noise_params=noise_params)
            
            # Forward pass
            if model_type == "vae":
                recon_x, mu, log_var = model(noisy_x)
                loss_dict = model.loss_function(recon_x, clean_x, mu, log_var)
            else:  # dvae
                recon_x, mu, log_var = model(noisy_x, clean_x)
                loss_dict = model.loss_function(recon_x, clean_x, mu, log_var)
            
            total_loss += loss_dict['loss'].item()
            total_recon_loss += loss_dict['recon_loss'].item()
            total_kl_loss += loss_dict['kl_loss'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_recon_loss = total_recon_loss / num_batches
    avg_kl_loss = total_kl_loss / num_batches
    
    # Create visualization directory
    vis_dir = os.path.join(save_dir, experiment_name)
    os.makedirs(vis_dir, exist_ok=True)
    
    # Visualize noise examples and reconstructions
    fig = visualize_noise_examples(
        clean_imgs[:8],
        noisy_imgs[:8],
        recon_imgs[:8],
        num_examples=8,
        title=f"{model_type.upper()}: {noise_type} noise (factor: {noise_factor})"
    )
    
    fig.savefig(os.path.join(vis_dir, "reconstructions.png"))
    plt.close(fig)
    
    # Visualize latent space
    try:
        latent_codes, labels = trainer.encode_dataset(test_loader)
        
        fig = visualize_latent_space(
            latent_codes,
            labels,
            method='tsne',
            title=f"{model_type.upper()} Latent Space (dim={latent_dim})"
        )
        
        fig.savefig(os.path.join(vis_dir, "latent_space.png"))
        plt.close(fig)
    except Exception as e:
        print(f"Error visualizing latent space: {e}")
    
    # Return metrics
    return {
        'model_type': model_type,
        'dataset': dataset_name,
        'latent_dim': latent_dim,
        'kl_weight': kl_weight,
        'hidden_dims': hidden_dims,
        'noise_type': noise_type,
        'noise_factor': noise_factor,
        'noisy_psnr': noisy_psnr.item(),
        'recon_psnr': recon_psnr.item(),
        'psnr_improvement': psnr_improvement.item(),
        'test_loss': avg_loss,
        'test_recon_loss': avg_recon_loss,
        'test_kl_loss': avg_kl_loss
    }


def main():
    args = parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Parse parameters for ablation study
    noise_factors = [float(f) for f in args.noise_factors.split(",")]
    latent_dims = [int(dim) for dim in args.latent_dims.split(",")]
    kl_weights = [float(w) for w in args.kl_weights.split(",")]
    hidden_dims_variants = [
        [int(dim) for dim in variant.split("-")] 
        for variant in args.hidden_dims_variants.split(",")
    ]
    
    # Set device
    device = torch.device(args.device)
    
    # Choose dataset function
    if args.dataset == "mnist":
        dataset_func = get_mnist_dataset
    else:  # cifar10
        dataset_func = get_cifar10_dataset
    
    # All experiment parameter combinations
    parameter_grid = list(itertools.product(
        noise_factors,        # noise_factor
        latent_dims,          # latent_dim
        kl_weights,           # kl_weight
        hidden_dims_variants  # hidden_dims
    ))
    
    # Possibly limit number of experiments
    if args.max_experiments is not None and len(parameter_grid) > args.max_experiments:
        print(f"Limiting to {args.max_experiments} experiments out of {len(parameter_grid)} possible combinations")
        parameter_grid = parameter_grid[:args.max_experiments]
    
    # Load datasets for each noise factor
    noise_datasets = {}
    for noise_factor in noise_factors:
        noise_params = {
            'noise_factor': noise_factor,
            'clip_min': 0.0,
            'clip_max': 1.0
        }
        
        # Add specific noise params if needed
        if args.noise_type == 'salt_and_pepper':
            noise_params['salt_vs_pepper'] = 0.5
        elif args.noise_type in ['block', 'line_h', 'line_v']:
            noise_params['block_size'] = 4
        
        print(f"Loading {args.dataset} dataset with {args.noise_type} noise (factor: {noise_factor})...")
        
        train_dataset = dataset_func(
            root=args.data_dir,
            train=True,
            noise_type=args.noise_type,
            noise_params=noise_params,
            download=True
        )
        
        test_dataset = dataset_func(
            root=args.data_dir,
            train=False,
            noise_type=args.noise_type,
            noise_params=noise_params,
            download=True
        )
        
        train_loader, val_loader = create_data_loaders(
            train_dataset,
            batch_size=args.batch_size,
            val_split=0.1,
            shuffle=True,
            num_workers=args.num_workers
        )
        
        test_loader, _ = create_data_loaders(
            test_dataset,
            batch_size=args.batch_size,
            val_split=0.0,
            shuffle=False,
            num_workers=args.num_workers
        )
        
        noise_datasets[noise_factor] = (train_loader, val_loader, test_loader)
    
    # Store all results
    all_results = []
    
    # Perform ablation studies
    for experiment_idx, (noise_factor, latent_dim, kl_weight, hidden_dims) in enumerate(parameter_grid):
        print(f"\n{'='*80}")
        print(f"Experiment {experiment_idx+1}/{len(parameter_grid)}: "
              f"noise_factor={noise_factor}, latent_dim={latent_dim}, "
              f"kl_weight={kl_weight}, hidden_dims={hidden_dims}")
        print(f"{'='*80}")
        
        # Get dataset loaders for this noise factor
        train_loader, val_loader, test_loader = noise_datasets[noise_factor]
        
        # Create experiment name
        experiment_name = f"dvae_{args.dataset}_noise{noise_factor}_latent{latent_dim}_kl{kl_weight}_" + \
                          f"hidden{'_'.join(str(d) for d in hidden_dims)}"
        
        # Train and evaluate DVAE
        dvae_results = train_and_evaluate(
            model_type="dvae",
            dataset_name=args.dataset,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            noise_type=args.noise_type,
            noise_factor=noise_factor,
            latent_dim=latent_dim,
            kl_weight=kl_weight,
            hidden_dims=hidden_dims,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_dir=args.save_dir,
            experiment_name=experiment_name,
            eval_only=args.eval_only
        )
        
        if dvae_results is not None:
            all_results.append(dvae_results)
    
    # Save all results
    results_path = os.path.join(args.save_dir, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Create summary plots
    print("\nCreating summary plots...")
    
    # Group results for plotting
    results_by_noise = {}
    results_by_latent = {}
    results_by_kl = {}
    results_by_hidden = {}
    
    # Group by different parameters
    for result in all_results:
        noise_factor = result['noise_factor']
        latent_dim = result['latent_dim']
        kl_weight = result['kl_weight']
        hidden_str = "_".join(str(d) for d in result['hidden_dims'])
        
        # Group by noise factor
        if noise_factor not in results_by_noise:
            results_by_noise[noise_factor] = []
        results_by_noise[noise_factor].append(result)
        
        # Group by latent dimension
        if latent_dim not in results_by_latent:
            results_by_latent[latent_dim] = []
        results_by_latent[latent_dim].append(result)
        
        # Group by KL weight
        if kl_weight not in results_by_kl:
            results_by_kl[kl_weight] = []
        results_by_kl[kl_weight].append(result)
        
        # Group by hidden dims
        if hidden_str not in results_by_hidden:
            results_by_hidden[hidden_str] = []
        results_by_hidden[hidden_str].append(result)
    
    # 1. Plot PSNR improvement vs noise factor for different latent dimensions
    plt.figure(figsize=(10, 6))
    for latent_dim in sorted(results_by_latent.keys()):
        # Group these results by noise factor
        noise_values = []
        psnr_values = []
        
        for result in results_by_latent[latent_dim]:
            noise_values.append(result['noise_factor'])
            psnr_values.append(result['psnr_improvement'])
        
        # Sort by noise factor
        sorted_pairs = sorted(zip(noise_values, psnr_values))
        noise_values = [x for x, _ in sorted_pairs]
        psnr_values = [y for _, y in sorted_pairs]
        
        plt.plot(noise_values, psnr_values, 'o-', label=f'Latent dim = {latent_dim}')
    
    plt.xlabel('Noise Factor')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Denoising Performance vs Noise Level')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psnr_vs_noise.png"))
    plt.close()
    
    # 2. Plot PSNR improvement vs latent dimension for different noise factors
    plt.figure(figsize=(10, 6))
    for noise_factor in sorted(results_by_noise.keys()):
        # Group these results by latent dimension
        latent_values = []
        psnr_values = []
        
        for result in results_by_noise[noise_factor]:
            latent_values.append(result['latent_dim'])
            psnr_values.append(result['psnr_improvement'])
        
        # Sort by latent dimension
        sorted_pairs = sorted(zip(latent_values, psnr_values))
        latent_values = [x for x, _ in sorted_pairs]
        psnr_values = [y for _, y in sorted_pairs]
        
        plt.plot(latent_values, psnr_values, 'o-', label=f'Noise factor = {noise_factor}')
    
    plt.xlabel('Latent Dimension')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Denoising Performance vs Latent Dimension')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psnr_vs_latent.png"))
    plt.close()
    
    # 3. Plot PSNR improvement vs KL weight for different latent dimensions
    plt.figure(figsize=(10, 6))
    for latent_dim in sorted(results_by_latent.keys()):
        # Group these results by KL weight
        kl_values = []
        psnr_values = []
        
        for result in results_by_latent[latent_dim]:
            kl_values.append(result['kl_weight'])
            psnr_values.append(result['psnr_improvement'])
        
        # Sort by KL weight
        sorted_pairs = sorted(zip(kl_values, psnr_values))
        kl_values = [x for x, _ in sorted_pairs]
        psnr_values = [y for _, y in sorted_pairs]
        
        plt.plot(kl_values, psnr_values, 'o-', label=f'Latent dim = {latent_dim}')
    
    plt.xlabel('KL Weight')
    plt.ylabel('PSNR Improvement (dB)')
    plt.title('Denoising Performance vs KL Weight')
    plt.xscale('log')  # Log scale for KL weights
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.save_dir, "psnr_vs_kl_weight.png"))
    plt.close()
    
    # 4. Table of results sorted by PSNR improvement
    sorted_results = sorted(all_results, key=lambda x: x['psnr_improvement'], reverse=True)
    
    # Print top 5 configurations
    print("\nTop 5 configurations by PSNR improvement:")
    print(f"{'Rank':<5} {'Latent':<8} {'KL Weight':<10} {'Hidden Dims':<20} {'Noise':<8} {'PSNR Gain':<10}")
    print("-" * 70)
    
    for i, result in enumerate(sorted_results[:5]):
        hidden_str = "-".join(str(d) for d in result['hidden_dims'])
        print(f"{i+1:<5} {result['latent_dim']:<8} {result['kl_weight']:<10.3f} "
              f"{hidden_str:<20} {result['noise_factor']:<8.2f} {result['psnr_improvement']:<10.2f}")
    
    # Create HTML report with all experiment results
    html_report = os.path.join(args.save_dir, "ablation_report.html")
    with open(html_report, 'w') as f:
        f.write("<html><head><title>DVAE Ablation Study</title>")
        f.write("<style>table {border-collapse: collapse; width: 100%;} ")
        f.write("th, td {border: 1px solid #ddd; padding: 8px; text-align: left;} ")
        f.write("tr:nth-child(even) {background-color: #f2f2f2;} ")
        f.write("th {background-color: #4CAF50; color: white;}</style></head>")
        f.write("<body><h1>DVAE Ablation Study Results</h1>")
        
        # Summary plots
        f.write("<h2>Summary Plots</h2>")
        f.write('<div style="display: flex; flex-wrap: wrap;">')
        f.write(f'<div style="flex: 50%;"><img src="psnr_vs_noise.png" width="100%"></div>')
        f.write(f'<div style="flex: 50%;"><img src="psnr_vs_latent.png" width="100%"></div>')
        f.write(f'<div style="flex: 50%;"><img src="psnr_vs_kl_weight.png" width="100%"></div>')
        f.write('</div>')
        
        # Table of all results
        f.write("<h2>All Experiment Results (Sorted by PSNR Improvement)</h2>")
        f.write("<table><tr><th>Rank</th><th>Latent Dim</th><th>KL Weight</th><th>Hidden Dims</th>")
        f.write("<th>Noise Factor</th><th>PSNR Improvement</th><th>Test Loss</th><th>Visualizations</th></tr>")
        
        for i, result in enumerate(sorted_results):
            hidden_str = "-".join(str(d) for d in result['hidden_dims'])
            experiment_name = f"dvae_{args.dataset}_noise{result['noise_factor']}_latent{result['latent_dim']}_kl{result['kl_weight']}_" + \
                             f"hidden{'_'.join(str(d) for d in result['hidden_dims'])}"
            
            f.write(f"<tr><td>{i+1}</td><td>{result['latent_dim']}</td><td>{result['kl_weight']:.3f}</td>")
            f.write(f"<td>{hidden_str}</td><td>{result['noise_factor']:.2f}</td>")
            f.write(f"<td>{result['psnr_improvement']:.2f} dB</td><td>{result['test_loss']:.2f}</td>")
            f.write(f'<td><a href="{experiment_name}/reconstructions.png">Reconstructions</a> | ')
            f.write(f'<a href="{experiment_name}/latent_space.png">Latent Space</a></td></tr>')
        
        f.write("</table></body></html>")
    
    print(f"\nAblation study complete! Results saved to {args.save_dir}")
    print(f"View complete report at: {html_report}")


if __name__ == "__main__":
    main() 