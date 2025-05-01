import os
import json
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

def loess_smooth(x, y, frac=0.5):
    """Apply LOESS smoothing (locally weighted regression)"""
    x_array = np.array(x)
    y_array = np.array(y)
    
    # Sort the data points by x values
    sort_indices = np.argsort(x_array)
    x_array = x_array[sort_indices]
    y_array = y_array[sort_indices]
    
    # Apply LOESS smoothing if we have enough points
    if len(x_array) >= 4:  # Need reasonable number of points for LOESS
        # Apply smoothing directly on linear scale
        smoothed = lowess(y_array, x_array, frac=frac, it=2, return_sorted=True)
        x_smooth = smoothed[:, 0]
        y_smooth = smoothed[:, 1]
    else:
        # Not enough points for LOESS, just return the original points
        x_smooth = x_array
        y_smooth = y_array
        
    return x_smooth, y_smooth

def main():
    # Define the directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Collect dataset sizes from directory names
    dataset_sizes = []
    for dirname in os.listdir(base_dir):
        if dirname.startswith("size_"):
            try:
                size = int(dirname.split("_")[1])
                dataset_sizes.append(size)
            except:
                continue
    
    dataset_sizes.sort()
    print(f"Found dataset sizes: {dataset_sizes}")
    
    # Dataset types to track
    dataset_types = [
        '1_noisy',
        '2_vae_recon',
        '3_dvae_recon',
        '4_cdvae_recon',
        '5_vae_recon_synthetic',
        '6_dvae_recon_synthetic',
        '7_cdvae_recon_synthetic'
    ]
    
    # Extract accuracies for each dataset type
    accuracies = {dtype: [] for dtype in dataset_types}
    size_data = {dtype: [] for dtype in dataset_types}
    
    for size in dataset_sizes:
        results_path = os.path.join(base_dir, f"size_{size}", "results.json")
        if os.path.exists(results_path):
            with open(results_path, 'r') as f:
                results = json.load(f)
                
            for dataset_type in dataset_types:
                if dataset_type in results:
                    accuracies[dataset_type].append(results[dataset_type]['test_accuracy'])
                    size_data[dataset_type].append(size)
                else:
                    print(f"Warning: Dataset type {dataset_type} not found in {results_path}")
    
    # Plot the results with smoothing
    plt.figure(figsize=(12, 8))
    
    # Use a more visually distinct color palette
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2']
    
    # Nicer labels for the legend
    nice_labels = {
        '1_noisy': 'Noisy Data',
        '2_vae_recon': 'VAE Reconstruction',
        '3_dvae_recon': 'DVAE Reconstruction',
        '4_cdvae_recon': 'CDVAE Reconstruction',
        '5_vae_recon_synthetic': 'VAE Recon + Synthetic',
        '6_dvae_recon_synthetic': 'DVAE Recon + Synthetic',
        '7_cdvae_recon_synthetic': 'CDVAE Recon + Synthetic'
    }
    
    # Set a better style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Configure grid lines (using linear scale)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    
    # Plot smoothed lines
    for i, (name, acc_list) in enumerate(accuracies.items()):
        if len(acc_list) > 0:
            # Create smooth fit using LOESS with increased smoothing
            x_smooth, y_smooth = loess_smooth(size_data[name], acc_list)
            
            # Plot the smoothed line
            plt.plot(x_smooth, y_smooth, 
                    color=colors[i], 
                    linewidth=3,  # Make lines slightly thicker
                    label=nice_labels.get(name, name))
    
    # Adding more descriptive labels and title
    plt.xlabel('Samples per Class', fontsize=15)
    plt.ylabel('Test Accuracy', fontsize=15)
    plt.title('Effect of Augmentation Strategies on Downstream Classification Accuracy (CIFAR-10)', fontsize=20)
    
    # Place legend inside the plot with larger font size
    plt.legend(loc='upper left', fontsize=11, framealpha=0.9)
    
    # Set reasonable x-axis limits with nice round numbers
    plt.xlim(0, max(dataset_sizes) + 10)
    
    # Set x-axis ticks at each dataset size point
    plt.xticks(dataset_sizes)
    
    plt.tight_layout()
    
    # Save the plot
    smoothed_plot_path = os.path.join(base_dir, 'augmentation_comparison_smoothed.png')
    plt.savefig(smoothed_plot_path, dpi=300)
    plt.savefig(os.path.join(base_dir, 'augmentation_comparison_smoothed.pdf'))
    
    print(f"Smoothed plot saved to: {smoothed_plot_path}")

if __name__ == "__main__":
    main() 