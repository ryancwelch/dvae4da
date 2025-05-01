import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the results from the JSON file
    results_path = os.path.join(base_dir, "noise_level_results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Extract the data
    noise_factors = results['noise_factors']
    noisy_accuracies = results['noisy_accuracies']
    vae_accuracies = results['vae_accuracies']
    dvae_accuracies = results['dvae_accuracies']
    cdvae_accuracies = results.get('cdvae_accuracies', [])  # Use get in case cdvae is not present
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    
    # Plot each line with markers and lines
    plt.plot(noise_factors, noisy_accuracies, 'o-', label='Noisy Data', linewidth=2)
    plt.plot(noise_factors, vae_accuracies, 's-', label='VAE Reconstructed', linewidth=2)
    plt.plot(noise_factors, dvae_accuracies, '^-', label='DVAE Reconstructed', linewidth=2)
    
    if cdvae_accuracies:
        plt.plot(noise_factors, cdvae_accuracies, 'd-', label='Conditional DVAE', linewidth=2)
    
    # Configure the plot appearance
    plt.grid(True, alpha=0.3)
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('Test Accuracy', fontsize=12)
    
    # Set the updated title
    plt.title('Effect of Noise Level on Downstream Classification Accuracy (MNIST)', fontsize=14)
    
    plt.legend(fontsize=11)
    plt.tight_layout()
    
    # Save the plot
    updated_plot_path = os.path.join(base_dir, "noise_level_analysis_updated.png")
    plt.savefig(updated_plot_path, dpi=300)
    plt.savefig(os.path.join(base_dir, "noise_level_analysis_updated.pdf"))
    
    print(f"Updated plot saved to {updated_plot_path}")

if __name__ == "__main__":
    main() 