import os
import json
import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the results from the JSON file
    results_path = os.path.join(base_dir, "results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Organize data by model type and noise type
    vae_gaussian = []
    dvae_gaussian = []
    vae_sp = []  # salt and pepper
    dvae_sp = []  # salt and pepper
    
    for result in results:
        model_type = result['model_type']
        noise_type = result['noise_type']
        
        if noise_type == 'gaussian':
            if model_type == 'vae':
                vae_gaussian.append(result)
            elif model_type == 'dvae':
                dvae_gaussian.append(result)
        elif noise_type == 'salt_and_pepper':
            if model_type == 'vae':
                vae_sp.append(result)
            elif model_type == 'dvae':
                dvae_sp.append(result)
    
    # Sort by noise factor
    vae_gaussian.sort(key=lambda x: x['noise_factor'])
    dvae_gaussian.sort(key=lambda x: x['noise_factor'])
    vae_sp.sort(key=lambda x: x['noise_factor'])
    dvae_sp.sort(key=lambda x: x['noise_factor'])
    
    # Create a figure with clean styling
    plt.figure(figsize=(14, 10))
    
    # Set a clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gaussian noise subplot
    plt.subplot(2, 1, 1)
    
    # Extract data points
    x_gaussian = [r['noise_factor'] for r in vae_gaussian]
    y_noisy_gaussian = [r['noisy_psnr'] for r in vae_gaussian]
    y_vae_gaussian = [r['recon_psnr'] for r in vae_gaussian]
    y_dvae_gaussian = [r['recon_psnr'] for r in dvae_gaussian]
    
    # Plot just the basic lines
    plt.plot(x_gaussian, y_noisy_gaussian, 'o--', color='#808080', linewidth=1.5, label='Noisy Input')
    plt.plot(x_gaussian, y_vae_gaussian, 'o-', color='#1f77b4', linewidth=2.5, label='VAE Reconstruction')
    plt.plot(x_gaussian, y_dvae_gaussian, 's-', color='#ff7f0e', linewidth=2.5, label='DVAE Reconstruction')
    
    # Configure the plot appearance - keeping it minimal
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Gaussian Noise', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Salt and pepper noise subplot
    plt.subplot(2, 1, 2)
    
    # Extract data points
    x_sp = [r['noise_factor'] for r in vae_sp]
    y_noisy_sp = [r['noisy_psnr'] for r in vae_sp]
    y_vae_sp = [r['recon_psnr'] for r in vae_sp]
    y_dvae_sp = [r['recon_psnr'] for r in dvae_sp]
    
    # Plot just the basic lines
    plt.plot(x_sp, y_noisy_sp, 'o--', color='#808080', linewidth=1.5, label='Noisy Input')
    plt.plot(x_sp, y_vae_sp, 'o-', color='#1f77b4', linewidth=2.5, label='VAE Reconstruction')
    plt.plot(x_sp, y_dvae_sp, 's-', color='#ff7f0e', linewidth=2.5, label='DVAE Reconstruction')
    
    # Configure the plot appearance - keeping it minimal
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Salt & Pepper Noise', fontsize=15)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Simple title
    plt.suptitle('Image Reconstruction Quality (CIFAR-10)', fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the clean plot
    clean_plot_path = os.path.join(base_dir, "clean_psnr_comparison.png")
    plt.savefig(clean_plot_path, dpi=300)
    plt.close()
    
    print(f"Clean PSNR comparison plot created at: {clean_plot_path}")

if __name__ == "__main__":
    main() 