import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

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
    
    noise_factors_gaussian = []
    noise_factors_sp = []
    
    for result in results:
        model_type = result['model_type']
        noise_type = result['noise_type']
        noise_factor = result['noise_factor']
        
        if noise_type == 'gaussian':
            if model_type == 'vae':
                vae_gaussian.append(result)
                if noise_factor not in noise_factors_gaussian:
                    noise_factors_gaussian.append(noise_factor)
            elif model_type == 'dvae':
                dvae_gaussian.append(result)
        elif noise_type == 'salt_and_pepper':
            if model_type == 'vae':
                vae_sp.append(result)
                if noise_factor not in noise_factors_sp:
                    noise_factors_sp.append(noise_factor)
            elif model_type == 'dvae':
                dvae_sp.append(result)
    
    # Sort by noise factor
    vae_gaussian.sort(key=lambda x: x['noise_factor'])
    dvae_gaussian.sort(key=lambda x: x['noise_factor'])
    vae_sp.sort(key=lambda x: x['noise_factor'])
    dvae_sp.sort(key=lambda x: x['noise_factor'])
    noise_factors_gaussian.sort()
    noise_factors_sp.sort()
    
    # Create a figure with rich styling
    plt.figure(figsize=(14, 10))
    
    # Set a clean and modern style
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Gaussian noise subplot with shaded difference area
    plt.subplot(2, 1, 1)
    
    # Extract data points
    x_gaussian = [r['noise_factor'] for r in vae_gaussian]
    y_noisy_gaussian = [r['noisy_psnr'] for r in vae_gaussian]
    y_vae_gaussian = [r['recon_psnr'] for r in vae_gaussian]
    y_dvae_gaussian = [r['recon_psnr'] for r in dvae_gaussian]
    
    # Plot base lines
    plt.plot(x_gaussian, y_noisy_gaussian, 'o--', color='#808080', linewidth=1.5, alpha=0.8, label='Noisy Input')
    plt.plot(x_gaussian, y_vae_gaussian, 'o-', color='#1f77b4', linewidth=2.5, label='VAE Reconstruction')
    plt.plot(x_gaussian, y_dvae_gaussian, 's-', color='#ff7f0e', linewidth=2.5, label='DVAE Reconstruction')
    
    # Fill area between DVAE and VAE to highlight the difference
    plt.fill_between(x_gaussian, y_vae_gaussian, y_dvae_gaussian, color='#ff7f0e', alpha=0.15, 
                    where=[dvae > vae for dvae, vae in zip(y_dvae_gaussian, y_vae_gaussian)],
                    interpolate=True, label='DVAE Advantage')
    
    # Add horizontal lines at important thresholds
    plt.axhline(y=20, color='green', linestyle=':', alpha=0.7, label='Good Quality (20dB)')
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='Acceptable Quality (10dB)')
    
    # Add data point annotations for DVAE
    for i, (x, y) in enumerate(zip(x_gaussian, y_dvae_gaussian)):
        plt.annotate(f"{y:.1f}dB", xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color='#ff7f0e', fontweight='bold')
    
    # Add improvement indicators as arrows between VAE and DVAE
    for i, (x, y_vae, y_dvae) in enumerate(zip(x_gaussian, y_vae_gaussian, y_dvae_gaussian)):
        if y_dvae > y_vae + 2:  # Only show arrows for significant improvements
            mid_y = (y_vae + y_dvae) / 2
            plt.annotate("", xy=(x, y_dvae), xytext=(x, y_vae), 
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='green', alpha=0.7),
                        annotation_clip=False)
            plt.annotate(f"+{y_dvae - y_vae:.1f}dB", xy=(x, mid_y), xytext=(5, 0), 
                        textcoords='offset points', ha='left', va='center',
                        fontsize=8, color='green', fontweight='bold')
    
    # Configure the plot appearance
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Gaussian Noise', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    # Set y-axis limits with some padding
    plt.ylim(min(min(y_noisy_gaussian), min(y_vae_gaussian), min(y_dvae_gaussian)) - 1, 
             max(max(y_noisy_gaussian), max(y_vae_gaussian), max(y_dvae_gaussian)) + 3)
    
    # Add reference text explaining PSNR
    plt.text(0.02, 0.02, "Higher PSNR = Better Image Quality", transform=plt.gca().transAxes, 
            fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.7))
    
    # Salt and pepper noise subplot with shaded difference area
    plt.subplot(2, 1, 2)
    
    # Extract data points
    x_sp = [r['noise_factor'] for r in vae_sp]
    y_noisy_sp = [r['noisy_psnr'] for r in vae_sp]
    y_vae_sp = [r['recon_psnr'] for r in vae_sp]
    y_dvae_sp = [r['recon_psnr'] for r in dvae_sp]
    
    # Plot base lines
    plt.plot(x_sp, y_noisy_sp, 'o--', color='#808080', linewidth=1.5, alpha=0.8, label='Noisy Input')
    plt.plot(x_sp, y_vae_sp, 'o-', color='#1f77b4', linewidth=2.5, label='VAE Reconstruction')
    plt.plot(x_sp, y_dvae_sp, 's-', color='#ff7f0e', linewidth=2.5, label='DVAE Reconstruction')
    
    # Fill area between DVAE and VAE to highlight the difference
    plt.fill_between(x_sp, y_vae_sp, y_dvae_sp, color='#ff7f0e', alpha=0.15, 
                    where=[dvae > vae for dvae, vae in zip(y_dvae_sp, y_vae_sp)],
                    interpolate=True, label='DVAE Advantage')
    
    # Add horizontal lines at important thresholds
    plt.axhline(y=20, color='green', linestyle=':', alpha=0.7, label='Good Quality (20dB)')
    plt.axhline(y=10, color='orange', linestyle=':', alpha=0.7, label='Acceptable Quality (10dB)')
    
    # Add data point annotations for DVAE
    for i, (x, y) in enumerate(zip(x_sp, y_dvae_sp)):
        plt.annotate(f"{y:.1f}dB", xy=(x, y), xytext=(0, 10), 
                    textcoords='offset points', ha='center', va='bottom',
                    fontsize=9, color='#ff7f0e', fontweight='bold')
    
    # Add improvement indicators as arrows between VAE and DVAE
    for i, (x, y_vae, y_dvae) in enumerate(zip(x_sp, y_vae_sp, y_dvae_sp)):
        if y_dvae > y_vae + 2:  # Only show arrows for significant improvements
            mid_y = (y_vae + y_dvae) / 2
            plt.annotate("", xy=(x, y_dvae), xytext=(x, y_vae), 
                        arrowprops=dict(arrowstyle="->", lw=1.5, color='green', alpha=0.7),
                        annotation_clip=False)
            plt.annotate(f"+{y_dvae - y_vae:.1f}dB", xy=(x, mid_y), xytext=(5, 0), 
                        textcoords='offset points', ha='left', va='center',
                        fontsize=8, color='green', fontweight='bold')
    
    # Configure the plot appearance
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Salt & Pepper Noise', fontsize=14, fontweight='bold')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10, loc='upper right')
    
    # Set y-axis limits with some padding
    plt.ylim(min(min(y_noisy_sp), min(y_vae_sp), min(y_dvae_sp)) - 1, 
             max(max(y_noisy_sp), max(y_vae_sp), max(y_dvae_sp)) + 3)
    
    # Add reference text explaining PSNR
    plt.text(0.02, 0.02, "Higher PSNR = Better Image Quality", transform=plt.gca().transAxes, 
            fontsize=9, style='italic', bbox=dict(facecolor='white', alpha=0.7))
    
    # Add a shared title for the figure
    plt.suptitle('Image Reconstruction Quality (PSNR) by Noise Level and Model', 
                fontsize=16, fontweight='bold', y=0.98)
    
    # Add an explanatory footnote
    footnote = "PSNR measures reconstruction quality in decibels (dB). Higher values indicate better quality.\n"
    footnote += "Noisy Input = original corrupted image, VAE/DVAE = reconstructed images after denoising."
    plt.figtext(0.5, 0.01, footnote, ha='center', fontsize=10, 
               bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save the enhanced plot
    enhanced_plot_path = os.path.join(base_dir, "enhanced_psnr_comparison.png")
    plt.savefig(enhanced_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Enhanced PSNR comparison plot created at: {enhanced_plot_path}")

if __name__ == "__main__":
    main() 