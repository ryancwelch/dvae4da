import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    # Define the directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Load the results from the JSON file
    results_path = os.path.join(base_dir, "results.json")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    # Set a nicer style
    plt.style.use('seaborn-v0_8-whitegrid')
    
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
    
    # 1. PLOT 1: PSNR Improvement vs Noise Factor (Line plot for each noise type)
    plt.figure(figsize=(12, 8))
    
    # Gaussian noise
    plt.subplot(1, 2, 1)
    plt.plot([r['noise_factor'] for r in vae_gaussian], 
             [r['psnr_improvement'] for r in vae_gaussian], 
             'o-', color='#1f77b4', linewidth=2, label='VAE')
    plt.plot([r['noise_factor'] for r in dvae_gaussian], 
             [r['psnr_improvement'] for r in dvae_gaussian], 
             's-', color='#ff7f0e', linewidth=2, label='DVAE')
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR Improvement (dB)', fontsize=12)
    plt.title('Gaussian Noise', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    # Salt and pepper noise
    plt.subplot(1, 2, 2)
    plt.plot([r['noise_factor'] for r in vae_sp], 
             [r['psnr_improvement'] for r in vae_sp], 
             'o-', color='#1f77b4', linewidth=2, label='VAE')
    plt.plot([r['noise_factor'] for r in dvae_sp], 
             [r['psnr_improvement'] for r in dvae_sp], 
             's-', color='#ff7f0e', linewidth=2, label='DVAE')
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR Improvement (dB)', fontsize=12)
    plt.title('Salt & Pepper Noise', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.suptitle('Denoising Performance Across Noise Levels', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    line_plot_path = os.path.join(base_dir, "noise_performance_by_type.png")
    plt.savefig(line_plot_path, dpi=300)
    plt.close()
    
    # 2. PLOT 2: Direct PSNR Values (Absolute reconstruction quality)
    plt.figure(figsize=(12, 8))
    
    # Gaussian noise
    plt.subplot(1, 2, 1)
    plt.plot([r['noise_factor'] for r in vae_gaussian], 
             [r['noisy_psnr'] for r in vae_gaussian], 
             'o--', color='gray', linewidth=1.5, alpha=0.7, label='Noisy Input')
    plt.plot([r['noise_factor'] for r in vae_gaussian], 
             [r['recon_psnr'] for r in vae_gaussian], 
             'o-', color='#1f77b4', linewidth=2, label='VAE Reconstruction')
    plt.plot([r['noise_factor'] for r in dvae_gaussian], 
             [r['recon_psnr'] for r in dvae_gaussian], 
             's-', color='#ff7f0e', linewidth=2, label='DVAE Reconstruction')
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Gaussian Noise', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    # Salt and pepper noise
    plt.subplot(1, 2, 2)
    plt.plot([r['noise_factor'] for r in vae_sp], 
             [r['noisy_psnr'] for r in vae_sp], 
             'o--', color='gray', linewidth=1.5, alpha=0.7, label='Noisy Input')
    plt.plot([r['noise_factor'] for r in vae_sp], 
             [r['recon_psnr'] for r in vae_sp], 
             'o-', color='#1f77b4', linewidth=2, label='VAE Reconstruction')
    plt.plot([r['noise_factor'] for r in dvae_sp], 
             [r['recon_psnr'] for r in dvae_sp], 
             's-', color='#ff7f0e', linewidth=2, label='DVAE Reconstruction')
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Salt & Pepper Noise', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    
    plt.suptitle('Image Quality (PSNR) Across Noise Levels', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    psnr_plot_path = os.path.join(base_dir, "absolute_psnr_values.png")
    plt.savefig(psnr_plot_path, dpi=300)
    plt.close()
    
    # 3. PLOT 3: Bar chart comparison at different noise levels
    width = 0.35
    
    # For Gaussian noise
    plt.figure(figsize=(15, 6))
    
    x = np.arange(len(noise_factors_gaussian))
    
    # Extract PSNR improvements
    vae_improvements = [r['psnr_improvement'] for r in vae_gaussian]
    dvae_improvements = [r['psnr_improvement'] for r in dvae_gaussian]
    
    plt.bar(x - width/2, vae_improvements, width, label='VAE', color='#1f77b4')
    plt.bar(x + width/2, dvae_improvements, width, label='DVAE', color='#ff7f0e')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR Improvement (dB)', fontsize=12)
    plt.title('Gaussian Noise: VAE vs DVAE Denoising Performance', fontsize=14)
    plt.xticks(x, [f'{nf:.1f}' for nf in noise_factors_gaussian])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(vae_improvements):
        plt.text(i - width/2, v + (0.5 if v > 0 else -0.5), f'{v:.1f}', 
                ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    for i, v in enumerate(dvae_improvements):
        plt.text(i + width/2, v + (0.5 if v > 0 else -0.5), f'{v:.1f}', 
                ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    bar_gaussian_path = os.path.join(base_dir, "gaussian_noise_comparison.png")
    plt.savefig(bar_gaussian_path, dpi=300)
    plt.close()
    
    # For Salt & Pepper noise
    plt.figure(figsize=(15, 6))
    
    x = np.arange(len(noise_factors_sp))
    
    # Extract PSNR improvements
    vae_improvements = [r['psnr_improvement'] for r in vae_sp]
    dvae_improvements = [r['psnr_improvement'] for r in dvae_sp]
    
    plt.bar(x - width/2, vae_improvements, width, label='VAE', color='#1f77b4')
    plt.bar(x + width/2, dvae_improvements, width, label='DVAE', color='#ff7f0e')
    
    # Add a horizontal line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3, linewidth=1)
    
    plt.xlabel('Noise Factor', fontsize=12)
    plt.ylabel('PSNR Improvement (dB)', fontsize=12)
    plt.title('Salt & Pepper Noise: VAE vs DVAE Denoising Performance', fontsize=14)
    plt.xticks(x, [f'{nf:.1f}' for nf in noise_factors_sp])
    plt.legend(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add values on top of bars
    for i, v in enumerate(vae_improvements):
        plt.text(i - width/2, v + (0.5 if v > 0 else -0.5), f'{v:.1f}', 
                ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    for i, v in enumerate(dvae_improvements):
        plt.text(i + width/2, v + (0.5 if v > 0 else -0.5), f'{v:.1f}', 
                ha='center', va='bottom' if v > 0 else 'top', fontsize=9)
    
    plt.tight_layout()
    
    # Save the plot
    bar_sp_path = os.path.join(base_dir, "salt_pepper_noise_comparison.png")
    plt.savefig(bar_sp_path, dpi=300)
    plt.close()
    
    # 4. PLOT 4: Heat map comparing performance across noise types and levels
    plt.figure(figsize=(12, 8))
    
    # Prepare data for heatmap
    noise_factors = sorted(list(set(noise_factors_gaussian + noise_factors_sp)))
    noise_types = ['Gaussian', 'Salt & Pepper']
    
    # Create matrix for VAE improvements
    vae_matrix = np.zeros((len(noise_types), len(noise_factors)))
    # Create matrix for DVAE improvements
    dvae_matrix = np.zeros((len(noise_types), len(noise_factors)))
    
    # Fill the matrices
    for i, result in enumerate(vae_gaussian):
        nf_index = noise_factors.index(result['noise_factor'])
        vae_matrix[0, nf_index] = result['psnr_improvement']
    
    for i, result in enumerate(dvae_gaussian):
        nf_index = noise_factors.index(result['noise_factor'])
        dvae_matrix[0, nf_index] = result['psnr_improvement']
    
    for i, result in enumerate(vae_sp):
        nf_index = noise_factors.index(result['noise_factor'])
        vae_matrix[1, nf_index] = result['psnr_improvement']
    
    for i, result in enumerate(dvae_sp):
        nf_index = noise_factors.index(result['noise_factor'])
        dvae_matrix[1, nf_index] = result['psnr_improvement']
    
    # Set up the figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # VAE heatmap
    sns.heatmap(vae_matrix, annot=True, cmap="RdBu_r", center=0, 
                xticklabels=[f'{nf:.1f}' for nf in noise_factors],
                yticklabels=noise_types, ax=ax1, fmt='.1f', cbar_kws={'label': 'PSNR Improvement (dB)'})
    ax1.set_title('VAE Denoising Performance', fontsize=14)
    ax1.set_xlabel('Noise Factor', fontsize=12)
    ax1.set_ylabel('Noise Type', fontsize=12)
    
    # DVAE heatmap
    sns.heatmap(dvae_matrix, annot=True, cmap="RdBu_r", center=0, 
                xticklabels=[f'{nf:.1f}' for nf in noise_factors],
                yticklabels=noise_types, ax=ax2, fmt='.1f', cbar_kws={'label': 'PSNR Improvement (dB)'})
    ax2.set_title('DVAE Denoising Performance', fontsize=14)
    ax2.set_xlabel('Noise Factor', fontsize=12)
    ax2.set_ylabel('Noise Type', fontsize=12)
    
    plt.suptitle('Denoising Performance (PSNR Improvement) Comparison', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the plot
    heatmap_path = os.path.join(base_dir, "performance_heatmap.png")
    plt.savefig(heatmap_path, dpi=300)
    plt.close()
    
    print(f"Improved plots created:")
    print(f"1. Line plots: {line_plot_path}")
    print(f"2. PSNR values: {psnr_plot_path}")
    print(f"3. Bar charts: {bar_gaussian_path} and {bar_sp_path}")
    print(f"4. Heatmap: {heatmap_path}")

if __name__ == "__main__":
    main() 