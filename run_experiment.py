#!/usr/bin/env python
"""
Run a complete DVAE experiment with training, evaluation, and comparison.

This script will:
1. Train a standard VAE on MNIST
2. Train a DVAE on noisy MNIST
3. Run classification comparison with VAE and DVAE augmented data
"""

import os
import argparse
import subprocess
import datetime


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run a complete DVAE experiment")
    
    # Dataset arguments
    parser.add_argument("--data-dir", type=str, default="./data", help="Directory to store datasets")
    parser.add_argument("--subsample", type=int, default=1000, 
                       help="Number of training samples to use for VAE/DVAE training")
    parser.add_argument("--num-classes", type=int, default=10, help="Number of classes in the dataset")
    
    # Noise arguments
    parser.add_argument("--noise-type", type=str, default="gaussian",
                       choices=["gaussian", "salt_and_pepper", "block", "line_h", "line_v"],
                       help="Type of noise to add")
    parser.add_argument("--noise-factor", type=float, default=0.2, 
                       help="Noise factor for DVAE training")
    
    # Model arguments
    parser.add_argument("--latent-dim", type=int, default=16, help="Dimension of the latent space")
    parser.add_argument("--hidden-dims", type=str, default="32,64,128,256", 
                       help="Dimensions of hidden layers (comma-separated)")
    
    # Training arguments
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train for")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer")
    
    # Classification arguments
    parser.add_argument("--class-subsample", type=int, default=100, 
                       help="Number of samples per class for classifier training")
    parser.add_argument("--aug-samples", type=int, default=100,
                       help="Number of synthetic samples per class for augmentation")
    parser.add_argument("--class-epochs", type=int, default=10,
                       help="Number of epochs to train classifiers")
    
    # Device arguments
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on (cuda or cpu)")
    
    # Run options
    parser.add_argument("--skip-vae", action="store_true", help="Skip VAE training")
    parser.add_argument("--skip-dvae", action="store_true", help="Skip DVAE training")
    parser.add_argument("--skip-classification", action="store_true", help="Skip classification comparison")
    
    # Output arguments
    parser.add_argument("--save-dir", type=str, default="results",
                       help="Directory to save results")
    
    return parser.parse_args()


def run_command(command, description=None):
    """Run a command and print output."""
    if description:
        print(f"\n{'=' * 80}\n{description}\n{'=' * 80}\n")
    
    print(f"Running command: {' '.join(command)}")
    result = subprocess.run(command, text=True)
    
    if result.returncode != 0:
        print(f"Command failed with exit code {result.returncode}")
        return False
    
    return True


def main():
    """Main function to run the experiment."""
    args = parse_args()
    
    # Create timestamp for experiment
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"experiment_{timestamp}"
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    vae_model_path = None
    dvae_model_path = None
    
    # 1. Train standard VAE on MNIST
    if not args.skip_vae:
        vae_experiment_name = f"vae_{timestamp}"
        vae_model_path = os.path.join(args.save_dir, vae_experiment_name, "models", "best.pt")
        
        command = [
            "python", "src/experiments/train_vae_mnist.py",
            "--data-dir", args.data_dir,
            "--batch-size", str(args.batch_size),
            "--subsample", str(args.subsample),
            "--img-size", "28",
            "--latent-dim", str(args.latent_dim),
            "--hidden-dims", args.hidden_dims,
            "--kl-weight", "1.0",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--early-stopping",
            "--patience", "5",
            "--save-interval", "5",
            "--save-dir", args.save_dir,
            "--experiment-name", vae_experiment_name,
            "--device", args.device
        ]
        
        success = run_command(command, "Training standard VAE on MNIST")
        if not success:
            print("VAE training failed, stopping experiment")
            return
    
    # 2. Train DVAE on noisy MNIST
    if not args.skip_dvae:
        dvae_experiment_name = f"dvae_{args.noise_type}_{args.noise_factor:.1f}_{timestamp}"
        dvae_model_path = os.path.join(args.save_dir, dvae_experiment_name, "models", "best.pt")
        
        command = [
            "python", "src/experiments/train_dvae_mnist.py",
            "--data-dir", args.data_dir,
            "--batch-size", str(args.batch_size),
            "--subsample", str(args.subsample),
            "--noise-type", args.noise_type,
            "--noise-factor", str(args.noise_factor),
            "--img-size", "28",
            "--latent-dim", str(args.latent_dim),
            "--hidden-dims", args.hidden_dims,
            "--kl-weight", "1.0",
            "--epochs", str(args.epochs),
            "--lr", str(args.lr),
            "--early-stopping",
            "--patience", "5",
            "--save-interval", "5",
            "--save-dir", args.save_dir,
            "--experiment-name", dvae_experiment_name,
            "--device", args.device
        ]
        
        success = run_command(command, f"Training DVAE on MNIST with {args.noise_type} noise (factor: {args.noise_factor})")
        if not success:
            print("DVAE training failed, stopping experiment")
            return
    
    # 3. Run classification comparison
    if not args.skip_classification:
        # We need both VAE and DVAE models for classification comparison
        if vae_model_path is None or dvae_model_path is None:
            print("Cannot run classification comparison without both VAE and DVAE models")
            print("Please provide model paths or train both models")
            return
        
        classification_dir = os.path.join(args.save_dir, f"classification_{timestamp}")
        
        command = [
            "python", "src/experiments/classification_experiment.py",
            "--data-dir", args.data_dir,
            "--num-classes", str(args.num_classes),
            "--subsample", str(args.class_subsample),
            "--vae-model", vae_model_path,
            "--dvae-model", dvae_model_path,
            "--aug-samples", str(args.aug_samples),
            "--epochs", str(args.class_epochs),
            "--lr", str(args.lr),
            "--batch-size", str(args.batch_size),
            "--device", args.device,
            "--save-dir", classification_dir
        ]
        
        success = run_command(command, "Running classification comparison")
        if not success:
            print("Classification comparison failed")
            return
    
    print("\nExperiment completed successfully!")
    print(f"Results saved in {args.save_dir}")


if __name__ == "__main__":
    main() 