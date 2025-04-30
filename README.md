# Denoising Variational Autoencoders for Robust Feature Learning and Data Augmentation

This project explores the use of Denoising Variational Autoencoders (DVAEs) for learning robust visual features and generating synthetic data to improve classification on limited, noisy datasets.

## Project Overview

The DVAE combines the denoising objective with VAE's variational learning to improve generative performance and feature extraction. The project aims to:

1. Train DVAE models on noisy, limited-size image datasets to learn meaningful latent representations that are robust to noise
2. Use the trained DVAE decoders to generate new (denoised) samples for data augmentation
3. Analyze the learned latent space via clustering and visualization
4. Evaluate the impact of DVAE-generated synthetic data on downstream classification tasks
5. Compare DVAEs against baseline models (standard VAE and traditional augmentation techniques)

## Setup

```bash
# Clone the repository
git clone [repository-url]
cd dvae4da

# Create a virtual environment (optional but recommended)
python -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure

- `src/` - Source code
  - `models/` - Model implementations (VAE, DVAE)
  - `data/` - Dataset loading and preprocessing
  - `utils/` - Utility functions
  - `experiments/` - Experiment scripts
- `notebooks/` - Jupyter notebooks for visualization and analysis
- `configs/` - Configuration files
- `results/` - Results and saved models

## Usage

[To be added as the project develops]

## References

- Im, D. J., Ahn, S., Memisevic, R., & Bengio, Y. (2016). Denoising Criterion for Variational Auto-Encoding Framework. AAAI.
- Kingma, D. P., & Welling, M. (2013). Auto-Encoding Variational Bayes. ICLR. 