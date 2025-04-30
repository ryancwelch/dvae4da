import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.models import get_conditional_vae_model

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load MNIST dataset
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

# Create a data loader
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

# Create a conditional VAE with smaller dimensions
model = get_conditional_vae_model(
    img_channels=1,
    img_size=28,
    hidden_dims=[16, 32],  # Use smaller dimensions to avoid size issues
    latent_dim=16,
    num_classes=10,
    kl_weight=1.0
)

model.to(device)

# Training setup
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 3  # Just train for a few epochs as a test

# Training loop
model.train()
for epoch in range(num_epochs):
    train_loss = 0
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, mu, log_var = model(images, labels)
        
        # Compute loss
        loss_dict = model.loss_function(recon_batch, images, mu, log_var)
        loss = loss_dict['loss']
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        
        if batch_idx % 100 == 0:
            print(f'Epoch: {epoch+1}/{num_epochs}, Batch: {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    avg_loss = train_loss / len(train_loader)
    print(f'Epoch: {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')

# Generate samples (one per class)
model.eval()
with torch.no_grad():
    # Create labels tensor (0-9)
    labels = torch.arange(10, device=device)
    
    # Generate samples
    samples = model.sample(10, labels, device)
    samples = samples.cpu()

# Plot samples
fig, axs = plt.subplots(1, 10, figsize=(12, 1.2))
for i in range(10):
    img = samples[i].squeeze().numpy()
    axs[i].imshow(img, cmap='gray')
    axs[i].set_title(f"{i}")
    axs[i].axis('off')

plt.tight_layout()
plt.savefig('conditional_vae_samples.png')
print("Samples saved to conditional_vae_samples.png")

# Save the model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_params': {
        'img_channels': 1,
        'img_size': 28,
        'hidden_dims': [16, 32],
        'latent_dim': 16,
        'num_classes': 10,
        'kl_weight': 1.0
    }
}, 'conditional_vae_model.pt')
print("Model saved to conditional_vae_model.pt") 