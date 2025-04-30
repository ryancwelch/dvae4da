import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional


class Encoder(nn.Module):
    """
    VAE Encoder network transforming input images to latent space parameters.
    """
    def __init__(self, 
                input_channels: int = 1, 
                hidden_dims: List[int] = None,
                latent_dim: int = 16,
                img_size: int = 28):
        """
        Args:
            input_channels: Number of channels in the input image
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            img_size: Size of the input image (assuming square)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128]
        
        # Build encoder layers
        modules = []
        
        # First conv layer
        modules.append(
            nn.Sequential(
                nn.Conv2d(input_channels, hidden_dims[0], kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(hidden_dims[0]),
                nn.LeakyReLU()
            )
        )
        
        # Additional conv layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Conv2d(hidden_dims[i], hidden_dims[i + 1], kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        self.encoder = nn.Sequential(*modules)
        
        # Calculate feature dimensions after convolutions
        self.feature_size = img_size // (2 ** len(hidden_dims))
        if self.feature_size < 1:
            # If we've over-downsampled, adjust the feature size
            self.feature_size = 1
            print(f"Warning: Image size {img_size} with {len(hidden_dims)} hidden layers results in feature_size < 1.")
            print(f"Setting feature_size to 1, but consider using fewer hidden layers or larger images.")
        
        self.flat_dim = hidden_dims[-1] * self.feature_size * self.feature_size
        
        # Projections to latent space
        self.fc_mu = nn.Linear(self.flat_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim, latent_dim)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the encoder.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        # Encode through convolutional layers
        features = self.encoder(x)
        
        # Reshape to flat vector
        batch_size = x.shape[0]
        features_shape = features.shape
        features = torch.flatten(features, start_dim=1)
        
        # Check for dimension mismatches
        actual_flat_dim = features.shape[1]
        if self.flat_dim != actual_flat_dim:
            print(f"Warning: Expected flat dimension {self.flat_dim} but got {actual_flat_dim}")
            print(f"Input shape: {x.shape}, Feature shape: {features_shape}")
            
            # Dynamically adjust FC layers if needed (first occurrence only)
            if not hasattr(self, '_dim_fixed'):
                self.flat_dim = actual_flat_dim
                self.fc_mu = nn.Linear(actual_flat_dim, self.fc_mu.out_features).to(x.device)
                self.fc_var = nn.Linear(actual_flat_dim, self.fc_var.out_features).to(x.device)
                self._dim_fixed = True
        
        # Project to latent distribution parameters
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        
        return mu, log_var


class Decoder(nn.Module):
    """
    VAE Decoder network transforming latent vectors to output images.
    """
    def __init__(self, 
                output_channels: int = 1, 
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                img_size: int = 28):
        """
        Args:
            output_channels: Number of channels in the output image
            hidden_dims: Dimensions of hidden layers (in reverse order from encoder)
            latent_dim: Dimension of the latent space
            img_size: Size of the output image (assuming square)
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [128, 64, 32]
        
        # Calculate initial feature map size
        self.num_upsamples = len(hidden_dims)
        self.feature_size = img_size // (2 ** self.num_upsamples)
        
        # Ensure feature size is at least 1
        if self.feature_size < 1:
            self.feature_size = 1
            print(f"Warning: Image size {img_size} with {self.num_upsamples} upsampling layers results in initial feature_size < 1.")
            print(f"Setting initial feature_size to 1, but consider using fewer hidden layers or larger images.")
        
        # Calculate flattened dimension
        self.flat_dim = hidden_dims[0] * self.feature_size * self.feature_size
        
        # Project from latent space to initial feature maps
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim, self.flat_dim),
            nn.LeakyReLU()
        )
        
        # Build decoder layers
        modules = []
        
        # Upsampling layers
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(
                        hidden_dims[i], 
                        hidden_dims[i + 1],
                        kernel_size=3, 
                        stride=2, 
                        padding=1,
                        output_padding=1
                    ),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )
        
        # Final layer to get to output channels with sigmoid activation
        modules.append(
            nn.Sequential(
                nn.ConvTranspose2d(
                    hidden_dims[-1],
                    output_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1
                ),
                nn.Sigmoid()
            )
        )
        
        self.decoder = nn.Sequential(*modules)
        
        # Calculate expected output size
        self.expected_output_size = self.feature_size * (2 ** (self.num_upsamples))
        self.target_img_size = img_size
        
        if self.expected_output_size != img_size:
            print(f"Note: Architecture will produce {self.expected_output_size}x{self.expected_output_size} " 
                  f"images, but target is {img_size}x{img_size}. Will use interpolation if needed.")
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the decoder.
        
        Args:
            z: Latent vector [B, latent_dim]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
        """
        # Project latent vector to initial feature maps
        features = self.latent_to_features(z)
        
        try:
            # Reshape to 3D feature maps
            features = features.view(-1, 
                                    features.shape[1] // (self.feature_size * self.feature_size),
                                    self.feature_size, 
                                    self.feature_size)
        except RuntimeError as e:
            # Handle reshape errors
            print(f"Reshape error in decoder: {e}")
            print(f"features shape: {features.shape}, flat_dim: {self.flat_dim}, feature_size: {self.feature_size}")
            
            # Attempt to adjust dimensions
            if not hasattr(self, '_dim_fixed'):
                batch_size = z.shape[0]
                channels = self.flat_dim // (self.feature_size * self.feature_size)
                features = features.view(batch_size, channels, self.feature_size, self.feature_size)
                self._dim_fixed = True
        
        # Decode through upsampling layers
        x_hat = self.decoder(features)
        
        # Ensure output matches target size
        if x_hat.shape[2:] != (self.target_img_size, self.target_img_size):
            if not hasattr(self, '_resize_warning_shown'):
                print(f"Resizing output from {x_hat.shape[2:]} to target {self.target_img_size}x{self.target_img_size}")
                self._resize_warning_shown = True
            
            x_hat = F.interpolate(x_hat, 
                                 size=(self.target_img_size, self.target_img_size),
                                 mode='bilinear', 
                                 align_corners=False)
        
        return x_hat


class VAE(nn.Module):
    """
    Variational Autoencoder model.
    """
    def __init__(self, 
                img_channels: int = 1, 
                img_size: int = 28,
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                kl_weight: float = 1.0):
        """
        Args:
            img_channels: Number of channels in the input/output image
            img_size: Size of the input/output image (assuming square)
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            kl_weight: Weight for the KL divergence term in the loss
        """
        super().__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        
        if hidden_dims is None:
            # Choose hidden dims based on image size
            if img_size <= 32:
                hidden_dims = [32, 64, 128]  # For smaller images like MNIST
            else:
                hidden_dims = [32, 64, 128, 256]  # For larger images
        
        # Encoder
        self.encoder = Encoder(
            input_channels=img_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )
        
        # Prepare decoder hidden dims (reversed)
        decoder_hidden_dims = hidden_dims.copy()
        decoder_hidden_dims.reverse()
        
        # Decoder
        self.decoder = Decoder(
            output_channels=img_channels,
            hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )
    
    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs to latent distribution parameters.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        return self.encoder(x)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to output images.
        
        Args:
            z: Latent vector [B, latent_dim]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
        """
        return self.decoder(z)
    
    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from latent distribution.
        
        Args:
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
            
        Returns:
            z: Sampled latent vector [B, latent_dim]
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + eps * std
        return z
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE.
        
        Args:
            x: Input image tensor [B, C, H, W]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_hat = self.decode(z)
        
        return x_hat, mu, log_var
    
    def sample(self, n_samples: int, device: torch.device) -> torch.Tensor:
        """
        Generate samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            device: Device to generate samples on
            
        Returns:
            samples: Generated image samples [n_samples, C, H, W]
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decode(z)
        return samples
    
    def loss_function(self, 
                     x_hat: torch.Tensor, 
                     x: torch.Tensor, 
                     mu: torch.Tensor, 
                     log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute VAE loss.
        
        Args:
            x_hat: Reconstructed image [B, C, H, W]
            x: Original image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # Reconstruction loss (using Mean Squared Error)
        recon_loss = F.mse_loss(x_hat, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class ConditionalVAE(VAE):
    """
    Conditional Variational Autoencoder.
    
    Extends the VAE by conditioning on class labels, allowing for 
    class-specific generation.
    """
    def __init__(self, 
                img_channels: int = 1, 
                img_size: int = 28,
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                num_classes: int = 10,
                kl_weight: float = 1.0):
        """
        Args:
            img_channels: Number of channels in the input/output image
            img_size: Size of the input/output image (assuming square)
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            num_classes: Number of classes for conditioning
            kl_weight: Weight for the KL divergence term in the loss
        """
        super(VAE, self).__init__()  # Call nn.Module.__init__()
        
        self.img_channels = img_channels
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.kl_weight = kl_weight
        self.num_classes = num_classes
        
        # Choose hidden dims based on image size if not provided
        if hidden_dims is None:
            if img_size <= 32:
                hidden_dims = [32, 64, 128]  # For smaller images like MNIST
            else:
                hidden_dims = [32, 64, 128, 256]  # For larger images
        
        # Create conditional encoder and decoder
        self.encoder = ConditionalEncoder(
            input_channels=img_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size,
            num_classes=num_classes
        )
        
        decoder_hidden_dims = hidden_dims.copy()
        decoder_hidden_dims.reverse()
        
        self.decoder = ConditionalDecoder(
            output_channels=img_channels,
            hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size,
            num_classes=num_classes
        )
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Conditional VAE.
        
        Args:
            x: Input image tensor [B, C, H, W]
            labels: Class labels [B]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        # Encode inputs with class conditioning
        mu, log_var = self.encoder(x, labels)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode latent vector to reconstruct image, with class conditioning
        x_hat = self.decoder(z, labels)
        
        return x_hat, mu, log_var
    
    def sample(self, n_samples: int, labels: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Generate class-conditional samples from the model.
        
        Args:
            n_samples: Number of samples to generate
            labels: Class labels for samples [n_samples]
            device: Device to generate samples on
            
        Returns:
            samples: Generated image samples [n_samples, C, H, W]
        """
        z = torch.randn(n_samples, self.latent_dim).to(device)
        samples = self.decoder(z, labels)
        return samples


class ConditionalEncoder(Encoder):
    """
    Conditional VAE Encoder that incorporates class information.
    """
    def __init__(self, 
                input_channels: int = 1, 
                hidden_dims: List[int] = None,
                latent_dim: int = 16,
                img_size: int = 28,
                num_classes: int = 10):
        """
        Args:
            input_channels: Number of channels in the input image
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            img_size: Size of the input image (assuming square)
            num_classes: Number of classes for conditioning
        """
        super().__init__(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )
        
        # Embedding for class labels
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Modify the latent projection to include class information
        self.fc_mu = nn.Linear(self.flat_dim + 64, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim + 64, latent_dim)
    
    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through the conditional encoder.
        
        Args:
            x: Input image tensor [B, C, H, W]
            labels: Class labels [B]
            
        Returns:
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        # Encode image through convolutional layers
        features = self.encoder(x)
        features = torch.flatten(features, start_dim=1)
        
        # Get class embedding
        class_embed = self.class_embedding(labels)
        
        # Concatenate image features with class embedding
        features = torch.cat([features, class_embed], dim=1)
        
        # Project to latent distribution parameters
        mu = self.fc_mu(features)
        log_var = self.fc_var(features)
        
        return mu, log_var


class ConditionalDecoder(Decoder):
    """
    Conditional VAE Decoder that incorporates class information.
    """
    def __init__(self, 
                output_channels: int = 1, 
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                img_size: int = 28,
                num_classes: int = 10):
        """
        Args:
            output_channels: Number of channels in the output image
            hidden_dims: Dimensions of hidden layers (in reverse order from encoder)
            latent_dim: Dimension of the latent space
            img_size: Size of the output image (assuming square)
            num_classes: Number of classes for conditioning
        """
        # Call the parent Decoder's init
        super().__init__(
            output_channels=output_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )
        
        # Embedding for class labels
        self.class_embedding = nn.Embedding(num_classes, 64)
        
        # Override the latent_to_features to include class embedding
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim + 64, self.flat_dim),
            nn.LeakyReLU()
        )
    
    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the conditional decoder.
        
        Args:
            z: Latent vector [B, latent_dim]
            labels: Class labels [B]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
        """
        # Get class embedding
        class_embed = self.class_embedding(labels)
        
        # Concatenate latent vector with class embedding
        z_cond = torch.cat([z, class_embed], dim=1)
        
        # Project to initial feature maps
        features = self.latent_to_features(z_cond)
        
        try:
            # Reshape to 3D feature maps
            features = features.view(-1, 
                                   features.shape[1] // (self.feature_size * self.feature_size),
                                   self.feature_size, 
                                   self.feature_size)
        except RuntimeError as e:
            # Handle reshape errors
            print(f"Reshape error in conditional decoder: {e}")
            print(f"features shape: {features.shape}, flat_dim: {self.flat_dim}, feature_size: {self.feature_size}")
            
            # Attempt to adjust dimensions
            if not hasattr(self, '_dim_fixed'):
                batch_size = z.shape[0]
                channels = self.flat_dim // (self.feature_size * self.feature_size)
                features = features.view(batch_size, channels, self.feature_size, self.feature_size)
                self._dim_fixed = True
        
        # Decode through upsampling layers
        x_hat = self.decoder(features)
        
        # Ensure output matches target size
        if x_hat.shape[2:] != (self.target_img_size, self.target_img_size):
            if not hasattr(self, '_resize_warning_shown'):
                print(f"Resizing conditional output from {x_hat.shape[2:]} to target {self.target_img_size}x{self.target_img_size}")
                self._resize_warning_shown = True
            x_hat = F.interpolate(x_hat, 
                                size=(self.target_img_size, self.target_img_size),
                                mode='bilinear', 
                                align_corners=False)
        
        return x_hat


def get_conditional_vae_model(img_channels: int = 1, 
                             img_size: int = 28,
                             hidden_dims: List[int] = None, 
                             latent_dim: int = 16,
                             num_classes: int = 10,
                             kl_weight: float = 1.0) -> ConditionalVAE:
    """
    Factory function to create a Conditional VAE model.
    
    Args:
        img_channels: Number of channels in the input/output image
        img_size: Size of the input/output image (assuming square)
        hidden_dims: Dimensions of hidden layers
        latent_dim: Dimension of the latent space
        num_classes: Number of classes for conditioning
        kl_weight: Weight for the KL divergence term in the loss
        
    Returns:
        Conditional VAE model
    """
    return ConditionalVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        num_classes=num_classes,
        kl_weight=kl_weight
    )

def get_vae_model(img_channels: int = 1, 
                 img_size: int = 28,
                 hidden_dims: List[int] = None, 
                 latent_dim: int = 16,
                 kl_weight: float = 1.0) -> VAE:
    """
    Factory function to create a VAE model.
    
    Args:
        img_channels: Number of channels in the input/output image
        img_size: Size of the input/output image (assuming square)
        hidden_dims: Dimensions of hidden layers
        latent_dim: Dimension of the latent space
        kl_weight: Weight for the KL divergence term in the loss
        
    Returns:
        VAE model
    """
    return VAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    ) 