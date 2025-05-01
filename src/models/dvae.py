import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional
from .vae import Encoder, Decoder, VAE  # Reuse components from standard VAE


class DVAE(VAE):
    """
    Denoising Variational Autoencoder model.
    
    Extends the standard VAE by training to denoise inputs.
    The encoder receives noisy input, while the decoder tries to reconstruct the clean input.
    """
    def __init__(self, 
                img_channels: int = 1, 
                img_size: int = 28,
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                kl_weight: float = 0.1):  # Reduced from 1.0 to 0.1 to prioritize reconstruction
        """
        Args:
            img_channels: Number of channels in the input/output image
            img_size: Size of the input/output image (assuming square)
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            kl_weight: Weight for the KL divergence term in the loss
        """
        # Initialize parent VAE class
        super().__init__(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
    
    def forward(self, noisy_x: torch.Tensor, clean_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the DVAE.
        
        Args:
            noisy_x: Noisy input image tensor [B, C, H, W]
            clean_x: Clean input image tensor (for training) [B, C, H, W]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        # Encode noisy input
        mu, log_var = self.encode(noisy_x)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode to reconstruct clean version
        x_hat = self.decode(z)
        
        return x_hat, mu, log_var
    
    def loss_function(self, 
                     x_hat: torch.Tensor, 
                     clean_x: torch.Tensor, 
                     mu: torch.Tensor, 
                     log_var: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute DVAE loss.
        
        Args:
            x_hat: Reconstructed image [B, C, H, W]
            clean_x: Clean target image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
            
        Returns:
            Dictionary containing total loss and individual components
        """
        # For DVAE, use a combination of MSE and L1 loss for better reconstruction
        # MSE loss focuses on larger errors, L1 loss preserves details
        mse_loss = F.mse_loss(x_hat, clean_x, reduction='mean')
        l1_loss = F.l1_loss(x_hat, clean_x, reduction='mean')
        
        # Combined reconstruction loss with higher weight on L1 for better details
        recon_loss = (mse_loss + 2.0 * l1_loss) * x_hat.shape[0] * x_hat.shape[1] * x_hat.shape[2] * x_hat.shape[3]
        
        # KL divergence loss with reduced weight
        kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Total loss - higher priority on reconstruction
        total_loss = recon_loss + self.kl_weight * kl_loss
        
        return {
            'loss': total_loss,
            'recon_loss': recon_loss,
            'kl_loss': kl_loss
        }


class ConditionalDVAE(DVAE):
    """
    Conditional Denoising Variational Autoencoder.
    
    Extends the DVAE by conditioning on class labels, allowing for 
    class-specific generation.
    """
    def __init__(self, 
                img_channels: int = 1, 
                img_size: int = 28,
                hidden_dims: List[int] = None, 
                latent_dim: int = 16,
                num_classes: int = 10,
                kl_weight: float = 0.05):  # Even lower KL weight for conditional model
        """
        Args:
            img_channels: Number of channels in the input/output image
            img_size: Size of the input/output image (assuming square)
            hidden_dims: Dimensions of hidden layers
            latent_dim: Dimension of the latent space
            num_classes: Number of classes for conditioning
            kl_weight: Weight for the KL divergence term in the loss
        """
        # Call parent DVAE __init__
        super().__init__(
            img_channels=img_channels,
            img_size=img_size,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            kl_weight=kl_weight
        )
        
        # Store additional attributes
        self.num_classes = num_classes
        
        # Override encoder and decoder with conditional versions
        self.encoder = ConditionalEncoder(
            input_channels=img_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size,
            num_classes=num_classes
        )
        
        decoder_hidden_dims = hidden_dims.copy() if hidden_dims else None
        if decoder_hidden_dims:
            decoder_hidden_dims.reverse()
        
        self.decoder = ConditionalDecoder(
            output_channels=img_channels,
            hidden_dims=decoder_hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size,
            num_classes=num_classes
        )
    
    def forward(self, noisy_x: torch.Tensor, labels: torch.Tensor, clean_x: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the Conditional DVAE.
        
        Args:
            noisy_x: Noisy input image tensor [B, C, H, W]
            labels: Class labels [B]
            clean_x: Clean input image tensor (for training) [B, C, H, W]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        # Encode noisy inputs with class conditioning
        mu, log_var = self.encoder(noisy_x, labels)
        
        # Sample from latent distribution
        z = self.reparameterize(mu, log_var)
        
        # Decode latent vector to reconstruct clean image, with class conditioning
        x_hat = self.decoder(z, labels)
        
        return x_hat, mu, log_var
    
    def encode(self, x: torch.Tensor, labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode inputs to latent distribution parameters with conditioning.
        
        Args:
            x: Input image tensor [B, C, H, W]
            labels: Class labels [B]
            
        Returns:
            mu: Mean of the latent distribution [B, latent_dim]
            log_var: Log variance of the latent distribution [B, latent_dim]
        """
        return self.encoder(x, labels)
    
    def conditional_decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decode latent vectors to output images with conditioning.
        
        Args:
            z: Latent vector [B, latent_dim]
            labels: Class labels [B]
            
        Returns:
            x_hat: Reconstructed image [B, C, H, W]
        """
        return self.decoder(z, labels)
    
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
        # Initialize parent Encoder
        super().__init__(
            input_channels=input_channels,
            hidden_dims=hidden_dims,
            latent_dim=latent_dim,
            img_size=img_size
        )
        
        # Embedding for class labels
        self.embedding_dim = 64
        self.class_embedding = nn.Embedding(num_classes, self.embedding_dim)
        
        # Calculate the actual flattened dimension by running a forward pass
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, img_size, img_size)
            dummy_features = self.encoder(dummy_input)
            self.flat_dim = dummy_features.flatten(start_dim=1).shape[1]
            print(f"ConditionalEncoder - Actual flat_dim calculated: {self.flat_dim}")
        
        # Replace the parent class's linear projections
        # First, remove the original linear layers to avoid having unused parameters
        del self.fc_mu
        del self.fc_var
        
        # Create new linear layers with correct input dimension
        self.fc_mu = nn.Linear(self.flat_dim + self.embedding_dim, latent_dim)
        self.fc_var = nn.Linear(self.flat_dim + self.embedding_dim, latent_dim)
    
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
        
        # Debug info
        if not hasattr(self, '_shape_printed'):
            print(f"Encoder flattened features shape: {features.shape}, flat_dim: {self.flat_dim}")
            self._shape_printed = True
        
        # Get class embedding
        class_embed = self.class_embedding(labels)
        
        # Debug info
        if not hasattr(self, '_embed_printed'):
            print(f"Class embedding shape: {class_embed.shape}")
            self._embed_printed = True
        
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
        
        # Store hidden_dims as class attribute to avoid AttributeError
        self.hidden_dims = hidden_dims if hidden_dims is not None else [32, 64, 128, 256]
        
        # Embedding for class labels
        self.embedding_dim = 64
        self.class_embedding = nn.Embedding(num_classes, self.embedding_dim)
        
        # Calculate the actual feature dimensions needed by decoder using a dummy pass
        with torch.no_grad():
            # First calculate the flat dimension needed
            self.feature_size = img_size // (2 ** len(self.hidden_dims)) if self.hidden_dims else img_size // 8
            
            # Ensure minimum feature size
            if self.feature_size < 1:
                self.feature_size = 1
                
            channels = self.hidden_dims[0] if self.hidden_dims and len(self.hidden_dims) > 0 else 64
            self.flat_dim = channels * self.feature_size * self.feature_size
            print(f"ConditionalDecoder - Calculated flat_dim: {self.flat_dim}, feature_size: {self.feature_size}")
        
        # Replace the original latent_to_features layer
        del self.latent_to_features
        
        # Override the latent_to_features to include class embedding
        self.latent_to_features = nn.Sequential(
            nn.Linear(latent_dim + self.embedding_dim, self.flat_dim),
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
        
        # Debug info
        if not hasattr(self, '_shape_printed'):
            print(f"Decoder input shapes - z: {z.shape}, class_embed: {class_embed.shape}")
            self._shape_printed = True
        
        # Concatenate latent vector with class embedding
        z_cond = torch.cat([z, class_embed], dim=1)
        
        # Project to initial feature maps
        features = self.latent_to_features(z_cond)
        
        # Debug info
        if not hasattr(self, '_features_printed'):
            print(f"Decoder features shape: {features.shape}, flat_dim: {self.flat_dim}, feature_size: {self.feature_size}")
            self._features_printed = True
        
        # Reshape to 3D feature maps
        batch_size = features.shape[0]
        channels = self.hidden_dims[0] if self.hidden_dims and len(self.hidden_dims) > 0 else 64
        features = features.view(batch_size, channels, self.feature_size, self.feature_size)
        
        # Decode through upsampling layers
        x_hat = self.decoder(features)
        
        # Debug output shape
        # print(f"ConditionalDecoder output shape: {x_hat.shape}")
        
        # Ensure output matches target size
        if x_hat.shape[2:] != (self.target_img_size, self.target_img_size):
            if not hasattr(self, '_resize_warning_shown'):
                print(f"Resizing conditional output from {x_hat.shape[2:]} to target {self.target_img_size}x{self.target_img_size}")
                self._resize_warning_shown = True
            x_hat = F.interpolate(x_hat, 
                                size=(self.target_img_size, self.target_img_size),
                                mode='bilinear', 
                                align_corners=False)
            # print(f"After resizing: {x_hat.shape}")
        
        return x_hat


def get_dvae_model(img_channels: int = 1, 
                  img_size: int = 28,
                  hidden_dims: List[int] = None, 
                  latent_dim: int = 16,
                  kl_weight: float = 0.1) -> DVAE:
    """
    Factory function to create a DVAE model.
    
    Args:
        img_channels: Number of channels in the input/output image
        img_size: Size of the input/output image (assuming square)
        hidden_dims: Dimensions of hidden layers
        latent_dim: Dimension of the latent space
        kl_weight: Weight for the KL divergence term in the loss
        
    Returns:
        DVAE model
    """
    return DVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        kl_weight=kl_weight
    )


def get_conditional_dvae_model(img_channels: int = 1, 
                              img_size: int = 28,
                              hidden_dims: List[int] = None, 
                              latent_dim: int = 16,
                              num_classes: int = 10,
                              kl_weight: float = 0.05) -> ConditionalDVAE:
    """
    Factory function to create a Conditional DVAE model.
    
    Args:
        img_channels: Number of channels in the input/output image
        img_size: Size of the input/output image (assuming square)
        hidden_dims: Dimensions of hidden layers
        latent_dim: Dimension of the latent space
        num_classes: Number of classes for conditioning
        kl_weight: Weight for the KL divergence term in the loss
        
    Returns:
        Conditional DVAE model
    """
    return ConditionalDVAE(
        img_channels=img_channels,
        img_size=img_size,
        hidden_dims=hidden_dims,
        latent_dim=latent_dim,
        num_classes=num_classes,
        kl_weight=kl_weight
    ) 