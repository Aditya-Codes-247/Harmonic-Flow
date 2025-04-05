import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Define default values in case config attributes are not available
DEFAULT_LATENT_DIM = 256
DEFAULT_HIDDEN_DIM = 1024
DEFAULT_DROPOUT = 0.1
DEFAULT_INSTRUMENTS = ["bass", "drums", "guitar", "piano"]

# Try to get values from config, fall back to defaults if not available
try:
    LATENT_DIM = config.LATENT_DIM
except AttributeError:
    LATENT_DIM = DEFAULT_LATENT_DIM

try:
    HIDDEN_DIM = config.HIDDEN_DIM
except AttributeError:
    HIDDEN_DIM = DEFAULT_HIDDEN_DIM

try:
    DROPOUT = config.DROPOUT
except AttributeError:
    DROPOUT = DEFAULT_DROPOUT

try:
    INSTRUMENTS = config.INSTRUMENTS
except AttributeError:
    INSTRUMENTS = DEFAULT_INSTRUMENTS

class StyleTransferModule(nn.Module):
    """
    Style transfer module that can apply stylistic characteristics of specific
    composers, eras, or genres.
    """
    def __init__(
        self,
        input_dim=LATENT_DIM,  # Using our fallback variable instead of direct config access
        style_dim=128,
        hidden_dim=HIDDEN_DIM,  # Using our fallback variable instead of direct config access
        num_styles=10,  # Number of pre-defined styles
        dropout=DROPOUT  # Using our fallback variable instead of direct config access
    ):
        super().__init__()
        
        # Style embeddings
        self.style_embeddings = nn.Parameter(
            torch.randn(num_styles, style_dim)
        )
        
        # Style encoder for extracting style from reference audio
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, style_dim)
        )
        
        # Adaptive instance normalization (AdaIN) layers
        self.adain_layers = nn.ModuleList([
            AdaINResBlock(input_dim, style_dim, dropout)
            for _ in range(4)
        ])
        
        # Final integration layer
        self.output_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def extract_style(self, reference_audio=None, style_idx=None):
        """
        Extract style code from reference audio or use pre-defined style.
        
        Args:
            reference_audio: Optional reference audio to extract style from
            style_idx: Optional index of pre-defined style
            
        Returns:
            style_code: Style code tensor
        """
        if reference_audio is not None:
            # Extract style from reference audio
            style_code = self.style_encoder(reference_audio)
        elif style_idx is not None:
            # Use pre-defined style
            style_code = self.style_embeddings[style_idx]
        else:
            # Default to first style
            style_code = self.style_embeddings[0]
        
        return style_code
        
    def forward(self, x, reference_audio=None, style_idx=None, interpolation_weight=1.0):
        """
        Forward pass.
        
        Args:
            x: Input latent representation
            reference_audio: Optional reference audio to extract style from
            style_idx: Optional index of pre-defined style
            interpolation_weight: Weight for style interpolation (0-1)
            
        Returns:
            output: Stylized output
        """
        # Extract style code
        style_code = self.extract_style(reference_audio, style_idx)
        
        # Apply style via AdaIN layers
        h = x
        for adain in self.adain_layers:
            h = adain(h, style_code)
        
        # Interpolate between input and stylized output
        h = x + interpolation_weight * (h - x)
        
        # Final integration
        output = self.output_layer(h)
        
        return output


class AdaINResBlock(nn.Module):
    """
    Residual block with adaptive instance normalization for style transfer.
    """
    def __init__(
        self,
        input_dim,
        style_dim,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Main layers
        self.norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.fc2 = nn.Linear(input_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Style modulation parameters
        self.style_modulation = nn.Linear(style_dim, input_dim * 2)  # scale and bias
    
    def forward(self, x, style_code):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            style_code: Style code for modulation
            
        Returns:
            out: Stylized output
        """
        # Compute style-dependent scale and bias
        style_params = self.style_modulation(style_code)
        scale, bias = torch.chunk(style_params, 2, dim=-1)
        
        # Apply AdaIN in residual path
        h = self.norm1(x)
        h = h * (scale + 1.0) + bias  # Scale and bias from style
        h = F.leaky_relu(h, 0.2)
        h = self.fc1(h)
        
        h = self.norm2(h)
        h = F.leaky_relu(h, 0.2)
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Residual connection
        out = x + h
        
        return out


class GenreAdaptiveLayer(nn.Module):
    """
    Genre-adaptive layer that fine-tunes outputs for specific genres.
    """
    def __init__(
        self,
        input_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_genres=10,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Genre embeddings
        self.genre_embeddings = nn.Embedding(num_genres, hidden_dim)
        
        # Genre-specific adaptation layers
        self.genre_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, input_dim)
            )
            for _ in range(num_genres)
        ])
        
        # Attention for selecting which genre-specific layer to use
        self.genre_attention = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, genre_idx=None):
        """
        Forward pass.
        
        Args:
            x: Input tensor
            genre_idx: Optional genre index for adaptation
            
        Returns:
            output: Genre-adapted output
        """
        batch_size = x.shape[0]
        
        if genre_idx is not None:
            # Use specified genre
            genre_emb = self.genre_embeddings(genre_idx)
            adapted_x = self.genre_layers[genre_idx](x)
            return adapted_x
        else:
            # Soft attention over all genres
            outputs = []
            attention_scores = []
            
            for i in range(len(self.genre_layers)):
                # Get genre embedding
                genre_emb = self.genre_embeddings.weight[i].unsqueeze(0).repeat(batch_size, 1)
                
                # Compute attention score
                concat_input = torch.cat([x, genre_emb], dim=1)
                attention = self.genre_attention(concat_input)
                attention_scores.append(attention)
                
                # Apply genre-specific adaptation
                adapted_x = self.genre_layers[i](x)
                outputs.append(adapted_x)
            
            # Stack outputs and attention scores
            outputs = torch.stack(outputs, dim=1)  # (batch_size, num_genres, input_dim)
            attention_scores = torch.cat(attention_scores, dim=1)  # (batch_size, num_genres)
            attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(2)
            
            # Weighted sum of genre-specific outputs
            output = torch.sum(outputs * attention_weights, dim=1)
            
            return output


class PostProcessingGAN(nn.Module):
    """
    GAN-based post-processing module for mixing, EQ, compression, and reverb.
    """
    def __init__(
        self,
        input_channels=len(config.INSTRUMENTS),
        hidden_dim=64,
        output_channels=len(config.INSTRUMENTS) + 1,  # +1 for mixed output
        kernel_size=7,
        num_layers=6,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Generator (U-Net architecture)
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.encoder_blocks.append(
            nn.Sequential(
                nn.Conv1d(input_channels, hidden_dim, kernel_size, padding=kernel_size//2),
                nn.LeakyReLU(0.2)
            )
        )
        
        current_dim = hidden_dim
        for i in range(num_layers - 1):
            next_dim = min(current_dim * 2, 512)
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv1d(current_dim, next_dim, kernel_size, stride=2, padding=kernel_size//2),
                    nn.BatchNorm1d(next_dim),
                    nn.LeakyReLU(0.2),
                    nn.Dropout(dropout)
                )
            )
            current_dim = next_dim
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_layers - 1):
            input_dim = current_dim
            output_dim = current_dim // 2
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose1d(input_dim * 2, output_dim, kernel_size, stride=2, padding=kernel_size//2, output_padding=1),
                    nn.BatchNorm1d(output_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                )
            )
            current_dim = output_dim
        
        # Final output layer
        self.final_layer = nn.Sequential(
            nn.ConvTranspose1d(current_dim * 2, output_channels, kernel_size, padding=kernel_size//2),
            nn.Tanh()
        )
        
        # Discriminator
        self.discriminator = nn.Sequential(
            nn.Conv1d(output_channels, hidden_dim, kernel_size, stride=2, padding=kernel_size//2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 2, hidden_dim * 4, kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim * 4),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 4, hidden_dim * 8, kernel_size, stride=2, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim * 8),
            nn.LeakyReLU(0.2),
            
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dim * 8, 1)
        )
    
    def generator(self, x):
        """
        Generator forward pass.
        
        Args:
            x: Input multi-track audio
            
        Returns:
            processed: Processed multi-track audio
        """
        # Apply encoder
        encoder_features = []
        h = x
        
        for block in self.encoder_blocks:
            h = block(h)
            encoder_features.append(h)
        
        # Apply decoder with skip connections
        for i, block in enumerate(self.decoder_blocks):
            skip_features = encoder_features[-(i+2)]
            h = torch.cat([h, skip_features], dim=1)
            h = block(h)
        
        # Final output layer with skip connection to input
        h = torch.cat([h, encoder_features[0]], dim=1)
        processed = self.final_layer(h)
        
        return processed
    
    def discriminator_forward(self, x):
        """
        Discriminator forward pass.
        
        Args:
            x: Input processed audio
            
        Returns:
            prediction: Real/fake prediction
        """
        return self.discriminator(x)
    
    def forward(self, x, mode='generator'):
        """
        Forward pass.
        
        Args:
            x: Input multi-track audio
            mode: 'generator' or 'discriminator'
            
        Returns:
            output: Model output
        """
        if mode == 'generator':
            return self.generator(x)
        elif mode == 'discriminator':
            return self.discriminator_forward(x)
        else:
            raise ValueError(f"Unknown mode: {mode}")


class SpatialAudioRenderer(nn.Module):
    """
    Spatial audio renderer for generating immersive 3D audio outputs.
    """
    def __init__(
        self,
        input_channels=len(config.INSTRUMENTS),
        output_channels=2,  # Stereo output (left and right)
        hidden_dim=64,
        num_azimuth=12,  # Angular resolution (number of azimuth positions)
        num_elevation=5,  # Elevation resolution
        kernel_size=7
    ):
        super().__init__()
        
        self.num_azimuth = num_azimuth
        self.num_elevation = num_elevation
        
        # Spatial parameter prediction network
        self.spatial_param_net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 2, input_channels * 3, kernel_size, padding=kernel_size//2),
            nn.Tanh()  # Output normalized spatial parameters
        )
        
        # HRTF (Head-Related Transfer Function) approximation using learned filters
        # In a real implementation, this would use a database of HRTFs
        self.hrtf_filters = nn.Parameter(
            torch.randn(num_azimuth * num_elevation, output_channels, kernel_size)
        )
        
        # Binaural rendering network
        self.binaural_renderer = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim, hidden_dim * 2, kernel_size, padding=kernel_size//2),
            nn.BatchNorm1d(hidden_dim * 2),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(hidden_dim * 2, output_channels, kernel_size, padding=kernel_size//2),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input multi-track audio (batch_size, num_instruments, time)
            
        Returns:
            binaural_output: Binaural (stereo) spatial audio output
        """
        batch_size, num_instruments, time_steps = x.shape
        
        # Predict spatial parameters (azimuth, elevation, distance) for each instrument
        spatial_params = self.spatial_param_net(x)
        spatial_params = spatial_params.view(batch_size, num_instruments, 3, time_steps)
        
        # Extract individual parameters
        azimuth = (spatial_params[:, :, 0, :] + 1) / 2 * (self.num_azimuth - 1)  # 0 to num_azimuth-1
        elevation = (spatial_params[:, :, 1, :] + 1) / 2 * (self.num_elevation - 1)  # 0 to num_elevation-1
        distance = (spatial_params[:, :, 2, :] + 1) / 2 * 0.9 + 0.1  # 0.1 to 1.0
        
        # Convert to indices for the HRTF filters
        azimuth_idx = azimuth.long()
        elevation_idx = elevation.long()
        
        # Clamp indices to valid range
        azimuth_idx = torch.clamp(azimuth_idx, 0, self.num_azimuth - 1)
        elevation_idx = torch.clamp(elevation_idx, 0, self.num_elevation - 1)
        
        # Compute HRTF filter indices
        hrtf_idx = elevation_idx * self.num_azimuth + azimuth_idx
        
        # Apply the appropriate HRTF filters to each instrument
        # For simplicity, we'll just use a weighted sum approach
        spatially_processed = []
        
        for b in range(batch_size):
            for i in range(num_instruments):
                # Get HRTF filter for this instrument's position
                hrtf_filter = self.hrtf_filters[hrtf_idx[b, i, 0]]  # Using first time step for simplicity
                
                # Apply filter using convolution
                filtered = F.conv1d(
                    x[b, i:i+1],
                    hrtf_filter.unsqueeze(1),
                    padding=self.hrtf_filters.size(2) // 2
                )
                
                # Apply distance attenuation
                dist_factor = distance[b, i, 0]  # Using first time step for simplicity
                filtered = filtered * dist_factor
                
                spatially_processed.append(filtered)
        
        # Stack and sum all processed instruments
        spatially_processed = torch.stack(spatially_processed, dim=1)
        summed = spatially_processed.sum(dim=1)
        
        # Apply final binaural rendering
        binaural_output = self.binaural_renderer(summed)
        
        return binaural_output


class StyleAndPostProcessingModule(nn.Module):
    """
    Complete style transfer and post-processing module.
    """
    def __init__(
        self,
        style_transfer=None,
        genre_adaptive=None,
        post_processing=None,
        spatial_renderer=None,
        input_dim=config.LATENT_DIM
    ):
        super().__init__()
        
        self.style_transfer = style_transfer if style_transfer else StyleTransferModule(input_dim=input_dim)
        self.genre_adaptive = genre_adaptive if genre_adaptive else GenreAdaptiveLayer(input_dim=input_dim)
        self.post_processing = post_processing if post_processing else PostProcessingGAN()
        self.spatial_renderer = spatial_renderer if spatial_renderer else SpatialAudioRenderer()
    
    def forward(self, x, reference_style=None, style_idx=None, genre_idx=None, interpolation_weight=1.0):
        """
        Forward pass.
        
        Args:
            x: Input multi-track audio in latent or waveform representation
            reference_style: Optional reference style
            style_idx: Optional style index
            genre_idx: Optional genre index
            interpolation_weight: Weight for style interpolation
            
        Returns:
            processed: Fully processed output
            outputs: Dictionary of intermediate outputs
        """
        # Apply style transfer if input is in latent space
        if len(x.shape) == 2 or (len(x.shape) == 3 and x.shape[1] == config.LATENT_DIM):
            # Input is in latent form
            stylized = self.style_transfer(x, reference_style, style_idx, interpolation_weight)
            genre_adapted = self.genre_adaptive(stylized, genre_idx)
            
            # The post-processing and spatial rendering expect waveform inputs,
            # so we would need to convert from latent to waveform here
            # For this implementation, we'll assume the conversion happens elsewhere
            
            outputs = {
                'stylized': stylized,
                'genre_adapted': genre_adapted
            }
            
            return genre_adapted, outputs
        else:
            # Input is in waveform form
            post_processed = self.post_processing(x, mode='generator')
            spatial_audio = self.spatial_renderer(post_processed)
            
            outputs = {
                'post_processed': post_processed,
                'spatial_audio': spatial_audio
            }
            
            return spatial_audio, outputs