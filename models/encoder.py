import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import AutoModel, AutoTokenizer

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class AudioEncoder(nn.Module):
    """
    Audio encoder that extracts musical features from raw audio.
    """
    def __init__(
        self, 
        input_dim=config.N_MELS, 
        hidden_dim=config.HIDDEN_DIM,
        latent_dim=config.LATENT_DIM, 
        num_layers=4,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Convolutional layers for processing spectrograms
        self.conv_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1 if i == 0 else 2**(i+4), 2**(i+5), kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(2**(i+5)),
                nn.LeakyReLU(0.2),
                nn.Dropout2d(dropout)
            ) for i in range(num_layers)
        ])
        
        # Calculate output size after conv layers
        self.calc_conv_output_dim = self._get_conv_output_dim(input_dim)
        
        # Linear layers
        self.linear_layers = nn.Sequential(
            nn.Linear(self.calc_conv_output_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
        
        # Tempo, rhythm, pitch, and timbre-specific branches
        self.tempo_branch = nn.Linear(hidden_dim, 64)  # Tempo features
        self.pitch_branch = nn.Linear(hidden_dim, 128)  # Pitch features
        self.timbre_branch = nn.Linear(hidden_dim, 128)  # Timbre features
        self.rhythm_branch = nn.Linear(hidden_dim, 128)  # Rhythm features
        
    def _get_conv_output_dim(self, input_dim):
        """Calculate output dimensions after conv layers."""
        # Create a sample input to determine output size
        sample_input = torch.zeros(1, 1, input_dim, 400)  # Assuming 400 time frames
        
        # Pass through conv layers
        x = sample_input
        for conv in self.conv_layers:
            x = conv(x)
        
        return x.numel()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input spectrogram tensor of shape (batch_size, time, freq)
        
        Returns:
            latent: Latent representation
            features: Dict containing tempo, pitch, timbre, and rhythm features
        """
        # Ensure correct input shape (batch, channels, freq, time)
        if len(x.shape) == 3:  # (batch, freq, time)
            x = x.unsqueeze(1)
        
        # Pass through conv layers
        for conv in self.conv_layers:
            x = conv(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Pass through linear layers
        x = self.linear_layers[0](x)  # Get hidden representation
        
        # Branch-specific features
        tempo_features = self.tempo_branch(x)
        pitch_features = self.pitch_branch(x)
        timbre_features = self.timbre_branch(x)
        rhythm_features = self.rhythm_branch(x)
        
        # Final latent representation
        latent = self.linear_layers[1:](x)
        
        features = {
            'tempo': tempo_features,
            'pitch': pitch_features,
            'timbre': timbre_features,
            'rhythm': rhythm_features
        }
        
        return latent, features


class TextEncoder(nn.Module):
    """
    Text encoder that extracts features from text prompts.
    Uses a pre-trained transformer model.
    """
    def __init__(
        self, 
        model_name="distilbert-base-uncased",
        latent_dim=config.LATENT_DIM,
        max_length=128,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Initialize tokenizer and transformer model
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.max_length = max_length
        
        # Freeze the transformer parameters
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        # Get transformer output dimension
        transformer_dim = self.transformer.config.hidden_size
        
        # Projection to latent dimension
        self.projection = nn.Sequential(
            nn.Linear(transformer_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, text):
        """
        Forward pass.
        
        Args:
            text: List of text prompts
        
        Returns:
            latent: Latent representation
        """
        # Tokenize text
        inputs = self.tokenizer(
            text, 
            padding='max_length', 
            truncation=True, 
            max_length=self.max_length, 
            return_tensors='pt'
        ).to(next(self.parameters()).device)
        
        # Pass through transformer
        with torch.no_grad():
            outputs = self.transformer(**inputs)
        
        # Get [CLS] token representation (sentence-level)
        cls_output = outputs.last_hidden_state[:, 0, :]
        
        # Project to latent dimension
        latent = self.projection(cls_output)
        
        return latent


class VisualEncoder(nn.Module):
    """
    Visual encoder that extracts features from images.
    Uses a pre-trained vision transformer model.
    """
    def __init__(
        self, 
        model_name="google/vit-base-patch16-224",
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Initialize vision transformer model
        self.vit = AutoModel.from_pretrained(model_name)
        
        # Freeze the transformer parameters
        for param in self.vit.parameters():
            param.requires_grad = False
        
        # Get transformer output dimension
        vit_dim = self.vit.config.hidden_size
        
        # Projection to latent dimension
        self.projection = nn.Sequential(
            nn.Linear(vit_dim, latent_dim * 2),
            nn.LayerNorm(latent_dim * 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 2, latent_dim)
        )
    
    def forward(self, images):
        """
        Forward pass.
        
        Args:
            images: Image tensor of shape (batch_size, channels, height, width)
        
        Returns:
            latent: Latent representation
        """
        # Pass through vision transformer
        with torch.no_grad():
            outputs = self.vit(images).last_hidden_state
        
        # Get [CLS] token representation (image-level)
        cls_output = outputs[:, 0, :]
        
        # Project to latent dimension
        latent = self.projection(cls_output)
        
        return latent


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that combines audio, text, and visual encoders.
    """
    def __init__(
        self,
        audio_encoder=None,
        text_encoder=None,
        visual_encoder=None,
        latent_dim=config.LATENT_DIM,
        dropout=config.DROPOUT
    ):
        super().__init__()
        
        # Initialize encoders if not provided
        self.audio_encoder = audio_encoder if audio_encoder else AudioEncoder(latent_dim=latent_dim)
        self.text_encoder = text_encoder if text_encoder else TextEncoder(latent_dim=latent_dim)
        self.visual_encoder = visual_encoder if visual_encoder else None
        
        # Fusion layer to combine multi-modal embeddings
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * (2 if visual_encoder else 1), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout)
        )
    
    def forward(self, audio=None, text=None, image=None):
        """
        Forward pass.
        
        Args:
            audio: Audio tensor
            text: Text prompts
            image: Image tensor
        
        Returns:
            latent: Fused latent representation
            features: Dict containing audio features
        """
        audio_latent, audio_features = None, None
        text_latent, image_latent = None, None
        
        # Get audio embeddings if provided
        if audio is not None:
            audio_latent, audio_features = self.audio_encoder(audio)
        
        # Get text embeddings if provided
        if text is not None and self.text_encoder is not None:
            text_latent = self.text_encoder(text)
        
        # Get image embeddings if provided
        if image is not None and self.visual_encoder is not None:
            image_latent = self.visual_encoder(image)
        
        # Fuse embeddings if multiple modalities are provided
        latents_to_fuse = []
        
        if audio_latent is not None:
            latents_to_fuse.append(audio_latent)
        
        if text_latent is not None:
            latents_to_fuse.append(text_latent)
            
        if image_latent is not None:
            latents_to_fuse.append(image_latent)
        
        if len(latents_to_fuse) > 1:
            # Concatenate latents
            combined_latent = torch.cat(latents_to_fuse, dim=1)
            
            # Fuse latents
            fused_latent = self.fusion(combined_latent)
        else:
            fused_latent = latents_to_fuse[0] if latents_to_fuse else None
        
        return fused_latent, audio_features 