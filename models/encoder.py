import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import transformers
from transformers import AutoModel, AutoTokenizer
from einops import rearrange
from config.config import Config

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class AudioEncoder(nn.Module):
    """
    Audio encoder that extracts musical features from raw audio.
    """
    def __init__(
        self, 
        input_dim=Config.n_mels, 
        hidden_dim=Config.hidden_dim,
        latent_dim=Config.latent_dim, 
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Project to latent space
        x = self.output_proj(x)
        
        return x


class TextEncoder(nn.Module):
    """
    Text encoder that extracts features from text prompts.
    Uses a pre-trained transformer model.
    """
    def __init__(
        self, 
        model_name="distilbert-base-uncased",
        latent_dim=Config.latent_dim,
        max_length=128,
        dropout=0.1
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
        latent_dim=Config.latent_dim,
        dropout=0.1
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


class EmotionEncoder(nn.Module):
    """Encoder for emotion labels."""
    
    def __init__(
        self,
        num_emotions=8,
        hidden_dim=Config.hidden_dim,
        latent_dim=Config.latent_dim,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_emotions, hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Emotion indices of shape (batch_size,)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x


class StyleEncoder(nn.Module):
    """Encoder for style labels."""
    
    def __init__(
        self,
        num_styles=10,
        hidden_dim=Config.hidden_dim,
        latent_dim=Config.latent_dim,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_styles, hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Style indices of shape (batch_size,)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x


class GenreEncoder(nn.Module):
    """Encoder for genre labels."""
    
    def __init__(
        self,
        num_genres=10,
        hidden_dim=Config.hidden_dim,
        latent_dim=Config.latent_dim,
        dropout=0.1
    ):
        super().__init__()
        
        self.embedding = nn.Embedding(num_genres, hidden_dim)
        self.proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Genre indices of shape (batch_size,)
            
        Returns:
            Latent representation of shape (batch_size, latent_dim)
        """
        x = self.embedding(x)
        x = self.dropout(x)
        x = self.proj(x)
        return x


class MultiModalEncoder(nn.Module):
    """
    Multi-modal encoder that combines audio, text, and visual encoders.
    """
    def __init__(
        self,
        input_dim=Config.n_mels,
        hidden_dim=Config.hidden_dim,
        latent_dim=Config.latent_dim,
        num_emotions=8,
        num_styles=10,
        num_genres=10,
        dropout=0.1
    ):
        super().__init__()
        
        self.audio_encoder = AudioEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.emotion_encoder = EmotionEncoder(
            num_emotions=num_emotions,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.style_encoder = StyleEncoder(
            num_styles=num_styles,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        self.genre_encoder = GenreEncoder(
            num_genres=num_genres,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            dropout=dropout
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 4, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim)
        )
    
    def forward(
        self,
        audio,
        emotion_idx=None,
        style_idx=None,
        genre_idx=None
    ):
        """
        Forward pass.
        
        Args:
            audio: Audio input of shape (batch_size, seq_len, input_dim)
            emotion_idx: Optional emotion indices of shape (batch_size,)
            style_idx: Optional style indices of shape (batch_size,)
            genre_idx: Optional genre indices of shape (batch_size,)
            
        Returns:
            Combined latent representation of shape (batch_size, latent_dim)
        """
        # Encode audio
        audio_latent = self.audio_encoder(audio)
        
        # Initialize emotion, style, and genre latents as zeros
        batch_size = audio.size(0)
        device = audio.device
        
        emotion_latent = torch.zeros(batch_size, self.audio_encoder.latent_dim, device=device)
        style_latent = torch.zeros(batch_size, self.audio_encoder.latent_dim, device=device)
        genre_latent = torch.zeros(batch_size, self.audio_encoder.latent_dim, device=device)
        
        # Encode conditional inputs if provided
        if emotion_idx is not None:
            emotion_latent = self.emotion_encoder(emotion_idx)
            
        if style_idx is not None:
            style_latent = self.style_encoder(style_idx)
            
        if genre_idx is not None:
            genre_latent = self.genre_encoder(genre_idx)
        
        # Concatenate all latents
        combined = torch.cat([
            audio_latent,
            emotion_latent,
            style_latent,
            genre_latent
        ], dim=-1)
        
        # Fuse latents
        fused = self.fusion(combined)
        
        return fused 