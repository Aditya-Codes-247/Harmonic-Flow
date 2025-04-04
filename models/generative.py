import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Sinusoidal position embeddings for diffusion model.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class AttentionBlock(nn.Module):
    """
    Self-attention block for processing sequential data.
    """
    def __init__(
        self,
        dim=Config.latent_dim,
        num_heads=Config.num_heads,
        dropout=Config.dropout
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Query, key, value projections
        self.qkv = nn.Linear(dim, dim * 3)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Layer normalization
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, dim)
        """
        # Layer normalization
        x = self.norm(x)
        
        # Project to query, key, value
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)
        
        # Compute attention scores
        dots = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply softmax
        attn = F.softmax(dots, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        # Project output
        out = self.proj(out)
        
        return out


class ConvBlock(nn.Module):
    """
    Convolutional block with normalization and activation.
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        use_norm=True,
        dropout=Config.dropout
    ):
        super().__init__()
        
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        
        self.use_norm = use_norm
        if use_norm:
            self.norm = nn.GroupNorm(8, out_channels)
            
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, seq_len)
            
        Returns:
            x: Output tensor of shape (batch_size, out_channels, seq_len)
        """
        x = self.conv(x)
        
        if self.use_norm:
            x = self.norm(x)
            
        x = self.activation(x)
        x = self.dropout(x)
        
        return x


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism that processes music at multiple temporal scales.
    """
    def __init__(
        self,
        input_dim=Config.latent_dim,
        hidden_dim=Config.hidden_dim,
        output_dim=Config.latent_dim,
        num_heads=Config.num_heads,
        num_layers=4,
        dropout=Config.dropout
    ):
        super().__init__()
        
        # Multi-scale processing
        self.scales = [
            {'name': 'micro', 'factor': 1},    # Notes/beats (original resolution)
            {'name': 'meso', 'factor': 4},     # Phrases/measures (4x downsampling)
            {'name': 'macro', 'factor': 16}    # Sections (16x downsampling)
        ]
        
        # Downsampling layers
        self.downsample_layers = nn.ModuleDict({
            scale['name']: nn.Sequential(
                nn.Conv1d(input_dim, input_dim, kernel_size=scale['factor'], stride=scale['factor'], padding=0),
                nn.GroupNorm(8, input_dim),
                nn.SiLU()
            ) if scale['factor'] > 1 else nn.Identity()
            for scale in self.scales
        })
        
        # Attention blocks for each scale
        self.attention_blocks = nn.ModuleDict({
            scale['name']: nn.ModuleList([
                AttentionBlock(
                    dim=input_dim,
                    num_heads=num_heads,
                    dropout=dropout
                )
                for _ in range(num_layers)
            ])
            for scale in self.scales
        })
        
        # Upsampling layers
        self.upsample_layers = nn.ModuleDict({
            scale['name']: nn.Sequential(
                nn.Upsample(scale_factor=scale['factor'], mode='nearest'),
                nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=1),
                nn.GroupNorm(8, input_dim),
                nn.SiLU()
            ) if scale['factor'] > 1 else nn.Identity()
            for scale in self.scales
        })
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(input_dim * len(self.scales), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            out: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Process each scale
        multi_scale_features = {}
        
        for scale in self.scales:
            scale_name = scale['name']
            
            # Convert to channel-first for convolutions
            x_scale = x.transpose(1, 2)  # (batch_size, input_dim, seq_len)
            
            # Downsample
            x_scale = self.downsample_layers[scale_name](x_scale)
            
            # Convert back to channel-last for attention
            x_scale = x_scale.transpose(1, 2)  # (batch_size, downsampled_seq_len, input_dim)
            
            # Apply attention blocks
            for attn_block in self.attention_blocks[scale_name]:
                x_scale = x_scale + attn_block(x_scale)
            
            # Convert to channel-first for upsampling
            x_scale = x_scale.transpose(1, 2)  # (batch_size, input_dim, downsampled_seq_len)
            
            # Upsample back to original resolution
            x_scale = self.upsample_layers[scale_name](x_scale)
            
            # Ensure upsampled tensor has the same sequence length as the input
            # This handles edge cases where upsampling doesn't result in exact dimensions
            if x_scale.size(2) != seq_len:
                # Either pad or trim
                if x_scale.size(2) < seq_len:
                    padding = seq_len - x_scale.size(2)
                    x_scale = F.pad(x_scale, (0, padding))
                else:
                    x_scale = x_scale[:, :, :seq_len]
            
            # Convert back to channel-last
            x_scale = x_scale.transpose(1, 2)  # (batch_size, seq_len, input_dim)
            
            multi_scale_features[scale_name] = x_scale
        
        # Concatenate features from all scales
        concat_features = torch.cat([
            multi_scale_features[scale['name']] for scale in self.scales
        ], dim=-1)
        
        # Integrate multi-scale features
        out = self.integration(concat_features)
        
        return out


class EmotionAwareLayer(nn.Module):
    """
    Emotion-aware layer to adjust musical parameters based on desired emotional tones.
    """
    def __init__(
        self,
        input_dim=Config.latent_dim,
        emotion_dim=8,  # 8 basic emotions
        hidden_dim=Config.hidden_dim,
        output_dim=Config.latent_dim,
        dropout=Config.dropout
    ):
        super().__init__()
        
        # Emotion embedding
        self.emotion_embedding = nn.Embedding(emotion_dim, input_dim)
        
        # Emotion modulation layers
        self.tempo_modulation = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Tempo scaling factor (0.5-1.5x)
        )
        
        self.dynamics_modulation = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Dynamics scaling factor (0.5-1.5x)
        )
        
        self.mode_modulation = nn.Sequential(
            nn.Linear(input_dim + input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1)  # Major vs. minor mode probability
        )
        
        # Integration layer
        self.integration = nn.Sequential(
            nn.Linear(input_dim + 4, hidden_dim),  # +4 for tempo, dynamics, and mode (2)
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x, emotion_idx):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            emotion_idx: Emotion indices of shape (batch_size,)
            
        Returns:
            out: Output tensor of shape (batch_size, seq_len, output_dim)
        """
        batch_size, seq_len, input_dim = x.shape
        
        # Get emotion embeddings
        emotion_emb = self.emotion_embedding(emotion_idx).unsqueeze(1).repeat(1, seq_len, 1)
        
        # Concatenate input with emotion embeddings
        x_emotion = torch.cat([x, emotion_emb], dim=-1)
        
        # Apply modulations
        tempo_factor = self.tempo_modulation(x_emotion)  # (batch_size, seq_len, 1)
        dynamics_factor = self.dynamics_modulation(x_emotion)  # (batch_size, seq_len, 1)
        mode_probs = self.mode_modulation(x_emotion)  # (batch_size, seq_len, 2)
        
        # Scale tempo and dynamics factors to desired range (0.5-1.5)
        tempo_factor = 0.5 + tempo_factor
        dynamics_factor = 0.5 + dynamics_factor
        
        # Concatenate features with modulation factors
        concat_features = torch.cat([
            x, tempo_factor, dynamics_factor, mode_probs
        ], dim=-1)
        
        # Integrate features
        out = self.integration(concat_features)
        
        return out


class MultiTrackDiffusionModel(nn.Module):
    """
    Multi-track diffusion model for generating coherent multi-track audio.
    """
    def __init__(
        self,
        input_dim=Config.latent_dim,
        hidden_dim=Config.hidden_dim,
        # Replace Config.diffusion_steps with a default value or the correct config reference
        num_steps=1000,  # Changed from Config.diffusion_steps to a default value
        beta_schedule="linear",
        num_instruments=len(Config.instruments),
        seq_length=128,
        dropout=Config.dropout
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_steps = num_steps
        self.num_instruments = num_instruments
        self.seq_length = seq_length
        
        # Set up diffusion schedule
        if beta_schedule == "linear":
            betas = torch.linspace(Config.diffusion_betas[0], Config.diffusion_betas[1], num_steps)
        elif beta_schedule == "cosine":
            # Cosine schedule as proposed in https://arxiv.org/abs/2102.09672
            steps = num_steps + 1
            x = torch.linspace(0, num_steps, steps)
            alphas_cumprod = torch.cos(((x / num_steps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        
        # Pre-compute diffusion parameters
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and posterior q(x_{t-1} | x_t, x_0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        sqrt_recip_alphas = torch.sqrt(1. / alphas)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        
        # Register buffers for efficient access during sampling
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', sqrt_alphas_cumprod)
        self.register_buffer('sqrt_one_minus_alphas_cumprod', sqrt_one_minus_alphas_cumprod)
        self.register_buffer('sqrt_recip_alphas', sqrt_recip_alphas)
        self.register_buffer('posterior_variance', posterior_variance)
        
        # Time embeddings
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbeddings(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Instrument embeddings
        self.instrument_embedding = nn.Embedding(num_instruments, hidden_dim)
        
        # UNet architecture
        # Down blocks
        self.down_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(input_dim if i == 0 else hidden_dim * (2 ** (i-1)), 
                         hidden_dim * (2 ** i), 
                         kernel_size=3, stride=2, padding=1),
                ConvBlock(hidden_dim * (2 ** i), hidden_dim * (2 ** i))
            )
            for i in range(3)  # 3 downsampling blocks
        ])
        
        # Middle block
        middle_dim = hidden_dim * (2 ** 2)  # 4x hidden_dim
        self.middle_block1 = ConvBlock(middle_dim, middle_dim)
        self.middle_attn = AttentionBlock(middle_dim)
        self.middle_block2 = ConvBlock(middle_dim, middle_dim)
        
        # Up blocks
        self.up_blocks = nn.ModuleList([
            nn.Sequential(
                ConvBlock(middle_dim if i == 0 else hidden_dim * (2 ** (2-i+1)), 
                         hidden_dim * (2 ** (2-i)), 
                         kernel_size=3, stride=1, padding=1),
                nn.Upsample(scale_factor=2, mode='nearest'),
                ConvBlock(hidden_dim * (2 ** (2-i)), hidden_dim * (2 ** (2-i)))
            )
            for i in range(3)  # 3 upsampling blocks
        ])
        
        # Final layers
        self.final_conv = nn.Sequential(
            ConvBlock(hidden_dim, hidden_dim),
            nn.Conv1d(hidden_dim, input_dim * num_instruments, kernel_size=1)
        )
        
        # Hierarchical attention for temporal coherence
        self.hierarchical_attention = HierarchicalAttention(
            input_dim=input_dim,
            output_dim=input_dim,
            num_heads=Config.num_heads
        )
        
        # Emotion-aware layer
        self.emotion_layer = EmotionAwareLayer(
            input_dim=input_dim,
            output_dim=input_dim
        )
        
    def forward(self, x, t, instrument_idx=None, emotion_idx=None):
        """
        Forward pass (denoising network).
        
        Args:
            x: Input tensor of shape (batch_size, num_instruments * input_dim, seq_length)
            t: Timestep indices of shape (batch_size,)
            instrument_idx: Optional instrument indices for conditional generation
            emotion_idx: Optional emotion indices for emotion-aware generation
            
        Returns:
            pred_noise: Predicted noise
        """
        batch_size = x.shape[0]
        
        # Time embedding
        t_emb = self.time_embedding(t)  # (batch_size, hidden_dim)
        
        # Reshape input for per-instrument processing if needed
        x_reshaped = x.view(batch_size, self.num_instruments, self.input_dim, -1)
        
        # Apply hierarchical attention for temporal coherence
        x_temporal = []
        for i in range(self.num_instruments):
            # Extract instrument track
            x_inst = x_reshaped[:, i]  # (batch_size, input_dim, seq_length)
            
            # Transpose for attention (batch_size, seq_length, input_dim)
            x_inst = x_inst.transpose(1, 2)
            
            # Apply hierarchical attention
            x_inst = self.hierarchical_attention(x_inst)
            
            # Apply emotion modulation if provided
            if emotion_idx is not None:
                x_inst = self.emotion_layer(x_inst, emotion_idx)
            
            # Transpose back (batch_size, input_dim, seq_length)
            x_inst = x_inst.transpose(1, 2)
            
            x_temporal.append(x_inst)
        
        # Concatenate processed instrument tracks
        x = torch.cat(x_temporal, dim=1)  # (batch_size, num_instruments * input_dim, seq_length)
        
        # Apply UNet
        # Down blocks
        skip_connections = []
        for down_block in self.down_blocks:
            x = down_block(x)
            skip_connections.append(x)
        
        # Middle blocks with attention
        x = self.middle_block1(x)
        
        # Reshape for attention
        b, c, s = x.shape
        x = x.transpose(1, 2).reshape(b, s, c)
        x = self.middle_attn(x)
        x = x.reshape(b, s, c).transpose(1, 2)
        
        x = self.middle_block2(x)
        
        # Up blocks with skip connections
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x)
            if i < len(skip_connections):
                # Add skip connection
                skip_x = skip_connections[-(i+1)]
                
                # Ensure compatible shapes (handle edge cases in sequence length)
                if x.size(2) != skip_x.size(2):
                    x = F.interpolate(x, size=skip_x.size(2), mode='nearest')
                    
                x = x + skip_x
        
        # Final layers
        pred_noise = self.final_conv(x)
        
        return pred_noise
    
    def q_sample(self, x_0, t):
        """
        Sample from q(x_t | x_0) (forward diffusion process).
        
        Args:
            x_0: Clean data
            t: Timestep indices
            
        Returns:
            x_t: Noisy data
            noise: Added noise
        """
        noise = torch.randn_like(x_0)
        
        # Extract coefficients for the given timesteps
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        
        # Sample from q(x_t | x_0)
        x_t = sqrt_alphas_cumprod_t * x_0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return x_t, noise
    
    @torch.no_grad()
    def p_sample(self, x_t, t, instrument_idx=None, emotion_idx=None):
        """
        Sample from p(x_{t-1} | x_t) (one step of the reverse diffusion process).
        
        Args:
            x_t: Current noisy data
            t: Current timestep index
            instrument_idx: Optional instrument indices for conditional generation
            emotion_idx: Optional emotion indices for emotion-aware generation
            
        Returns:
            x_{t-1}: Sample from p(x_{t-1} | x_t)
        """
        # Predict noise component
        pred_noise = self.forward(x_t, t, instrument_idx, emotion_idx)
        
        # Extract coefficients
        sqrt_recip_alphas_t = self.sqrt_recip_alphas[t].view(-1, 1, 1)
        betas_t = self.betas[t].view(-1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1)
        posterior_variance_t = self.posterior_variance[t].view(-1, 1, 1)
        
        # Compute the mean for p(x_{t-1} | x_t)
        p_mean = sqrt_recip_alphas_t * (x_t - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        
        # Add noise if t > 0, otherwise return the mean
        if t[0] > 0:
            noise = torch.randn_like(x_t)
            return p_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return p_mean
    
    @torch.no_grad()
    def sample(self, batch_size=1, instrument_idx=None, emotion_idx=None):
        """
        Sample new data by running the complete reverse diffusion process.
        
        Args:
            batch_size: Number of samples to generate
            instrument_idx: Optional instrument indices for conditional generation
            emotion_idx: Optional emotion indices for emotion-aware generation
            
        Returns:
            samples: Generated samples
        """
        device = next(self.parameters()).device
        
        # Start with random noise
        x = torch.randn(
            batch_size, 
            self.num_instruments * self.input_dim, 
            self.seq_length, 
            device=device
        )
        
        # Iteratively denoise
        for t_idx in reversed(range(0, self.num_steps)):
            t = torch.full((batch_size,), t_idx, device=device, dtype=torch.long)
            x = self.p_sample(x, t, instrument_idx, emotion_idx)
        
        return x


class GenerativeModule(nn.Module):
    """
    Generative module that generates audio from latent representations.
    Uses a transformer-based architecture with self-attention.
    """
    def __init__(
        self,
        input_dim=Config.latent_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=6,
        num_heads=Config.num_heads,
        dropout=Config.dropout
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding
        self.pos_encoder = nn.Parameter(torch.randn(1, 1000, hidden_dim))
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.Sequential(
                AttentionBlock(hidden_dim, num_heads, dropout),
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            for _ in range(num_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, input_dim)
            
        Returns:
            Output tensor of shape (batch_size, seq_len, input_dim)
        """
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pos_encoder[:, :x.size(1)]
        
        # Pass through transformer layers
        for layer in self.layers:
            x = x + layer(x)
        
        # Project to output space
        x = self.output_proj(x)
        
        return x
    
    def generate(self, z, max_length=1000):
        """
        Generate audio from latent representation.
        
        Args:
            z: Latent tensor of shape (batch_size, input_dim)
            max_length: Maximum sequence length to generate
            
        Returns:
            Generated audio tensor of shape (batch_size, max_length, input_dim)
        """
        batch_size = z.size(0)
        device = z.device
        
        # Initialize output tensor
        output = torch.zeros(batch_size, max_length, self.input_dim, device=device)
        
        # Set first token to latent representation
        output[:, 0] = z
        
        # Generate tokens autoregressively
        for t in range(1, max_length):
            # Get current sequence
            current = output[:, :t]
            
            # Generate next token
            next_token = self.forward(current)
            output[:, t] = next_token[:, -1]
        
        return output