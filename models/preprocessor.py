import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import Config
from utils import extract_features

class SourceSeparationBlock(nn.Module):
    """
    Source separation block that separates audio into different instrument tracks.
    Uses a U-Net architecture with attention mechanisms.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=len(Config.instruments),
        base_channels=32,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        
        # Encoder path
        self.encoder_blocks = nn.ModuleList()
        current_channels = input_channels
        
        for i in range(num_layers):
            out_channels = base_channels * (2 ** i)
            self.encoder_blocks.append(
                nn.Sequential(
                    nn.Conv2d(current_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(dropout)
                )
            )
            current_channels = out_channels
        
        # Decoder path
        self.decoder_blocks = nn.ModuleList()
        self.attention_blocks = nn.ModuleList()
        
        for i in range(num_layers - 1, -1, -1):
            in_channels = base_channels * (2 ** (i + 1))
            out_channels = base_channels * (2 ** i)
            
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, 4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Dropout2d(dropout)
                )
            )
            
            # Attention block
            self.attention_blocks.append(
                nn.Sequential(
                    nn.Conv2d(out_channels * 2, out_channels, 1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(out_channels, out_channels, 1),
                    nn.Sigmoid()
                )
            )
        
        # Final output layer
        self.final_conv = nn.Conv2d(base_channels, output_channels, 1)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Separated instrument tracks of shape (batch_size, num_instruments, height, width)
        """
        # Encoder path
        encoder_outputs = []
        current = x
        
        for encoder_block in self.encoder_blocks:
            current = encoder_block(current)
            encoder_outputs.append(current)
            if current.size(2) > 2:  # Don't downsample if too small
                current = F.max_pool2d(current, 2)
        
        # Decoder path
        current = encoder_outputs[-1]
        
        for i, (decoder_block, attention_block) in enumerate(zip(self.decoder_blocks, self.attention_blocks)):
            # Upsample
            current = decoder_block(current)
            
            # Skip connection with attention
            if i < len(encoder_outputs) - 1:
                skip = encoder_outputs[-(i + 2)]
                attention = attention_block(torch.cat([current, skip], 1))
                current = current * attention + skip * (1 - attention)
        
        # Final output
        output = self.final_conv(current)
        
        return output


class FeatureExtractor(nn.Module):
    """
    Neural feature extractor for extracting musical features from audio.
    """
    def __init__(
        self,
        input_dim=Config.n_mels,
        hidden_dim=Config.hidden_dim,
        dropout=Config.dropout
    ):
        super().__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2),
        )
        
        # Calculate output size after conv layers
        self.calc_conv_output_dim = self._get_conv_output_dim(input_dim)
        
        # Feature-specific heads
        self.chord_head = nn.Sequential(
            nn.Linear(self.calc_conv_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 24)  # 12 major + 12 minor chords
        )
        
        self.genre_head = nn.Sequential(
            nn.Linear(self.calc_conv_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 10)  # 10 common music genres
        )
        
        self.tempo_head = nn.Sequential(
            nn.Linear(self.calc_conv_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)  # Tempo as BPM
        )
        
        self.emotion_head = nn.Sequential(
            nn.Linear(self.calc_conv_output_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 8)  # 8 emotion categories
        )
    
    def _get_conv_output_dim(self, input_dim):
        """Calculate output dimensions after conv layers."""
        # Create a sample input to determine output size
        sample_input = torch.zeros(1, 1, input_dim, 400)  # Assuming 400 time frames
        
        # Pass through conv layers
        x = self.conv_layers(sample_input)
        
        return x.numel()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input spectrogram of shape (batch_size, freq, time)
            
        Returns:
            features: Dictionary of extracted features
        """
        # Ensure correct input shape
        if len(x.shape) == 3:  # (batch, freq, time)
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Pass through conv layers
        x = self.conv_layers(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Extract features
        chord_probs = F.softmax(self.chord_head(x), dim=1)
        genre_probs = F.softmax(self.genre_head(x), dim=1)
        tempo = F.relu(self.tempo_head(x))  # Tempo should be positive
        emotion_probs = F.softmax(self.emotion_head(x), dim=1)
        
        features = {
            'chord_progressions': chord_probs,
            'genre': genre_probs,
            'tempo': tempo,
            'emotion': emotion_probs
        }
        
        return features


class TemporalAnalyzer(nn.Module):
    """
    Temporal analyzer to identify long-term patterns in music.
    Uses a transformer-based approach to model temporal dependencies.
    """
    def __init__(
        self,
        input_dim=Config.latent_dim,
        hidden_dim=Config.hidden_dim,
        num_layers=4,
        num_heads=Config.num_heads,
        dropout=Config.dropout
    ):
        super().__init__()
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True
        )
        
        self.transformer = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers
        )
        
        # Motif detection head
        self.motif_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim)
        )
        
    def forward(self, x, mask=None):
        """
        Forward pass.
        
        Args:
            x: Input sequence of shape (batch_size, seq_len, features)
            mask: Optional attention mask
            
        Returns:
            temporal_features: Enhanced temporal features
            motifs: Detected motifs
        """
        # Pass through transformer
        temporal_features = self.transformer(x, src_key_padding_mask=mask)
        
        # Detect motifs
        motifs = self.motif_head(temporal_features)
        
        return temporal_features, motifs


class PreProcessingModule(nn.Module):
    """
    Complete pre-processing module that integrates source separation,
    feature extraction, and temporal analysis.
    """
    def __init__(
        self,
        source_separation=None,
        feature_extractor=None,
        temporal_analyzer=None
    ):
        super().__init__()
        
        self.source_separation = source_separation if source_separation else SourceSeparationBlock()
        self.feature_extractor = feature_extractor if feature_extractor else FeatureExtractor()
        self.temporal_analyzer = temporal_analyzer if temporal_analyzer else TemporalAnalyzer()
    
    def forward(self, mixed_audio, spectrograms=None):
        """
        Forward pass.
        
        Args:
            mixed_audio: Mixed audio input
            spectrograms: Optional pre-computed spectrograms
            
        Returns:
            processed: Dictionary containing all processed outputs
        """
        # Source separation
        if spectrograms is None:
            # Compute spectrograms if not provided
            if isinstance(mixed_audio, torch.Tensor):
                audio_np = mixed_audio.cpu().numpy()
            else:
                audio_np = mixed_audio
            
            # Compute spectrogram
            spectrogram = librosa.feature.melspectrogram(
                y=audio_np.squeeze()[0] if len(audio_np.shape) > 1 else audio_np,
                sr=Config.sample_rate,
                n_fft=Config.n_fft,
                hop_length=Config.hop_length,
                n_mels=Config.n_mels
            )
            
            # Convert to log scale
            spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
            
            # Convert to tensor
            spectrogram = torch.tensor(spectrogram, dtype=torch.float).unsqueeze(0).unsqueeze(0)
        else:
            spectrogram = spectrograms
        
        # Apply source separation
        separated_sources = self.source_separation(spectrogram)
        
        # Extract features for each separated source
        features = {}
        for instrument, source in separated_sources.items():
            features[instrument] = self.feature_extractor(source)
        
        # Combine features across time for temporal analysis
        # For simplicity, we'll just use the piano features here
        piano_features = features.get('piano', next(iter(features.values())))
        chord_probs = piano_features['chord_progressions']
        
        # Create a sequence of chord probabilities
        seq_len = chord_probs.size(0)
        chord_sequence = chord_probs.view(1, seq_len, -1)  # (1, seq_len, num_chords)
        
        # Apply temporal analysis
        temporal_features, motifs = self.temporal_analyzer(chord_sequence)
        
        processed = {
            'separated_sources': separated_sources,
            'features': features,
            'temporal_features': temporal_features,
            'motifs': motifs
        }
        
        return processed 