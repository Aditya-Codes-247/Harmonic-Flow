import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import librosa
import numpy as np

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from utils import extract_features

class SourceSeparationBlock(nn.Module):
    """
    Source separation block to separate audio into constituent tracks.
    This implementation provides a basic UNet-based approach for source separation.
    In a production environment, you would integrate a pre-trained model like Demucs.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=len(config.INSTRUMENTS),
        base_channels=32,
        depth=5
    ):
        super().__init__()
        
        # Encoder (downsampling)
        self.encoder = nn.ModuleList()
        self.encoder.append(nn.Conv2d(input_channels, base_channels, kernel_size=7, padding=3))
        
        for i in range(depth - 1):
            in_channels = base_channels * (2 ** i)
            out_channels = base_channels * (2 ** (i + 1))
            self.encoder.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        # Decoder (upsampling)
        self.decoder = nn.ModuleList()
        for i in range(depth - 1):
            in_channels = base_channels * (2 ** (depth - i - 1))
            out_channels = base_channels * (2 ** (depth - i - 2))
            self.decoder.append(
                nn.Sequential(
                    nn.LeakyReLU(0.2),
                    nn.ConvTranspose2d(in_channels * 2, out_channels, kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels)
                )
            )
        
        # Final layer
        self.final = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(base_channels * 2, output_channels, kernel_size=7, padding=3),
            nn.Tanh()
        )
        
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input mixed audio spectrogram of shape (batch_size, channels, freq, time)
            
        Returns:
            separated: Dictionary of separated tracks
        """
        # Store encoder outputs for skip connections
        encoder_outputs = []
        
        # Encoder forward pass
        for enc_layer in self.encoder:
            x = enc_layer(x)
            encoder_outputs.append(x)
        
        # Decoder forward pass with skip connections
        for i, dec_layer in enumerate(self.decoder):
            skip_connection = encoder_outputs[-(i + 2)]  # Skip connection from encoder
            x = torch.cat([x, skip_connection], dim=1)  # Concatenate along channel dimension
            x = dec_layer(x)
        
        # Final layer
        x = torch.cat([x, encoder_outputs[0]], dim=1)  # Skip connection from first encoder layer
        mask = self.final(x)  # (batch_size, num_instruments, freq, time)
        
        # Apply mask to input to get separated sources
        separated = {}
        for i, instrument in enumerate(config.INSTRUMENTS):
            separated[instrument] = mask[:, i:i+1, :, :] * x
        
        return separated


class FeatureExtractor(nn.Module):
    """
    Neural feature extractor for extracting musical features from audio.
    """
    def __init__(
        self,
        input_dim=config.N_MELS,
        hidden_dim=config.HIDDEN_DIM,
        dropout=config.DROPOUT
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
        input_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM,
        num_layers=4,
        num_heads=config.NUM_HEADS,
        dropout=config.DROPOUT
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
                sr=config.SAMPLE_RATE,
                n_fft=config.N_FFT,
                hop_length=config.HOP_LENGTH,
                n_mels=config.N_MELS
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