import os
import torch
import torchaudio
import librosa
import numpy as np
from pathlib import Path

def load_audio(file_path, sample_rate=44100):
    """Load audio file and resample if necessary."""
    waveform, sr = torchaudio.load(file_path)
    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
    return waveform

def extract_features(waveform, sample_rate=44100, n_fft=2048, hop_length=512):
    """Extract audio features from waveform."""
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    # Compute spectrogram
    spec = torchaudio.transforms.Spectrogram(
        n_fft=n_fft,
        hop_length=hop_length,
        power=2.0
    )(waveform)
    
    return spec

def compute_mel_spectrogram(waveform, sample_rate=44100, n_mels=128, n_fft=2048, hop_length=512):
    """Compute mel spectrogram from waveform."""
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        n_mels=n_mels
    )
    
    # Convert to mono if stereo
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)
    
    mel_spec = mel_transform(waveform)
    return mel_spec

def get_track_dirs(data_dir, split='train'):
    """Get all track directories for a given split."""
    split_dir = Path(data_dir) / split
    if not split_dir.exists():
        raise ValueError(f"Split directory {split_dir} does not exist")
    
    track_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    return sorted(track_dirs) 