import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
from pathlib import Path
import json
import librosa

import config
from utils import load_audio, extract_features, compute_mel_spectrogram, get_track_dirs, get_project_root
from config.config import Config

class AudioTransforms:
    def __init__(self):
        self.sample_rate = Config.sample_rate
        self.n_fft = Config.n_fft
        self.hop_length = Config.hop_length
        self.n_mels = Config.n_mels
    
    def __call__(self, waveform):
        # Apply any audio transformations here
        return waveform

class SlakhDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Get all track directories
        self.track_dirs = get_track_dirs(self.data_dir, split)
        
        # Load instrument mappings
        self.instruments = Config.instruments
    
    def __len__(self):
        return len(self.track_dirs)
    
    def __getitem__(self, idx):
        track_dir = self.track_dirs[idx]
        
        # Load mixed audio
        mixed_path = track_dir / 'mixed' / 'audio.wav'
        mixed_audio = load_audio(mixed_path, Config.sample_rate)
        
        # Load individual instrument audio
        audio_dict = {}
        for instrument in self.instruments:
            instrument_path = track_dir / instrument / 'audio.wav'
            if instrument_path.exists():
                audio = load_audio(instrument_path, Config.sample_rate)
                audio_dict[instrument] = audio
        
        # Apply transforms if any
        if self.transform:
            mixed_audio = self.transform(mixed_audio)
            for instrument in audio_dict:
                audio_dict[instrument] = self.transform(audio_dict[instrument])
        
        return {
            'mixed_audio': mixed_audio,
            'audio': audio_dict,
            'sample_rate': Config.sample_rate,
            'path': str(mixed_path)
        }

def create_dataloader(data_dir, batch_size, split='train', num_workers=4, drop_last=True):
    """Create data loader for the Slakh2100 dataset."""
    # Get project root to handle paths correctly
    project_root = get_project_root()
    data_dir = project_root / data_dir
    
    # Create dataset
    transform = AudioTransforms()
    dataset = SlakhDataset(data_dir, split=split, transform=transform)
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    
    return loader

class Slakh2100Dataset(Dataset):
    """Dataset for Slakh2100 multi-track audio."""
    
    def __init__(
        self, 
        data_dir, 
        segment_duration=config.SEGMENT_DURATION,
        sample_rate=config.SAMPLE_RATE,
        instruments=config.INSTRUMENTS,
        transform=None,
        split="train",
        random_segments=True
    ):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing the Slakh2100 dataset
            segment_duration: Duration of audio segments in seconds
            sample_rate: Target sample rate
            instruments: List of instrument tracks to load
            transform: Optional transform to apply to the data
            split: Dataset split ('train', 'validation', or 'test')
            random_segments: Whether to use random segments during training
        """
        self.data_dir = Path(data_dir)
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.instruments = instruments
        self.transform = transform
        self.split = split
        self.random_segments = random_segments
        
        # Get track directories
        self.track_dirs = get_track_dirs(self.data_dir / split)
        print(f"Found {len(self.track_dirs)} tracks in {split} set.")
        
        # Pre-compute metadata for faster loading
        self.metadata = self._load_metadata()
        
    def _load_metadata(self):
        """Load and precompute metadata for all tracks."""
        metadata = []
        
        for track_dir in self.track_dirs:
            track_metadata = {}
            track_metadata["path"] = track_dir
            
            # Check that all instrument files exist
            all_instruments_exist = True
            instrument_durations = {}
            
            for instrument in self.instruments:
                instrument_path = track_dir / f"{instrument}.wav"
                if not instrument_path.exists():
                    print(f"Warning: {instrument}.wav not found in {track_dir}")
                    all_instruments_exist = False
                    break
                
                # Get audio duration
                try:
                    info = torchaudio.info(instrument_path)
                    duration = info.num_frames / info.sample_rate
                    instrument_durations[instrument] = duration
                except Exception as e:
                    print(f"Error reading {instrument_path}: {e}")
                    all_instruments_exist = False
                    break
            
            if not all_instruments_exist:
                continue
            
            # Use minimum duration across all instruments
            min_duration = min(instrument_durations.values())
            track_metadata["duration"] = min_duration
            
            # Calculate number of possible segments
            if min_duration >= self.segment_duration:
                num_segments = int(min_duration // self.segment_duration)
                track_metadata["num_segments"] = num_segments
                metadata.append(track_metadata)
            else:
                print(f"Skipping {track_dir}: duration {min_duration}s < segment {self.segment_duration}s")
        
        return metadata
    
    def __len__(self):
        """Return the number of segments in the dataset."""
        return sum(m["num_segments"] for m in self.metadata)
    
    def __getitem__(self, idx):
        """Get the idx-th segment."""
        # Find which track this segment belongs to
        track_idx = 0
        while idx >= self.metadata[track_idx]["num_segments"]:
            idx -= self.metadata[track_idx]["num_segments"]
            track_idx += 1
        
        track_metadata = self.metadata[track_idx]
        track_dir = track_metadata["path"]
        
        # Determine segment offset
        if self.random_segments and self.split == "train":
            # For training, use random segments
            max_offset = track_metadata["duration"] - self.segment_duration
            offset = random.uniform(0, max_offset)
        else:
            # For validation and test, use fixed segments
            offset = idx * self.segment_duration
        
        # Load audio for each instrument
        audio_data = {}
        mel_spectrograms = {}
        features = {}
        
        for instrument in self.instruments:
            instrument_path = track_dir / f"{instrument}.wav"
            audio = load_audio(
                instrument_path, 
                sample_rate=self.sample_rate,
                duration=self.segment_duration,
                offset=offset
            )
            
            if audio is None:
                # Return a zero tensor if audio loading failed
                audio = torch.zeros(1, int(self.segment_duration * self.sample_rate))
            
            # Ensure correct shape
            if audio.shape[1] < int(self.segment_duration * self.sample_rate):
                # Pad if necessary
                padding = int(self.segment_duration * self.sample_rate) - audio.shape[1]
                audio = torch.nn.functional.pad(audio, (0, padding))
            
            audio_data[instrument] = audio
            
            # Compute mel spectrogram
            mel_spec = compute_mel_spectrogram(audio)
            mel_spectrograms[instrument] = mel_spec
            
            # Extract features
            inst_features = extract_features(audio, sample_rate=self.sample_rate)
            features[instrument] = inst_features
        
        # Create a mixed track by summing all instruments
        mixed_audio = sum(audio_data.values())
        
        # Apply transformations if any
        if self.transform:
            mixed_audio = self.transform(mixed_audio)
            
        # Compute mel spectrogram for mixed audio
        mixed_mel = compute_mel_spectrogram(mixed_audio)
        
        # Extract features for mixed audio
        mixed_features = extract_features(mixed_audio, sample_rate=self.sample_rate)
        
        return {
            "track_id": track_dir.name,
            "offset": offset,
            "audio": audio_data,
            "mixed_audio": mixed_audio,
            "mel_spectrograms": mel_spectrograms,
            "mixed_mel": mixed_mel,
            "features": features,
            "mixed_features": mixed_features
        }

def create_dataloader(
    data_dir, 
    batch_size=config.BATCH_SIZE, 
    split="train", 
    num_workers=4, 
    pin_memory=True,
    drop_last=True
):
    """Create a dataloader for the Slakh2100 dataset."""
    dataset = Slakh2100Dataset(
        data_dir=data_dir,
        split=split,
        random_segments=(split == "train")
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == "train"),
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last
    )
    
    return dataloader

class AudioTransforms:
    """Audio transformation class for data augmentation."""
    
    def __init__(self, sample_rate=config.SAMPLE_RATE, p=0.5):
        """Initialize audio transforms."""
        self.sample_rate = sample_rate
        self.p = p
        
    def time_stretch(self, audio, rate_range=(0.8, 1.2)):
        """Apply time stretching."""
        if random.random() < self.p:
            rate = random.uniform(*rate_range)
            audio_np = audio.squeeze().numpy()
            stretched = librosa.effects.time_stretch(audio_np, rate=rate)
            
            # Ensure output length is same as input
            if len(stretched) > len(audio_np):
                stretched = stretched[:len(audio_np)]
            else:
                stretched = np.pad(stretched, (0, max(0, len(audio_np) - len(stretched))))
                
            return torch.tensor(stretched, dtype=torch.float).unsqueeze(0)
        return audio
    
    def pitch_shift(self, audio, n_steps_range=(-2, 2)):
        """Apply pitch shifting."""
        if random.random() < self.p:
            n_steps = random.uniform(*n_steps_range)
            audio_np = audio.squeeze().numpy()
            shifted = librosa.effects.pitch_shift(
                audio_np, 
                sr=self.sample_rate, 
                n_steps=n_steps
            )
            return torch.tensor(shifted, dtype=torch.float).unsqueeze(0)
        return audio
    
    def add_noise(self, audio, noise_level_range=(0.001, 0.005)):
        """Add random noise."""
        if random.random() < self.p:
            noise_level = random.uniform(*noise_level_range)
            noise = torch.randn_like(audio) * noise_level
            return audio + noise
        return audio
    
    def __call__(self, audio):
        """Apply transformations."""
        audio = self.time_stretch(audio)
        audio = self.pitch_shift(audio)
        audio = self.add_noise(audio)
        return audio 