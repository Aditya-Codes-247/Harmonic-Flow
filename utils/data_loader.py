import os
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from pathlib import Path
from config.config import Config

class SlakhDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None):
        """
        Initialize the Slakh2100 dataset.
        
        Args:
            root_dir (str): Root directory of the dataset
            split (str): Dataset split ('train', 'validation', or 'test')
            transform: Optional transform to apply to the audio
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.transform = transform
        
        # Get all audio files in the split directory
        self.audio_files = list(self.root_dir / split / 'mixed' / 'audio').glob('*.wav')
        
        # Load instrument mappings
        self.instruments = ['bass', 'drums', 'guitar', 'piano']
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        # Get mixed audio path
        mixed_audio_path = self.audio_files[idx]
        
        # Get corresponding instrument audio paths
        audio_dict = {}
        for instrument in self.instruments:
            instrument_path = mixed_audio_path.parent.parent / instrument / 'audio' / mixed_audio_path.name
            if instrument_path.exists():
                audio_dict[instrument] = instrument_path
        
        # Load mixed audio
        mixed_audio, sample_rate = torchaudio.load(mixed_audio_path)
        
        # Load individual instrument audio
        instrument_audio = {}
        for instrument, path in audio_dict.items():
            audio, _ = torchaudio.load(path)
            instrument_audio[instrument] = audio
        
        # Apply transforms if any
        if self.transform:
            mixed_audio = self.transform(mixed_audio)
            for instrument in instrument_audio:
                instrument_audio[instrument] = self.transform(instrument_audio[instrument])
        
        return {
            'mixed_audio': mixed_audio,
            'audio': instrument_audio,
            'sample_rate': sample_rate,
            'path': str(mixed_audio_path)
        } 