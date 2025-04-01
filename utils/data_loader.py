import torch
from torch.utils.data import Dataset
import os
import torchaudio
from config.config import Config

class SlakhDataset(Dataset):
    def __init__(self, data_dir, split='train'):
        self.data_dir = os.path.join(data_dir, split)
        self.tracks = sorted([d for d in os.listdir(self.data_dir) if d.startswith('Track')])
        
    def __len__(self):
        return len(self.tracks)
        
    def __getitem__(self, idx):
        track = self.tracks[idx]
        track_dir = os.path.join(self.data_dir, track)
        
        # Load audio files
        audio_files = {}
        for instrument in ['bass', 'drums', 'guitar', 'piano']:
            file_path = os.path.join(track_dir, f'{instrument}.wav')
            waveform, sample_rate = torchaudio.load(file_path)
            if sample_rate != Config.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sample_rate, Config.sample_rate)
            audio_files[instrument] = waveform
            
        return audio_files 