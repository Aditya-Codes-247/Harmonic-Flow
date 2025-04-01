import os
import json
import random
import numpy as np
import torch
import torchaudio
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import config

def set_seed(seed):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    """Save model checkpoint."""
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss
    }
    torch.save(checkpoint, path)

def load_checkpoint(model, optimizer=None, scheduler=None, path=None):
    """Load model checkpoint."""
    if not path or not os.path.exists(path):
        return 0, 0, float('inf')
    
    checkpoint = torch.load(path, map_location=config.DEVICE)
    model.load_state_dict(checkpoint['model'])
    
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler and checkpoint['scheduler'] and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def load_audio(file_path, sample_rate=config.SAMPLE_RATE, duration=None, offset=0.0):
    """Load audio file and resample if necessary."""
    try:
        if duration:
            waveform, sr = torchaudio.load(file_path, frame_offset=int(offset*sample_rate), 
                                          num_frames=int(duration*sample_rate))
        else:
            waveform, sr = torchaudio.load(file_path)
        
        # Convert to mono if stereo
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        # Resample if necessary
        if sr != sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, sample_rate)
            
        return waveform
    except Exception as e:
        print(f"Error loading audio {file_path}: {e}")
        return None

def extract_features(audio, sample_rate=config.SAMPLE_RATE):
    """Extract musical features from audio."""
    # Convert to numpy for librosa
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    features = {}
    
    # Extract tempo and beat information
    tempo, beat_frames = librosa.beat.beat_track(y=audio_np, sr=sample_rate)
    features['tempo'] = tempo
    features['beat_frames'] = librosa.frames_to_time(beat_frames, sr=sample_rate).tolist()
    
    # Extract pitch distribution using chroma features
    chroma = librosa.feature.chroma_cqt(y=audio_np, sr=sample_rate)
    features['pitch_distribution'] = chroma.mean(axis=1).tolist()
    
    # Extract timbral features
    mfcc = librosa.feature.mfcc(y=audio_np, sr=sample_rate, n_mfcc=13)
    features['timbral_features'] = {
        'mfcc_mean': mfcc.mean(axis=1).tolist(),
        'mfcc_std': mfcc.std(axis=1).tolist()
    }
    
    # Extract rhythmic patterns
    onset_env = librosa.onset.onset_strength(y=audio_np, sr=sample_rate)
    features['rhythmic_patterns'] = librosa.feature.tempogram(onset_envelope=onset_env, sr=sample_rate).mean(axis=1).tolist()
    
    # Extract spectral features
    spectral_centroid = librosa.feature.spectral_centroid(y=audio_np, sr=sample_rate)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio_np, sr=sample_rate)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_np, sr=sample_rate)
    
    features['spectral_features'] = {
        'centroid_mean': float(spectral_centroid.mean()),
        'bandwidth_mean': float(spectral_bandwidth.mean()),
        'rolloff_mean': float(spectral_rolloff.mean())
    }
    
    # Estimate chord progressions (simplified)
    chroma_cqt = librosa.feature.chroma_cqt(y=audio_np, sr=sample_rate)
    frames_per_chord = 20  # Arbitrary number of frames to estimate a chord
    num_chords = chroma_cqt.shape[1] // frames_per_chord
    
    chord_progression = []
    for i in range(num_chords):
        chord_segment = chroma_cqt[:, i*frames_per_chord:(i+1)*frames_per_chord]
        chord_notes = np.mean(chord_segment, axis=1)
        chord_progression.append(chord_notes.tolist())
    
    features['chord_progressions'] = chord_progression
    
    return features

def compute_mel_spectrogram(audio, n_mels=config.N_MELS, n_fft=config.N_FFT, hop_length=config.HOP_LENGTH, sample_rate=config.SAMPLE_RATE):
    """Compute mel spectrogram from audio."""
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio

    # Compute mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio_np, 
        sr=sample_rate, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        n_mels=n_mels
    )
    
    # Convert to log scale
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    return torch.tensor(log_mel_spec, dtype=torch.float)

def visualize_spectrogram(audio, title="Mel Spectrogram", save_path=None):
    """Visualize and optionally save a spectrogram."""
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    plt.figure(figsize=(10, 4))
    mel_spec = librosa.feature.melspectrogram(y=audio_np, sr=config.SAMPLE_RATE)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    librosa.display.specshow(
        mel_spec_db, 
        sr=config.SAMPLE_RATE, 
        x_axis='time', 
        y_axis='mel', 
        cmap='viridis'
    )
    
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def source_separation(mixed_audio):
    """Placeholder for source separation using a pre-trained model like Demucs."""
    # In a real implementation, this would use a model like Demucs or similar
    # For now, we'll return a dictionary with the original audio duplicated for each track
    if isinstance(mixed_audio, torch.Tensor):
        audio_tensor = mixed_audio
    else:
        audio_tensor = torch.tensor(mixed_audio)
    
    tracks = {}
    for instrument in config.INSTRUMENTS:
        # This is a placeholder - actual source separation would go here
        tracks[instrument] = audio_tensor.clone()
    
    return tracks

def get_track_dirs(data_dir):
    """Get a list of track directories."""
    return [p for p in Path(data_dir).iterdir() if p.is_dir() and p.name.startswith('Track')]

def log_metrics(writer, metrics, step, prefix='train'):
    """Log metrics to TensorBoard."""
    for key, value in metrics.items():
        writer.add_scalar(f'{prefix}/{key}', value, step)

def normalize_audio(audio, target_level=-25):
    """Normalize audio to a target RMS level."""
    if isinstance(audio, torch.Tensor):
        audio_np = audio.squeeze().numpy()
    else:
        audio_np = audio
    
    # Calculate current RMS
    rms = np.sqrt(np.mean(audio_np**2))
    
    # Calculate target RMS based on target level in dB
    target_rms = 10**(target_level/20)
    
    # Apply gain
    if rms > 0:
        gain = target_rms / rms
        normalized_audio = audio_np * gain
    else:
        normalized_audio = audio_np
    
    return torch.tensor(normalized_audio, dtype=torch.float) if isinstance(audio, torch.Tensor) else normalized_audio 