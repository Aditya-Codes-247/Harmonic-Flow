# This file makes the utils directory a Python package 

import os
import random
import torch
import numpy as np
from pathlib import Path

from .audio import load_audio, extract_features, compute_mel_spectrogram, get_track_dirs

def set_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def get_project_root():
    """Get the absolute path to the project root directory."""
    return Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'epoch': epoch,
        'step': step,
        'loss': loss
    }, path)

def load_checkpoint(model, optimizer, scheduler, path):
    """Load model checkpoint."""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint['epoch'], checkpoint['step'], checkpoint['loss']

def log_metrics(writer, metrics, step, prefix=''):
    """Log metrics to tensorboard."""
    for name, value in metrics.items():
        writer.add_scalar(f'{prefix}/{name}', value, step)

def visualize_spectrogram(spec, title=None, save_path=None):
    """Visualize spectrogram."""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(10, 4))
    plt.imshow(spec, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    if title:
        plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show() 