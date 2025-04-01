import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

def plot_spectrogram(spectrogram, title="Spectrogram", save_path=None):
    """Plot a spectrogram."""
    plt.figure(figsize=(10, 4))
    plt.imshow(spectrogram, aspect='auto', origin='lower')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_waveform(waveform, sample_rate, title="Waveform", save_path=None):
    """Plot a waveform."""
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.numpy())
    plt.title(title)
    plt.xlabel('Sample')
    plt.ylabel('Amplitude')
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_attention_weights(attention_weights, title="Attention Weights", save_path=None):
    """Plot attention weights."""
    plt.figure(figsize=(8, 8))
    sns.heatmap(attention_weights, cmap='viridis')
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def plot_latent_space(latent_vectors, labels=None, title="Latent Space Visualization", save_path=None):
    """Plot latent space vectors using t-SNE."""
    from sklearn.manifold import TSNE
    
    # Convert to numpy if tensor
    if torch.is_tensor(latent_vectors):
        latent_vectors = latent_vectors.detach().cpu().numpy()
    
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    latent_2d = tsne.fit_transform(latent_vectors)
    
    # Plot
    plt.figure(figsize=(10, 8))
    if labels is not None:
        scatter = plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=labels, cmap='tab10')
        plt.colorbar(scatter)
    else:
        plt.scatter(latent_2d[:, 0], latent_2d[:, 1])
    
    plt.title(title)
    if save_path:
        plt.savefig(save_path)
    plt.close()

def log_tensorboard_metrics(writer, metrics, step):
    """Log metrics to TensorBoard."""
    for name, value in metrics.items():
        writer.add_scalar(name, value, step)

def plot_training_history(history, save_path=None):
    """Plot training history metrics."""
    plt.figure(figsize=(12, 4))
    
    # Plot losses
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    # Plot accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close() 