import torch

class Config:
    # Training parameters
    batch_size = 32
    num_workers = 4
    epochs = 100
    lr = 0.001
    weight_decay = 1e-5
    
    # Model parameters
    input_dim = 512
    latent_dim = 256
    hidden_dim = 512
    
    # Audio parameters
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    segment_duration = 4.0  # Duration in seconds
    
    # Dataset parameters
    instruments = ['bass', 'drums', 'guitar', 'piano']
    
    # Paths
    data_dir = 'data/slakh2100'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    plot_dir = 'plots'
    
    # Constants for backwards compatibility
    SEGMENT_DURATION = 4.0
    SAMPLE_RATE = 44100
    BATCH_SIZE = 32
    INSTRUMENTS = ['bass', 'drums', 'guitar', 'piano']
    LATENT_DIM = 256
    HIDDEN_DIM = 512
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 