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
    
    # Dataset parameters
    instruments = ['bass', 'drums', 'guitar', 'piano']
    
    # Paths
    data_dir = 'data/slakh2100'
    checkpoint_dir = 'checkpoints'
    log_dir = 'logs'
    plot_dir = 'plots'
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 