import torch

class Config:
    # Audio parameters
    sample_rate = 22050
    n_fft = 2048
    hop_length = 512
    n_mels = 80
    
    # Model parameters
    latent_dim = 256
    hidden_dim = 1024
    
    # Training parameters
    batch_size = 8
    num_workers = 4
    epochs = 100
    lr = 5e-5
    weight_decay = 1e-6
    grad_accum_steps = 4
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu' 