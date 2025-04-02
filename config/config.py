import torch

class Config:
    """Configuration class for the HarmonicFlow model."""
    
    # Training parameters
    batch_size = 32
    num_workers = 4
    epochs = 100
    lr = 0.001
    weight_decay = 0.0001
    grad_accum_steps = 1
    val_check_interval = 0.25
    seed = 42
    
    # Model parameters
    input_dim = 128  # Number of mel bands
    latent_dim = 256
    hidden_dim = 512
    
    # Audio parameters
    sample_rate = 44100
    n_fft = 2048
    hop_length = 512
    n_mels = 128
    segment_duration = 4.0  # Duration of audio segments in seconds
    
    # Dataset parameters
    instruments = [
        "drums",
        "bass",
        "piano",
        "guitar",
        "strings",
        "synth",
        "vocal",
        "other"
    ]
    
    # Paths
    data_dir = "data/slakh2100"
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    plot_dir = "plots"
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Constants for backwards compatibility
    SEGMENT_DURATION = segment_duration
    SAMPLE_RATE = sample_rate
    BATCH_SIZE = batch_size
    INSTRUMENTS = instruments
    LATENT_DIM = latent_dim
    HIDDEN_DIM = hidden_dim
    N_MELS = n_mels 