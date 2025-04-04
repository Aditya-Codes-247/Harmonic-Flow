import os
import torch
from pathlib import Path

# Dataset paths
DATA_ROOT = Path("data/slakh2100")
TRAIN_DIR = DATA_ROOT / "train"
VALID_DIR = DATA_ROOT / "validation"
TEST_DIR = DATA_ROOT / "test"

# Audio processing
SAMPLE_RATE = 22050
HOP_LENGTH = 512
N_FFT = 2048
N_MELS = 128
SEGMENT_DURATION = 10  # seconds
AUDIO_LENGTH = SAMPLE_RATE * SEGMENT_DURATION

# Model parameters
LATENT_DIM = 256
NUM_HEADS = 8
NUM_LAYERS = 12
HIDDEN_DIM = 1024
DROPOUT = 0.1
LEARNING_RATE = 5e-5
ADAM_BETAS = (0.9, 0.999)
WEIGHT_DECAY = 1e-6

# Training parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
GRADIENT_ACCUMULATION_STEPS = 2
WARM_UP_STEPS = 1000
VAL_CHECK_INTERVAL = 0.25
CHECKPOINT_DIR = "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Random seed for reproducibility
SEED = 42

# Instrument tracks
INSTRUMENTS = ["bass", "drums", "guitar", "piano"]

# Feature extraction
FEATURE_KEYS = [
    "tempo", 
    "pitch_distribution", 
    "timbral_features", 
    "rhythmic_patterns", 
    "chord_progressions"
]

# Multi-modal settings
TEXT_EMBEDDING_DIM = 768
VISUAL_EMBEDDING_DIM = 1024

# Diffusion model settings
# Define diffusion_steps at the module level instead of in a Config class
diffusion_steps = 1000  # This can now be accessed as config.diffusion_steps
DIFFUSION_BETAS = (1e-4, 0.02)

# Post-processing parameters
REVERB_SETTINGS = {
    "room_size": 0.7,
    "damping": 0.5,
    "wet_level": 0.3,
    "dry_level": 0.8
}

# Create necessary directories
os.makedirs(CHECKPOINT_DIR, exist_ok=True)