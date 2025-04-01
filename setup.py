import subprocess
import sys
import os

def install_dependencies():
    # Get the directory containing setup.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    
    # First, uninstall existing packages to avoid conflicts
    packages_to_remove = [
        "torch", "torchaudio", "torchvision", "pytorch-lightning",
        "numpy", "tensorboard", "tqdm", "matplotlib", "scikit-learn",
        "transformers", "pillow"
    ]
    
    for package in packages_to_remove:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", package])
        except subprocess.CalledProcessError:
            pass  # Package might not be installed
    
    # Install PyTorch with CUDA support
    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "torch==2.0.0+cu117",
        "torchaudio==2.0.0+cu117",
        "torchvision==0.15.0+cu117",
        "--extra-index-url", "https://download.pytorch.org/whl/cu117"
    ])
    
    # Install other dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", requirements_path])

def setup_cuda():
    # Set CUDA environment variables to avoid duplicate registration
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TF logging
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use first GPU

def create_directories():
    # Create necessary directories
    directories = ['checkpoints', 'logs', 'plots', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

if __name__ == "__main__":
    print("Setting up CUDA environment...")
    setup_cuda()
    
    print("Creating necessary directories...")
    create_directories()
    
    print("Installing dependencies...")
    install_dependencies()
    
    print("Setup completed successfully!") 