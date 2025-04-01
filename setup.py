import subprocess
import sys
import os

def install_dependencies():
    # Get the directory containing setup.py
    current_dir = os.path.dirname(os.path.abspath(__file__))
    requirements_path = os.path.join(current_dir, 'requirements.txt')
    
    # First, uninstall existing torch packages to avoid conflicts
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "torch", "torchaudio", "torchvision", "pytorch-lightning"])
    
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

if __name__ == "__main__":
    print("Setting up CUDA environment...")
    setup_cuda()
    
    print("Installing dependencies...")
    install_dependencies()
    
    print("Setup completed successfully!") 