import os
import gdown
import subprocess
from tqdm import tqdm

def setup_dataset():
    # Create data directory
    os.makedirs('data', exist_ok=True)
    
    # Download test set
    file_id = '1Xo-bGORndJhenHvzf3eY5Lt0XCTgElZP'
    output = 'data/slakh2100-testset-22050.tar.xz'
    
    print("Downloading test set...")
    url = f'https://drive.google.com/uc?id={file_id}'
    gdown.download(url, output, quiet=False)
    
    # Verify file exists and check its size
    if os.path.exists(output):
        size = os.path.getsize(output)
        print(f"Downloaded file size: {size / (1024*1024*1024):.2f} GB")
        
        # Extract the test set
        print("Extracting test set...")
        subprocess.run(['tar', '-xf', output, '-C', 'data'], check=True)
        print("Extraction successful!")
    else:
        print("Download failed - file not found")

if __name__ == "__main__":
    setup_dataset() 