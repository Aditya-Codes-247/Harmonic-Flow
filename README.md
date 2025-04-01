# HarmonicFlow: Audio-to-Music Generative AI Model

HarmonicFlow is a state-of-the-art generative AI model for multi-track music generation and processing. It can generate coherent multi-instrument music tracks, perform style transfer between genres, and create immersive spatial audio outputs.

## Features

- **Multi-modal Input**: Process audio, text prompts, and visual references to generate music
- **Multi-track Generation**: Create coherent bass, drums, guitar, and piano tracks simultaneously
- **Dynamic Latent Space**: Adaptively improve outputs based on user feedback
- **Neural Audio Graphs**: Model relationships between instruments and musical elements
- **Hierarchical Attention**: Process music at multiple temporal scales for structural coherence
- **Emotion-Aware Generation**: Adjust musical parameters to reflect desired emotional tones
- **Style Transfer**: Apply stylistic characteristics of specific composers, eras, or genres
- **Genre-Adaptive Layers**: Fine-tune outputs for specific musical genres
- **Spatial Audio Rendering**: Generate immersive 3D audio outputs with binaural rendering

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/harmonicflow.git
cd harmonicflow
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Download the Slakh2100 dataset following the instructions in [data/README.md](data/README.md).

## Dataset Preparation

HarmonicFlow is trained on the [Slakh2100](https://www.slakh.com/) dataset, which contains 2100 high-quality, multi-track songs with separate stems for different instruments.

1. Follow the instructions in the original [dataset README](data/README.md) to download and extract the dataset.
2. Ensure that the data is organized in the `data/slakh2100/` directory with the following structure:
```
data/slakh2100/
 └─── train/
       └─── Track00001/
             └─── bass.wav
             └─── drums.wav
             └─── guitar.wav
             └─── piano.wav
       ...
 └─── validation/
       ...
 └─── test/
       ...
```

## Training

To train the HarmonicFlow model from scratch:

```bash
python train.py --data_dir data/slakh2100 --batch_size 16 --epochs 100
```

Additional training options:
```
--lr: Learning rate (default: 5e-5)
--hidden_dim: Hidden dimension size (default: 1024)
--latent_dim: Latent dimension size (default: 256)
--checkpoint_dir: Directory to save checkpoints (default: checkpoints)
--resume: Resume training from latest checkpoint
--seed: Random seed for reproducibility (default: 42)
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir logs
```

## Evaluation

To evaluate a trained model:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --data_dir data/slakh2100 --compute_metrics
```

To generate samples with specific style and emotion:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --style 2 --emotion 1 --num_samples 5
```

To generate samples using text prompts:

```bash
python evaluate.py --checkpoint checkpoints/best.pt --use_text --text_prompt "An energetic jazz composition with walking bass and brushed drums"
```

## Model Architecture

HarmonicFlow consists of several integrated modules:

1. **Input Layer**:
   - Audio Encoder: Processes raw audio into feature representations
   - Text Encoder: Processes text prompts using transformer models
   - Visual Encoder: Extracts features from reference images or videos

2. **Pre-Processing Module**:
   - Source Separation: Separates audio into constituent tracks
   - Feature Extraction: Identifies musical elements, chord progressions, etc.
   - Temporal Analysis: Captures long-term dependencies and motifs

3. **Latent Space Module**:
   - Dynamic Latent Space: Adapts based on feedback
   - Neural Audio Graph: Models relationships between instruments and patterns

4. **Generative Module**:
   - Multi-Track Diffusion Model: Generates coherent multi-instrument tracks
   - Hierarchical Attention: Processes music at multiple temporal scales
   - Emotion-Aware Layer: Adjusts parameters to reflect desired emotions

5. **Style Transfer and Post-Processing**:
   - Style Transfer: Applies characteristics of specific styles
   - Genre-Adaptive Layers: Fine-tunes for specific genres
   - Post-Processing GAN: Handles mixing, EQ, compression, etc.
   - Spatial Audio Renderer: Creates immersive 3D audio outputs

## Example Usage

### Basic generation:

```python
import torch
from models.harmonicflow import HarmonicFlow

# Initialize model
model = HarmonicFlow().to(device)

# Load checkpoint
checkpoint = torch.load("checkpoints/best.pt", map_location=device)
model.load_state_dict(checkpoint["model"])
model.eval()

# Generate with text prompt
generated_audio = model.generate(
    text=["A calm piano melody with jazz drums and acoustic bass"],
    style_idx=torch.tensor([3]),  # Jazz style
    emotion_idx=torch.tensor([2]),  # Calm/peaceful emotion
    batch_size=1
)

# Process the generated output
# ...
```

### Using with custom audio input:

```python
import torchaudio
from utils import compute_mel_spectrogram

# Load reference audio
waveform, sample_rate = torchaudio.load("reference.wav")
if sample_rate != 22050:
    waveform = torchaudio.functional.resample(waveform, sample_rate, 22050)

# Compute mel spectrogram
mel_spec = compute_mel_spectrogram(waveform)

# Generate new tracks based on reference
generated_audio = model.generate(
    audio=mel_spec,
    genre_idx=torch.tensor([5]),  # Rock genre
    batch_size=1
)
```

## Citation

If you use HarmonicFlow in your research, please cite:

```
@article{harmonicflow2023,
  title={HarmonicFlow: Multi-track Music Generation with Neural Audio Graphs and Hierarchical Attention},
  author={Your Name},
  journal={arXiv preprint arXiv:XXXX.XXXXX},
  year={2023}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Slakh2100 Dataset](https://www.slakh.com/) for providing high-quality multi-track data
- PyTorch team for the excellent deep learning framework
- The research community for advancements in music generation techniques

# Download instructions

## Test data
If you are only interested in inference, you can download just the Slakh2100 test-set that we used. It is available at the following [link](https://drive.google.com/file/d/1Xo-bGORndJhenHvzf3eY5Lt0XCTgElZP/view?usp=sharing).
```bash
# Move into this directory
cd data/

# Extract dataset here
tar -xvf slakh2100-testset-22050.tar.xz
```
The test set alone should occupy around 7GB of memory. 

## Complete dataset (train, validation, and test sets)
If you are interested in training some models of your own, you need to download the complete dataset.

Instructions for downloading are the following:

 1. Download the compressed data for each stem (bass, drums, etc.)
 2. Extract the data in this folder (i. e. `data/`)
 3. [optional] Delete the compressed data 
 4. Run the shell script `convert_data_format.sh` 

In the sections below, you can find a more precise description for each of these steps.

### 1. Download compressed data
You can download the data we used in our experiments from the following links:
 
 - [Bass Data](https://drive.google.com/file/d/1T7rbuwyqR73K__0L3nF550rVBXgrpYVT/view?usp=sharing)
 - [Drums Data](https://drive.google.com/file/d/1vieJQdvN22YrTdBMMvZXw1xr1rdko1pm/view?usp=sharing)
 - [Guitar Data](https://drive.google.com/file/d/1Uo3iN4lIecJ8SJulEKlhD96Bgd2CGf8F/view?usp=sharing)
 - [Piano Data](https://drive.google.com/file/d/1w3Zou4oL_DfdJm1o_Y-qLCvVYG8W6J62/view?usp=sharing)

Move the downloaded files into the `data/` directory.

```
data/bass_22050.tar.xz
data/drums_22050.tar.xz
data/guitar_22050.tar.xz
data/piano_22050.tar.xz
```

### 2. Extract data
```bash
# Move inside this directory
cd data/

# Decompress and extract data
tar -xvf bass_22050.tar.xz 
tar -xvf drums_22050.tar.xz
tar -xvf guitar_22050.tar.xz
tar -xvf piano_22050.tar.xz
```
This step might take a while, especially depending on your hardware. If you have a fast internet connection, consider instead downloading the zipped versions from [here](https://drive.google.com/drive/folders/1lCr93-47J3lsm_X5sBWGc9J1UfAz9pJE?usp=sharing).

After the extraction of all the sources dataset, you should have four directories:
```
data/bass_22050/
data/drums_22050/
data/guitar_22050/
data/piano_22050/
```

### 3. Delete compressed data
To free up some space it is possible now to delete the compressed version of the data. It will no longer be necessary.
```
rm data/*_22050.tar.xz 
```

### 4. Convert data
Before being able to use the dataset for training, it is necessary to run the following command:
```bash
# Move inside this directory
cd data/

 # Make script executable 
chmod +x ./convert_data_format.sh

# Convert the format of your data
./convert_data_format.sh
```
This command will convert the downloaded data into a format that the training script can digest. In particular, after running everything, your `data/` directory should contain the `slakh2100` folder, organized in the following fashion:
```
data/
 └─── slakh2100/
       └─── train/
             └─── Track00001/
                   └─── bass.wav
                   └─── drums.wav
                   └─── guitar.wav
                   └─── piano.wav
            ...
      ...
```
> ⚠️ **NOTE:**
> After running the script, the space occupied by `data/` should not change drastically, since all the files are hard-links, and are not actually copied.

