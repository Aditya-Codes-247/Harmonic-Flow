import os
import argparse
import numpy as np
import torch
import torchaudio
import librosa
import librosa.display
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import soundfile as sf
from pathlib import Path

import config
from data_loader import create_dataloader
from models.harmonicflow import HarmonicFlow
from utils import set_seed, load_checkpoint, visualize_spectrogram, compute_mel_spectrogram, normalize_audio

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate HarmonicFlow model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/slakh2100", help="Path to Slakh2100 dataset")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Output directory for generated audio")
    parser.add_argument("--split", type=str, default="test", choices=["test", "validation"], help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=10, help="Number of samples to generate")
    
    # Model parameters
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--latent_dim", type=int, default=config.LATENT_DIM, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM, help="Hidden dimension")
    
    # Generation parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--style", type=int, default=None, help="Style index to use (0-9)")
    parser.add_argument("--genre", type=int, default=None, help="Genre index to use (0-9)")
    parser.add_argument("--emotion", type=int, default=None, help="Emotion index to use (0-7)")
    parser.add_argument("--use_text", action="store_true", help="Use text prompts for generation")
    parser.add_argument("--text_prompt", type=str, default=None, help="Text prompt for generation")
    
    # Evaluation metrics
    parser.add_argument("--compute_metrics", action="store_true", help="Compute evaluation metrics")
    
    return parser.parse_args()

def audio_to_waveform(audio_latent, sample_rate=config.SAMPLE_RATE):
    """
    Convert audio latent representation to waveform.
    This is a placeholder function that would normally use a vocoder or other synthesis method.
    """
    # In a real implementation, this would use a trained audio decoder or vocoder
    # Here we'll create a very simple sine-based synthesis
    
    # Extract frequency and amplitude information from latent
    # This is just a placeholder - in reality, this would be a learned mapping
    freq_base = 220.0  # A3 as base frequency
    duration = 10.0  # seconds
    
    time_points = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    
    latent_np = audio_latent.cpu().numpy() if isinstance(audio_latent, torch.Tensor) else audio_latent
    
    # Use first half of latent as frequency modulation, second half as amplitude
    half_point = latent_np.shape[-1] // 2
    freq_mod = latent_np[..., :half_point]
    amp_mod = latent_np[..., half_point:]
    
    # Normalize
    freq_mod = (freq_mod - np.min(freq_mod)) / (np.max(freq_mod) - np.min(freq_mod) + 1e-8)
    amp_mod = (amp_mod - np.min(amp_mod)) / (np.max(amp_mod) - np.min(amp_mod) + 1e-8) * 0.8 + 0.2
    
    # Generate waveform
    waveform = np.zeros(len(time_points))
    
    for i in range(min(10, freq_mod.shape[-1])):  # Use up to 10 harmonics
        freq = freq_base * (i + 1) * (0.5 + freq_mod[i] * 0.5)  # Scale frequency
        amp = amp_mod[i]
        waveform += amp * np.sin(2 * np.pi * freq * time_points)
    
    # Normalize
    waveform = waveform / np.max(np.abs(waveform))
    
    return waveform

def compute_metrics(original, generated):
    """
    Compute evaluation metrics between original and generated audio.
    """
    # Convert to numpy if tensors
    if isinstance(original, torch.Tensor):
        original = original.cpu().numpy()
    if isinstance(generated, torch.Tensor):
        generated = generated.cpu().numpy()
    
    # Mean Squared Error
    mse = np.mean((original - generated) ** 2)
    
    # Signal-to-Noise Ratio (SNR)
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - generated) ** 2)
    snr = 10 * np.log10(signal_power / noise_power) if noise_power > 0 else float('inf')
    
    # Compute spectrograms
    orig_spec = librosa.stft(original.squeeze())
    gen_spec = librosa.stft(generated.squeeze())
    
    orig_mag = np.abs(orig_spec)
    gen_mag = np.abs(gen_spec)
    
    # Spectral Convergence
    spec_convergence = np.linalg.norm(orig_mag - gen_mag, 'fro') / np.linalg.norm(orig_mag, 'fro')
    
    # Log-Spectral Distance
    lsd = np.mean(np.sqrt(np.mean((20 * np.log10(orig_mag + 1e-8) - 20 * np.log10(gen_mag + 1e-8)) ** 2, axis=0)))
    
    metrics = {
        'mse': float(mse),
        'snr_db': float(snr),
        'spectral_convergence': float(spec_convergence),
        'log_spectral_distance': float(lsd)
    }
    
    return metrics

def evaluate_reconstruction(model, test_loader, args):
    """
    Evaluate model reconstruction capability on test data.
    """
    model.eval()
    
    all_metrics = {instrument: [] for instrument in config.INSTRUMENTS}
    all_metrics['mixed'] = []
    
    os.makedirs(os.path.join(args.output_dir, "reconstructions"), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(test_loader, desc="Evaluating reconstruction")):
            if i >= args.num_samples:
                break
                
            # Move data to device
            mixed_audio = batch["mixed_audio"].to(next(model.parameters()).device)
            mixed_mel = batch["mixed_mel"].to(next(model.parameters()).device)
            target_audio = {k: v.to(next(model.parameters()).device) for k, v in batch["audio"].items()}
            
            # Get track ID and offset for filename
            track_id = batch["track_id"][0]
            offset = batch["offset"][0].item()
            
            # Generate reconstruction
            reconstructed, outputs = model.generate(
                audio=mixed_mel,
                style_idx=args.style if args.style is not None else None,
                genre_idx=args.genre if args.genre is not None else None,
                emotion_idx=args.emotion if args.emotion is not None else None,
                return_all=True
            )
            
            # Convert to waveform
            reconstructed_audio = {}
            
            for j, instrument in enumerate(config.INSTRUMENTS):
                # Extract instrument-specific latent
                inst_latent = reconstructed[:, j * config.LATENT_DIM:(j+1) * config.LATENT_DIM, :]
                
                # Convert to waveform
                inst_audio = audio_to_waveform(inst_latent[0])
                reconstructed_audio[instrument] = inst_audio
                
                # Compute metrics
                original = target_audio[instrument][0, 0].cpu().numpy()
                metrics = compute_metrics(original, inst_audio)
                all_metrics[instrument].append(metrics)
                
                # Save audio
                output_path = os.path.join(
                    args.output_dir, 
                    "reconstructions", 
                    f"{track_id}_{offset:.1f}_{instrument}_reconstructed.wav"
                )
                sf.write(output_path, inst_audio, config.SAMPLE_RATE)
                
                # Save original
                orig_path = os.path.join(
                    args.output_dir, 
                    "reconstructions", 
                    f"{track_id}_{offset:.1f}_{instrument}_original.wav"
                )
                sf.write(orig_path, original, config.SAMPLE_RATE)
                
                # Save spectrograms
                visualize_spectrogram(
                    original,
                    title=f"Original {instrument.capitalize()}",
                    save_path=os.path.join(
                        args.output_dir, 
                        "reconstructions", 
                        f"{track_id}_{offset:.1f}_{instrument}_original_spec.png"
                    )
                )
                
                visualize_spectrogram(
                    inst_audio,
                    title=f"Reconstructed {instrument.capitalize()}",
                    save_path=os.path.join(
                        args.output_dir, 
                        "reconstructions", 
                        f"{track_id}_{offset:.1f}_{instrument}_reconstructed_spec.png"
                    )
                )
            
            # Mix reconstructed audio
            mixed_reconstructed = sum(reconstructed_audio.values())
            
            # Compute metrics for mixed audio
            original_mixed = mixed_audio[0, 0].cpu().numpy()
            mixed_metrics = compute_metrics(original_mixed, mixed_reconstructed)
            all_metrics['mixed'].append(mixed_metrics)
            
            # Save mixed audio
            output_path = os.path.join(
                args.output_dir, 
                "reconstructions", 
                f"{track_id}_{offset:.1f}_mixed_reconstructed.wav"
            )
            sf.write(output_path, mixed_reconstructed, config.SAMPLE_RATE)
            
            # Save original mixed
            orig_path = os.path.join(
                args.output_dir, 
                "reconstructions", 
                f"{track_id}_{offset:.1f}_mixed_original.wav"
            )
            sf.write(orig_path, original_mixed, config.SAMPLE_RATE)
    
    # Calculate average metrics
    avg_metrics = {instrument: {} for instrument in config.INSTRUMENTS}
    avg_metrics['mixed'] = {}
    
    for instrument in all_metrics.keys():
        for metric in all_metrics[instrument][0].keys():
            avg_metrics[instrument][metric] = np.mean([m[metric] for m in all_metrics[instrument]])
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, "reconstruction_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(avg_metrics, f, indent=2)
    
    print("Reconstruction evaluation completed!")
    print(f"Average MSE (mixed): {avg_metrics['mixed']['mse']:.6f}")
    print(f"Average SNR (mixed): {avg_metrics['mixed']['snr_db']:.2f} dB")
    
    return avg_metrics

def generate_samples(model, args, text_prompts=None):
    """
    Generate random samples from the model.
    """
    model.eval()
    
    num_samples = args.num_samples
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    # Set up style, genre, and emotion indices if provided
    style_idx = torch.tensor([args.style], device=next(model.parameters()).device) if args.style is not None else None
    genre_idx = torch.tensor([args.genre], device=next(model.parameters()).device) if args.genre is not None else None
    emotion_idx = torch.tensor([args.emotion], device=next(model.parameters()).device) if args.emotion is not None else None
    
    # Default text prompts if not provided
    if text_prompts is None and args.use_text:
        if args.text_prompt:
            text_prompts = [args.text_prompt]
        else:
            text_prompts = [
                "A happy, upbeat jazz song with acoustic bass and drums",
                "A sad piano ballad with gentle accompaniment",
                "An energetic rock song with electric guitar and heavy drums",
                "A calm ambient piece with soft piano melodies",
                "A funky groove with slap bass and tight drums",
                "A classical piece with dramatic piano and percussion",
                "A reggae song with dub bass and laid-back drums",
                "A blues song with walking bass and brush drums",
                "A techno track with pulsing bass and electronic drums",
                "A country song with picked guitar and gentle rhythm section"
            ]
            text_prompts = text_prompts[:num_samples]
    
    with torch.no_grad():
        for i in range(num_samples):
            # Set up text prompt if using text
            text = [text_prompts[i]] if args.use_text and i < len(text_prompts) else None
            
            # Generate sample
            sample = model.generate(
                batch_size=1,
                text=text,
                style_idx=style_idx,
                genre_idx=genre_idx,
                emotion_idx=emotion_idx
            )[0]
            
            # Convert to waveform
            sample_audio = {}
            
            for j, instrument in enumerate(config.INSTRUMENTS):
                # Extract instrument-specific latent
                inst_latent = sample[j * config.LATENT_DIM:(j+1) * config.LATENT_DIM, :]
                
                # Convert to waveform
                inst_audio = audio_to_waveform(inst_latent)
                sample_audio[instrument] = inst_audio
                
                # Save audio
                output_path = os.path.join(
                    args.output_dir,
                    "samples",
                    f"sample_{i+1}_{instrument}.wav"
                )
                sf.write(output_path, inst_audio, config.SAMPLE_RATE)
                
                # Save spectrogram
                visualize_spectrogram(
                    inst_audio,
                    title=f"Generated {instrument.capitalize()} (Sample {i+1})",
                    save_path=os.path.join(
                        args.output_dir,
                        "samples",
                        f"sample_{i+1}_{instrument}_spec.png"
                    )
                )
            
            # Mix sample audio
            mixed_sample = sum(sample_audio.values())
            
            # Normalize
            mixed_sample = normalize_audio(mixed_sample)
            
            # Save mixed audio
            output_path = os.path.join(
                args.output_dir,
                "samples",
                f"sample_{i+1}_mixed.wav"
            )
            sf.write(output_path, mixed_sample, config.SAMPLE_RATE)
            
            # Save mixed spectrogram
            visualize_spectrogram(
                mixed_sample,
                title=f"Generated Mixed Audio (Sample {i+1})",
                save_path=os.path.join(
                    args.output_dir,
                    "samples",
                    f"sample_{i+1}_mixed_spec.png"
                )
            )
            
            # Save metadata
            metadata = {
                "sample_id": i+1,
                "style_idx": args.style,
                "genre_idx": args.genre,
                "emotion_idx": args.emotion,
                "text_prompt": text_prompts[i] if text is not None else None
            }
            
            with open(os.path.join(args.output_dir, "samples", f"sample_{i+1}_metadata.json"), 'w') as f:
                json.dump(metadata, f, indent=2)
    
    print(f"Generated {num_samples} samples successfully!")

def evaluate(args):
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    model = HarmonicFlow(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(config.DEVICE)
    
    # Load checkpoint
    _, _, _ = load_checkpoint(model, path=args.checkpoint)
    print(f"Model loaded from {args.checkpoint}")
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create test data loader if evaluating reconstruction
    if args.compute_metrics:
        test_loader = create_dataloader(
            data_dir=args.data_dir,
            batch_size=1,  # Always use batch size 1 for evaluation
            split=args.split,
            num_workers=2,
            drop_last=False
        )
        
        # Evaluate reconstruction
        metrics = evaluate_reconstruction(model, test_loader, args)
    
    # Generate samples
    generate_samples(model, args)
    
    print(f"Evaluation completed! Results saved to {args.output_dir}")

if __name__ == "__main__":
    args = parse_args()
    evaluate(args) 