import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

from models.encoder import MultiModalEncoder
from models.preprocessor import PreProcessingModule
from models.latent import LatentSpaceModule
from models.generative import GenerativeModule
from models.style import StyleAndPostProcessingModule

class HarmonicFlow(nn.Module):
    """
    Complete HarmonicFlow model that integrates all components.
    """
    def __init__(
        self,
        encoder=None,
        preprocessor=None,
        latent=None,
        generative=None,
        style_post=None,
        latent_dim=config.LATENT_DIM,
        hidden_dim=config.HIDDEN_DIM
    ):
        super().__init__()
        
        # Initialize components if not provided
        self.encoder = encoder if encoder else MultiModalEncoder(latent_dim=latent_dim)
        self.preprocessor = preprocessor if preprocessor else PreProcessingModule()
        self.latent = latent if latent else LatentSpaceModule(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.generative = generative if generative else GenerativeModule(input_dim=latent_dim, hidden_dim=hidden_dim)
        self.style_post = style_post if style_post else StyleAndPostProcessingModule(input_dim=latent_dim)
        
        # Integration layers
        self.latent_to_diffusion = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim * len(config.INSTRUMENTS))
        )
        
    def encode(self, audio=None, text=None, image=None):
        """
        Encode inputs to latent representation.
        
        Args:
            audio: Audio input
            text: Text prompt
            image: Image input
            
        Returns:
            latent: Latent representation
            features: Audio features
        """
        return self.encoder(audio, text, image)
    
    def preprocess(self, mixed_audio, spectrograms=None):
        """
        Preprocess audio input.
        
        Args:
            mixed_audio: Mixed audio input
            spectrograms: Optional pre-computed spectrograms
            
        Returns:
            processed: Preprocessed output
        """
        return self.preprocessor(mixed_audio, spectrograms)
        
    def generate(
        self,
        latent=None,
        audio=None,
        text=None,
        image=None,
        feedback=None,
        instrument_idx=None,
        emotion_idx=None,
        style_idx=None,
        genre_idx=None,
        batch_size=1,
        return_all=False
    ):
        """
        Generate output from latent representation or input modalities.
        
        Args:
            latent: Optional latent representation
            audio: Optional audio input
            text: Optional text prompt
            image: Optional image input
            feedback: Optional feedback signal
            instrument_idx: Optional instrument indices
            emotion_idx: Optional emotion indices
            style_idx: Optional style indices
            genre_idx: Optional genre indices
            batch_size: Batch size for generation
            return_all: Whether to return all intermediate outputs
            
        Returns:
            generated: Generated output
            outputs: Dictionary of all outputs if return_all=True
        """
        all_outputs = {}
        
        # Get latent representation either directly or from inputs
        if latent is None and (audio is not None or text is not None or image is not None):
            latent, features = self.encode(audio, text, image)
            all_outputs['encoder'] = {'latent': latent, 'features': features}
        
        if latent is None:
            # Generate from random latent if no input is provided
            latent = torch.randn(batch_size, config.LATENT_DIM, device=next(self.parameters()).device)
        
        # Process through latent space module
        integrated_latent, latent_outputs = self.latent(latent, feedback)
        all_outputs['latent'] = latent_outputs
        
        # Convert latent to diffusion model input
        diffusion_input = self.latent_to_diffusion(integrated_latent)
        diffusion_input = diffusion_input.view(
            diffusion_input.size(0),
            len(config.INSTRUMENTS) * config.LATENT_DIM,
            -1  # Sequence length determined by the module
        )
        
        # Generate with diffusion model
        generated_raw = self.generative.generate(
            batch_size=batch_size,
            instrument_idx=instrument_idx,
            emotion_idx=emotion_idx
        )
        all_outputs['generative'] = {'raw_output': generated_raw}
        
        # Apply style transfer and post-processing
        processed, style_outputs = self.style_post(
            generated_raw,
            style_idx=style_idx,
            genre_idx=genre_idx
        )
        all_outputs['style_post'] = style_outputs
        
        if return_all:
            return processed, all_outputs
        else:
            return processed
    
    def forward(
        self,
        audio=None,
        text=None,
        image=None,
        feedback=None,
        instrument_idx=None,
        emotion_idx=None,
        style_idx=None,
        genre_idx=None,
        return_all=False
    ):
        """
        Forward pass.
        
        Args:
            audio: Audio input
            text: Text prompt
            image: Image input
            feedback: Feedback signal
            instrument_idx: Instrument indices
            emotion_idx: Emotion indices
            style_idx: Style indices
            genre_idx: Genre indices
            return_all: Whether to return all intermediate outputs
            
        Returns:
            output: Model output
            outputs: Dictionary of all outputs if return_all=True
        """
        return self.generate(
            audio=audio,
            text=text,
            image=image,
            feedback=feedback,
            instrument_idx=instrument_idx,
            emotion_idx=emotion_idx,
            style_idx=style_idx,
            genre_idx=genre_idx,
            batch_size=audio.size(0) if audio is not None else 1,
            return_all=return_all
        )
    
    def train_step(
        self,
        audio_batch,
        mel_batch=None,
        text_batch=None,
        image_batch=None,
        instrument_idx=None,
        emotion_idx=None,
        style_idx=None,
        genre_idx=None
    ):
        """
        Training step.
        
        Args:
            audio_batch: Batch of audio data
            mel_batch: Optional batch of mel spectrograms
            text_batch: Optional batch of text prompts
            image_batch: Optional batch of images
            instrument_idx: Optional instrument indices
            emotion_idx: Optional emotion indices
            style_idx: Optional style indices
            genre_idx: Optional genre indices
            
        Returns:
            loss_dict: Dictionary of losses
            outputs: Outputs for monitoring
        """
        batch_size = audio_batch["mixed_audio"].size(0)
        device = next(self.parameters()).device
        
        # Move data to device
        mixed_audio = audio_batch["mixed_audio"].to(device)
        target_audio = {k: v.to(device) for k, v in audio_batch["audio"].items()}
        
        # Optional inputs
        if mel_batch is not None:
            mixed_mel = mel_batch["mixed_mel"].to(device)
            target_mel = {k: v.to(device) for k, v in mel_batch["mel_spectrograms"].items()}
        else:
            mixed_mel = audio_batch["mixed_mel"].to(device)
            target_mel = audio_batch["mel_spectrograms"]
            
        if text_batch is not None:
            text = text_batch.to(device)
        else:
            text = None
        
        if image_batch is not None:
            image = image_batch.to(device)
        else:
            image = None
        
        # Encode inputs
        latent, features = self.encode(mixed_mel, text, image)
        
        # Process through latent space
        integrated_latent, _ = self.latent(latent)
        
        # Convert to diffusion input
        diffusion_input = self.latent_to_diffusion(integrated_latent)
        diffusion_input = diffusion_input.view(
            batch_size,
            len(config.INSTRUMENTS) * config.LATENT_DIM,
            -1
        )
        
        # Sample timestep
        t = torch.randint(0, self.generative.diffusion_model.num_steps, (batch_size,), device=device)
        
        # Create target for diffusion
        target_diffusion = torch.cat(
            [target_mel[instrument] for instrument in config.INSTRUMENTS],
            dim=1
        )
        
        # Add noise according to timestep (forward process)
        noisy_input, noise = self.generative.diffusion_model.q_sample(target_diffusion, t)
        
        # Predict noise
        pred_noise = self.generative(
            noisy_input, 
            t, 
            instrument_idx=instrument_idx, 
            emotion_idx=emotion_idx
        )
        
        # Diffusion loss
        diffusion_loss = F.mse_loss(pred_noise, noise)
        
        # Reconstruction loss for latent space
        latent_recon = self.latent.dynamic_latent.decode(latent)
        latent_loss = F.mse_loss(latent_recon, mixed_mel.reshape(batch_size, -1))
        
        # KL divergence for VAE
        mu = self.latent.dynamic_latent.mu
        logvar = self.latent.dynamic_latent.logvar
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / batch_size
        
        # Style transfer loss (if style_idx is provided)
        if style_idx is not None:
            # Extract style from a reference
            style_code = self.style_post.style_transfer.extract_style(style_idx=style_idx)
            stylized = self.style_post.style_transfer(integrated_latent, style_idx=style_idx)
            
            # Style loss (simplified)
            style_loss = F.mse_loss(stylized, integrated_latent)
        else:
            style_loss = torch.tensor(0.0, device=device)
        
        # Compute total loss
        total_loss = diffusion_loss + 0.1 * latent_loss + 0.01 * kl_loss + 0.05 * style_loss
        
        # Return losses and outputs
        loss_dict = {
            "total_loss": total_loss,
            "diffusion_loss": diffusion_loss,
            "latent_loss": latent_loss,
            "kl_loss": kl_loss,
            "style_loss": style_loss
        }
        
        outputs = {
            "latent": latent,
            "integrated_latent": integrated_latent,
            "pred_noise": pred_noise,
            "target_noise": noise
        }
        
        return loss_dict, outputs 