import os
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random

import config
from data_loader import create_dataloader, AudioTransforms
from models.harmonicflow import HarmonicFlow
from utils import set_seed, save_checkpoint, load_checkpoint, log_metrics, visualize_spectrogram

def parse_args():
    parser = argparse.ArgumentParser(description="Train HarmonicFlow model")
    
    # Data parameters
    parser.add_argument("--data_dir", type=str, default="data/slakh2100", help="Path to Slakh2100 dataset")
    parser.add_argument("--batch_size", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    
    # Model parameters
    parser.add_argument("--latent_dim", type=int, default=config.LATENT_DIM, help="Latent dimension")
    parser.add_argument("--hidden_dim", type=int, default=config.HIDDEN_DIM, help="Hidden dimension")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=config.WEIGHT_DECAY, help="Weight decay")
    parser.add_argument("--grad_accum_steps", type=int, default=config.GRADIENT_ACCUMULATION_STEPS, 
                        help="Gradient accumulation steps")
    parser.add_argument("--checkpoint_dir", type=str, default=config.CHECKPOINT_DIR, help="Checkpoint directory")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--seed", type=int, default=config.SEED, help="Random seed")
    parser.add_argument("--val_check_interval", type=float, default=config.VAL_CHECK_INTERVAL, 
                        help="Validation check interval (in epochs)")
    
    # Logging parameters
    parser.add_argument("--log_dir", type=str, default="logs", help="Log directory")
    parser.add_argument("--log_every", type=int, default=100, help="Log metrics every N steps")
    parser.add_argument("--save_every", type=int, default=1000, help="Save checkpoint every N steps")
    
    return parser.parse_args()

def train_epoch(model, train_loader, optimizer, scheduler, epoch, args, writer, start_step=0):
    model.train()
    
    running_loss = 0.0
    running_diffusion_loss = 0.0
    running_latent_loss = 0.0
    running_kl_loss = 0.0
    running_style_loss = 0.0
    
    step = start_step
    
    # Create progress bar
    tqdm_loader = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
    
    # Reset gradients
    optimizer.zero_grad()
    
    for batch_idx, batch in enumerate(tqdm_loader):
        # Prepare random conditional indices for training
        batch_size = batch["mixed_audio"].size(0)
        
        # Random emotion indices (1-8) or None
        if random.random() < 0.5:
            emotion_idx = torch.randint(0, 8, (batch_size,), device=config.DEVICE)
        else:
            emotion_idx = None
            
        # Random style indices (1-10) or None
        if random.random() < 0.5:
            style_idx = torch.randint(0, 10, (batch_size,), device=config.DEVICE)
        else:
            style_idx = None
            
        # Random genre indices (1-10) or None
        if random.random() < 0.5:
            genre_idx = torch.randint(0, 10, (batch_size,), device=config.DEVICE)
        else:
            genre_idx = None
            
        # Forward pass
        loss_dict, outputs = model.train_step(
            audio_batch=batch,
            emotion_idx=emotion_idx,
            style_idx=style_idx,
            genre_idx=genre_idx
        )
        
        # Scale loss based on gradient accumulation
        loss = loss_dict["total_loss"] / args.grad_accum_steps
        
        # Backward pass
        loss.backward()
        
        # Update running losses
        running_loss += loss_dict["total_loss"].item()
        running_diffusion_loss += loss_dict["diffusion_loss"].item()
        running_latent_loss += loss_dict["latent_loss"].item()
        running_kl_loss += loss_dict["kl_loss"].item()
        running_style_loss += loss_dict["style_loss"].item()
        
        # Update parameters every grad_accum_steps or at the last batch
        if (batch_idx + 1) % args.grad_accum_steps == 0 or (batch_idx + 1) == len(train_loader):
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Update parameters
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            # Increment step
            step += 1
            
            # Log metrics
            if step % args.log_every == 0:
                # Calculate average losses
                avg_loss = running_loss / args.log_every
                avg_diffusion_loss = running_diffusion_loss / args.log_every
                avg_latent_loss = running_latent_loss / args.log_every
                avg_kl_loss = running_kl_loss / args.log_every
                avg_style_loss = running_style_loss / args.log_every
                
                # Log to tensorboard
                metrics = {
                    "loss": avg_loss,
                    "diffusion_loss": avg_diffusion_loss,
                    "latent_loss": avg_latent_loss,
                    "kl_loss": avg_kl_loss,
                    "style_loss": avg_style_loss,
                    "learning_rate": scheduler.get_last_lr()[0]
                }
                log_metrics(writer, metrics, step, prefix="train")
                
                # Update progress bar
                tqdm_loader.set_postfix(loss=avg_loss)
                
                # Reset running losses
                running_loss = 0.0
                running_diffusion_loss = 0.0
                running_latent_loss = 0.0
                running_kl_loss = 0.0
                running_style_loss = 0.0
                
            # Save checkpoint
            if step % args.save_every == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=step,
                    loss=running_loss / (batch_idx + 1),
                    path=os.path.join(args.checkpoint_dir, f"step_{step}.pt")
                )
                
    return step

def validate(model, val_loader, epoch, args, writer, step):
    model.eval()
    
    val_loss = 0.0
    val_diffusion_loss = 0.0
    val_latent_loss = 0.0
    val_kl_loss = 0.0
    val_style_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Validation {epoch+1}"):
            # Forward pass
            loss_dict, outputs = model.train_step(audio_batch=batch)
            
            # Accumulate losses
            val_loss += loss_dict["total_loss"].item()
            val_diffusion_loss += loss_dict["diffusion_loss"].item()
            val_latent_loss += loss_dict["latent_loss"].item()
            val_kl_loss += loss_dict["kl_loss"].item()
            val_style_loss += loss_dict["style_loss"].item()
    
    # Calculate average losses
    num_batches = len(val_loader)
    avg_val_loss = val_loss / num_batches
    avg_val_diffusion_loss = val_diffusion_loss / num_batches
    avg_val_latent_loss = val_latent_loss / num_batches
    avg_val_kl_loss = val_kl_loss / num_batches
    avg_val_style_loss = val_style_loss / num_batches
    
    # Log metrics to tensorboard
    metrics = {
        "loss": avg_val_loss,
        "diffusion_loss": avg_val_diffusion_loss,
        "latent_loss": avg_val_latent_loss,
        "kl_loss": avg_val_kl_loss,
        "style_loss": avg_val_style_loss
    }
    log_metrics(writer, metrics, step, prefix="val")
    
    print(f"Validation loss: {avg_val_loss:.6f}")
    
    # Generate samples
    with torch.no_grad():
        # Generate random samples
        batch_size = 4
        samples = model.generate(batch_size=batch_size)
        
        # Log sample spectrograms
        for i in range(min(batch_size, 4)):
            sample = samples[i].cpu().numpy()
            
            # Visualize spectrograms
            for j, instrument in enumerate(config.INSTRUMENTS):
                sample_instrument = sample[j * config.LATENT_DIM:(j+1) * config.LATENT_DIM]
                
                # Save spectrograms
                sample_path = os.path.join(args.log_dir, f"samples/epoch_{epoch}_step_{step}_{instrument}_{i}.png")
                os.makedirs(os.path.dirname(sample_path), exist_ok=True)
                
                visualize_spectrogram(
                    sample_instrument, 
                    title=f"Generated {instrument.capitalize()} (Sample {i+1})",
                    save_path=sample_path
                )
                
                # Log to tensorboard
                writer.add_image(
                    f"samples/{instrument}_{i}",
                    np.transpose(plt.imread(sample_path), (2, 0, 1)),
                    step
                )
    
    return avg_val_loss

def train(args):
    # Set random seeds
    set_seed(args.seed)
    
    # Create directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Create tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Create data augmentation
    train_transform = AudioTransforms()
    
    # Create data loaders
    train_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split="train",
        num_workers=args.num_workers,
        drop_last=True
    )
    
    val_loader = create_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        split="validation",
        num_workers=args.num_workers,
        drop_last=False
    )
    
    # Create model
    model = HarmonicFlow(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    ).to(config.DEVICE)
    
    # Create optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=config.ADAM_BETAS,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate scheduler
    total_steps = len(train_loader) // args.grad_accum_steps * args.epochs
    warmup_steps = config.WARM_UP_STEPS
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Load checkpoint if resuming
    start_epoch = 0
    start_step = 0
    best_val_loss = float('inf')
    
    if args.resume:
        checkpoint_path = os.path.join(args.checkpoint_dir, "latest.pt")
        if os.path.exists(checkpoint_path):
            start_epoch, start_step, best_val_loss = load_checkpoint(
                model, optimizer, scheduler, checkpoint_path
            )
            print(f"Resuming from epoch {start_epoch}, step {start_step}")
    
    # Log model parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Training loop
    print(f"Starting training for {args.epochs} epochs")
    
    for epoch in range(start_epoch, args.epochs):
        # Train for one epoch
        step = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            args=args,
            writer=writer,
            start_step=start_step if epoch == start_epoch else 0
        )
        
        # Reset start_step after first epoch if resuming
        start_step = 0
        
        # Validate
        if (epoch + 1) % int(1 / args.val_check_interval) == 0 or (epoch + 1) == args.epochs:
            val_loss = validate(
                model=model,
                val_loader=val_loader,
                epoch=epoch,
                args=args,
                writer=writer,
                step=step
            )
            
            # Save checkpoint if validation loss improved
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    step=step,
                    loss=val_loss,
                    path=os.path.join(args.checkpoint_dir, "best.pt")
                )
        
        # Always save latest checkpoint
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch + 1,  # Save next epoch to resume from
            step=step,
            loss=val_loss if (epoch + 1) % int(1 / args.val_check_interval) == 0 else 0.0,
            path=os.path.join(args.checkpoint_dir, "latest.pt")
        )
    
    writer.close()
    print("Training completed!")

if __name__ == "__main__":
    args = parse_args()
    train(args) 