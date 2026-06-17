"""
Stage 1: Contrastive Pretraining Trainer
Trains the shared emotion space using NT-Xent loss
Saves projection heads separately for transfer to finetuning
"""
import os
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import numpy as np

from cmsmet.models.losses import CrossModalContrastiveLoss
from cmsmet.config import ExperimentConfig
from cmsmet.training.checkpoint_manager import CheckpointManager


class ContrastivePretrainingTrainer:
    """Trainer for Stage 1: Contrastive Pretraining"""
    
    def __init__(
        self,
        model: nn.Module,
        config: ExperimentConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            model: SharedEmotionSpace model
            config: Experiment configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = model
        self.config = config
        self.device = torch.device(device)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"Initialized ContrastivePretrainingTrainer on {self.device}")
        
        # Initialize loss function
        self.loss_fn = CrossModalContrastiveLoss(
            temperature=config.contrastive.temperature,
            loss_type=config.contrastive.loss_type,
        )
        self.logger.info(f"Using {config.contrastive.loss_type} loss with temperature={config.contrastive.temperature}")
        
        # Initialize optimizer and scheduler
        self.optimizer, self.scheduler = self._build_optimizer()
        self.logger.info(f"Optimizer: {config.training.optimizer}, LR: {config.training.learning_rate}")
        self.logger.info(f"Scheduler: {config.training.scheduler}")
        
        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            config.checkpoint_dir,
            config.experiment_name,
        )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Gradient accumulation
        self.accumulation_steps = getattr(config, 'accumulation_steps', 1)
        
        # Mixed precision training (saves ~3x memory on NVIDIA GPUs)
        self.use_amp = self.device.type == 'cuda'
        self.scaler = GradScaler() if self.use_amp else None
        if self.use_amp:
            self.logger.info("Using mixed precision (AMP) training for memory efficiency")
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.embedding_norms = []  # Track if embeddings are collapsing
        
        self.logger.info(
            f"Model has {self._count_parameters()} trainable parameters"
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging to file and console"""
        logger = logging.getLogger("contrastive_trainer")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers
        logger.handlers = []
        
        # File handler
        log_file = os.path.join(self.config.log_dir, "contrastive_training.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        return logger
    
    def _build_optimizer(self) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """Build optimizer and scheduler"""
        
        # Only AdamW for fair comparison
        if self.config.training.optimizer.lower() != "adamw":
            raise ValueError(
                f"Only AdamW allowed for fair comparison, got {self.config.training.optimizer}"
            )
        
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.training.learning_rate,
            weight_decay=self.config.training.weight_decay,
        )
        
        # Only cosine annealing for fair comparison
        if self.config.training.scheduler.lower() != "cosine":
            raise ValueError(
                f"Only cosine scheduler allowed for fair comparison, got {self.config.training.scheduler}"
            )
        
        # Cosine annealing with warmup
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.training.num_epochs,
        )
        
        return optimizer, scheduler
    
    def _count_parameters(self) -> int:
        """Count total trainable parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def _get_encoder_projection_states(self) -> Dict[str, Dict]:
        """Extract encoder and projection head states from model"""
        states = {}
        
        # Handle full models with encoders (HuBERT, MERT)
        if hasattr(self.model, 'speech_encoder'):
            states['speech_encoder'] = self.model.speech_encoder.hubert.state_dict()
            states['speech_projection'] = self.model.speech_encoder.projection.state_dict()
        
        if hasattr(self.model, 'music_encoder'):
            states['music_encoder'] = self.model.music_encoder.mert.state_dict()
            states['music_projection'] = self.model.music_encoder.projection.state_dict()
        
        # Handle embedding-only models (just projection heads)
        if hasattr(self.model, 'speech_proj'):
            states['speech_projection'] = self.model.speech_proj.state_dict()
        
        if hasattr(self.model, 'music_proj'):
            states['music_projection'] = self.model.music_proj.state_dict()
        
        # If no recognized components, just save the whole model
        if not states:
            states['model'] = self.model.state_dict()
        
        return states
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch
        
        Args:
            train_loader: Training dataloader with paired samples
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        embedding_norms = []
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}/{self.config.training.num_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            # Handle both dict and PairedBatch formats
            if hasattr(batch, 'to'):  # PairedBatch
                batch = batch.to(self.device)
                speech_waveform = batch.speech_waveform
                music_waveform = batch.music_waveform
                speech_emotion = batch.speech_emotion
            else:  # Dict format
                speech_waveform = batch['speech_waveform'].to(self.device)
                music_waveform = batch['music_waveform'].to(self.device)
                speech_emotion = batch.get('speech_emotion', None)
                if speech_emotion is not None:
                    speech_emotion = speech_emotion.to(self.device)
            
            # Forward pass with automatic mixed precision
            if self.use_amp:
                with autocast(dtype=torch.float16):
                    outputs = self.model(
                        speech_waveform=speech_waveform,
                        music_waveform=music_waveform,
                    )
                    
                    # Compute loss
                    loss_output = self.loss_fn(
                        outputs['speech_embeddings'],
                        outputs['music_embeddings'],
                        emotion_labels=speech_emotion if speech_emotion is not None else None,
                    )
                    loss = loss_output['loss']
                    
                    # Normalize loss by accumulation steps
                    loss = loss / self.accumulation_steps
                
                # Backward with gradient scaling
                self.scaler.scale(loss).backward()
            else:
                # Forward pass without AMP
                outputs = self.model(
                    speech_waveform=speech_waveform,
                    music_waveform=music_waveform,
                )
                
                # Compute loss
                loss_output = self.loss_fn(
                    outputs['speech_embeddings'],
                    outputs['music_embeddings'],
                    emotion_labels=speech_emotion if speech_emotion is not None else None,
                )
                loss = loss_output['loss']
                
                # Normalize loss by accumulation steps
                loss = loss / self.accumulation_steps
                
                # Backward pass
                loss.backward()
            
            # Optimizer step only after accumulation_steps batches
            is_accumulation_step = (batch_idx + 1) % self.accumulation_steps == 0
            is_last_batch = (batch_idx + 1) == len(train_loader)
            
            if is_accumulation_step or is_last_batch:
                # Gradient clipping
                if self.config.training.gradient_clip > 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.training.gradient_clip,
                    )
                
                # Optimizer step
                if self.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Clear CUDA cache to free memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
            
            # Tracking
            total_loss += loss.item() * self.accumulation_steps  # Undo normalization for logging
            num_batches += 1
            self.global_step += 1
            
            # Track embedding norms for collapse detection
            speech_norm = outputs['speech_embeddings'].norm(dim=1).mean().item()
            music_norm = outputs['music_embeddings'].norm(dim=1).mean().item()
            embedding_norms.append((speech_norm, music_norm))
            
            # Log every N steps
            if (batch_idx + 1) % self.config.training.log_every_n_steps == 0:
                avg_loss = total_loss / num_batches
                self.logger.info(
                    f"Step {self.global_step}: Loss = {avg_loss:.4f}, "
                    f"Speech norm = {speech_norm:.4f}, Music norm = {music_norm:.4f}"
                )
            
            pbar.set_postfix({'loss': f'{loss.item() * self.accumulation_steps:.4f}'})
        
        # Epoch summary
        epoch_loss = total_loss / num_batches
        self.train_losses.append(epoch_loss)
        
        # Average embedding norms
        avg_speech_norm = np.mean([n[0] for n in embedding_norms])
        avg_music_norm = np.mean([n[1] for n in embedding_norms])
        self.embedding_norms.append((avg_speech_norm, avg_music_norm))
        
        self.logger.info(
            f"Epoch {self.current_epoch + 1} Summary: "
            f"Loss = {epoch_loss:.4f}, "
            f"Speech norm = {avg_speech_norm:.4f}, "
            f"Music norm = {avg_music_norm:.4f}"
        )
        
        return {
            'loss': epoch_loss,
            'speech_norm': avg_speech_norm,
            'music_norm': avg_music_norm,
        }
    
    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Validate on validation set
        
        Args:
            val_loader: Validation dataloader
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validating"):
            # Handle both dict and PairedBatch formats
            if hasattr(batch, 'to'):  # PairedBatch
                batch = batch.to(self.device)
                speech_waveform = batch.speech_waveform
                music_waveform = batch.music_waveform
                speech_emotion = batch.speech_emotion
            else:  # Dict format
                speech_waveform = batch['speech_waveform'].to(self.device)
                music_waveform = batch['music_waveform'].to(self.device)
                speech_emotion = batch.get('speech_emotion', None)
                if speech_emotion is not None:
                    speech_emotion = speech_emotion.to(self.device)
            
            outputs = self.model(
                speech_waveform=speech_waveform,
                music_waveform=music_waveform,
            )
            
            loss_output = self.loss_fn(
                outputs['speech_embeddings'],
                outputs['music_embeddings'],
                emotion_labels=speech_emotion if speech_emotion is not None else None,
            )
            loss = loss_output['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        val_loss = total_loss / num_batches
        self.val_losses.append(val_loss)
        
        self.logger.info(f"Validation Loss = {val_loss:.4f}")
        
        return {'loss': val_loss}
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ):
        """
        Train for multiple epochs
        
        Args:
            train_loader: Training dataloader
            val_loader: Optional validation dataloader
        """
        
        self.logger.info("="*80)
        self.logger.info("STAGE 1: CONTRASTIVE PRETRAINING")
        self.logger.info("="*80)
        self.logger.info(f"Experiment: {self.config.experiment_name}")
        self.logger.info(f"Total epochs: {self.config.training.num_epochs}")
        self.logger.info(f"Batch size: {self.config.data.batch_size}")
        self.logger.info(f"Learning rate: {self.config.training.learning_rate}")
        self.logger.info(f"Weight decay: {self.config.training.weight_decay}")
        self.logger.info(f"Temperature: {self.config.contrastive.temperature}")
        self.logger.info("="*80 + "\n")
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            
            # Validate
            if val_loader is not None:
                val_metrics = self.validate(val_loader)
            else:
                val_metrics = {}
            
            # Scheduler step
            self.scheduler.step()
            
            # Check if best model
            current_loss = val_metrics.get('loss', train_metrics['loss'])
            is_best = current_loss < self.best_loss
            
            if is_best:
                self.best_loss = current_loss
                self.patience_counter = 0
                self.logger.info(f"[OK] New best model! Loss = {self.best_loss:.4f}")
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0 or is_best:
                self._save_checkpoint(is_best=is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.training.patience:
                self.logger.info(
                    f"Early stopping after {epoch + 1} epochs "
                    f"(patience={self.config.training.patience})"
                )
                break
        
        self.logger.info("\n" + "="*80)
        self.logger.info("CONTRASTIVE PRETRAINING COMPLETED")
        self.logger.info(f"Best loss: {self.best_loss:.4f}")
        self.logger.info("="*80)
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save checkpoint with separated projection heads"""
        
        # Get state dicts
        states = self._get_encoder_projection_states()
        
        # For embedding-only models, just save the model state
        if 'model' in states:
            checkpoint_path = self.checkpoint_manager.save_contrastive_checkpoint(
                model=self.model,
                model_state=states['model'],
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                config=self.config,
                epoch=self.current_epoch,
                global_step=self.global_step,
                best_loss=self.best_loss,
                metadata={'is_best': is_best},
            )
        else:
            # For full models with encoders, save separated components
            checkpoint_path = self.checkpoint_manager.save_contrastive_checkpoint(
                model=self.model,
                speech_encoder_state=states.get('speech_encoder'),
                music_encoder_state=states.get('music_encoder'),
                speech_projection_state=states.get('speech_projection'),
                music_projection_state=states.get('music_projection'),
                optimizer_state=self.optimizer.state_dict(),
                scheduler_state=self.scheduler.state_dict(),
                config=self.config,
                epoch=self.current_epoch,
                global_step=self.global_step,
                best_loss=self.best_loss,
                metadata={'is_best': is_best},
            )
        
        self.logger.info(f"Saved checkpoint: {checkpoint_path}")
    
    def extract_projection_heads(self) -> Dict[str, torch.nn.Module]:
        """
        Extract learned projection heads for transfer to finetuning
        
        Returns:
            Dict with speech and music projection modules
        """
        # Support both full models (with encoders) and embedding-only models
        projections = {}
        if hasattr(self.model, 'speech_encoder') and hasattr(self.model.speech_encoder, 'projection'):
            projections['speech_projection'] = self.model.speech_encoder.projection
        elif hasattr(self.model, 'speech_proj'):
            projections['speech_projection'] = self.model.speech_proj
        elif hasattr(self.model, 'speech_projection'):
            projections['speech_projection'] = self.model.speech_projection

        if hasattr(self.model, 'music_encoder') and hasattr(self.model.music_encoder, 'projection'):
            projections['music_projection'] = self.model.music_encoder.projection
        elif hasattr(self.model, 'music_proj'):
            projections['music_projection'] = self.model.music_proj
        elif hasattr(self.model, 'music_projection'):
            projections['music_projection'] = self.model.music_projection

        # Fallback to returning the whole model if no projection modules detected
        if not projections:
            projections['model'] = self.model

        return projections
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get summary of training"""
        return {
            'total_epochs': self.current_epoch + 1,
            'total_steps': self.global_step,
            'best_loss': self.best_loss,
            'final_train_loss': self.train_losses[-1] if self.train_losses else None,
            'final_val_loss': self.val_losses[-1] if self.val_losses else None,
            'training_time': datetime.now().isoformat(),
        }
