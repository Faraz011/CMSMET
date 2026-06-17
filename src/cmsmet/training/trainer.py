"""
Training utilities for CMSMET
"""
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from datetime import datetime

from cmsmet.models.encoders import SharedEmotionSpace, HuBERTEncoder, MERTEncoder
from cmsmet.models.losses import CrossModalContrastiveLoss
from cmsmet.config import ExperimentConfig


class Trainer:
    """Main trainer for CMSMET cross-modal emotion transfer"""
    
    def __init__(
        self,
        config: ExperimentConfig,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Args:
            config: Experiment configuration
            device: Device to use ('cuda' or 'cpu')
        """
        self.config = config
        self.device = torch.device(device)
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"Initialized trainer on {self.device}")
        
        # Initialize model
        self.model = self._build_model()
        self.logger.info(f"Model built with {self._count_parameters()} parameters")
        
        # Initialize loss
        self.loss_fn = CrossModalContrastiveLoss(
            temperature=config.contrastive.temperature,
            loss_type=config.contrastive.loss_type,
        )
        
        # Initialize optimizer
        self.optimizer, self.scheduler = self._build_optimizer()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_loss = float('inf')
        self.patience_counter = 0
        
        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging"""
        logger = logging.getLogger("cmsmet_trainer")
        logger.setLevel(logging.INFO)
        
        # File handler
        log_file = os.path.join(self.config.log_dir, "training.log")
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
    
    def _build_model(self) -> SharedEmotionSpace:
        """Build the shared emotion space model"""
        # Create encoders
        speech_encoder = HuBERTEncoder(
            model_name=self.config.encoder.speech_encoder_model,
            output_dim=self.config.encoder.embedding_dim,
            pretrained=self.config.encoder.speech_use_pretrained,
        ).to(self.device)
        
        music_encoder = MERTEncoder(
            model_name=self.config.encoder.music_encoder_model,
            output_dim=self.config.encoder.embedding_dim,
            pretrained=self.config.encoder.music_use_pretrained,
        ).to(self.device)
        
        # Create shared space
        model = SharedEmotionSpace(
            speech_encoder=speech_encoder,
            music_encoder=music_encoder,
            embedding_dim=self.config.encoder.embedding_dim,
        )
        
        return model.to(self.device)
    
    def _build_optimizer(self) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """Build optimizer and scheduler"""
        if self.config.training.optimizer == "adamw":
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        elif self.config.training.optimizer == "adam":
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.config.training.learning_rate,
                weight_decay=self.config.training.weight_decay,
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.training.optimizer}")
        
        # Scheduler
        if self.config.training.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.training.num_epochs,
            )
        elif self.config.training.scheduler == "linear":
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                total_iters=self.config.training.num_epochs,
            )
        else:
            scheduler = optim.lr_scheduler.ConstantLR(optimizer)
        
        return optimizer, scheduler
    
    def _count_parameters(self) -> int:
        """Count total parameters"""
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def train_epoch(
        self,
        train_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Train for one epoch
        
        Returns:
            Dict with training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            speech_waveform = batch['waveform'].to(self.device)
            emotion_labels = batch.get('emotion', None)
            if emotion_labels is not None:
                emotion_labels = emotion_labels.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Assume first half is speech, second half is music in batch
            batch_size = speech_waveform.shape[0] // 2
            speech_audio = speech_waveform[:batch_size]
            music_audio = speech_waveform[batch_size:]
            
            outputs = self.model(
                speech_waveform=speech_audio,
                music_waveform=music_audio,
            )
            
            # Compute loss
            loss_output = self.loss_fn(
                outputs['speech_embeddings'],
                outputs['music_embeddings'],
                emotion_labels=emotion_labels[:batch_size] if emotion_labels is not None else None,
            )
            loss = loss_output['loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.training.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.training.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            if self.global_step % self.config.training.log_every_n_steps == 0:
                self.logger.info(
                    f"Step {self.global_step}: loss = {avg_loss:.4f}"
                )
        
        return {
            'train_loss': total_loss / num_batches,
            'lr': self.optimizer.param_groups[0]['lr'],
        }
    
    @torch.no_grad()
    def evaluate(
        self,
        val_loader: DataLoader,
    ) -> Dict[str, float]:
        """
        Evaluate on validation set
        
        Returns:
            Dict with validation metrics
        """
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(val_loader, desc="Validation"):
            speech_waveform = batch['waveform'].to(self.device)
            emotion_labels = batch.get('emotion', None)
            if emotion_labels is not None:
                emotion_labels = emotion_labels.to(self.device)
            
            # Split batch
            batch_size = speech_waveform.shape[0] // 2
            speech_audio = speech_waveform[:batch_size]
            music_audio = speech_waveform[batch_size:]
            
            outputs = self.model(
                speech_waveform=speech_audio,
                music_waveform=music_audio,
            )
            
            loss_output = self.loss_fn(
                outputs['speech_embeddings'],
                outputs['music_embeddings'],
                emotion_labels=emotion_labels[:batch_size] if emotion_labels is not None else None,
            )
            loss = loss_output['loss']
            
            total_loss += loss.item()
            num_batches += 1
        
        return {
            'val_loss': total_loss / num_batches,
        }
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            Dict with training history
        """
        self.logger.info(f"Starting training for {self.config.training.num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'lr': [],
        }
        
        for epoch in range(self.config.training.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            history['lr'].append(train_metrics['lr'])
            
            self.logger.info(f"Epoch {epoch + 1} - loss: {train_metrics['train_loss']:.4f}")
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                self.logger.info(f"Epoch {epoch + 1} - val_loss: {val_metrics['val_loss']:.4f}")
                
                # Check early stopping
                if val_metrics['val_loss'] < self.best_loss - self.config.training.min_delta:
                    self.best_loss = val_metrics['val_loss']
                    self.patience_counter = 0
                    self._save_checkpoint(is_best=True)
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.config.training.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            
            # Save checkpoint
            if (epoch + 1) % self.config.training.save_every_n_epochs == 0:
                self._save_checkpoint()
            
            # Step scheduler
            self.scheduler.step()
        
        self.logger.info("Training completed")
        return history
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config.to_dict(),
        }
        
        if is_best:
            path = os.path.join(self.config.checkpoint_dir, 'best_model.pt')
        else:
            path = os.path.join(
                self.config.checkpoint_dir,
                f'checkpoint_epoch_{self.current_epoch:03d}.pt'
            )
        
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.logger.info(f"Loaded checkpoint from {path}")
