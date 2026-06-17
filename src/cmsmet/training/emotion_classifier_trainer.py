"""
Trainer for downstream emotion classification tasks
Integrates with SER-standard evaluation metrics (UA, WA, confusion matrix, etc.)
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

from cmsmet.models.classifiers import EmotionClassifier, DualEmotionClassifier
from cmsmet.evaluation.metrics import EmotionMetrics, ExperimentResults


class EmotionClassificationTrainer:
    """Trainer for emotion classification with SER-standard evaluation"""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        num_epochs: int = 50,
        patience: int = 10,
        min_delta: float = 0.001,
        gradient_clip: float = 1.0,
        log_every_n_steps: int = 10,
        checkpoint_dir: str = "checkpoints",
        seed: int = 42,
    ):
        """
        Args:
            model: EmotionClassifier or DualEmotionClassifier instance
            device: Device to use ('cuda' or 'cpu')
            learning_rate: Learning rate for optimizer
            weight_decay: Weight decay (L2 regularization)
            num_epochs: Maximum number of training epochs
            patience: Early stopping patience
            min_delta: Minimum improvement for early stopping
            gradient_clip: Gradient clipping threshold
            log_every_n_steps: Logging frequency
            checkpoint_dir: Directory to save checkpoints
            seed: Random seed for reproducibility
        """
        self.model = model.to(device)
        self.device = torch.device(device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_epochs = num_epochs
        self.patience = patience
        self.min_delta = min_delta
        self.gradient_clip = gradient_clip
        self.log_every_n_steps = log_every_n_steps
        self.checkpoint_dir = checkpoint_dir
        self.seed = seed
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Setup logging
        self.logger = self._setup_logger()
        self.logger.info(f"Initialized trainer on {self.device} with seed {seed}")
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(
            f"Model parameters: {total_params:,} total, {trainable_params:,} trainable"
        )
        
        # Optimizer and scheduler
        self.optimizer, self.scheduler = self._build_optimizer()
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Tracking
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_accuracy = 0.0
        self.patience_counter = 0
        
        # Metrics
        num_classes = model.num_classes if hasattr(model, 'num_classes') else 4
        self.metrics_engine = EmotionMetrics(
            num_classes=num_classes,
            class_names=['happy', 'sad', 'angry', 'neutral'] if num_classes == 4 else 
                       ['neutral', 'happy', 'sad', 'angry', 'frustrated']
        )
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        logger = logging.getLogger(f"Trainer_{self.seed}")
        logger.setLevel(logging.INFO)
        
        # Remove existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)
        
        # File handler
        log_file = os.path.join(self.checkpoint_dir, f"train_seed_{self.seed}.log")
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
        return logger
    
    def _build_optimizer(self) -> Tuple[optim.Optimizer, optim.lr_scheduler.LRScheduler]:
        """Build optimizer and scheduler"""
        optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
        )
        
        return optimizer, scheduler
    
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
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch + 1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Prepare batch
            waveform = batch['waveform'].to(self.device)
            emotion = batch['emotion'].long().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            outputs = self.model(waveform)
            logits = outputs['logits']
            
            # Compute loss
            loss = self.loss_fn(logits, emotion)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.gradient_clip,
                )
            
            self.optimizer.step()
            
            # Metrics
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            # Log
            avg_loss = total_loss / num_batches
            pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
            
            if self.global_step % self.log_every_n_steps == 0:
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
        Evaluate on validation set using SER metrics
        
        Returns:
            Dict with SER evaluation metrics (UA, WA, per-class F1, etc.)
        """
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(val_loader, desc=f"Epoch {self.current_epoch + 1} [Val]")
        for batch in pbar:
            waveform = batch['waveform'].to(self.device)
            emotion = batch['emotion'].long().to(self.device)
            
            # Forward pass
            outputs = self.model(waveform)
            logits = outputs['logits']
            
            # Loss
            loss = self.loss_fn(logits, emotion)
            total_loss += loss.item()
            num_batches += 1
            
            # Predictions
            predictions = logits.argmax(dim=-1)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(emotion.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
        
        # Aggregate predictions
        all_predictions = np.concatenate(all_predictions)
        all_targets = np.concatenate(all_targets)
        
        # Compute SER metrics
        metrics = self.metrics_engine.compute_metrics(all_predictions, all_targets)
        metrics['val_loss'] = total_loss / num_batches
        
        return metrics
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
    ) -> Tuple[Dict[str, List], Optional[Dict]]:
        """
        Train the model
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
        
        Returns:
            Tuple of (history, best_metrics)
            - history: Dict with training history
            - best_metrics: Best validation metrics (SER standard)
        """
        self.logger.info(f"Starting training for {self.num_epochs} epochs")
        
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_ua': [],
            'val_wa': [],
            'lr': [],
        }
        
        best_metrics = None
        
        for epoch in range(self.num_epochs):
            self.current_epoch = epoch
            
            # Train
            train_metrics = self.train_epoch(train_loader)
            history['train_loss'].append(train_metrics['train_loss'])
            history['lr'].append(train_metrics['lr'])
            
            log_msg = f"Epoch {epoch + 1} - train_loss: {train_metrics['train_loss']:.4f}"
            
            # Validate
            if val_loader is not None:
                val_metrics = self.evaluate(val_loader)
                history['val_loss'].append(val_metrics['val_loss'])
                history['val_ua'].append(val_metrics['ua'])
                history['val_wa'].append(val_metrics['wa'])
                
                log_msg += f" | val_loss: {val_metrics['val_loss']:.4f}"
                log_msg += f" | val_ua: {val_metrics['ua']:.4f} | val_wa: {val_metrics['wa']:.4f}"
                
                self.logger.info(log_msg)
                
                # Check early stopping based on UA (primary SER metric)
                current_ua = val_metrics['ua']
                if current_ua > self.best_val_accuracy + self.min_delta:
                    self.best_val_accuracy = current_ua
                    self.patience_counter = 0
                    best_metrics = val_metrics
                    self._save_checkpoint(is_best=True)
                    self.logger.info(f"New best UA: {current_ua:.4f}")
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= self.patience:
                        self.logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            else:
                self.logger.info(log_msg)
            
            # Step scheduler
            self.scheduler.step()
        
        self.logger.info("Training completed")
        return history, best_metrics
    
    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'seed': self.seed,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_accuracy': self.best_val_accuracy,
        }
        
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        if is_best:
            path = os.path.join(self.checkpoint_dir, f'best_model_seed_{self.seed}.pt')
        else:
            path = os.path.join(
                self.checkpoint_dir,
                f'checkpoint_epoch_{self.current_epoch:03d}_seed_{self.seed}.pt'
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
        self.best_val_accuracy = checkpoint.get('best_val_accuracy', 0.0)
        self.logger.info(f"Loaded checkpoint from {path}")
