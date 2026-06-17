"""
CMSMET Paper Experiment: Baseline (HuBERT-only) vs CMSMET (HuBERT+MERT SupCon+LP-FT)
======================================================================================

Compares two approaches:
  Baseline:     Random projection + simple end-to-end training (HuBERT speech only)
  Contrastive:  SupCon pre-trained projection (HuBERT speech + MERT music) + LP-FT

Usage:
    python scripts/run_paper_experiment.py
    python scripts/run_paper_experiment.py --seeds 42 123 456 --device cuda
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from copy import deepcopy
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# Add src to path for DeeperProjectionHead
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from cmsmet.models.encoders import DeeperProjectionHead

# ============================================================================
# CONFIGURATION
# ============================================================================

# Shared architecture (identical for baseline and contrastive)
SHARED_CONFIG = {
    'embedding_dim': 768,       # HuBERT / MERT output
    'projection_hidden': 256,   # Projection intermediate
    'projection_output': 128,   # Projection final = contrastive space
    'num_classes': 4,           # happy, sad, angry, neutral
    'dropout': 0.2,             # Must match Stage 1
    'num_epochs': 100,
    'batch_size': 64,
    'gradient_clip': 1.0,
    'patience': 15,
    'loss': 'cross_entropy',
    'label_smoothing': 0.1,     # Prevents LP overfitting on well-separated features
}

# Baseline: random init + direct fine-tuning (no LP phase)
BASELINE_CONFIG = {
    **SHARED_CONFIG,
    'optimizer': 'adamw',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'alpha': 1.0,              # Classification only
    'lp_epochs': 0,            # No LP — direct fine-tuning
    'lp_lr': 1e-4,
    'ft_proj_lr': 1e-5,
}

# Contrastive: SupCon pretrained + direct fine-tuning (no LP phase)
CONTRASTIVE_CONFIG = {
    **SHARED_CONFIG,
    'optimizer': 'adamw',
    'learning_rate': 1e-4,
    'weight_decay': 1e-5,
    'scheduler': 'cosine',
    'alpha': 1.0,              # Classification only during Stage 3
    'lp_epochs': 0,            # No LP — direct fine-tuning
    'lp_lr': 1e-4,
    'ft_proj_lr': 1e-5,        # Fine-tuning LR for projection (10x lower)
}

# For backward compat — points to contrastive config (used by Stage 1)
FIXED_CONFIG = CONTRASTIVE_CONFIG

# IEMOCAP emotion mapping: 4-class standard
# happy(+excited)=0, sad=1, angry=2, neutral=3 (frustrated dropped)
EMOTION_NAMES = ['happy', 'sad', 'angry', 'neutral']


# ============================================================================
# DATASET
# ============================================================================

class EmbeddingDataset(Dataset):
    """Pre-extracted HuBERT embeddings + IEMOCAP emotion labels."""

    def __init__(self, embeddings: np.ndarray, labels: np.ndarray, sessions: np.ndarray):
        """
        Args:
            embeddings: (N, 768) HuBERT embeddings
            labels: (N,) emotion labels 0-3
            sessions: (N,) session IDs 1-5
        """
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.sessions = sessions  # Keep as numpy for indexing

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]


# ============================================================================
# MODEL — Identical architecture for both baseline and contrastive
# ============================================================================

class EmotionClassifierHead(nn.Module):
    """
    Projection + Classifier for emotion recognition.

    Architecture:
        Input [768] -> DeeperProjectionHead(768, 256, 128) -> Classifier(128,4)

    The ONLY difference between baseline and contrastive:
    - Baseline: projection initialized with RANDOM weights
    - Contrastive: projection initialized with Stage 1 LEARNED weights
    """

    def __init__(self, config: dict):
        super().__init__()

        # Projection heads — MUST match Stage 1 architecture exactly
        self.speech_projection = DeeperProjectionHead(
            input_dim=config['embedding_dim'],
            hidden_dim=config['projection_hidden'],
            output_dim=config['projection_output'],
            dropout=config['dropout'],
        )
        self.music_projection = DeeperProjectionHead(
            input_dim=config['embedding_dim'],
            hidden_dim=config['projection_hidden'],
            output_dim=config['projection_output'],
            dropout=config['dropout'],
        )

        # Classifier head
        self.classifier = nn.Linear(config['projection_output'], config['num_classes'])

    def forward(self, x, modality='speech'):
        """
        Args:
            x: (batch_size, 768) embeddings
        Returns:
            logits: (batch_size, 4) emotion logits
        """
        if modality == 'speech':
            proj = self.speech_projection(x)
        else:
            proj = self.music_projection(x)
        logits = self.classifier(proj)
        return logits


def create_baseline_model(config: dict) -> EmotionClassifierHead:
    """Create model with RANDOM projection (baseline)."""
    model = EmotionClassifierHead(config)
    return model


def create_contrastive_model(config: dict, checkpoint) -> EmotionClassifierHead:
    """Create model with Stage 1 PRETRAINED projection (contrastive).

    Args:
        config: Model configuration.
        checkpoint: Either a file path (str) or a state dict (dict) with keys
                    'speech_projection' and optionally 'music_projection'.
    """
    model = EmotionClassifierHead(config)

    # Accept either a path or pre-loaded dict
    if isinstance(checkpoint, str):
        ckpt = torch.load(checkpoint, map_location='cpu')
    else:
        ckpt = checkpoint

    if 'speech_projection' not in ckpt:
        raise ValueError(f"Checkpoint missing 'speech_projection'. Keys: {list(ckpt.keys())}")

    # Load speech projection
    proj_state = ckpt['speech_projection']
    model.speech_projection.load_state_dict(proj_state)

    # Load music projection if available
    music_proj_state = ckpt.get('music_projection', None)
    if music_proj_state is not None:
        model.music_projection.load_state_dict(music_proj_state)

    return model


# ============================================================================
# PER-FOLD STAGE 1 SUPCON PRE-TRAINING (leak-free)
# ============================================================================

class ContrastiveProjectionModel(nn.Module):
    """Contrastive model with deeper projection heads for per-fold Stage 1."""

    def __init__(self, speech_dim=768, music_dim=768, proj_dim=128, dropout=0.2):
        super().__init__()
        self.speech_projection = DeeperProjectionHead(
            input_dim=speech_dim, hidden_dim=256,
            output_dim=proj_dim, dropout=dropout,
        )
        self.music_projection = DeeperProjectionHead(
            input_dim=music_dim, hidden_dim=256,
            output_dim=proj_dim, dropout=dropout,
        )

    def forward(self, speech_emb, music_emb):
        speech_proj = self.speech_projection(speech_emb)
        music_proj = self.music_projection(music_emb)
        speech_proj = nn.functional.normalize(speech_proj, dim=-1)
        music_proj = nn.functional.normalize(music_proj, dim=-1)
        return speech_proj, music_proj


def compute_supcon_loss(speech_proj, music_proj, labels, temperature=0.07):
    """Supervised Contrastive Loss across both modalities."""
    batch_size = speech_proj.shape[0]
    device = speech_proj.device
    features = torch.cat([speech_proj, music_proj], dim=0)       # [2B, D]
    all_labels = torch.cat([labels, labels], dim=0)               # [2B]
    sim_matrix = torch.matmul(features, features.T) / temperature # [2B, 2B]
    label_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).float()
    self_mask = torch.eye(2 * batch_size, device=device)
    label_mask = label_mask * (1 - self_mask)
    logits_max, _ = sim_matrix.max(dim=1, keepdim=True)
    logits = sim_matrix - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    num_positives = label_mask.sum(dim=1)
    mean_log_prob = (label_mask * log_prob).sum(dim=1) / (num_positives + 1e-8)
    valid = num_positives > 0
    loss = -mean_log_prob[valid].mean()
    return loss


class FoldPairedDataset(Dataset):
    """Pairs speech embeddings with randomly-sampled same-emotion music embeddings."""

    def __init__(self, speech_embs: np.ndarray, speech_labels: np.ndarray,
                 music_by_emo: Dict[int, List[torch.Tensor]], augment: bool = True):
        self.speech_embs = speech_embs
        self.speech_labels = speech_labels
        self.augment = augment
        # Store music arrays for each emotion class
        self.music_by_emo_np = {}
        for emo, tensors in music_by_emo.items():
            if tensors:
                self.music_by_emo_np[emo] = [t.numpy() if isinstance(t, torch.Tensor) else t for t in tensors]
            else:
                self.music_by_emo_np[emo] = []
        # Build valid indices (speech samples that have matching music)
        self.valid_indices = []
        for i in range(len(self.speech_embs)):
            emo = int(self.speech_labels[i])
            if emo in self.music_by_emo_np and self.music_by_emo_np[emo]:
                self.valid_indices.append(i)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        speech_idx = self.valid_indices[idx]
        emo = int(self.speech_labels[speech_idx])
        speech_emb = torch.tensor(self.speech_embs[speech_idx], dtype=torch.float32)
        music_options = self.music_by_emo_np[emo]
        music_emb = torch.tensor(music_options[np.random.randint(len(music_options))], dtype=torch.float32)
        if self.augment:
            if np.random.rand() < 0.5:
                speech_emb = speech_emb + torch.randn_like(speech_emb) * 0.02
            if np.random.rand() < 0.5:
                music_emb = music_emb + torch.randn_like(music_emb) * 0.02
        return speech_emb, music_emb, emo


def train_stage1_for_fold(
    speech_embeddings: np.ndarray,
    speech_labels: np.ndarray,
    speech_sessions: np.ndarray,
    test_session: int,
    music_by_emo: Dict[int, List[torch.Tensor]],
    config: dict,
    device: torch.device,
    logger: logging.Logger,
    num_epochs: int = 50,
    temperature: float = 0.07,
) -> dict:
    """
    Train Stage 1 SupCon projection heads for ONE LOSO fold,
    excluding the test session from speech data to prevent leakage.

    Returns a state dict with keys 'speech_projection' and 'music_projection'.
    """
    # --- Filter speech to exclude test session ---
    train_mask = speech_sessions != test_session
    fold_embs = speech_embeddings[train_mask]
    fold_labels = speech_labels[train_mask]

    n_excluded = (~train_mask).sum()
    logger.info(f"  [Stage1-Fold] Training SupCon excluding session {test_session} "
                f"({train_mask.sum()} train, {n_excluded} held-out)")

    # --- Dataset & Loader ---
    dataset = FoldPairedDataset(fold_embs, fold_labels, music_by_emo, augment=True)
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True, num_workers=0)

    # --- Model ---
    model = ContrastiveProjectionModel(
        speech_dim=config['embedding_dim'],
        music_dim=config['embedding_dim'],
        proj_dim=config['projection_output'],
        dropout=config['dropout'],
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    best_loss = float('inf')
    best_state = None

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        for speech_b, music_b, label_b in loader:
            speech_b = speech_b.to(device)
            music_b = music_b.to(device)
            label_b = torch.tensor(label_b, dtype=torch.long, device=device) if not isinstance(label_b, torch.Tensor) else label_b.to(device)

            speech_proj, music_proj = model(speech_b, music_b)
            loss = compute_supcon_loss(speech_proj, music_proj, label_b, temperature)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        scheduler.step()

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {
                'speech_projection': deepcopy(model.speech_projection.state_dict()),
                'music_projection': deepcopy(model.music_projection.state_dict()),
            }

        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  [Stage1-Fold] Epoch {epoch+1}/{num_epochs} "
                        f"Loss: {avg_loss:.4f} (best: {best_loss:.4f})")

    logger.info(f"  [Stage1-Fold] Done. Best loss: {best_loss:.4f}")
    return best_state


# ============================================================================
# TRAINER
# ============================================================================

def compute_nt_xent_loss(q, k, temp=0.07):
    """
    q: [B, D] - query embeddings (normalized)
    k: [B, D] - key embeddings (normalized)
    """
    batch_size = q.shape[0]
    # Cosine similarity matrix: shape [B, B]
    logits = torch.matmul(q, k.T) / temp
    # Labels: diagonal elements are positive pairs
    labels = torch.arange(batch_size, device=q.device)
    loss_q = nn.functional.cross_entropy(logits, labels)
    loss_k = nn.functional.cross_entropy(logits.T, labels)
    return (loss_q + loss_k) / 2.0


class FairTrainer:
    """Trains a model with FIXED hyperparameters for fair comparison."""

    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        model_type: str,  # 'baseline' or 'contrastive'
        logger: logging.Logger,
        music_by_emo: Dict[int, List[torch.Tensor]],
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_type = model_type
        self.logger = logger
        self.music_by_emo = music_by_emo
        self.alpha = config.get('alpha', 0.5)

        # Loss — IDENTICAL for both (label smoothing prevents LP overfitting)
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.0))

        # Check if we use LP-FT
        self.lp_epochs = self.config.get('lp_epochs', 0)
        
        if self.lp_epochs > 0:
            self.setup_optimizer_and_scheduler('lp', self.lp_epochs)
        else:
            self.setup_optimizer_and_scheduler('ft', self.config['num_epochs'])

        # Mixed precision
        self.use_amp = device.type == 'cuda'
        self.scaler = GradScaler('cuda') if self.use_amp else None

        # Tracking
        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        # Log config
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"  Model type: {model_type}")
        self.logger.info(f"  Parameters: {total_params:,} total, {trainable_params:,} trainable")
        if self.lp_epochs > 0:
            self.logger.info(f"  LP-FT Setup: lp_epochs={self.lp_epochs}, lp_lr={config.get('lp_lr', 1e-3)}, ft_proj_lr={config.get('ft_proj_lr', 1e-5)}")
        else:
            self.logger.info(f"  Optimizer: AdamW (lr={config['learning_rate']}, wd={config['weight_decay']})")
            self.logger.info(f"  Scheduler: CosineAnnealing (T_max={config['num_epochs']})")
        self.logger.info(f"  Loss: CrossEntropy (unweighted) with Joint Regularization (alpha={self.alpha})")
        self.logger.info(f"  AMP: {self.use_amp}")

    def train_epoch(self, train_loader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for embeddings, labels in train_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(embeddings, modality='speech')
                    loss_class = self.loss_fn(logits, labels)

                    if self.alpha < 1.0:
                        # Sample matching music embeddings
                        music_batch_list = []
                        for label in labels:
                            label_item = label.item()
                            m_embs = self.music_by_emo.get(label_item, [])
                            if len(m_embs) > 0:
                                idx = np.random.randint(0, len(m_embs))
                                music_batch_list.append(m_embs[idx])
                            else:
                                music_batch_list.append(torch.zeros(self.config['embedding_dim'], dtype=torch.float32))
                        music_embeddings = torch.stack(music_batch_list).to(self.device)

                        # Compute contrastive regularizer
                        speech_proj = self.model.speech_projection(embeddings)
                        music_proj = self.model.music_projection(music_embeddings)
                        speech_proj = nn.functional.normalize(speech_proj, dim=-1)
                        music_proj = nn.functional.normalize(music_proj, dim=-1)
                        loss_contrastive = compute_nt_xent_loss(speech_proj, music_proj, temp=0.07)

                        loss = self.alpha * loss_class + (1.0 - self.alpha) * loss_contrastive
                    else:
                        loss = loss_class

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(embeddings, modality='speech')
                loss_class = self.loss_fn(logits, labels)

                if self.alpha < 1.0:
                    # Sample matching music embeddings
                    music_batch_list = []
                    for label in labels:
                        label_item = label.item()
                        m_embs = self.music_by_emo.get(label_item, [])
                        if len(m_embs) > 0:
                            idx = np.random.randint(0, len(m_embs))
                            music_batch_list.append(m_embs[idx])
                        else:
                            music_batch_list.append(torch.zeros(self.config['embedding_dim'], dtype=torch.float32))
                    music_embeddings = torch.stack(music_batch_list).to(self.device)

                    # Compute contrastive regularizer
                    speech_proj = self.model.speech_projection(embeddings)
                    music_proj = self.model.music_projection(music_embeddings)
                    speech_proj = nn.functional.normalize(speech_proj, dim=-1)
                    music_proj = nn.functional.normalize(music_proj, dim=-1)
                    loss_contrastive = compute_nt_xent_loss(speech_proj, music_proj, temp=0.07)

                    loss = self.alpha * loss_class + (1.0 - self.alpha) * loss_contrastive
                else:
                    loss = loss_class

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> Dict:
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds = []
        all_labels = []

        for embeddings, labels in val_loader:
            embeddings = embeddings.to(self.device)
            labels = labels.to(self.device)

            if self.use_amp:
                with autocast(device_type='cuda', dtype=torch.float16):
                    logits = self.model(embeddings, modality='speech')
                    loss = self.loss_fn(logits, labels)
            else:
                logits = self.model(embeddings, modality='speech')
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Compute metrics
        wa = (all_preds == all_labels).mean()

        # UA: per-class accuracy averaged
        per_class_acc = []
        for c in range(self.config['num_classes']):
            mask = all_labels == c
            if mask.sum() > 0:
                per_class_acc.append((all_preds[mask] == c).mean())
            else:
                per_class_acc.append(0.0)
        ua = np.mean(per_class_acc)

        # Confusion matrix
        conf_matrix = np.zeros((self.config['num_classes'], self.config['num_classes']), dtype=int)
        for true_l, pred_l in zip(all_labels, all_preds):
            conf_matrix[int(true_l), int(pred_l)] += 1

        return {
            'loss': total_loss / max(n_batches, 1),
            'ua': float(ua),
            'wa': float(wa),
            'per_class_acc': [float(x) for x in per_class_acc],
            'confusion_matrix': conf_matrix.tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
        }

    def setup_optimizer_and_scheduler(self, phase: str, T_max: int):
        if phase == 'lp':
            # Phase 1: Linear Probing
            # Freeze projections
            for param in self.model.speech_projection.parameters():
                param.requires_grad = False
            for param in self.model.music_projection.parameters():
                param.requires_grad = False
            # Unfreeze classifier
            for param in self.model.classifier.parameters():
                param.requires_grad = True
                
            # Optimizer on classifier parameters only
            self.optimizer = optim.AdamW(
                [p for p in self.model.parameters() if p.requires_grad],
                lr=self.config.get('lp_lr', 1e-3),
                weight_decay=self.config['weight_decay'],
            )
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=1e-6,
            )
            self.logger.info(f"  [LP Phase] Freezing projection heads. Optimizer LR={self.config.get('lp_lr', 1e-3)}")
        else:
            # Phase 2: Fine-Tuning
            # Unfreeze projections
            for param in self.model.speech_projection.parameters():
                param.requires_grad = True
            for param in self.model.music_projection.parameters():
                param.requires_grad = True
            # Classifier stays trainable
            for param in self.model.classifier.parameters():
                param.requires_grad = True
                
            # Discriminative learning rates
            proj_params = list(self.model.speech_projection.parameters()) + list(self.model.music_projection.parameters())
            clf_params = list(self.model.classifier.parameters())
            
            self.optimizer = optim.AdamW([
                {'params': proj_params, 'lr': self.config.get('ft_proj_lr', 1e-5)},
                {'params': clf_params, 'lr': self.config['learning_rate']}
            ], weight_decay=self.config['weight_decay'])
            
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=T_max,
                eta_min=1e-6,
            )
            self.logger.info(f"  [FT Phase] Unfreezing projection heads. Projection LR={self.config.get('ft_proj_lr', 1e-5)}, Classifier LR={self.config['learning_rate']}")

    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> Dict:
        """Full training loop with LP-FT protocol.
        
        LP phase: runs for exactly lp_epochs, NO early stopping.
        FT phase: runs with early stopping (patience).
        """

        self.logger.info(f"  Training for up to {self.config['num_epochs']} epochs (patience={self.config['patience']})")

        best_metrics = None
        current_phase = 'lp' if self.lp_epochs > 0 else 'ft'

        for epoch in range(self.config['num_epochs']):
            # Transition from LP to FT
            if self.lp_epochs > 0 and epoch == self.lp_epochs:
                self.setup_optimizer_and_scheduler('ft', self.config['num_epochs'] - self.lp_epochs)
                current_phase = 'ft'
                # Reset tracking for FT phase — LP best doesn't count
                self.patience_counter = 0
                self.best_val_loss = float('inf')
                self.best_model_state = None
                best_metrics = None

            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            val_loss = val_metrics['loss']
            val_ua = val_metrics['ua']
            val_wa = val_metrics['wa']

            # Track best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_model_state = deepcopy(self.model.state_dict())
                best_metrics = val_metrics
                best_metrics['best_epoch'] = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Log every 10 epochs or on improvement or on phase transition
            is_transition_epoch = (self.lp_epochs > 0 and (epoch == self.lp_epochs - 1 or epoch == self.lp_epochs))
            if (epoch + 1) % 10 == 0 or self.patience_counter == 0 or is_transition_epoch:
                marker = " \u2605" if self.patience_counter == 0 else ""
                phase_str = f"[{current_phase.upper()}] " if self.lp_epochs > 0 else ""
                self.logger.info(
                    f"    {phase_str}Epoch {epoch+1:3d} | Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | UA: {val_ua:.4f} | WA: {val_wa:.4f}{marker}"
                )

            # Early stopping — ONLY during FT phase (never during LP)
            if current_phase == 'ft' and self.patience_counter >= self.config['patience']:
                self.logger.info(f"    Early stopping at epoch {epoch+1} (patience={self.config['patience']})")
                break

        # Restore best model
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        # Final eval with best model
        final_metrics = self.validate(val_loader)
        final_metrics['best_epoch'] = best_metrics.get('best_epoch', 0) if best_metrics else 0
        final_metrics['total_epochs'] = epoch + 1

        self.logger.info(
            f"  [OK] Best model: Epoch {final_metrics['best_epoch']} | "
            f"UA: {final_metrics['ua']:.4f} | WA: {final_metrics['wa']:.4f}"
        )

        return final_metrics


# ============================================================================
# DATA LOADING — Build embeddings + labels from IEMOCAP
# ============================================================================

def load_data(project_root: Path, logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load pre-extracted HuBERT embeddings and IEMOCAP emotion labels, combining train and test splits correctly.
    """
    sys.path.insert(0, str(project_root / 'src'))
    from cmsmet.data.iemocap import IEMOCAPDataset

    # Paths to embeddings
    emb_dir = project_root / 'outputs' / 'embeddings'
    train_emb_path = emb_dir / 'iemocap_embeddings.npy'
    test_emb_path = emb_dir / 'iemocap_test_embeddings.npy'

    if not train_emb_path.exists():
        raise FileNotFoundError(f"Train embeddings not found: {train_emb_path}")
    if not test_emb_path.exists():
        raise FileNotFoundError(f"Test embeddings not found: {test_emb_path}")

    train_emb_data = np.load(str(train_emb_path), allow_pickle=True).item()
    test_emb_data = np.load(str(test_emb_path), allow_pickle=True).item()

    logger.info(f"Loaded {len(train_emb_data)} train embeddings and {len(test_emb_data)} test embeddings")

    # Load datasets
    iemocap_root = project_root / 'IEMOCAP_full_release'
    
    train_dataset = IEMOCAPDataset(
        root_dir=str(iemocap_root),
        split='train',
        sr=16000,
        min_duration=0.5,
        max_duration=30.0,
        use_4class=True,
        seed=42,
    )
    test_dataset = IEMOCAPDataset(
        root_dir=str(iemocap_root),
        split='test',
        sr=16000,
        min_duration=0.5,
        max_duration=30.0,
        use_4class=True,
        seed=42,
    )

    logger.info(f"Loaded {len(train_dataset)} train samples and {len(test_dataset)} test samples from IEMOCAP")

    # Match embeddings to labels for train
    train_samples_count = min(len(train_emb_data), len(train_dataset))
    # Match embeddings to labels for test
    test_samples_count = min(len(test_emb_data), len(test_dataset))
    
    total_samples = train_samples_count + test_samples_count
    embeddings = np.zeros((total_samples, 768), dtype=np.float32)
    labels = np.zeros(total_samples, dtype=np.int64)
    sessions = np.zeros(total_samples, dtype=np.int64)

    valid_count = 0
    
    # Add train samples
    for idx in range(train_samples_count):
        if idx not in train_emb_data:
            continue
        sample = train_dataset.samples[idx]
        emotion = sample['emotion']
        if emotion < 0 or emotion >= 4:
            continue
        embeddings[valid_count] = train_emb_data[idx]
        labels[valid_count] = emotion
        sessions[valid_count] = sample['session']
        valid_count += 1

    # Add test samples
    for idx in range(test_samples_count):
        if idx not in test_emb_data:
            continue
        sample = test_dataset.samples[idx]
        emotion = sample['emotion']
        if emotion < 0 or emotion >= 4:
            continue
        embeddings[valid_count] = test_emb_data[idx]
        labels[valid_count] = emotion
        sessions[valid_count] = sample['session']
        valid_count += 1

    embeddings = embeddings[:valid_count]
    labels = labels[:valid_count]
    sessions = sessions[:valid_count]

    logger.info(f"Matched {valid_count} total samples with valid embeddings and labels")

    # Class distribution
    for c in range(4):
        count = (labels == c).sum()
        logger.info(f"  Class {c} ({EMOTION_NAMES[c]}): {count} samples ({100*count/valid_count:.1f}%)")

    # Session distribution
    for s in range(1, 6):
        count = (sessions == s).sum()
        logger.info(f"  Session {s}: {count} samples")

    return embeddings, labels, sessions


# ============================================================================
# MUSIC EMBEDDINGS LOADING
# ============================================================================

def load_deam_music_embeddings(project_root: Path, logger: logging.Logger) -> Dict[int, List[torch.Tensor]]:
    """Load DEAM music embeddings (HuBERT) and group them by emotion category."""
    deam_emb_path = project_root / 'outputs' / 'embeddings' / 'deam_embeddings.npy'
    if not deam_emb_path.exists():
        logger.warning(f"HuBERT DEAM embeddings not found at {deam_emb_path}.")
        return {0: [], 1: [], 2: [], 3: []}

    logger.info(f"Loading HuBERT DEAM music embeddings from {deam_emb_path}...")
    music_embs = np.load(str(deam_emb_path), allow_pickle=True).item()

    from cmsmet.data.deam_fixed import DEAMDataset
    deam_ds = DEAMDataset(root_dir=str(project_root / 'deam_data'), split="train", metadata_only=True)

    music_by_emo = {0: [], 1: [], 2: [], 3: []}
    for idx in range(len(music_embs)):
        if idx < len(deam_ds):
            sample = deam_ds.samples[idx]
            emo = sample['emotion']
            if emo in music_by_emo:
                music_by_emo[emo].append(torch.tensor(music_embs[idx], dtype=torch.float32))

    total_music = sum(len(v) for v in music_by_emo.values())
    logger.info(f"Loaded {total_music} HuBERT music embeddings: "
                f"Happy: {len(music_by_emo[0])}, Sad: {len(music_by_emo[1])}, Angry: {len(music_by_emo[2])}, Neutral: {len(music_by_emo[3])}")

    return music_by_emo


def load_deam_mert_embeddings(project_root: Path, logger: logging.Logger) -> Dict[int, List[torch.Tensor]]:
    """Load DEAM music embeddings extracted with MERT-v1-95M and group by emotion."""
    mert_emb_path = project_root / 'outputs' / 'embeddings' / 'deam_mert_embeddings.npy'
    if not mert_emb_path.exists():
        logger.warning(f"MERT DEAM embeddings not found at {mert_emb_path}. "
                       f"Run: python scripts/extract_music_embeddings_mert.py first.")
        logger.warning("Falling back to HuBERT music embeddings for SupCon.")
        return load_deam_music_embeddings(project_root, logger)

    logger.info(f"Loading MERT DEAM music embeddings from {mert_emb_path}...")
    music_embs = np.load(str(mert_emb_path), allow_pickle=True).item()

    # Use deam_fixed with encoder='mert' for consistent metadata
    from cmsmet.data.deam_fixed import DEAMDataset
    deam_ds = DEAMDataset(root_dir=str(project_root / 'deam_data'), split="train",
                          encoder="mert", metadata_only=True)

    music_by_emo = {0: [], 1: [], 2: [], 3: []}
    for idx in range(len(music_embs)):
        if idx < len(deam_ds):
            sample = deam_ds.samples[idx]
            emo = sample['emotion']
            if emo in music_by_emo:
                music_by_emo[emo].append(torch.tensor(music_embs[idx], dtype=torch.float32))

    total_music = sum(len(v) for v in music_by_emo.values())
    logger.info(f"Loaded {total_music} MERT music embeddings: "
                f"Happy: {len(music_by_emo[0])}, Sad: {len(music_by_emo[1])}, Angry: {len(music_by_emo[2])}, Neutral: {len(music_by_emo[3])}")

    # Verify dimension
    if total_music > 0:
        first_key = next(k for k, v in music_by_emo.items() if v)
        dim = music_by_emo[first_key][0].shape[0]
        logger.info(f"  MERT embedding dimension: {dim}")

    return music_by_emo


# ============================================================================
# LOSO EVALUATION — Leave-One-Session-Out
# ============================================================================

def run_loso_experiment(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sessions: np.ndarray,
    baseline_config: dict,
    contrastive_config: dict,
    seeds: List[int],
    device: torch.device,
    logger: logging.Logger,
    music_by_emo_hubert: Dict[int, List[torch.Tensor]],
    music_by_emo_mert: Dict[int, List[torch.Tensor]],
) -> Dict:
    """
    Run full LOSO experiment for BOTH baseline and contrastive models.

    Baseline:     random projection + simple end-to-end training (no LP-FT)
    Contrastive:  SupCon pre-trained projection (MERT music) + LP-FT

    Stage 1 SupCon is retrained PER FOLD, excluding the test session
    to prevent data leakage. Uses MERT music embeddings.

    5-fold LOSO x N seeds = 5N runs per model type.

    Returns dict with all results.
    """
    results = {
        'baseline': {'folds': [], 'all_ua': [], 'all_wa': []},
        'contrastive': {'folds': [], 'all_ua': [], 'all_wa': []},
    }

    # Config map per model type
    config_map = {
        'baseline': baseline_config,
        'contrastive': contrastive_config,
    }
    # Music embeddings map: MERT for SupCon pre-training, HuBERT for baseline (unused but needed for interface)
    music_map = {
        'baseline': music_by_emo_hubert,
        'contrastive': music_by_emo_mert,
    }

    unique_sessions = sorted(np.unique(sessions))
    logger.info(f"\n{'='*70}")
    logger.info(f"LOSO EXPERIMENT: {len(unique_sessions)} folds x {len(seeds)} seeds = {len(unique_sessions)*len(seeds)} runs per model")
    logger.info(f"  Baseline:     simple end-to-end (lp_epochs=0)")
    logger.info(f"  Contrastive:  SupCon(MERT) + LP-FT (lp_epochs={contrastive_config['lp_epochs']})")
    logger.info(f"{'='*70}\n")

    for fold_idx, test_session in enumerate(unique_sessions):
        test_mask = sessions == test_session
        train_mask = ~test_mask

        train_emb = embeddings[train_mask]
        train_labels = labels[train_mask]
        test_emb = embeddings[test_mask]
        test_labels = labels[test_mask]

        logger.info(f"--- FOLD {fold_idx+1}/5: Test Session {test_session} (train={train_mask.sum()}, test={test_mask.sum()}) ---")

        # ===== Per-fold Stage 1: train SupCon using MERT music embeddings =====
        fold_ckpt = train_stage1_for_fold(
            speech_embeddings=embeddings,
            speech_labels=labels,
            speech_sessions=sessions,
            test_session=test_session,
            music_by_emo=music_by_emo_mert,   # Use MERT music for SupCon
            config=contrastive_config,
            device=device,
            logger=logger,
        )

        for model_type in ['baseline', 'contrastive']:
            fold_results = {'session': int(test_session), 'seed_results': []}
            cfg = config_map[model_type]
            mus = music_map[model_type]

            for seed in seeds:
                # Set seed for reproducibility
                torch.manual_seed(seed)
                np.random.seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed(seed)

                # Create dataset
                train_dataset = EmbeddingDataset(train_emb, train_labels, sessions[train_mask])
                test_dataset = EmbeddingDataset(test_emb, test_labels, sessions[test_mask])

                train_loader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True, num_workers=0, pin_memory=True)
                test_loader = DataLoader(test_dataset, batch_size=cfg['batch_size'], shuffle=False, num_workers=0, pin_memory=True)

                # Create model
                if model_type == 'baseline':
                    model = create_baseline_model(cfg)
                else:
                    model = create_contrastive_model(cfg, fold_ckpt)

                # Train with model-specific config
                trainer = FairTrainer(model, cfg, device, model_type, logger, mus)
                metrics = trainer.train(train_loader, test_loader)

                fold_results['seed_results'].append({
                    'seed': seed,
                    'ua': metrics['ua'],
                    'wa': metrics['wa'],
                    'per_class_acc': metrics['per_class_acc'],
                    'confusion_matrix': metrics['confusion_matrix'],
                    'best_epoch': metrics['best_epoch'],
                    'total_epochs': metrics['total_epochs'],
                })

                results[model_type]['all_ua'].append(metrics['ua'])
                results[model_type]['all_wa'].append(metrics['wa'])

            # Compute fold statistics
            fold_uas = [r['ua'] for r in fold_results['seed_results']]
            fold_was = [r['wa'] for r in fold_results['seed_results']]
            fold_results['ua_mean'] = float(np.mean(fold_uas))
            fold_results['ua_std'] = float(np.std(fold_uas))
            fold_results['wa_mean'] = float(np.mean(fold_was))
            fold_results['wa_std'] = float(np.std(fold_was))

            results[model_type]['folds'].append(fold_results)

            logger.info(
                f"  [{model_type.upper():12s}] Fold {fold_idx+1} -- "
                f"UA: {fold_results['ua_mean']:.4f}+/-{fold_results['ua_std']:.4f} | "
                f"WA: {fold_results['wa_mean']:.4f}+/-{fold_results['wa_std']:.4f}"
            )

    # Compute overall statistics
    for model_type in ['baseline', 'contrastive']:
        all_ua = results[model_type]['all_ua']
        all_wa = results[model_type]['all_wa']
        results[model_type]['overall_ua_mean'] = float(np.mean(all_ua))
        results[model_type]['overall_ua_std'] = float(np.std(all_ua))
        results[model_type]['overall_wa_mean'] = float(np.mean(all_wa))
        results[model_type]['overall_wa_std'] = float(np.std(all_wa))

    # Print summary
    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"  Baseline (HuBERT-only, simple): UA = {results['baseline']['overall_ua_mean']:.4f} +/- {results['baseline']['overall_ua_std']:.4f}")
    logger.info(f"  CMSMET  (MERT+SupCon+LP-FT):    UA = {results['contrastive']['overall_ua_mean']:.4f} +/- {results['contrastive']['overall_ua_std']:.4f}")

    improvement = results['contrastive']['overall_ua_mean'] - results['baseline']['overall_ua_mean']
    logger.info(f"  Improvement:                   ΔUA = {improvement:+.4f} ({improvement*100:+.2f}%)")

    # Paired t-test
    from scipy import stats
    t_stat, p_value = stats.ttest_rel(results['contrastive']['all_ua'], results['baseline']['all_ua'])
    results['significance'] = {'t_stat': float(t_stat), 'p_value': float(p_value)}
    logger.info(f"  Statistical significance:      t={t_stat:.3f}, p={p_value:.4f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(n.s.)'}")
    logger.info(f"{'='*70}\n")

    return results


# ============================================================================
# PAPER-READY OUTPUT
# ============================================================================

def save_paper_results(results: Dict, output_dir: Path, logger: logging.Logger):
    """Save results in paper-ready format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full JSON results
    json_path = output_dir / 'experiment_results.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved full results to {json_path}")

    # 2. Summary table (for paper)
    summary_path = output_dir / 'paper_table.txt'
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("Table: Cross-Modal Speech-Music Emotion Transfer Results\n")
        f.write("Dataset: IEMOCAP (4-class), Evaluation: 5-fold LOSO\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"{'Model':<30} {'UA':>12} {'WA':>12} {'p-value':>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Baseline (HuBERT-only)':<30} "
                f"{results['baseline']['overall_ua_mean']:.4f}+/-{results['baseline']['overall_ua_std']:.4f}  "
                f"{results['baseline']['overall_wa_mean']:.4f}+/-{results['baseline']['overall_wa_std']:.4f}  "
                f"{'--':>10}\n")
        f.write(f"{'CMSMET (SupCon+LP-FT)':<30} "
                f"{results['contrastive']['overall_ua_mean']:.4f}+/-{results['contrastive']['overall_ua_std']:.4f}  "
                f"{results['contrastive']['overall_wa_mean']:.4f}+/-{results['contrastive']['overall_wa_std']:.4f}  "
                f"{results['significance']['p_value']:.4f}\n")
        f.write("-" * 70 + "\n")

        improvement = results['contrastive']['overall_ua_mean'] - results['baseline']['overall_ua_mean']
        f.write(f"\nImprovement: ΔUA = {improvement:+.4f} ({improvement*100:+.2f}%)\n")
        f.write(f"Statistical significance: t={results['significance']['t_stat']:.3f}, p={results['significance']['p_value']:.4f}\n")

        # Per-fold breakdown
        f.write(f"\n\n{'='*80}\nPer-Fold Breakdown\n{'='*80}\n\n")
        f.write(f"{'Fold':<8} {'Session':<10} {'Baseline UA':>14} {'Contrastive UA':>16} {'Δ':>10}\n")
        f.write("-" * 60 + "\n")
        for i in range(5):
            b_fold = results['baseline']['folds'][i]
            c_fold = results['contrastive']['folds'][i]
            delta = c_fold['ua_mean'] - b_fold['ua_mean']
            f.write(f"{'Fold '+str(i+1):<8} {b_fold['session']:<10} "
                    f"{b_fold['ua_mean']:.4f}±{b_fold['ua_std']:.4f}  "
                    f"{c_fold['ua_mean']:.4f}±{c_fold['ua_std']:.4f}  "
                    f"{delta:+.4f}\n")

        # Per-class accuracy
        f.write(f"\n\n{'='*80}\nPer-Class Accuracy (averaged across folds and seeds)\n{'='*80}\n\n")
        for model_type in ['baseline', 'contrastive']:
            f.write(f"\n{model_type.upper()}:\n")
            all_per_class = []
            for fold in results[model_type]['folds']:
                for seed_result in fold['seed_results']:
                    all_per_class.append(seed_result['per_class_acc'])
            all_per_class = np.array(all_per_class)
            for c in range(4):
                mean_acc = all_per_class[:, c].mean()
                std_acc = all_per_class[:, c].std()
                f.write(f"  {EMOTION_NAMES[c]:<12}: {mean_acc:.4f} ± {std_acc:.4f}\n")

    logger.info(f"Saved paper table to {summary_path}")

    # 3. LaTeX table
    latex_path = output_dir / 'paper_table.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Cross-Modal Speech-Music Emotion Transfer Results on IEMOCAP (4-class, 5-fold LOSO)}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\begin{tabular}{lcc}\n")
        f.write("\\toprule\n")
        f.write("Model & UA (\\%) & WA (\\%) \\\\\n")
        f.write("\\midrule\n")
        f.write(f"Baseline (HuBERT-only) & "
                f"${results['baseline']['overall_ua_mean']*100:.2f} \\pm {results['baseline']['overall_ua_std']*100:.2f}$ & "
                f"${results['baseline']['overall_wa_mean']*100:.2f} \\pm {results['baseline']['overall_wa_std']*100:.2f}$ \\\\\n")
        f.write(f"\\textbf{{CMSMET (SupCon+LP-FT)}} & "
                f"$\\mathbf{{{results['contrastive']['overall_ua_mean']*100:.2f} \\pm {results['contrastive']['overall_ua_std']*100:.2f}}}$ & "
                f"$\\mathbf{{{results['contrastive']['overall_wa_mean']*100:.2f} \\pm {results['contrastive']['overall_wa_std']*100:.2f}}}$ \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")

        improvement = results['contrastive']['overall_ua_mean'] - results['baseline']['overall_ua_mean']
        f.write(f"\\vspace{{2mm}}\n")
        f.write(f"\\footnotesize{{$\\Delta$UA = {improvement*100:+.2f}\\%, "
                f"$p = {results['significance']['p_value']:.4f}$ (paired t-test)}}\n")
        f.write("\\end{table}\n")

    logger.info(f"Saved LaTeX table to {latex_path}")

    # 4. Confusion matrices (aggregated)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        for idx, model_type in enumerate(['baseline', 'contrastive']):
            # Aggregate confusion matrices
            total_cm = np.zeros((4, 4))
            for fold in results[model_type]['folds']:
                for seed_result in fold['seed_results']:
                    total_cm += np.array(seed_result['confusion_matrix'])

            # Normalize rows
            row_sums = total_cm.sum(axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1
            norm_cm = total_cm / row_sums

            ax = axes[idx]
            im = ax.imshow(norm_cm, cmap='Blues', vmin=0, vmax=1)
            ax.set_xticks(range(4))
            ax.set_yticks(range(4))
            ax.set_xticklabels(EMOTION_NAMES, fontsize=10)
            ax.set_yticklabels(EMOTION_NAMES, fontsize=10)
            ax.set_xlabel('Predicted', fontsize=12)
            ax.set_ylabel('True', fontsize=12)

            title = 'Baseline (random proj.)' if model_type == 'baseline' else 'Contrastive (pretrained proj.)'
            ua = results[model_type]['overall_ua_mean']
            ax.set_title(f'{title}\nUA = {ua:.4f}', fontsize=12, fontweight='bold')

            for i in range(4):
                for j in range(4):
                    color = 'white' if norm_cm[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{norm_cm[i,j]:.2f}', ha='center', va='center', color=color, fontsize=11)

            plt.colorbar(im, ax=ax, fraction=0.046)

        plt.tight_layout()
        cm_path = output_dir / 'confusion_matrices.png'
        plt.savefig(cm_path, dpi=200, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved confusion matrices to {cm_path}")

    except Exception as e:
        logger.warning(f"Could not generate plots: {e}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMSMET Paper Experiment")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456],
                        help='Random seeds (default: 42 123 456)')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device: cuda, cpu, or auto')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Output directory (default: outputs/paper_results)')
    parser.add_argument('--regenerate', action='store_true',
                        help='Only regenerate paper tables and plots from existing results json')
    args = parser.parse_args()

    # Paths
    project_root = Path(__file__).parent.parent

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = project_root / 'outputs' / 'paper_results'
    output_dir.mkdir(parents=True, exist_ok=True)

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)

    # Logger
    logger = logging.getLogger('paper_experiment')
    logger.setLevel(logging.INFO)
    logger.handlers = []

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    # File handler
    log_path = output_dir / 'experiment.log'
    fh = logging.FileHandler(str(log_path), mode='w', encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)

    # Header
    logger.info("=" * 70)
    logger.info("CMSMET PAPER EXPERIMENT (LEAK-FREE: per-fold Stage 1)")
    logger.info("Baseline (HuBERT-only, simple) vs CMSMET (HuBERT+MERT, SupCon+LP-FT)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"Stage 1: SupCon retrained per fold with MERT music (leak-free)")
    logger.info(f"Baseline config: {json.dumps(BASELINE_CONFIG, indent=2)}")
    logger.info(f"Contrastive config: {json.dumps(CONTRASTIVE_CONFIG, indent=2)}")
    logger.info("")

    # Regenerate results if requested
    if args.regenerate:
        json_path = output_dir / 'experiment_results.json'
        if not json_path.exists():
            logger.error(f"Results JSON not found at {json_path}")
            return 1
        logger.info(f"Loading existing results from {json_path}...")
        with open(json_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        save_paper_results(results, output_dir, logger)
        logger.info("Regeneration complete!")
        return 0

    # Load data
    logger.info("Loading speech data...")
    embeddings, labels, sessions = load_data(project_root, logger)

    logger.info("Loading music data...")
    music_by_emo_hubert = load_deam_music_embeddings(project_root, logger)
    music_by_emo_mert = load_deam_mert_embeddings(project_root, logger)

    # Run experiment
    start_time = time.time()
    results = run_loso_experiment(
        embeddings=embeddings,
        labels=labels,
        sessions=sessions,
        baseline_config=BASELINE_CONFIG,
        contrastive_config=CONTRASTIVE_CONFIG,
        seeds=args.seeds,
        device=device,
        logger=logger,
        music_by_emo_hubert=music_by_emo_hubert,
        music_by_emo_mert=music_by_emo_mert,
    )
    elapsed = time.time() - start_time

    # Add metadata
    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seeds': args.seeds,
        'baseline_config': BASELINE_CONFIG,
        'contrastive_config': CONTRASTIVE_CONFIG,
        'elapsed_seconds': elapsed,
        'stage1_mode': 'per-fold SupCon with MERT music (leak-free)',
        'n_samples': len(embeddings),
        'speech_encoder': 'HuBERT-base (facebook/hubert-base-ls960)',
        'music_encoder': 'MERT-v1-95M (m-a-p/MERT-v1-95M)',
    }

    # Save results
    save_paper_results(results, output_dir, logger)

    logger.info(f"\nTotal time: {elapsed/60:.1f} minutes")
    logger.info("Done! Paper-ready results saved.")

    return 0


if __name__ == '__main__':
    sys.exit(main())
