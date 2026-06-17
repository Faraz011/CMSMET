#!/usr/bin/env python3
"""
CMSMET End-to-End Experiment: HuBERT UNFROZEN + Joint Contrastive Regularization
==================================================================================
Stage 1: SupCon on pre-extracted embeddings (fast, unchanged)
Stage 3: Raw audio → HuBERT (UNFROZEN) → Projection → Classifier

Comparison:
  Baseline:     HuBERT(unfrozen) + RANDOM projection + L_CE only
  Contrastive:  HuBERT(unfrozen) + SUPCON projection + L_CE + λ*L_SupCon
  
The contrastive model maintains cross-modal alignment during fine-tuning
via a joint loss: L_total = L_CE + λ * L_SupCon(speech, music)
"""

import sys
import os
import argparse
import logging
import json
import time
from pathlib import Path
from datetime import datetime
from copy import deepcopy
from typing import Dict, List, Tuple, Optional, Set

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
from cmsmet.models.encoders import DeeperProjectionHead
from cmsmet.data.iemocap_fixed import IEMOCAPDataset, collate_audio_batch

from transformers import HubertModel

# ============================================================================
# CONFIGS — Identical for baseline and contrastive (only projection init differs)
# ============================================================================

CONFIG = {
    'embedding_dim': 768,
    'projection_hidden': 256,
    'projection_output': 128,
    'num_classes': 4,
    'dropout': 0.2,
    'num_epochs': 15,
    'batch_size': 4,
    'gradient_clip': 1.0,
    'patience': 5,
    'label_smoothing': 0.1,
    'grad_accum_steps': 16,     # Effective batch = 4 * 16 = 64
    'max_audio_len': 80000,     # 5s at 16kHz
    'num_unfrozen_layers': 2,
    'hubert_lr': 2e-5,
    'projection_lr': 1e-4,
    'classifier_lr': 1e-3,
    'weight_decay': 0.01,
    'lambda_contrastive': 0.1,  # Joint contrastive regularization weight
    'supcon_epochs': 20,
    'supcon_temperature': 0.07,
    'supcon_batch_size': 256,
}

EMOTION_NAMES = ['happy', 'sad', 'angry', 'neutral']


# ============================================================================
# END-TO-END MODEL — HuBERT + Projection + Classifier
# ============================================================================

class EndToEndModel(nn.Module):
    """HuBERT (top layers unfrozen) + DeeperProjectionHead + Classifier."""

    def __init__(self, config: dict):
        super().__init__()
        num_unfrozen = config.get('num_unfrozen_layers', 2)

        # HuBERT encoder
        self.hubert = HubertModel.from_pretrained(
            "facebook/hubert-base-ls960",
            output_hidden_states=True,
        )

        # Freeze everything first
        for param in self.hubert.parameters():
            param.requires_grad = False

        # Unfreeze last N transformer layers + final layer norm
        total_layers = len(self.hubert.encoder.layers)  # 12 for base
        self.unfrozen_layer_indices = list(range(total_layers - num_unfrozen, total_layers))
        for idx in self.unfrozen_layer_indices:
            for param in self.hubert.encoder.layers[idx].parameters():
                param.requires_grad = True
        # Unfreeze final layer norm if it exists
        if hasattr(self.hubert.encoder, 'layer_norm'):
            for param in self.hubert.encoder.layer_norm.parameters():
                param.requires_grad = True

        # Enable gradient checkpointing on unfrozen layers only
        self.hubert.gradient_checkpointing_enable()

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

        # Classifier
        self.classifier = nn.Linear(config['projection_output'], config['num_classes'])
        self.max_audio_len = config.get('max_audio_len', 80000)

    def get_hubert_embedding(self, waveform: torch.Tensor) -> torch.Tensor:
        """Raw audio → mean-pooled HuBERT embedding (768-dim)."""
        if waveform.shape[-1] > self.max_audio_len:
            waveform = waveform[..., :self.max_audio_len]
        outputs = self.hubert(waveform, output_hidden_states=True, return_dict=True)
        hidden = outputs.last_hidden_state  # (batch, seq, 768)
        return hidden.mean(dim=1)           # (batch, 768)

    def forward(self, waveform: torch.Tensor, return_proj: bool = False):
        embedding = self.get_hubert_embedding(waveform)
        proj = self.speech_projection(embedding)
        logits = self.classifier(proj)
        if return_proj:
            return logits, proj
        return logits


def create_baseline_e2e(config: dict) -> EndToEndModel:
    """Random projection init."""
    return EndToEndModel(config)


def create_contrastive_e2e(config: dict, supcon_state: dict) -> EndToEndModel:
    """SupCon-pretrained projection init. Music projection FROZEN as reference."""
    model = EndToEndModel(config)
    model.speech_projection.load_state_dict(supcon_state['speech_projection'])
    if 'music_projection' in supcon_state:
        model.music_projection.load_state_dict(supcon_state['music_projection'])
    # Freeze music projection — used only as contrastive reference
    for param in model.music_projection.parameters():
        param.requires_grad = False
    return model


# ============================================================================
# END-TO-END TRAINER
# ============================================================================

class E2ETrainer:
    """Trainer with discriminative LRs, gradient accumulation, AMP.
    Supports joint contrastive regularization for contrastive model."""

    def __init__(self, model, config, device, model_type, logger,
                 music_by_emo=None):
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.model_type = model_type
        self.logger = logger
        self.music_by_emo = music_by_emo  # None for baseline, dict for contrastive
        self.lambda_c = config.get('lambda_contrastive', 0.1)

        self.loss_fn = nn.CrossEntropyLoss(
            label_smoothing=config.get('label_smoothing', 0.0)
        )
        self.grad_accum = config.get('grad_accum_steps', 1)

        # Only include trainable params in optimizer
        hubert_params = [p for p in self.model.hubert.parameters() if p.requires_grad]
        proj_params = [p for p in (list(self.model.speech_projection.parameters()) +
                       list(self.model.music_projection.parameters())) if p.requires_grad]
        clf_params = list(self.model.classifier.parameters())

        param_groups = []
        if hubert_params:
            param_groups.append({'params': hubert_params, 'lr': config['hubert_lr']})
        if proj_params:
            param_groups.append({'params': proj_params, 'lr': config['projection_lr']})
        param_groups.append({'params': clf_params, 'lr': config['classifier_lr']})

        self.optimizer = optim.AdamW(param_groups, weight_decay=config['weight_decay'])

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config['num_epochs'], eta_min=1e-7
        )
        self.scaler = GradScaler()

        self.best_val_loss = float('inf')
        self.best_model_state = None
        self.patience_counter = 0

        total = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mode = f'JOINT (λ={self.lambda_c})' if music_by_emo else 'CE-only'
        logger.info(f"  [{model_type.upper()}] Params: {total:,} total, {trainable:,} trainable")
        logger.info(f"    Loss: {mode}")
        logger.info(f"    LR: HuBERT={config['hubert_lr']}, Proj={config['projection_lr']}, Clf={config['classifier_lr']}")
        logger.info(f"    Batch={config['batch_size']} x GradAccum={self.grad_accum} = effective {config['batch_size']*self.grad_accum}")

    def _sample_music_batch(self, labels):
        """Sample matching music embeddings for a batch of emotion labels."""
        batch_music = []
        for label in labels.cpu().numpy():
            emo = int(label)
            music_list = self.music_by_emo.get(emo, [])
            if music_list:
                idx = np.random.randint(len(music_list))
                batch_music.append(music_list[idx])
            else:
                batch_music.append(torch.zeros(self.config['embedding_dim']))
        return torch.stack(batch_music).to(self.device)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_sc = 0.0
        n_batches = 0
        self.optimizer.zero_grad()

        for i, batch in enumerate(train_loader):
            waveforms = batch['waveform'].to(self.device)
            labels = batch['emotion'].to(self.device)

            with torch.cuda.amp.autocast():
                if self.music_by_emo is not None:
                    # JOINT: L_CE + λ * L_SupCon
                    logits, speech_proj = self.model(waveforms, return_proj=True)
                    ce_loss = self.loss_fn(logits, labels)

                    # Get music projections (frozen path)
                    music_embs = self._sample_music_batch(labels)
                    with torch.no_grad():
                        music_proj = self.model.music_projection(music_embs)
                    # Normalize for contrastive loss
                    speech_proj_n = nn.functional.normalize(speech_proj, dim=-1)
                    music_proj_n = nn.functional.normalize(music_proj, dim=-1)
                    sc_loss = compute_supcon_loss(
                        speech_proj_n, music_proj_n, labels,
                        self.config.get('supcon_temperature', 0.07)
                    )
                    loss = (ce_loss + self.lambda_c * sc_loss) / self.grad_accum
                    total_ce += ce_loss.item()
                    total_sc += sc_loss.item()
                else:
                    # BASELINE: L_CE only
                    logits = self.model(waveforms)
                    loss = self.loss_fn(logits, labels) / self.grad_accum

            self.scaler.scale(loss).backward()

            if (i + 1) % self.grad_accum == 0 or (i + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config['gradient_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.grad_accum
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        if self.music_by_emo and n_batches > 0:
            self._last_ce = total_ce / n_batches
            self._last_sc = total_sc / n_batches
        return avg_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_preds, all_labels = [], []

        for batch in val_loader:
            waveforms = batch['waveform'].to(self.device)
            labels = batch['emotion'].to(self.device)

            with torch.cuda.amp.autocast():
                logits = self.model(waveforms)
                loss = self.loss_fn(logits, labels)

            total_loss += loss.item()
            n_batches += 1
            all_preds.extend(logits.float().argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        wa = float((all_preds == all_labels).mean())

        per_class_acc = []
        for c in range(self.config['num_classes']):
            mask = all_labels == c
            per_class_acc.append(float((all_preds[mask] == c).mean()) if mask.sum() > 0 else 0.0)
        ua = float(np.mean(per_class_acc))

        return {
            'loss': total_loss / max(n_batches, 1),
            'ua': ua, 'wa': wa,
            'per_class_acc': per_class_acc,
            'confusion_matrix': confusion_matrix(all_labels, all_preds, labels=list(range(4))).tolist(),
            'predictions': all_preds.tolist(),
            'labels': all_labels.tolist(),
        }

    def train(self, train_loader, val_loader):
        self.logger.info(f"  Training {self.config['num_epochs']} epochs (patience={self.config['patience']})")
        best_metrics = None

        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)
            self.scheduler.step()

            vl, vu, vw = val_metrics['loss'], val_metrics['ua'], val_metrics['wa']

            if vl < self.best_val_loss:
                self.best_val_loss = vl
                self.best_model_state = deepcopy(self.model.state_dict())
                best_metrics = val_metrics
                best_metrics['best_epoch'] = epoch + 1
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            if (epoch + 1) % 3 == 0 or self.patience_counter == 0:
                mark = " \u2605" if self.patience_counter == 0 else ""
                extra = ""
                if self.music_by_emo and hasattr(self, '_last_ce'):
                    extra = f" [CE:{self._last_ce:.3f} SC:{self._last_sc:.3f}]"
                self.logger.info(
                    f"    Epoch {epoch+1:2d} | Train: {train_loss:.4f}{extra} | "
                    f"Val: {vl:.4f} | UA: {vu:.4f} | WA: {vw:.4f}{mark}"
                )

            if self.patience_counter >= self.config['patience']:
                self.logger.info(f"    Early stop at epoch {epoch+1}")
                break

        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        final = self.validate(val_loader)
        final['best_epoch'] = best_metrics.get('best_epoch', 0) if best_metrics else 0
        final['total_epochs'] = epoch + 1

        self.logger.info(
            f"  [OK] Best: Epoch {final['best_epoch']} | UA: {final['ua']:.4f} | WA: {final['wa']:.4f}"
        )
        return final


# ============================================================================
# STAGE 1 SUPCON (uses pre-extracted embeddings — unchanged from existing)
# ============================================================================

class ContrastiveProjectionModel(nn.Module):
    def __init__(self, dim=768, proj_dim=128, dropout=0.2):
        super().__init__()
        self.speech_projection = DeeperProjectionHead(dim, 256, proj_dim, dropout)
        self.music_projection = DeeperProjectionHead(dim, 256, proj_dim, dropout)

    def forward(self, speech, music):
        sp = nn.functional.normalize(self.speech_projection(speech), dim=-1)
        mp = nn.functional.normalize(self.music_projection(music), dim=-1)
        return sp, mp


def compute_supcon_loss(speech_proj, music_proj, labels, temperature=0.07):
    B = speech_proj.shape[0]
    device = speech_proj.device
    features = torch.cat([speech_proj, music_proj], dim=0)
    all_labels = torch.cat([labels, labels], dim=0)
    sim = torch.matmul(features, features.T) / temperature
    label_mask = (all_labels.unsqueeze(0) == all_labels.unsqueeze(1)).float()
    self_mask = torch.eye(2 * B, device=device)
    label_mask = label_mask * (1 - self_mask)
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()
    exp_logits = torch.exp(logits) * (1 - self_mask)
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    n_pos = label_mask.sum(dim=1)
    mean_log_prob = (label_mask * log_prob).sum(dim=1) / (n_pos + 1e-8)
    return -mean_log_prob[n_pos > 0].mean()


class FoldPairedDataset(Dataset):
    def __init__(self, speech_embs, speech_labels, music_by_emo):
        self.speech_embs = speech_embs
        self.speech_labels = speech_labels
        self.music_by_emo = {
            emo: [t.numpy() if isinstance(t, torch.Tensor) else t for t in tensors]
            for emo, tensors in music_by_emo.items() if tensors
        }
        self.valid_indices = [
            i for i in range(len(speech_embs))
            if int(speech_labels[i]) in self.music_by_emo
        ]

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        emo = int(self.speech_labels[i])
        speech = torch.tensor(self.speech_embs[i], dtype=torch.float32)
        music_opts = self.music_by_emo[emo]
        music = torch.tensor(music_opts[np.random.randint(len(music_opts))], dtype=torch.float32)
        # Light augmentation
        speech = speech + torch.randn_like(speech) * 0.02
        music = music + torch.randn_like(music) * 0.02
        return speech, music, emo


def train_stage1_for_fold(speech_embs, speech_labels, speech_sessions,
                          test_session, music_by_emo, config, device, logger):
    """Train SupCon projection for one LOSO fold (leak-free)."""
    mask = speech_sessions != test_session
    fold_embs = speech_embs[mask]
    fold_labels = speech_labels[mask]
    logger.info(f"  [Stage1] SupCon excluding session {test_session} "
                f"({mask.sum()} train, {(~mask).sum()} held-out)")

    ds = FoldPairedDataset(fold_embs, fold_labels, music_by_emo)
    loader = DataLoader(ds, batch_size=config['supcon_batch_size'], shuffle=True, num_workers=0)

    model = ContrastiveProjectionModel(
        config['embedding_dim'], config['projection_output'], config['dropout']
    ).to(device)
    opt = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, config['supcon_epochs'], 1e-6)

    best_loss, best_state = float('inf'), None
    for epoch in range(config['supcon_epochs']):
        model.train()
        epoch_loss, nb = 0.0, 0
        for sb, mb, lb in loader:
            sb, mb = sb.to(device), mb.to(device)
            lb = torch.tensor(lb, dtype=torch.long, device=device) if not isinstance(lb, torch.Tensor) else lb.to(device)
            sp, mp = model(sb, mb)
            loss = compute_supcon_loss(sp, mp, lb, config['supcon_temperature'])
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            epoch_loss += loss.item(); nb += 1
        avg = epoch_loss / max(nb, 1); sched.step()
        if avg < best_loss:
            best_loss = avg
            best_state = {
                'speech_projection': deepcopy(model.speech_projection.state_dict()),
                'music_projection': deepcopy(model.music_projection.state_dict()),
            }
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(f"  [Stage1] Epoch {epoch+1}/{config['supcon_epochs']} "
                        f"Loss: {avg:.4f} (best: {best_loss:.4f})")
    logger.info(f"  [Stage1] Done. Best loss: {best_loss:.4f}")
    return best_state


# ============================================================================
# DATA LOADING
# ============================================================================

def load_embeddings(project_root, logger):
    """Load pre-extracted HuBERT embeddings for Stage 1 SupCon."""
    from cmsmet.data.iemocap import IEMOCAPDataset
    emb_dir = project_root / 'outputs' / 'embeddings'
    train_data = np.load(str(emb_dir / 'iemocap_embeddings.npy'), allow_pickle=True).item()
    test_data = np.load(str(emb_dir / 'iemocap_test_embeddings.npy'), allow_pickle=True).item()
    iemocap_root = project_root / 'IEMOCAP_full_release'

    train_ds = IEMOCAPDataset(root_dir=str(iemocap_root), split='train', sr=16000, use_4class=True, seed=42)
    test_ds = IEMOCAPDataset(root_dir=str(iemocap_root), split='test', sr=16000, use_4class=True, seed=42)

    total = min(len(train_data), len(train_ds)) + min(len(test_data), len(test_ds))
    embs = np.zeros((total, 768), dtype=np.float32)
    labels = np.zeros(total, dtype=np.int64)
    sessions = np.zeros(total, dtype=np.int64)
    count = 0

    for idx in range(min(len(train_data), len(train_ds))):
        if idx not in train_data: continue
        s = train_ds.samples[idx]
        if s['emotion'] < 0 or s['emotion'] >= 4: continue
        embs[count], labels[count], sessions[count] = train_data[idx], s['emotion'], s['session']
        count += 1
    for idx in range(min(len(test_data), len(test_ds))):
        if idx not in test_data: continue
        s = test_ds.samples[idx]
        if s['emotion'] < 0 or s['emotion'] >= 4: continue
        embs[count], labels[count], sessions[count] = test_data[idx], s['emotion'], s['session']
        count += 1

    logger.info(f"Loaded {count} embeddings for Stage 1 SupCon")
    return embs[:count], labels[:count], sessions[:count]


def load_mert_music(project_root, logger):
    """Load MERT music embeddings grouped by emotion."""
    path = project_root / 'outputs' / 'embeddings' / 'deam_mert_embeddings.npy'
    if not path.exists():
        logger.warning(f"MERT embeddings not found: {path}")
        return {0: [], 1: [], 2: [], 3: []}
    music_embs = np.load(str(path), allow_pickle=True).item()
    from cmsmet.data.deam_fixed import DEAMDataset
    deam_ds = DEAMDataset(root_dir=str(project_root / 'deam_data'), split="train",
                          encoder="mert", metadata_only=True)
    by_emo = {0: [], 1: [], 2: [], 3: []}
    for idx in range(len(music_embs)):
        if idx < len(deam_ds):
            emo = deam_ds.samples[idx]['emotion']
            if emo in by_emo:
                by_emo[emo].append(torch.tensor(music_embs[idx], dtype=torch.float32))
    total = sum(len(v) for v in by_emo.values())
    logger.info(f"Loaded {total} MERT music embeddings")
    return by_emo


# ============================================================================
# LOSO EXPERIMENT
# ============================================================================

def run_loso_e2e(project_root, config, seeds, device, logger, num_folds=5):
    """LOSO with HuBERT unfrozen."""

    # Load pre-extracted embeddings for Stage 1 SupCon
    embs, labels, sessions = load_embeddings(project_root, logger)
    music_by_emo = load_mert_music(project_root, logger)

    iemocap_root = project_root / 'IEMOCAP_full_release'
    results = {
        'baseline': {'folds': [], 'all_ua': [], 'all_wa': []},
        'contrastive': {'folds': [], 'all_ua': [], 'all_wa': []},
    }

    logger.info(f"\n{'='*70}")
    logger.info(f"E2E LOSO: {num_folds} folds x {len(seeds)} seeds")
    logger.info(f"HuBERT: UNFROZEN (last {config.get('num_unfrozen_layers',2)} layers, lr={config['hubert_lr']})")
    logger.info(f"{'='*70}\n")

    for fold_idx in range(num_folds):
        test_session = fold_idx + 1
        test_speakers = {f'Ses{test_session:02d}M', f'Ses{test_session:02d}F'}

        # Load audio datasets for this fold
        train_ds = IEMOCAPDataset(
            root_dir=str(iemocap_root), split='train', sr=16000,
            test_speakers=test_speakers, use_4class=True, seed=42,
        )
        test_ds = IEMOCAPDataset(
            root_dir=str(iemocap_root), split='test', sr=16000,
            test_speakers=test_speakers, use_4class=True, seed=42,
        )

        train_loader = DataLoader(
            train_ds, batch_size=config['batch_size'], shuffle=True,
            num_workers=2, pin_memory=True, collate_fn=collate_audio_batch,
        )
        test_loader = DataLoader(
            test_ds, batch_size=config['batch_size'], shuffle=False,
            num_workers=2, pin_memory=True, collate_fn=collate_audio_batch,
        )

        logger.info(f"--- FOLD {fold_idx+1}/5: Test {test_speakers} "
                     f"(train={len(train_ds)}, test={len(test_ds)}) ---")

        # Stage 1: SupCon on embeddings (fast)
        supcon_state = train_stage1_for_fold(
            embs, labels, sessions, test_session,
            music_by_emo, config, device, logger,
        )

        # Stage 3: End-to-end fine-tuning
        for model_type in ['baseline', 'contrastive']:
            fold_results = {'seed_results': [], 'ua_scores': [], 'wa_scores': []}

            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)

                if model_type == 'baseline':
                    model = create_baseline_e2e(config)
                else:
                    model = create_contrastive_e2e(config, supcon_state)

                # Contrastive model uses joint training (CE + λ*SupCon)
                mbemo = music_by_emo if model_type == 'contrastive' else None
                trainer = E2ETrainer(model, config, device, model_type, logger,
                                     music_by_emo=mbemo)
                metrics = trainer.train(train_loader, test_loader)

                fold_results['seed_results'].append(metrics)
                fold_results['ua_scores'].append(metrics['ua'])
                fold_results['wa_scores'].append(metrics['wa'])
                results[model_type]['all_ua'].append(metrics['ua'])
                results[model_type]['all_wa'].append(metrics['wa'])

                # Free GPU memory
                del model, trainer
                torch.cuda.empty_cache()

            fold_results['ua_mean'] = float(np.mean(fold_results['ua_scores']))
            fold_results['ua_std'] = float(np.std(fold_results['ua_scores']))
            fold_results['wa_mean'] = float(np.mean(fold_results['wa_scores']))
            fold_results['wa_std'] = float(np.std(fold_results['wa_scores']))
            results[model_type]['folds'].append(fold_results)

            logger.info(f"  [{model_type.upper():12s}] Fold {fold_idx+1} -- "
                        f"UA: {fold_results['ua_mean']:.4f}+/-{fold_results['ua_std']:.4f}")

    # Overall statistics
    from scipy import stats
    for mt in ['baseline', 'contrastive']:
        results[mt]['overall_ua_mean'] = float(np.mean(results[mt]['all_ua']))
        results[mt]['overall_ua_std'] = float(np.std(results[mt]['all_ua']))
        results[mt]['overall_wa_mean'] = float(np.mean(results[mt]['all_wa']))
        results[mt]['overall_wa_std'] = float(np.std(results[mt]['all_wa']))

    if len(results['contrastive']['all_ua']) >= 2:
        t_stat, p_value = stats.ttest_rel(results['contrastive']['all_ua'], results['baseline']['all_ua'])
    else:
        t_stat, p_value = 0.0, 1.0
    results['significance'] = {'t_stat': float(t_stat), 'p_value': float(p_value)}
    improvement = results['contrastive']['overall_ua_mean'] - results['baseline']['overall_ua_mean']

    logger.info(f"\n{'='*70}")
    logger.info("FINAL RESULTS (HuBERT UNFROZEN)")
    logger.info(f"{'='*70}")
    logger.info(f"  Baseline (random proj):    UA = {results['baseline']['overall_ua_mean']:.4f} +/- {results['baseline']['overall_ua_std']:.4f}")
    logger.info(f"  CMSMET   (SupCon proj):    UA = {results['contrastive']['overall_ua_mean']:.4f} +/- {results['contrastive']['overall_ua_std']:.4f}")
    logger.info(f"  Improvement:               dUA = {improvement:+.4f} ({improvement*100:+.2f}%)")
    sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else '(n.s.)'
    logger.info(f"  Significance:              t={t_stat:.3f}, p={p_value:.4f} {sig}")
    logger.info(f"{'='*70}\n")

    return results


# ============================================================================
# RESULTS SAVING
# ============================================================================

def save_results(results, output_dir, logger):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON
    with open(output_dir / 'e2e_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Text table
    with open(output_dir / 'e2e_paper_table.txt', 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CMSMET End-to-End Results (HuBERT UNFROZEN)\n")
        f.write("IEMOCAP 4-class, 5-fold LOSO\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"{'Model':<30} {'UA':>14} {'WA':>14} {'p-value':>10}\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'Baseline (random proj)':<30} "
                f"{results['baseline']['overall_ua_mean']:.4f}+/-{results['baseline']['overall_ua_std']:.4f}  "
                f"{results['baseline']['overall_wa_mean']:.4f}+/-{results['baseline']['overall_wa_std']:.4f}  "
                f"{'--':>10}\n")
        f.write(f"{'CMSMET (SupCon proj)':<30} "
                f"{results['contrastive']['overall_ua_mean']:.4f}+/-{results['contrastive']['overall_ua_std']:.4f}  "
                f"{results['contrastive']['overall_wa_mean']:.4f}+/-{results['contrastive']['overall_wa_std']:.4f}  "
                f"{results['significance']['p_value']:.4f}\n")
        f.write("-" * 70 + "\n")
        imp = results['contrastive']['overall_ua_mean'] - results['baseline']['overall_ua_mean']
        f.write(f"\nImprovement: dUA = {imp:+.4f} ({imp*100:+.2f}%)\n")
        f.write(f"Significance: t={results['significance']['t_stat']:.3f}, p={results['significance']['p_value']:.4f}\n")

        # Per-fold
        f.write(f"\n{'='*80}\nPer-Fold Breakdown\n{'='*80}\n\n")
        f.write(f"{'Fold':<8} {'Baseline UA':>14} {'CMSMET UA':>14} {'Delta':>10}\n")
        f.write("-" * 50 + "\n")
        for i in range(len(results['baseline']['folds'])):
            bu = results['baseline']['folds'][i]['ua_mean']
            cu = results['contrastive']['folds'][i]['ua_mean']
            f.write(f"Fold {i+1:<3} {bu:>14.4f} {cu:>14.4f} {cu-bu:>+10.4f}\n")

    # LaTeX table
    with open(output_dir / 'e2e_paper_table.tex', 'w') as f:
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{CMSMET E2E Results (HuBERT Unfrozen)}\n")
        f.write("\\begin{tabular}{lcc}\n\\toprule\n")
        f.write("Model & UA (\\%) & WA (\\%) \\\\\n\\midrule\n")
        f.write(f"Baseline (random proj) & "
                f"${results['baseline']['overall_ua_mean']*100:.2f} \\pm {results['baseline']['overall_ua_std']*100:.2f}$ & "
                f"${results['baseline']['overall_wa_mean']*100:.2f} \\pm {results['baseline']['overall_wa_std']*100:.2f}$ \\\\\n")
        f.write(f"\\textbf{{CMSMET (SupCon)}} & "
                f"$\\mathbf{{{results['contrastive']['overall_ua_mean']*100:.2f} \\pm {results['contrastive']['overall_ua_std']*100:.2f}}}$ & "
                f"$\\mathbf{{{results['contrastive']['overall_wa_mean']*100:.2f} \\pm {results['contrastive']['overall_wa_std']*100:.2f}}}$ \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    logger.info(f"Results saved to {output_dir}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="CMSMET E2E (HuBERT Unfrozen)")
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--output-dir', type=str, default=None)
    parser.add_argument('--folds', type=int, default=5, help='Number of LOSO folds (1-5)')
    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / 'outputs' / 'paper_results_e2e'
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)

    logger = logging.getLogger('e2e_experiment')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)
    fh = logging.FileHandler(str(output_dir / 'e2e_experiment.log'), mode='w', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s | %(message)s'))
    logger.addHandler(fh)

    logger.info("=" * 70)
    logger.info("CMSMET END-TO-END EXPERIMENT (HuBERT UNFROZEN)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Seeds: {args.seeds}")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2)}")

    start = time.time()
    results = run_loso_e2e(project_root, CONFIG, args.seeds, device, logger, num_folds=args.folds)
    elapsed = time.time() - start

    results['metadata'] = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'seeds': args.seeds,
        'config': CONFIG,
        'elapsed_seconds': elapsed,
        'hubert': 'UNFROZEN (facebook/hubert-base-ls960)',
        'music_encoder': 'MERT-v1-95M (frozen, Stage 1 only)',
    }

    save_results(results, output_dir, logger)
    logger.info(f"Total time: {elapsed/60:.1f} minutes")
    return 0


if __name__ == '__main__':
    sys.exit(main())
