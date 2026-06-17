"""
Checkpoint Manager for saving and loading model states
Handles different training phases (contrastive, finetuning)
"""
import os
import json
import torch
from pathlib import Path
from typing import Dict, Optional, Any
from datetime import datetime
import hashlib


class CheckpointManager:
    """Manages checkpoint saving/loading with phase tracking and verification"""
    
    def __init__(self, checkpoint_dir: str, experiment_name: str):
        """
        Args:
            checkpoint_dir: Directory to save checkpoints
            experiment_name: Name of experiment for organizing outputs
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.experiment_name = experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Phase-specific subdirectories
        self.contrastive_dir = self.checkpoint_dir / "contrastive"
        self.finetuning_dir = self.checkpoint_dir / "finetuning"
        
        self.contrastive_dir.mkdir(parents=True, exist_ok=True)
        self.finetuning_dir.mkdir(parents=True, exist_ok=True)
    
    def save_contrastive_checkpoint(
        self,
        model: torch.nn.Module,
        speech_encoder_state: Optional[Dict] = None,
        music_encoder_state: Optional[Dict] = None,
        speech_projection_state: Optional[Dict] = None,
        music_projection_state: Optional[Dict] = None,
        optimizer_state: Optional[Dict] = None,
        scheduler_state: Optional[Dict] = None,
        config: Any = None,
        epoch: int = 0,
        global_step: int = 0,
        best_loss: float = float('inf'),
        metadata: Optional[Dict] = None,
        model_state: Optional[Dict] = None,
    ) -> str:
        """
        Save checkpoint from contrastive pretraining phase
        
        Args:
            model: Full model
            speech_encoder_state: HuBERT encoder state dict
            music_encoder_state: MERT encoder state dict
            speech_projection_state: Speech projection head state dict
            music_projection_state: Music projection head state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            config: Experiment config
            epoch: Current epoch
            global_step: Global training step
            best_loss: Best loss so far
            metadata: Additional metadata to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'phase': 'contrastive',
            'epoch': epoch,
            'global_step': global_step,
            'best_loss': best_loss,
            'timestamp': datetime.now().isoformat(),
            
            # Individual component state dicts (may be None for embedding-only runs)
            'speech_encoder_state': speech_encoder_state,
            'music_encoder_state': music_encoder_state,
            'speech_projection_state': speech_projection_state,
            'music_projection_state': music_projection_state,
            'model_state': model_state,
            
            # Optimizer and scheduler
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            
            # Configuration
            'config': config.to_dict() if hasattr(config, 'to_dict') else (config.__dict__ if config is not None else {}),
            
            # Metadata
            'metadata': metadata or {},
        }
        
        # Save checkpoint
        checkpoint_path = (
            self.contrastive_dir / f"epoch_{epoch:03d}_step_{global_step:06d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best checkpoint if this is the best so far
        if metadata and metadata.get('is_best', False):
            best_path = self.contrastive_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Save metadata about best checkpoint
            best_metadata = {
                'epoch': epoch,
                'global_step': global_step,
                'best_loss': best_loss,
                'timestamp': checkpoint['timestamp'],
            }
            with open(self.contrastive_dir / "best_model_info.json", 'w') as f:
                json.dump(best_metadata, f, indent=2)
        
        return str(checkpoint_path)
    
    def save_finetuning_checkpoint(
        self,
        model: torch.nn.Module,
        speech_encoder_state: Dict,
        music_encoder_state: Dict,
        speech_projection_state: Dict,
        music_projection_state: Dict,
        classifier_state: Dict,
        optimizer_state: Dict,
        scheduler_state: Dict,
        config: Any,
        epoch: int,
        global_step: int,
        best_accuracy: float,
        frozen_projection: bool,
        metadata: Optional[Dict] = None,
    ) -> str:
        """
        Save checkpoint from finetuning phase
        
        Args:
            model: Full model
            speech_encoder_state: HuBERT encoder state dict
            music_encoder_state: MERT encoder state dict
            speech_projection_state: Speech projection head state dict
            music_projection_state: Music projection head state dict
            classifier_state: Classification head state dict
            optimizer_state: Optimizer state dict
            scheduler_state: Scheduler state dict
            config: Experiment config
            epoch: Current epoch
            global_step: Global training step
            best_accuracy: Best accuracy so far
            frozen_projection: Whether projection heads were frozen
            metadata: Additional metadata to save
        
        Returns:
            Path to saved checkpoint
        """
        checkpoint = {
            'phase': 'finetuning',
            'epoch': epoch,
            'global_step': global_step,
            'best_accuracy': best_accuracy,
            'frozen_projection': frozen_projection,
            'timestamp': datetime.now().isoformat(),
            
            # Encoder components
            'speech_encoder_state': speech_encoder_state,
            'music_encoder_state': music_encoder_state,
            
            # Projection heads (may be frozen)
            'speech_projection_state': speech_projection_state,
            'music_projection_state': music_projection_state,
            
            # Classification head (NEW)
            'classifier_state': classifier_state,
            
            # Optimizer and scheduler
            'optimizer_state': optimizer_state,
            'scheduler_state': scheduler_state,
            
            # Configuration
            'config': config.to_dict() if hasattr(config, 'to_dict') else config.__dict__,
            
            # Metadata
            'metadata': metadata or {},
        }
        
        # Save checkpoint
        checkpoint_path = (
            self.finetuning_dir / f"epoch_{epoch:03d}_step_{global_step:06d}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        
        # Also save as best checkpoint if this is the best so far
        if metadata and metadata.get('is_best', False):
            best_path = self.finetuning_dir / "best_model.pt"
            torch.save(checkpoint, best_path)
            
            # Save metadata about best checkpoint
            best_metadata = {
                'epoch': epoch,
                'global_step': global_step,
                'best_accuracy': best_accuracy,
                'timestamp': checkpoint['timestamp'],
                'frozen_projection': frozen_projection,
            }
            with open(self.finetuning_dir / "best_model_info.json", 'w') as f:
                json.dump(best_metadata, f, indent=2)
        
        return str(checkpoint_path)
    
    def load_contrastive_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint from contrastive phase
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dict with all checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify phase
        if checkpoint.get('phase') != 'contrastive':
            raise ValueError(
                f"Expected contrastive phase, got {checkpoint.get('phase')}"
            )
        
        return checkpoint
    
    def load_finetuning_checkpoint(self, checkpoint_path: str) -> Dict:
        """
        Load checkpoint from finetuning phase
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Dict with all checkpoint data
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Verify phase
        if checkpoint.get('phase') != 'finetuning':
            raise ValueError(
                f"Expected finetuning phase, got {checkpoint.get('phase')}"
            )
        
        return checkpoint
    
    def get_best_contrastive_checkpoint(self) -> Optional[str]:
        """Get path to best contrastive checkpoint"""
        best_path = self.contrastive_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None
    
    def get_best_finetuning_checkpoint(self) -> Optional[str]:
        """Get path to best finetuning checkpoint"""
        best_path = self.finetuning_dir / "best_model.pt"
        return str(best_path) if best_path.exists() else None
    
    def get_latest_contrastive_checkpoint(self) -> Optional[str]:
        """Get path to latest contrastive checkpoint"""
        checkpoints = sorted(self.contrastive_dir.glob("epoch_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None
    
    def get_latest_finetuning_checkpoint(self) -> Optional[str]:
        """Get path to latest finetuning checkpoint"""
        checkpoints = sorted(self.finetuning_dir.glob("epoch_*.pt"))
        return str(checkpoints[-1]) if checkpoints else None
    
    def get_projection_heads(self, checkpoint_path: str) -> Dict:
        """
        Extract only projection heads from checkpoint
        Useful for transferring pretrained projections to finetuning
        
        Args:
            checkpoint_path: Path to contrastive checkpoint
        
        Returns:
            Dict with projection head state dicts
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        return {
            'speech_projection_state': checkpoint['speech_projection_state'],
            'music_projection_state': checkpoint['music_projection_state'],
        }
    
    def verify_checkpoint_integrity(self, checkpoint_path: str) -> bool:
        """
        Verify checkpoint has all required fields
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            True if checkpoint is valid
        """
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            phase = checkpoint.get('phase')
            if phase == 'contrastive':
                required = [
                    'phase', 'epoch', 'global_step', 'best_loss',
                    'speech_encoder_state', 'music_encoder_state',
                    'speech_projection_state', 'music_projection_state',
                    'optimizer_state', 'scheduler_state', 'config'
                ]
            elif phase == 'finetuning':
                required = [
                    'phase', 'epoch', 'global_step', 'best_accuracy',
                    'frozen_projection', 'speech_encoder_state', 'music_encoder_state',
                    'speech_projection_state', 'music_projection_state',
                    'classifier_state', 'optimizer_state', 'scheduler_state', 'config'
                ]
            else:
                return False
            
            for field in required:
                if field not in checkpoint:
                    print(f"Missing field: {field}")
                    return False
            
            return True
        
        except Exception as e:
            print(f"Error verifying checkpoint: {e}")
            return False
    
    def compute_checkpoint_hash(self, checkpoint_path: str) -> str:
        """
        Compute hash of checkpoint for data integrity verification
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            MD5 hash of checkpoint
        """
        hash_md5 = hashlib.md5()
        with open(checkpoint_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    
    def save_metadata_report(
        self,
        filename: str,
        phase: str,
        data: Dict,
    ) -> str:
        """
        Save metadata report as JSON
        
        Args:
            filename: Name of report file
            phase: Training phase ('contrastive' or 'finetuning')
            data: Data to save
        
        Returns:
            Path to saved report
        """
        if phase == 'contrastive':
            report_dir = self.contrastive_dir
        else:
            report_dir = self.finetuning_dir
        
        report_path = report_dir / filename
        with open(report_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        return str(report_path)
