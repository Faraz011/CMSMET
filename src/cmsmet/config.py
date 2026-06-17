"""
CMSMET Configuration Module
Cross-Modal Speech-Music Emotion Transfer via Contrastive Learning
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json
from pathlib import Path


@dataclass
class DataConfig:
    """Data loading configuration"""
    iemocap_root: str = r"C:\Users\Faraz\CMSMET\IEMOCAP_full_release"
    deam_root: str = r"C:\Users\Faraz\CMSMET\deam_data"
    
    # Sample rates
    speech_sr: int = 16000
    music_sr: int = 16000
    
    # Audio preprocessing
    min_duration: float = 0.5  # seconds
    max_duration: float = 30.0  # seconds
    
    # Train/val/test split
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15
    
    # Dataloader settings
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True


@dataclass
class EncoderConfig:
    """Encoder model configuration"""
    # Speech encoder (HuBERT)
    speech_encoder_model: str = "facebook/hubert-base-ls960"
    speech_encoder_output_dim: int = 768
    speech_use_pretrained: bool = True
    
    # Music encoder (MERT)
    music_encoder_model: str = "m-a-p/MERT-v1-95M"
    music_encoder_output_dim: int = 256
    music_use_pretrained: bool = True
    
    # Shared embedding space
    embedding_dim: int = 128
    dropout: float = 0.2


@dataclass
class ContrastiveLossConfig:
    """Contrastive learning loss configuration"""
    temperature: float = 0.07
    # SupCon (Supervised Contrastive Loss) - uses emotion labels for better contrastive pairs
    loss_type: str = "supcon"  # or "nt_xent", "triplet"


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 50
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_steps: int = 1000
    
    # Optimization
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Regularization
    gradient_clip: float = 1.0
    
    # Gradient accumulation for memory efficiency
    # Effective batch = batch_size * accumulation_steps
    accumulation_steps: int = 1
    
    # Device
    device: str = "cuda"
    
    # Checkpointing
    save_every_n_epochs: int = 5
    log_every_n_steps: int = 100
    
    # Early stopping
    patience: int = 10
    min_delta: float = 1e-4


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    # Metrics to compute
    compute_correlation: bool = True
    compute_concordance_cc: bool = True
    
    # Cross-modal transfer evaluation
    probe_classifier_hidden_dim: int = 256
    probe_lr: float = 1e-3
    probe_epochs: int = 20
    
    # Save outputs
    save_embeddings: bool = True
    save_visualizations: bool = True


@dataclass
class ExperimentConfig:
    """Complete experiment configuration"""
    # Naming
    experiment_name: str = "cmsmet_v1"
    run_id: Optional[str] = None
    seed: int = 42
    
    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    contrastive: ContrastiveLossConfig = field(default_factory=ContrastiveLossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Paths
    output_root: str = r"C:\Users\Faraz\CMSMET\outputs"
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    
    def __post_init__(self):
        """Setup paths after initialization"""
        Path(self.output_root).mkdir(parents=True, exist_ok=True)
        if self.checkpoint_dir is None:
            self.checkpoint_dir = str(
                Path(self.output_root) / self.experiment_name / "checkpoints"
            )
        if self.log_dir is None:
            self.log_dir = str(
                Path(self.output_root) / self.experiment_name / "logs"
            )
        Path(self.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict:
        """Convert config to dictionary"""
        return {
            "experiment_name": self.experiment_name,
            "run_id": self.run_id,
            "seed": self.seed,
            "data": self.data.__dict__,
            "encoder": self.encoder.__dict__,
            "contrastive": self.contrastive.__dict__,
            "training": self.training.__dict__,
            "evaluation": self.evaluation.__dict__,
            "output_root": self.output_root,
        }
    
    def save(self, path: str):
        """Save config to JSON"""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_json(cls, path: str) -> "ExperimentConfig":
        """Load config from JSON"""
        with open(path, 'r') as f:
            d = json.load(f)
        return cls(**d)


# Default configurations for different experiment setups
CONFIG_SPEECH_ONLY = ExperimentConfig(
    experiment_name="speech_only_baseline"
)

CONFIG_MUSIC_ONLY = ExperimentConfig(
    experiment_name="music_only_baseline"
)

CONFIG_CONTRASTIVE = ExperimentConfig(
    experiment_name="contrastive_shared_space"
)

CONFIG_CROSS_MODAL_TRANSFER = ExperimentConfig(
    experiment_name="cross_modal_transfer"
)
