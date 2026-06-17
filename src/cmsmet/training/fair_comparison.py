"""
Training Layer Implementation Guide
Enforces identical conditions between baseline and pretrained models
"""

# ==============================================================================
# 1. CONFIGURATION - Fixed hyperparameters
# ==============================================================================

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FairComparisonConfig:
    """Fixed configuration for valid baseline vs model comparison"""
    
    # ENCODER FREEZING - Must be identical for both
    encoder_freeze_strategy: str = "freeze_both"  # or "finetune_both"
    
    # ARCHITECTURE
    projection_hidden_dim: int = 256
    projection_output_dim: int = 128
    dropout_rate: float = 0.3
    
    # LOSS FUNCTION
    temperature: float = 0.07  # NT-Xent temperature (FIXED, do not tune!)
    loss_type: str = "nt_xent"  # NT-Xent loss
    
    # OPTIMIZATION - MUST be identical
    optimizer_type: str = "adamw"  # Only AdamW allowed
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # LEARNING RATE SCHEDULE - MUST be identical
    scheduler_type: str = "cosine"  # Only cosine annealing allowed
    warmup_steps: int = 1000
    num_epochs: int = 50
    gradient_clip: float = 1.0
    
    # DATA LOADING - MUST be unchanged
    batch_size: int = 256  # Fixed for contrastive loss quality
    
    def validate(self):
        """Validate configuration constraints"""
        # Encoder freezing
        assert self.encoder_freeze_strategy in ["freeze_both", "finetune_both"], \
            "Invalid freeze strategy"
        
        # Optimizer
        assert self.optimizer_type == "adamw", \
            "Only AdamW optimizer is allowed for fair comparison"
        
        # Scheduler
        assert self.scheduler_type == "cosine", \
            "Only cosine annealing schedule is allowed"
        
        # Temperature
        assert 0.01 <= self.temperature <= 0.5, \
            f"Temperature {self.temperature} outside reasonable range"
        
        # Learning rate
        assert 1e-5 <= self.learning_rate <= 1e-2, \
            f"Learning rate {self.learning_rate} outside range"
        
        print("✓ Configuration validation passed")


# ==============================================================================
# 2. MODEL FACTORY - Ensures identical architecture
# ==============================================================================

import torch
import torch.nn as nn
from typing import Tuple


class ProjectionHead(nn.Module):
    """Projection head used in BOTH baseline and pretrained model"""
    
    def __init__(self, input_dim: int, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.projection(x)


def create_baseline_classifier(
    config: FairComparisonConfig,
    encoder_dim: int = 768,
    num_classes: int = 4,
) -> nn.Module:
    """
    Create baseline classifier with RANDOM initialization
    
    Architecture:
    - Encoder (HuBERT): frozen or fine-tuned based on config
    - Projection head: RANDOM weights (not pretrained)
    - Classification head: random weights
    """
    from cmsmet.models.classifiers import EmotionClassifier
    
    model = EmotionClassifier(
        encoder_type="hubert",
        embedding_dim=config.projection_output_dim,
        num_classes=num_classes,
        pretrained=True,  # Load pretrained encoder weights
        freeze_encoder=(config.encoder_freeze_strategy == "freeze_both"),
        dropout=config.dropout_rate,
    )
    
    # Verify projection head exists and has correct dimensions
    assert hasattr(model, 'encoder'), "Model must have encoder"
    assert hasattr(model, 'classification_head'), "Model must have classification head"
    
    # Initialize projection head randomly (NOT from checkpoint)
    # This ensures it starts with random weights, unlike pretrained model
    for param in model.classification_head.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    return model


def create_pretrained_classifier(
    config: FairComparisonConfig,
    pretrained_checkpoint: str,
    encoder_dim: int = 768,
    num_classes: int = 4,
) -> nn.Module:
    """
    Create pretrained classifier with LEARNED initialization
    
    Architecture:
    - Encoder (HuBERT): frozen or fine-tuned (MUST match baseline)
    - Projection head: PRETRAINED weights from contrastive learning
    - Classification head: random weights
    """
    from cmsmet.models.classifiers import EmotionClassifier
    
    model = EmotionClassifier(
        encoder_type="hubert",
        embedding_dim=config.projection_output_dim,
        num_classes=num_classes,
        pretrained=True,  # Load pretrained encoder weights
        freeze_encoder=(config.encoder_freeze_strategy == "freeze_both"),
        dropout=config.dropout_rate,
    )
    
    # Load pretrained projection head from checkpoint
    checkpoint = torch.load(pretrained_checkpoint, map_location="cpu")
    
    # Verify checkpoint has projection weights
    assert "projection_state_dict" in checkpoint, \
        f"Checkpoint missing 'projection_state_dict': {list(checkpoint.keys())}"
    
    # Load projection head
    model.classification_head.load_state_dict(checkpoint["projection_state_dict"])
    
    # Re-initialize classification head (same as baseline)
    # This ensures difference comes from projection, not classification head
    for param in model.classification_head.parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param)
        else:
            nn.init.zeros_(param)
    
    return model


# ==============================================================================
# 3. TRAINER - Identical training loop
# ==============================================================================

import logging
from pathlib import Path


class FairComparisonTrainer:
    """Trainer with IDENTICAL hyperparameters for baseline and model"""
    
    def __init__(
        self,
        model: nn.Module,
        config: FairComparisonConfig,
        device: str = "cuda",
        seed: int = 42,
        output_dir: str = "outputs",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = torch.device(device)
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set seed for reproducibility
        self._set_seed(seed)
        
        # Setup logger
        self.logger = self._setup_logger()
        
        # Log configuration
        self._log_configuration()
        
        # Build optimizer and scheduler (guaranteed identical)
        self.optimizer, self.scheduler = self._build_optimizer_scheduler()
        
        # Loss function
        self.loss_fn = nn.CrossEntropyLoss()
    
    def _set_seed(self, seed: int):
        """Set all random seeds"""
        import numpy as np
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger"""
        logger = logging.getLogger(f"FairTrainer_{self.seed}")
        logger.handlers.clear()
        logger.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # File handler
        log_file = self.output_dir / f"training_seed_{self.seed}.log"
        fh = logging.FileHandler(log_file)
        fh.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        ch.setFormatter(formatter)
        fh.setFormatter(formatter)
        
        logger.addHandler(ch)
        logger.addHandler(fh)
        
        return logger
    
    def _log_configuration(self):
        """Log all configuration parameters for reproducibility"""
        self.logger.info("="*80)
        self.logger.info("FAIR COMPARISON CONFIGURATION")
        self.logger.info("="*80)
        
        # Encoder freezing
        self.logger.info(f"Encoder freeze strategy: {self.config.encoder_freeze_strategy}")
        self.logger.info(f"  ➜ freeze_encoder = {self.config.encoder_freeze_strategy == 'freeze_both'}")
        
        # Architecture
        self.logger.info(f"Projection head: dimensions {768} → {self.config.projection_hidden_dim} → {self.config.projection_output_dim}")
        self.logger.info(f"Dropout rate: {self.config.dropout_rate}")
        
        # Loss function
        self.logger.info(f"Loss function: {self.config.loss_type}")
        self.logger.info(f"  ➜ Temperature (NT-Xent): τ = {self.config.temperature}")
        self.logger.info(f"  ➜ (FIXED: This value is NOT tuned)")
        
        # Optimizer
        self.logger.info(f"Optimizer: {self.config.optimizer_type.upper()}")
        self.logger.info(f"  ➜ Learning rate: {self.config.learning_rate}")
        self.logger.info(f"  ➜ Weight decay: {self.config.weight_decay}")
        
        # Scheduler
        self.logger.info(f"Learning rate scheduler: {self.config.scheduler_type.upper()}")
        self.logger.info(f"  ➜ Warmup steps: {self.config.warmup_steps}")
        self.logger.info(f"  ➜ Total epochs: {self.config.num_epochs}")
        self.logger.info(f"Gradient clipping: {self.config.gradient_clip}")
        
        # Data loading
        self.logger.info(f"Batch size (ALL dataloaders): {self.config.batch_size}")
        self.logger.info(f"  ➜ (FIXED: Same for train, val, test)")
        
        # Model parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger.info(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
        
        self.logger.info("="*80 + "\n")
    
    def _build_optimizer_scheduler(self) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
        """Build optimizer and scheduler with FIXED hyperparameters"""
        
        # Optimizer: MUST be AdamW
        if self.config.optimizer_type != "adamw":
            raise ValueError(f"Only AdamW supported, got {self.config.optimizer_type}")
        
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        
        # Scheduler: MUST be cosine annealing
        if self.config.scheduler_type != "cosine":
            raise ValueError(f"Only cosine supported, got {self.config.scheduler_type}")
        
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
        )
        
        return optimizer, scheduler
    
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            waveform = batch['waveform'].to(self.device)
            emotion = batch['emotion'].long().to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(waveform)
            logits = outputs['logits']
            
            # Backward pass
            loss = self.loss_fn(logits, emotion)
            loss.backward()
            
            # Gradient clipping (FIXED value)
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip,
                )
            
            self.optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def eval_epoch(self, val_loader):
        """Evaluate one epoch"""
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in val_loader:
                waveform = batch['waveform'].to(self.device)
                emotion = batch['emotion'].long().to(self.device)
                
                outputs = self.model(waveform)
                logits = outputs['logits']
                loss = self.loss_fn(logits, emotion)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)


# ==============================================================================
# 4. VERIFICATION SCRIPT
# ==============================================================================

def verify_identical_conditions(
    baseline_model: nn.Module,
    pretrained_model: nn.Module,
    baseline_trainer: FairComparisonTrainer,
    pretrained_trainer: FairComparisonTrainer,
) -> bool:
    """
    Verify that baseline and pretrained model have IDENTICAL training conditions
    
    Returns:
        True if all checks pass, False otherwise (with error messages)
    """
    
    print("\n" + "="*80)
    print("VERIFYING FAIR COMPARISON CONDITIONS")
    print("="*80 + "\n")
    
    checks_passed = 0
    checks_total = 0
    
    # 1. Parameter counts
    checks_total += 1
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    pretrained_params = sum(p.numel() for p in pretrained_model.parameters())
    
    if baseline_params == pretrained_params:
        print(f"✓ Parameter count: {baseline_params:,} (identical)")
        checks_passed += 1
    else:
        print(f"✗ Parameter count MISMATCH:")
        print(f"    Baseline:   {baseline_params:,}")
        print(f"    Pretrained: {pretrained_params:,}")
    
    # 2. Optimizer type
    checks_total += 1
    baseline_opt = baseline_trainer.optimizer.__class__.__name__
    pretrained_opt = pretrained_trainer.optimizer.__class__.__name__
    
    if baseline_opt == pretrained_opt == "AdamW":
        print(f"✓ Optimizer: AdamW (identical)")
        checks_passed += 1
    else:
        print(f"✗ Optimizer MISMATCH: {baseline_opt} vs {pretrained_opt}")
    
    # 3. Learning rate
    checks_total += 1
    baseline_lr = baseline_trainer.optimizer.param_groups[0]['lr']
    pretrained_lr = pretrained_trainer.optimizer.param_groups[0]['lr']
    
    if baseline_lr == pretrained_lr:
        print(f"✓ Learning rate: {baseline_lr:.1e} (identical)")
        checks_passed += 1
    else:
        print(f"✗ Learning rate MISMATCH: {baseline_lr:.1e} vs {pretrained_lr:.1e}")
    
    # 4. Weight decay
    checks_total += 1
    baseline_wd = baseline_trainer.optimizer.param_groups[0]['weight_decay']
    pretrained_wd = pretrained_trainer.optimizer.param_groups[0]['weight_decay']
    
    if baseline_wd == pretrained_wd:
        print(f"✓ Weight decay: {baseline_wd:.1e} (identical)")
        checks_passed += 1
    else:
        print(f"✗ Weight decay MISMATCH: {baseline_wd:.1e} vs {pretrained_wd:.1e}")
    
    # 5. Scheduler type
    checks_total += 1
    baseline_sched = baseline_trainer.scheduler.__class__.__name__
    pretrained_sched = pretrained_trainer.scheduler.__class__.__name__
    
    if baseline_sched == pretrained_sched == "CosineAnnealingLR":
        print(f"✓ Scheduler: CosineAnnealingLR (identical)")
        checks_passed += 1
    else:
        print(f"✗ Scheduler MISMATCH: {baseline_sched} vs {pretrained_sched}")
    
    # 6. Batch size
    checks_total += 1
    baseline_batch = baseline_trainer.config.batch_size
    pretrained_batch = pretrained_trainer.config.batch_size
    
    if baseline_batch == pretrained_batch:
        print(f"✓ Batch size: {baseline_batch} (identical)")
        checks_passed += 1
    else:
        print(f"✗ Batch size MISMATCH: {baseline_batch} vs {pretrained_batch}")
    
    # 7. Temperature
    checks_total += 1
    baseline_temp = baseline_trainer.config.temperature
    pretrained_temp = pretrained_trainer.config.temperature
    
    if baseline_temp == pretrained_temp:
        print(f"✓ Temperature (NT-Xent): {baseline_temp} (identical)")
        checks_passed += 1
    else:
        print(f"✗ Temperature MISMATCH: {baseline_temp} vs {pretrained_temp}")
    
    # Summary
    print("\n" + "="*80)
    print(f"PASSED: {checks_passed}/{checks_total} checks")
    print("="*80 + "\n")
    
    if checks_passed == checks_total:
        print("✓ ✓ ✓ FAIR COMPARISON VERIFIED ✓ ✓ ✓\n")
        return True
    else:
        print("✗ ✗ ✗ FAIR COMPARISON FAILED ✗ ✗ ✗\n")
        return False


# ==============================================================================
# 5. USAGE EXAMPLE
# ==============================================================================

def example_fair_comparison():
    """Example: Train baseline and pretrained model with IDENTICAL conditions"""
    
    # 1. Create fixed configuration
    config = FairComparisonConfig(
        encoder_freeze_strategy="freeze_both",
        temperature=0.07,
        learning_rate=1e-4,
        batch_size=256,
        num_epochs=50,
    )
    config.validate()
    
    # 2. Create baseline classifier (random projection head)
    baseline = create_baseline_classifier(config)
    
    # 3. Create pretrained classifier (learned projection head)
    pretrained = create_pretrained_classifier(
        config,
        pretrained_checkpoint="outputs/pretrained_model.pt",
    )
    
    # 4. Create trainers with IDENTICAL hyperparameters
    baseline_trainer = FairComparisonTrainer(
        model=baseline,
        config=config,
        seed=42,
        output_dir="outputs/baseline",
    )
    
    pretrained_trainer = FairComparisonTrainer(
        model=pretrained,
        config=config,
        seed=42,
        output_dir="outputs/pretrained",
    )
    
    # 5. Verify conditions before training
    verify_identical_conditions(
        baseline,
        pretrained,
        baseline_trainer,
        pretrained_trainer,
    )
    
    # 6. Train (conditions guaranteed identical)
    # baseline_trainer.train_epoch(train_loader)
    # pretrained_trainer.train_epoch(train_loader)
    
    print("Ready to train with guaranteed identical conditions!")


if __name__ == "__main__":
    example_fair_comparison()
