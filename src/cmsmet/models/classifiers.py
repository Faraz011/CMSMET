"""
Downstream emotion classification models
- EmotionClassifier: Speech/music → emotion logits
"""
import torch
import torch.nn as nn
from typing import Optional, Dict
from cmsmet.models.encoders import HuBERTEncoder, MERTEncoder


class EmotionClassifier(nn.Module):
    """Speech/Music emotion classifier with pretrained encoders"""
    
    def __init__(
        self,
        encoder_type: str = "hubert",  # 'hubert' or 'mert'
        model_name: Optional[str] = None,
        embedding_dim: int = 128,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_encoder: bool = False,
        dropout: float = 0.3,
    ):
        """
        Args:
            encoder_type: Type of encoder ('hubert' for speech, 'mert' for music)
            model_name: Model identifier (uses defaults if None)
            embedding_dim: Output embedding dimension from encoder
            num_classes: Number of emotion classes (4 for standard, 5 for full)
            pretrained: Whether to load pretrained encoder weights
            freeze_encoder: Whether to freeze encoder backbone
            dropout: Dropout rate for classification head
        """
        super().__init__()
        
        self.encoder_type = encoder_type
        self.num_classes = num_classes
        
        # Initialize encoder based on type
        if encoder_type == "hubert":
            if model_name is None:
                model_name = "facebook/hubert-base-ls960"
            self.encoder = HuBERTEncoder(
                model_name=model_name,
                output_dim=embedding_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_encoder,
                use_last_hidden=True,
            )
            input_type = "speech"
        elif encoder_type == "mert":
            if model_name is None:
                model_name = "m-a-p/MERT-v1-330M"
            self.encoder = MERTEncoder(
                model_name=model_name,
                output_dim=embedding_dim,
                pretrained=pretrained,
                freeze_backbone=freeze_encoder,
            )
            input_type = "music"
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        self.input_type = input_type
        self.embedding_dim = embedding_dim
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(
        self,
        waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            waveform: Audio waveform (B, T) at appropriate sample rate
                     16kHz for HuBERT, 24kHz for MERT
        
        Returns:
            Dict with:
            - 'embeddings': (B, embedding_dim) speech/music embeddings
            - 'logits': (B, num_classes) emotion class logits
        """
        # Encode
        embeddings = self.encoder(waveform)
        
        # Classify
        logits = self.classification_head(embeddings)
        
        return {
            'embeddings': embeddings,
            'logits': logits,
        }
    
    def freeze_encoder(self):
        """Freeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters"""
        for param in self.encoder.parameters():
            param.requires_grad = True


class DualEmotionClassifier(nn.Module):
    """Joint speech+music emotion classifier with contrastive fusion"""
    
    def __init__(
        self,
        embedding_dim: int = 128,
        num_classes: int = 4,
        pretrained: bool = True,
        freeze_encoders: bool = False,
        dropout: float = 0.3,
        fusion_method: str = "concat",  # 'concat', 'attention', 'gating'
    ):
        """
        Args:
            embedding_dim: Shared embedding dimension
            num_classes: Number of emotion classes
            pretrained: Whether to load pretrained encoder weights
            freeze_encoders: Whether to freeze encoder backbones
            dropout: Dropout rate
            fusion_method: How to fuse speech+music embeddings
        """
        super().__init__()
        
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.fusion_method = fusion_method
        
        # Encoders
        self.speech_encoder = HuBERTEncoder(
            output_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_encoders,
            use_last_hidden=True,
        )
        self.music_encoder = MERTEncoder(
            output_dim=embedding_dim,
            pretrained=pretrained,
            freeze_backbone=freeze_encoders,
        )
        
        # Fusion layer
        if fusion_method == "concat":
            fusion_dim = embedding_dim * 2
            self.fusion_layer = None
        elif fusion_method == "attention":
            fusion_dim = embedding_dim
            self.fusion_layer = nn.MultiheadAttention(
                embed_dim=embedding_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
        elif fusion_method == "gating":
            fusion_dim = embedding_dim
            self.fusion_layer = nn.Sequential(
                nn.Linear(embedding_dim * 2, embedding_dim),
                nn.Sigmoid(),
            )
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )
    
    def forward(
        self,
        speech_waveform: torch.Tensor,
        music_waveform: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            speech_waveform: Speech audio (B, T) at 16kHz
            music_waveform: Music audio (B, T) at 24kHz
        
        Returns:
            Dict with 'embeddings' and 'logits'
        """
        # Encode modalities
        speech_emb = self.speech_encoder(speech_waveform)  # (B, embedding_dim)
        music_emb = self.music_encoder(music_waveform)      # (B, embedding_dim)
        
        # Fuse
        if self.fusion_method == "concat":
            fused = torch.cat([speech_emb, music_emb], dim=-1)  # (B, 2*embedding_dim)
        elif self.fusion_method == "attention":
            # Use speech as query, music as key/value
            attn_out, _ = self.fusion_layer(
                speech_emb.unsqueeze(1),
                music_emb.unsqueeze(1),
                music_emb.unsqueeze(1),
            )
            fused = attn_out.squeeze(1) + speech_emb  # Residual
        elif self.fusion_method == "gating":
            gate = self.fusion_layer(torch.cat([speech_emb, music_emb], dim=-1))
            fused = gate * speech_emb + (1 - gate) * music_emb
        
        # Classify
        logits = self.classification_head(fused)
        
        return {
            'speech_embeddings': speech_emb,
            'music_embeddings': music_emb,
            'fused_embeddings': fused,
            'logits': logits,
        }
    
    def freeze_encoders(self):
        """Freeze encoder parameters"""
        for param in self.speech_encoder.parameters():
            param.requires_grad = False
        for param in self.music_encoder.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """Unfreeze encoder parameters"""
        for param in self.speech_encoder.parameters():
            param.requires_grad = True
        for param in self.music_encoder.parameters():
            param.requires_grad = True
