"""
Contrastive Learning Loss Functions
NT-Xent (Normalized Temperature-scaled Cross Entropy), Triplet Loss, etc.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent)
    Used in contrastive learning (SimCLR, CLIP, etc.)
    
    Pulls similar samples together in embedding space while pushing apart dissimilar ones.
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        """
        Args:
            temperature: Temperature parameter for scaling logits
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings_a: torch.Tensor,
        embeddings_b: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute NT-Xent loss between two embedding sets
        
        Args:
            embeddings_a: Embeddings from first modality (batch_size, embedding_dim)
            embeddings_b: Embeddings from second modality (batch_size, embedding_dim)
            labels: Optional labels for harder mining (batch_size,)
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize embeddings
        embeddings_a = F.normalize(embeddings_a, p=2, dim=1)
        embeddings_b = F.normalize(embeddings_b, p=2, dim=1)
        
        batch_size = embeddings_a.shape[0]
        device = embeddings_a.device
        
        # Compute similarity matrix (batch_size, batch_size)
        # logits_ab: similarity between a and b
        logits_ab = torch.mm(embeddings_a, embeddings_b.t()) / self.temperature
        logits_ba = logits_ab.t()
        
        # Create labels for positive pairs (diagonal is positive)
        # Target should be the class index, not a one-hot matrix
        target = torch.arange(batch_size, device=device, dtype=torch.long)
        
        # Compute loss
        loss_ab = F.cross_entropy(logits_ab, target, reduction=self.reduction)
        loss_ba = F.cross_entropy(logits_ba, target, reduction=self.reduction)
        
        loss = (loss_ab + loss_ba) / 2
        
        return loss


class TripletLoss(nn.Module):
    """Triplet loss for contrastive learning"""
    
    def __init__(self, margin: float = 0.2, reduction: str = "mean"):
        """
        Args:
            margin: Margin for triplet loss
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            anchor: Anchor embeddings (batch_size, embedding_dim)
            positive: Positive embeddings (batch_size, embedding_dim)
            negative: Negative embeddings (batch_size, embedding_dim)
        
        Returns:
            loss: Scalar loss value
        """
        # Normalize
        anchor = F.normalize(anchor, p=2, dim=1)
        positive = F.normalize(positive, p=2, dim=1)
        negative = F.normalize(negative, p=2, dim=1)
        
        # Compute distances
        pos_dist = torch.norm(anchor - positive, p=2, dim=1)
        neg_dist = torch.norm(anchor - negative, p=2, dim=1)
        
        # Triplet loss
        losses = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        
        if self.reduction == "mean":
            return losses.mean()
        else:
            return losses.sum()


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss
    Extends NT-Xent to use label information for harder mining
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        """
        Args:
            temperature: Temperature parameter
            reduction: 'mean' or 'sum'
        """
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: Embeddings of shape (batch_size, embedding_dim)
            labels: Labels of shape (batch_size,)
        
        Returns:
            loss: Scalar loss value
        """
        embeddings = F.normalize(embeddings, p=2, dim=1)
        batch_size = embeddings.shape[0]
        
        # Compute similarity matrix
        similarity = torch.mm(embeddings, embeddings.t()) / self.temperature
        
        # Create mask for positive pairs (same label)
        mask = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).float()
        
        # Remove self-similarity from positive pairs
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size, device=embeddings.device).view(-1, 1),
            0
        )
        mask = mask * logits_mask
        
        # Compute probabilities
        exp_logits = torch.exp(similarity) * logits_mask
        log_prob = similarity - torch.log(exp_logits.sum(1, keepdim=True) + 1e-10)
        
        # Compute loss (mean over positive pairs)
        loss = -(mask * log_prob).sum(1) / (mask.sum(1) + 1e-10)
        
        if self.reduction == "mean":
            return loss.mean()
        else:
            return loss.sum()


class CrossModalContrastiveLoss(nn.Module):
    """
    Contrastive loss for cross-modal learning (speech-music)
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        loss_type: str = "nt_xent",
        use_labels: bool = True,
    ):
        """
        Args:
            temperature: Temperature for softmax
            loss_type: 'nt_xent', 'triplet', or 'supcon'
            use_labels: Whether to use label information
        """
        super().__init__()
        self.temperature = temperature
        self.loss_type = loss_type
        self.use_labels = use_labels
        
        if loss_type == "nt_xent":
            self.loss_fn = NTXentLoss(temperature=temperature)
        elif loss_type == "triplet":
            self.loss_fn = TripletLoss(margin=0.2)
        elif loss_type == "supcon":
            self.loss_fn = SupConLoss(temperature=temperature)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        speech_embeddings: torch.Tensor,
        music_embeddings: torch.Tensor,
        emotion_labels: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Compute cross-modal contrastive loss
        
        Args:
            speech_embeddings: Speech embeddings (batch_size, embedding_dim)
            music_embeddings: Music embeddings (batch_size, embedding_dim)
            emotion_labels: Emotion labels (batch_size,) for supervised learning
        
        Returns:
            Dict with loss and auxiliary info
        """
        # Normalize embeddings
        speech_embeddings = F.normalize(speech_embeddings, p=2, dim=1)
        music_embeddings = F.normalize(music_embeddings, p=2, dim=1)
        
        if self.loss_type == "nt_xent":
            # NT-Xent between modalities
            loss_s2m = self.loss_fn(speech_embeddings, music_embeddings)
            loss_m2s = self.loss_fn(music_embeddings, speech_embeddings)
            loss = (loss_s2m + loss_m2s) / 2
        
        elif self.loss_type == "triplet" and emotion_labels is not None:
            # Hard triplet mining
            batch_size = speech_embeddings.shape[0]
            # Use different emotion samples as negatives
            neg_mask = torch.ne(emotion_labels.unsqueeze(0), emotion_labels.unsqueeze(1))
            
            losses = []
            for i in range(batch_size):
                neg_indices = torch.where(neg_mask[i])[0]
                if len(neg_indices) > 0:
                    neg_idx = neg_indices[torch.randint(len(neg_indices), (1,))]
                    loss_i = self.loss_fn(
                        speech_embeddings[i].unsqueeze(0),
                        music_embeddings[i].unsqueeze(0),
                        music_embeddings[neg_idx].unsqueeze(0),
                    )
                    losses.append(loss_i)
            
            loss = torch.stack(losses).mean() if losses else torch.tensor(0.0)
        
        elif self.loss_type == "supcon" and emotion_labels is not None:
            # Concatenate modalities and use supervised contrastive loss
            combined_embeddings = torch.cat([speech_embeddings, music_embeddings], dim=0)
            combined_labels = torch.cat([emotion_labels, emotion_labels], dim=0)
            loss = self.loss_fn(combined_embeddings, combined_labels)
        
        else:
            # Fallback to NT-Xent
            loss = self.loss_fn(speech_embeddings, music_embeddings)
        
        return {
            'loss': loss,
            'loss_type': self.loss_type,
        }
