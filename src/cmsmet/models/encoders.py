"""
Audio Encoders for Speech and Music
- HuBERT for speech
- MERT for music
"""
import torch
import torch.nn as nn
from typing import Tuple, Optional
from transformers import HubertModel, AutoModel
import warnings

warnings.filterwarnings('ignore')


class ResidualBlock(nn.Module):
    """Residual Block with LayerNorm, GELU, and Dropout"""
    def __init__(self, dim: int, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.drop = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc1(x)
        x = self.ln1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.ln2(x)
        x = self.drop(x)
        return residual + x


class DeeperProjectionHead(nn.Module):
    """Deeper projection head with LayerNorm, GELU, and Residual connections"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.2):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.res_blocks = nn.Sequential(
            ResidualBlock(hidden_dim, dropout),
            ResidualBlock(hidden_dim, dropout),
        )
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_layer(x)
        x = self.res_blocks(x)
        x = self.output_layer(x)
        return x


class HuBERTEncoder(nn.Module):
    """HuBERT for speech emotion encoding"""
    
    def __init__(
        self,
        model_name: str = "facebook/hubert-base-ls960",
        output_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_last_hidden: bool = True,
    ):
        """
        Args:
            model_name: HuBERT model identifier
            output_dim: Output embedding dimension
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze HuBERT backbone
            use_last_hidden: Whether to use last hidden state (else use mean pooling)
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_last_hidden = use_last_hidden
        
        # Load pretrained HuBERT
        self.hubert = HubertModel.from_pretrained(
            model_name,
            output_hidden_states=True,
        )
        self.hubert_dim = self.hubert.config.hidden_size
        
        # Enable gradient checkpointing to save memory during training
        self.hubert.gradient_checkpointing = True
        
        if freeze_backbone:
            for param in self.hubert.parameters():
                param.requires_grad = False
        
        # Projection to output dimension
        self.projection = DeeperProjectionHead(
            input_dim=self.hubert_dim,
            hidden_dim=256,
            output_dim=output_dim,
            dropout=0.2,
        )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        waveform: torch.Tensor,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform: Audio tensor of shape (batch_size, audio_length)
            return_dict: If True, returns dict with embeddings and hidden states
        
        Returns:
            embeddings: Shape (batch_size, output_dim)
            attention_mask: Shape (batch_size, seq_length) if return_dict=True
        """
        # Ensure audio is float32 and on same device
        waveform = waveform.float().to(self.hubert.device)
        
        # When encoder is frozen, run in inference mode
        if not self.hubert.training:
            with torch.no_grad():
                outputs = self.hubert(
                    waveform,
                    output_hidden_states=True,
                    return_dict=True,
                )
        else:
            # Get HuBERT outputs
            outputs = self.hubert(
                waveform,
                output_hidden_states=True,
                return_dict=True,
            )
        
        # Use last hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_dim)
        
        # Mean pooling over sequence
        if self.use_last_hidden:
            # Use [CLS]-like token (first token)
            pooled = hidden_states[:, 0, :]
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Project to output dimension
        embeddings = self.projection(pooled)
        embeddings = self.norm(embeddings)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        if return_dict:
            return {
                'embeddings': embeddings,
                'hidden_states': hidden_states,
            }
        else:
            return embeddings, hidden_states


class MERTEncoder(nn.Module):
    """MERT for music emotion encoding"""
    
    def __init__(
        self,
        model_name: str = "m-a-p/MERT-v1-95M",
        output_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        use_last_hidden: bool = True,
    ):
        """
        Args:
            model_name: MERT model identifier
            output_dim: Output embedding dimension
            pretrained: Whether to load pretrained weights
            freeze_backbone: Whether to freeze MERT backbone
            use_last_hidden: Whether to use last hidden state
        """
        super().__init__()
        
        self.model_name = model_name
        self.output_dim = output_dim
        self.use_last_hidden = use_last_hidden
        
        # Load pretrained MERT
        try:
            self.mert = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                output_hidden_states=True,
            )
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
            print("Using HuBERT as fallback for music")
            self.mert = HubertModel.from_pretrained(
                "facebook/hubert-base-ls960",
                output_hidden_states=True,
            )
        
        # Enable gradient checkpointing to save memory during training
        self.mert.gradient_checkpointing = True
        
        if hasattr(self.mert, 'config'):
            self.mert_dim = self.mert.config.hidden_size
        else:
            self.mert_dim = 768  # Default
        
        if freeze_backbone:
            for param in self.mert.parameters():
                param.requires_grad = False
        
        # Projection to output dimension
        self.projection = DeeperProjectionHead(
            input_dim=self.mert_dim,
            hidden_dim=256,
            output_dim=output_dim,
            dropout=0.2,
        )
        
        self.norm = nn.LayerNorm(output_dim)
    
    def forward(
        self,
        waveform: torch.Tensor,
        return_dict: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            waveform: Audio tensor of shape (batch_size, audio_length)
            return_dict: If True, returns dict with embeddings and hidden states
        
        Returns:
            embeddings: Shape (batch_size, output_dim)
            attention_mask: Shape (batch_size, seq_length) if return_dict=True
        """
        # Ensure audio is float32 and on same device
        waveform = waveform.float().to(self.mert.device)
        
        # When encoder is frozen, run in inference mode
        if not self.mert.training:
            with torch.no_grad():
                try:
                    outputs = self.mert(
                        waveform,
                        output_hidden_states=True,
                        return_dict=True,
                    )
                except Exception as e:
                    # Fallback if MERT has different interface
                    print(f"Error in MERT forward: {e}, using last 768 dimension")
                    outputs = self.mert(
                        input_values=waveform,
                        output_hidden_states=True,
                        return_dict=True,
                    )
        else:
            # Get MERT outputs
            try:
                outputs = self.mert(
                    waveform,
                    output_hidden_states=True,
                    return_dict=True,
                )
            except Exception as e:
                # Fallback if MERT has different interface
                print(f"Error in MERT forward: {e}, using last 768 dimension")
                outputs = self.mert(
                    input_values=waveform,
                    output_hidden_states=True,
                    return_dict=True,
                )
        
        # Use last hidden states
        hidden_states = outputs.hidden_states[-1]  # (batch_size, seq_length, hidden_dim)
        
        # Mean pooling over sequence
        if self.use_last_hidden:
            pooled = hidden_states[:, 0, :]
        else:
            pooled = hidden_states.mean(dim=1)
        
        # Project to output dimension
        embeddings = self.projection(pooled)
        embeddings = self.norm(embeddings)
        
        # L2 normalize
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
        
        if return_dict:
            return {
                'embeddings': embeddings,
                'hidden_states': hidden_states,
            }
        else:
            return embeddings, hidden_states


class SharedEmotionSpace(nn.Module):
    """Shared emotion embedding space with projection heads"""
    
    def __init__(
        self,
        speech_encoder: HuBERTEncoder,
        music_encoder: MERTEncoder,
        embedding_dim: int = 128,
    ):
        """
        Args:
            speech_encoder: HuBERT encoder
            music_encoder: MERT encoder
            embedding_dim: Shared embedding dimension
        """
        super().__init__()
        
        self.speech_encoder = speech_encoder
        self.music_encoder = music_encoder
        self.embedding_dim = embedding_dim
    
    def forward_speech(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Encode speech audio
        
        Returns:
            embeddings: Shape (batch_size, embedding_dim)
            encoder_output: Dict with hidden states
        """
        embeddings, hidden_states = self.speech_encoder(
            waveform,
            return_dict=False,
        )
        return embeddings, {'hidden_states': hidden_states}
    
    def forward_music(self, waveform: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Encode music audio
        
        Returns:
            embeddings: Shape (batch_size, embedding_dim)
            encoder_output: Dict with hidden states
        """
        embeddings, hidden_states = self.music_encoder(
            waveform,
            return_dict=False,
        )
        return embeddings, {'hidden_states': hidden_states}
    
    def forward(
        self,
        speech_waveform: Optional[torch.Tensor] = None,
        music_waveform: Optional[torch.Tensor] = None,
    ) -> dict:
        """
        Forward pass for both modalities
        
        Args:
            speech_waveform: Speech audio tensor
            music_waveform: Music audio tensor
        
        Returns:
            Dict with embeddings and auxiliary outputs
        """
        output = {}
        
        if speech_waveform is not None:
            speech_emb, speech_aux = self.forward_speech(speech_waveform)
            output['speech_embeddings'] = speech_emb
            output['speech_hidden_states'] = speech_aux['hidden_states']
        
        if music_waveform is not None:
            music_emb, music_aux = self.forward_music(music_waveform)
            output['music_embeddings'] = music_emb
            output['music_hidden_states'] = music_aux['hidden_states']
        
        return output
