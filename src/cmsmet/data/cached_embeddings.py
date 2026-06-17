"""
Cached Embedding Dataset - stores precomputed embeddings in memory
This allows training on embeddings instead of raw waveforms
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import pickle


class CachedEmbeddingDataset(Dataset):
    """Dataset that stores embeddings in memory for fast access"""
    
    def __init__(self, embeddings: np.ndarray, emotions: np.ndarray):
        """
        Args:
            embeddings: (N, embedding_dim) float32 array
            emotions: (N,) int32 array
        """
        self.embeddings = torch.from_numpy(embeddings).float()
        self.emotions = torch.from_numpy(emotions).long()
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx: int) -> Dict:
        return {
            'embedding': self.embeddings[idx],
            'emotion': self.emotions[idx],
        }


class PairedEmbeddingDataset(Dataset):
    """Dataset with paired speech and music embeddings from same emotion"""
    
    def __init__(
        self,
        speech_embeddings: np.ndarray,
        speech_emotions: np.ndarray,
        music_embeddings: np.ndarray,
        music_emotions: np.ndarray,
    ):
        """
        Args:
            speech_embeddings: (N_speech, speech_dim) array
            speech_emotions: (N_speech,) emotion indices
            music_embeddings: (N_music, music_dim) array
            music_emotions: (N_music,) emotion indices
        """
        self.speech_emb = torch.from_numpy(speech_embeddings).float()
        self.speech_emo = torch.from_numpy(speech_emotions).long()
        
        self.music_emb = torch.from_numpy(music_embeddings).float()
        self.music_emo = torch.from_numpy(music_emotions).long()
        
        # Create emotion-based indexing for matching
        self.emotion_to_music_idx = {}
        for idx, emo in enumerate(music_emotions):
            emo = int(emo)
            if emo not in self.emotion_to_music_idx:
                self.emotion_to_music_idx[emo] = []
            self.emotion_to_music_idx[emo].append(idx)
    
    def __len__(self):
        return len(self.speech_emb)
    
    def __getitem__(self, idx: int) -> Dict:
        speech_idx = idx
        speech_embedding = self.speech_emb[speech_idx]
        speech_emotion = self.speech_emo[speech_idx]
        
        # Match music from same emotion
        music_emotion_idx_list = self.emotion_to_music_idx.get(int(speech_emotion), [])
        if music_emotion_idx_list:
            music_idx = np.random.choice(music_emotion_idx_list)
        else:
            # Fallback to random music
            music_idx = np.random.randint(len(self.music_emb))
        
        music_embedding = self.music_emb[music_idx]
        
        return {
            'speech_embedding': speech_embedding,
            'music_embedding': music_embedding,
            'speech_emotion': speech_emotion,
        }


def precompute_and_cache_embeddings(
    iemocap_dataset,
    deam_dataset,
    encoder_speech,
    encoder_music,
    device: str = "cpu",
) -> tuple:
    """
    Precompute embeddings from raw audio datasets and cache in memory
    
    Args:
        iemocap_dataset: Raw IEMOCAP dataset
        deam_dataset: Raw DEAM dataset
        encoder_speech: Frozen speech encoder (on CPU for memory efficiency)
        encoder_music: Frozen music encoder (on CPU for memory efficiency)
        device: Device for encoding (cpu recommended)
    
    Returns:
        (speech_embeddings, speech_emotions, music_embeddings, music_emotions)
    """
    
    print("\nPrecomputing embeddings...")
    print(f"Encoding on device: {device}")
    
    encoder_speech = encoder_speech.to(device).eval()
    encoder_music = encoder_music.to(device).eval()
    
    speech_embeddings = []
    speech_emotions = []
    
    music_embeddings = []
    music_emotions = []
    
    with torch.no_grad():
        # Process speech (IEMOCAP)
        print("Encoding IEMOCAP...")
        for idx in range(len(iemocap_dataset)):
            try:
                sample = iemocap_dataset[idx]
                
                if isinstance(sample, dict):
                    waveform = sample['waveform']
                    emotion = sample['emotion']
                else:
                    waveform = sample.waveform
                    emotion = sample.emotion
                
                # Convert to tensor if needed
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform).float()
                
                # Add batch dimension and encode
                waveform = waveform.unsqueeze(0).to(device)
                emb, _ = encoder_speech(waveform)
                
                speech_embeddings.append(emb.cpu().numpy())
                speech_emotions.append(int(emotion))
                
                if (idx + 1) % 500 == 0:
                    print(f"  {idx + 1} / {len(iemocap_dataset)} IEMOCAP samples")
                
            except Exception as e:
                print(f"Error encoding IEMOCAP sample {idx}: {e}")
                continue
        
        speech_embeddings = np.concatenate(speech_embeddings, axis=0).astype(np.float32)
        speech_emotions = np.array(speech_emotions, dtype=np.int32)
        print(f"[OK] IEMOCAP embeddings: {speech_embeddings.shape}")
        
        # Process music (DEAM)
        print("Encoding DEAM...")
        for idx in range(len(deam_dataset)):
            try:
                sample = deam_dataset[idx]
                
                if isinstance(sample, dict):
                    waveform = sample['waveform']
                    emotion = sample['emotion']
                else:
                    waveform = sample.waveform
                    emotion = sample.emotion
                
                # Convert to tensor if needed
                if isinstance(waveform, np.ndarray):
                    waveform = torch.from_numpy(waveform).float()
                
                # Add batch dimension and encode
                waveform = waveform.unsqueeze(0).to(device)
                emb, _ = encoder_music(waveform)
                
                music_embeddings.append(emb.cpu().numpy())
                music_emotions.append(int(emotion))
                
                if (idx + 1) % 500 == 0:
                    print(f"  {idx + 1} / {len(deam_dataset)} DEAM samples")
                
            except Exception as e:
                print(f"Error encoding DEAM sample {idx}: {e}")
                continue
        
        music_embeddings = np.concatenate(music_embeddings, axis=0).astype(np.float32)
        music_emotions = np.array(music_emotions, dtype=np.int32)
        print(f"[OK] DEAM embeddings: {music_embeddings.shape}")
    
    return speech_embeddings, speech_emotions, music_embeddings, music_emotions
