"""
Data Pairing Utilities for Contrastive Learning
Pairs speech samples with music samples for cross-modal contrastive learning
"""
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from typing import Tuple, List, Dict, Optional
import numpy as np
from dataclasses import dataclass


@dataclass
class PairedBatch:
    """Batch containing paired speech and music samples"""
    speech_waveform: torch.Tensor      # (B, audio_len)
    music_waveform: torch.Tensor       # (B, audio_len)
    speech_emotion: Optional[torch.Tensor]  # (B,)
    music_emotion: Optional[torch.Tensor]   # (B,)
    speech_sample_id: List[str]        # Sample identifiers
    music_sample_id: List[str]
    
    def to(self, device: torch.device):
        """Move batch to device"""
        return PairedBatch(
            speech_waveform=self.speech_waveform.to(device),
            music_waveform=self.music_waveform.to(device),
            speech_emotion=self.speech_emotion.to(device) if self.speech_emotion is not None else None,
            music_emotion=self.music_emotion.to(device) if self.music_emotion is not None else None,
            speech_sample_id=self.speech_sample_id,
            music_sample_id=self.music_sample_id,
        )


class EmotionAwarePairer:
    """
    Pairs speech and music samples based on emotion similarity
    For contrastive learning: matched emotions are positive pairs
    """
    
    def __init__(self, emotion_mapping: Optional[Dict[str, int]] = None):
        """
        Args:
            emotion_mapping: Map from emotion name to emotion ID
        """
        self.emotion_mapping = emotion_mapping or {
            'ang': 0, 'hap': 1, 'neu': 2, 'sad': 3,  # IEMOCAP
            'fru': 0, 'exc': 1, 'oth': 2,            # IEMOCAP variants
        }
    
    def create_pairing_strategy(
        self,
        speech_dataset,
        music_dataset,
        pairing_type: str = "matched_emotion",
    ) -> Dict[int, Tuple[int, int]]:
        """
        Create pairing strategy between speech and music samples
        
        Args:
            speech_dataset: Speech dataset with emotion labels
            music_dataset: Music dataset with emotion labels
            pairing_type: 'matched_emotion' or 'random'
        
        Returns:
            Dict mapping index to (speech_idx, music_idx) tuple
        """
        if pairing_type == "matched_emotion":
            return self._create_matched_emotion_pairing(speech_dataset, music_dataset)
        elif pairing_type == "random":
            return self._create_random_pairing(len(speech_dataset), len(music_dataset))
        else:
            raise ValueError(f"Unknown pairing type: {pairing_type}")
    
    def _create_matched_emotion_pairing(
        self,
        speech_dataset,
        music_dataset,
    ) -> Dict[int, Tuple[int, int]]:
        """Pair samples with same emotion when possible"""
        
        # Group samples by emotion
        speech_by_emotion = {}
        music_by_emotion = {}
        
        for idx, sample in enumerate(speech_dataset.samples):
            emotion = sample.get('emotion', 'unknown')
            if emotion not in speech_by_emotion:
                speech_by_emotion[emotion] = []
            speech_by_emotion[emotion].append(idx)
        
        for idx, sample in enumerate(music_dataset.samples):
            emotion = sample.get('emotion', 'unknown')
            if emotion not in music_by_emotion:
                music_by_emotion[emotion] = []
            music_by_emotion[emotion].append(idx)
        
        # Create pairings
        pairings = {}
        pair_idx = 0
        
        for emotion, speech_indices in speech_by_emotion.items():
            music_indices = music_by_emotion.get(emotion, [])
            
            if not music_indices:
                # No matching music, use random from entire music dataset
                music_indices = list(range(len(music_dataset)))
            
            for speech_idx in speech_indices:
                # Cycle through available music with this emotion
                music_idx = music_indices[pair_idx % len(music_indices)]
                pairings[pair_idx] = (speech_idx, music_idx)
                pair_idx += 1
        
        return pairings
    
    def _create_random_pairing(
        self,
        num_speech: int,
        num_music: int,
    ) -> Dict[int, Tuple[int, int]]:
        """Random pairing strategy"""
        
        pairings = {}
        for idx in range(min(num_speech, num_music)):
            # Random music sample for each speech sample
            music_idx = np.random.randint(0, num_music)
            pairings[idx] = (idx, music_idx)
        
        return pairings


class PairedDataset(Dataset):
    """
    Dataset that pairs speech and music samples for contrastive training
    """
    
    def __init__(
        self,
        speech_dataset,
        music_dataset,
        pairing_strategy: Dict[int, Tuple[int, int]],
    ):
        """
        Args:
            speech_dataset: Dataset with speech samples
            music_dataset: Dataset with music samples
            pairing_strategy: Dict mapping pair_idx to (speech_idx, music_idx)
        """
        self.speech_dataset = speech_dataset
        self.music_dataset = music_dataset
        self.pairing_strategy = pairing_strategy
        self.pair_indices = sorted(pairing_strategy.keys())
    
    def __len__(self) -> int:
        return len(self.pair_indices)
    
    def __getitem__(self, idx: int) -> Dict:
        """
        Get paired speech and music samples
        
        Returns:
            Dict with:
                - speech_waveform: (audio_len,)
                - music_waveform: (audio_len,)
                - speech_emotion: int
                - music_emotion: int
                - speech_sample_id: str
                - music_sample_id: str
        """
        pair_idx = self.pair_indices[idx]
        speech_idx, music_idx = self.pairing_strategy[pair_idx]
        
        # Get speech sample
        speech_sample = self.speech_dataset[speech_idx]
        
        # Get music sample
        music_sample = self.music_dataset[music_idx]
        
        return {
            'speech_waveform': speech_sample['waveform'],
            'music_waveform': music_sample['waveform'],
            'speech_emotion': speech_sample.get('emotion', -1),
            'music_emotion': music_sample.get('emotion', -1),
            'speech_sample_id': str(speech_idx),
            'music_sample_id': str(music_idx),
        }


class PairedBatchCollator:
    """Collate paired samples into batches"""
    
    def __init__(self, pad_waveform: bool = True):
        """
        Args:
            pad_waveform: Whether to pad waveforms to max length in batch
        """
        self.pad_waveform = pad_waveform
    
    def __call__(self, batch: List[Dict]) -> PairedBatch:
        """
        Collate list of paired samples
        
        Args:
            batch: List of dicts from PairedDataset
        
        Returns:
            PairedBatch with padded tensors
        """
        speech_waveforms = []
        music_waveforms = []
        speech_emotions = []
        music_emotions = []
        speech_ids = []
        music_ids = []
        
        # Collect samples
        for sample in batch:
            speech_waveforms.append(sample['speech_waveform'])
            music_waveforms.append(sample['music_waveform'])
            speech_emotions.append(sample['speech_emotion'])
            music_emotions.append(sample['music_emotion'])
            speech_ids.append(sample['speech_sample_id'])
            music_ids.append(sample['music_sample_id'])
        
        # Pad if needed
        if self.pad_waveform:
            speech_waveforms = self._pad_sequence(speech_waveforms)
            music_waveforms = self._pad_sequence(music_waveforms)
        else:
            # Stack assumes same length
            speech_waveforms = torch.stack(speech_waveforms)
            music_waveforms = torch.stack(music_waveforms)
        
        # Convert to tensors - handle both raw values and tensors
        # Extract scalar values if items are tensors
        speech_emotions_list = [int(e.item()) if isinstance(e, torch.Tensor) else int(e) for e in speech_emotions]
        music_emotions_list = [int(e.item()) if isinstance(e, torch.Tensor) else int(e) for e in music_emotions]
        
        speech_emotions = torch.tensor(speech_emotions_list, dtype=torch.long)
        music_emotions = torch.tensor(music_emotions_list, dtype=torch.long)
        
        return PairedBatch(
            speech_waveform=speech_waveforms,
            music_waveform=music_waveforms,
            speech_emotion=speech_emotions,
            music_emotion=music_emotions,
            speech_sample_id=speech_ids,
            music_sample_id=music_ids,
        )
    
    @staticmethod
    def _pad_sequence(sequences: List[torch.Tensor]) -> torch.Tensor:
        """Pad sequences to max length"""
        max_len = max(seq.shape[0] for seq in sequences)
        padded = []
        
        for seq in sequences:
            if seq.shape[0] < max_len:
                padding = torch.zeros(max_len - seq.shape[0], dtype=seq.dtype)
                seq = torch.cat([seq, padding])
            padded.append(seq)
        
        return torch.stack(padded)


def create_paired_dataloader(
    speech_dataset,
    music_dataset,
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    pairing_type: str = "matched_emotion",
) -> Tuple[DataLoader, Dict]:
    """
    Create dataloader for contrastive learning with paired speech-music samples
    
    Args:
        speech_dataset: Speech audio dataset
        music_dataset: Music audio dataset
        batch_size: Training batch size
        num_workers: Number of workers for data loading
        shuffle: Whether to shuffle pairings
        pairing_type: How to pair samples ('matched_emotion' or 'random')
    
    Returns:
        Tuple of (dataloader, pairing_metadata)
    """
    
    # Create pairing strategy
    pairer = EmotionAwarePairer()
    pairing_strategy = pairer.create_pairing_strategy(
        speech_dataset,
        music_dataset,
        pairing_type=pairing_type,
    )
    
    # Create paired dataset
    paired_dataset = PairedDataset(
        speech_dataset,
        music_dataset,
        pairing_strategy,
    )
    
    # Create dataloader
    collator = PairedBatchCollator(pad_waveform=True)
    dataloader = DataLoader(
        paired_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collator,
    )
    
    # Metadata about pairing
    metadata = {
        'pairing_type': pairing_type,
        'num_pairs': len(paired_dataset),
        'speech_samples': len(speech_dataset),
        'music_samples': len(music_dataset),
        'batch_size': batch_size,
    }
    
    return dataloader, metadata
