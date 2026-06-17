"""
DEAM Dataset Loader with Encoder-Aware Resampling and Consistent Chunking  
Database for Emotion Analysis using Physiological Signals (DEAM)

CRITICAL FIXES:
1. Encoder-specific resampling: HuBERT @ 16kHz, MERT @ 24kHz
2. Fixed chunk parameters: 10s segments, 5s stride (ALWAYS consistent)
3. Label space: Continuous V/A only (no categorical mixing)
"""
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import csv


class DEAMDataset(Dataset):
    """
    DEAM dataset loader with proper resampling per encoder and consistent chunking
    
    Chunking policy (MUST be consistent across all experiments):
    - Segment duration: 10 seconds
    - Segment stride: 5 seconds (50% overlap)
    - Canonical sample rate: Encoder-specific (16kHz for HuBERT, 24kHz for MERT)
    """
    
    # FIXED: Encoder-specific sample rates to prevent resampling inconsistency
    ENCODER_SAMPLE_RATES = {
        'hubert': 16000,    # HuBERT expects 16kHz
        'mert': 24000,      # MERT expects 24kHz
        'default': 16000,   # Default if encoder not specified
    }
    
    # Sliding window parameters (MUST be consistent)
    SEGMENT_DURATION = 10.0  # seconds
    SEGMENT_STRIDE = 5.0     # seconds (50% overlap)
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sr: Optional[int] = None,  # If None, uses encoder default
        encoder: str = "hubert",  # 'hubert' or 'mert'
        segment_duration: float = SEGMENT_DURATION,  # FIXED: 10s
        segment_stride: float = SEGMENT_STRIDE,      # FIXED: 5s
        min_segment_duration: float = 5.0,
        split_ratio: float = 0.8,  # 80/20 train/test
        cache_path: Optional[str] = None,
        seed: int = 42,
        metadata_only: bool = False,
    ):
        """
        Args:
            root_dir: Path to DEAM dataset root
            split: 'train', 'test', or 'all'
            sr: Sample rate override. If None, uses encoder default
            encoder: 'hubert' (16kHz) or 'mert' (24kHz)
            segment_duration: Window size in seconds (FIXED: 10s)
            segment_stride: Step size in seconds (FIXED: 5s)
            min_segment_duration: Minimum segment length to keep
            split_ratio: Train/test split ratio
            cache_path: Path to cache processed data
            seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.split = split
        self.seed = seed
        
        # FIXED: Use encoder-specific sample rate if sr not override
        self.encoder = encoder.lower()
        if sr is None:
            self.sr = self.ENCODER_SAMPLE_RATES.get(self.encoder, self.ENCODER_SAMPLE_RATES['default'])
        else:
            self.sr = sr
        
        # FIXED: Sliding window parameters (MUST be constant)
        self.segment_duration = segment_duration
        self.segment_stride = segment_stride
        self.min_segment_duration = min_segment_duration
        self.split_ratio = split_ratio
        
        print(f"DEAMDataset initialized:")
        print(f"  Encoder: {self.encoder}")
        print(f"  Sample rate: {self.sr}kHz (encoder-specific)")
        print(f"  Segment duration: {self.segment_duration}s")
        print(f"  Segment stride: {self.segment_stride}s")
        
        self.metadata_only = metadata_only
        self.samples = []
        self._load_dataset(cache_path)
    
    def _get_annotation_dir(self) -> Path:
        """Find annotation directory (handles multiple directory structure variants)"""
        candidates = [
            self.root_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level",
            self.root_dir / "DEAM_Annotations" / "annotations" / "annotations_averaged_per_song" / "song_level",
            self.root_dir / "annotations" / "annotations averaged per song" / "song_level",
            self.root_dir / "song_level",
        ]
        
        for path in candidates:
            if path.exists():
                print(f"Found annotation dir: {path}")
                return path
        
        raise FileNotFoundError(f"Could not find DEAM annotation directory in {self.root_dir}")
    
    def _get_audio_dir(self) -> Path:
        """Find audio directory (handles multiple directory structure variants)"""
        candidates = [
            self.root_dir / "DEAM_audio" / "MEMD_audio",
            self.root_dir / "DEAM_audio" / "audio",
            self.root_dir / "audio",
            self.root_dir / "music_speech_files",
        ]
        
        for path in candidates:
            if path.exists():
                print(f"Found audio dir: {path}")
                return path
        
        raise FileNotFoundError(f"Could not find DEAM audio directory in {self.root_dir}")
    
    def _load_dataset(self, cache_path: Optional[str] = None):
        """Load dataset with fixed chunking and encoder-specific resampling"""
        
        # Check cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
                self.samples = cached_data.get('samples', [])
                # Verify cached data uses same parameters
                if cached_data.get('sr') != self.sr:
                    print(f"Warning: Cache has sr={cached_data.get('sr')} but loader uses sr={self.sr}")
                if cached_data.get('segment_duration') != self.segment_duration:
                    print(f"Warning: Cache has segment_duration={cached_data.get('segment_duration')} but loader uses {self.segment_duration}")
            
            # Ensure cached samples have 'emotion' key populated dynamically
            for sample in self.samples:
                v_norm = sample['valence']
                a_norm = sample['arousal']
                if v_norm >= 0.5 and a_norm >= 0.5:
                    sample['emotion'] = 0  # happy
                elif v_norm < 0.5 and a_norm < 0.5:
                    sample['emotion'] = 1  # sad
                elif v_norm < 0.5 and a_norm >= 0.5:
                    sample['emotion'] = 2  # angry
                else:
                    sample['emotion'] = 3  # neutral
            return
        
        annotation_dir = self._get_annotation_dir()
        audio_dir = self._get_audio_dir()
        
        # Load all CSV annotation files
        print(f"Loading annotations from {annotation_dir}")
        csv_files = sorted(annotation_dir.glob("*.csv"))
        print(f"Found {len(csv_files)} annotation files")
        
        all_tracks = []
        
        for csv_file in csv_files:
            try:
                with open(csv_file, 'r', encoding='utf-8') as f:
                    # Skip BOM if present
                    content = f.read()
                    if content.startswith('\ufeff'):
                        content = content[1:]
                    
                    # Parse CSV with potential spaces in column names
                    lines = content.strip().split('\n')
                    if not lines:
                        continue
                    
                    # Parse header
                    header = [col.strip() for col in lines[0].split(',')]
                    print(f"  CSV header: {header}")
                    
                    # Find column indices (handling spaces)
                    song_id_idx = None
                    valence_idx = None
                    arousal_idx = None
                    
                    for i, col in enumerate(header):
                        col_lower = col.lower()
                        if 'song_id' in col_lower or col_lower == 'id':
                            song_id_idx = i
                        elif 'valence_mean' in col_lower or col_lower == 'valence':
                            valence_idx = i
                        elif 'arousal_mean' in col_lower or col_lower == 'arousal':
                            arousal_idx = i
                    
                    if song_id_idx is None or valence_idx is None or arousal_idx is None:
                        print(f"  Warning: Could not find required columns in {csv_file}")
                        continue
                    
                    # Parse data rows
                    for line in lines[1:]:
                        if not line.strip():
                            continue
                        
                        values = [v.strip() for v in line.split(',')]
                        try:
                            song_id = int(values[song_id_idx])
                            valence = float(values[valence_idx])
                            arousal = float(values[arousal_idx])
                            
                            all_tracks.append({
                                'id': song_id,
                                'valence': valence,
                                'arousal': arousal,
                            })
                        except (ValueError, IndexError) as e:
                            continue
            
            except Exception as e:
                print(f"Error reading {csv_file}: {e}")
                continue
        
        print(f"Loaded {len(all_tracks)} tracks with annotations")
        
        # Split into train/test (reproducible)
        rng = np.random.RandomState(self.seed)
        all_ids = [t['id'] for t in all_tracks]
        rng.shuffle(all_ids)
        split_idx = int(len(all_ids) * self.split_ratio)
        train_ids = set(all_ids[:split_idx])
        test_ids = set(all_ids[split_idx:])
        
        print(f"Split: {len(train_ids)} train, {len(test_ids)} test")
        
        # Process each track
        segment_samples = int(self.segment_duration * self.sr)
        stride_samples = int(self.segment_stride * self.sr)
        
        for track in all_tracks:
            track_id = track['id']
            
            # Check split
            if self.split == "train" and track_id not in train_ids:
                continue
            elif self.split == "test" and track_id not in test_ids:
                continue
            
            # Find audio file (try multiple extensions and formats)
            audio_candidates = [
                audio_dir / f"{track_id}.mp3",
                audio_dir / f"{track_id}.wav",
                audio_dir / f"{str(track_id).zfill(4)}.mp3",
                audio_dir / f"{str(track_id).zfill(4)}.wav",
            ]
            
            audio_path = None
            for candidate in audio_candidates:
                if candidate.exists():
                    audio_path = candidate
                    break
            
            if audio_path is None:
                continue
            
            # Load audio, resample to encoder-specific sr, and chunk
            try:
                # Normalize V/A from [1, 9] to [0, 1]
                valence_norm = (track['valence'] - 1.0) / 8.0
                arousal_norm = (track['arousal'] - 1.0) / 8.0
                
                # Map valence and arousal to quadrant-based emotion class
                if valence_norm >= 0.5 and arousal_norm >= 0.5:
                    emotion_cat = 0  # happy
                elif valence_norm < 0.5 and arousal_norm < 0.5:
                    emotion_cat = 1  # sad
                elif valence_norm < 0.5 and arousal_norm >= 0.5:
                    emotion_cat = 2  # angry
                else:
                    emotion_cat = 3  # neutral

                if self.metadata_only:
                    # Get duration without loading raw audio (much faster)
                    duration = librosa.get_duration(path=str(audio_path))
                    num_segments = 1 + max(0, int((duration - self.segment_duration) // self.segment_stride))
                    for seg_idx in range(num_segments):
                        self.samples.append({
                            'audio': None,
                            'sr': self.sr,
                            'valence': valence_norm,
                            'arousal': arousal_norm,
                            'emotion': emotion_cat,
                            'track_id': track_id,
                            'segment_idx': seg_idx,
                        })
                else:
                    # Load at native rate first, then resample
                    y, sr_native = librosa.load(str(audio_path), sr=None, mono=True)
                    
                    # Resample to encoder-specific rate
                    if sr_native != self.sr:
                        y = librosa.resample(y, orig_sr=sr_native, target_sr=self.sr)
                    
                    # Create chunks with fixed window and stride
                    num_segments = 1 + max(0, (len(y) - segment_samples) // stride_samples)
                    
                    for seg_idx in range(num_segments):
                        start_idx = seg_idx * stride_samples
                        end_idx = start_idx + segment_samples
                        
                        if end_idx > len(y):
                            # Pad final segment
                            segment = np.pad(y[start_idx:], (0, end_idx - len(y)))
                        else:
                            segment = y[start_idx:end_idx]
                        
                        # Check segment duration
                        seg_duration = len(segment) / self.sr
                        if seg_duration >= self.min_segment_duration:
                            self.samples.append({
                                'audio': segment,
                                'sr': self.sr,  # FIXED: Encoder-specific
                                'valence': valence_norm,
                                'arousal': arousal_norm,
                                'emotion': emotion_cat,
                                'track_id': track_id,
                                'segment_idx': seg_idx,
                            })
            
            except Exception as e:
                print(f"Error processing track {track_id}: {e}")
                continue
        
        print(f"Created {len(self.samples)} segments for split '{self.split}'")
        
        # Cache with metadata
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'samples': self.samples,
                    'sr': self.sr,
                    'encoder': self.encoder,
                    'segment_duration': self.segment_duration,
                    'segment_stride': self.segment_stride,
                }, f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return audio tensor and continuous V/A labels plus mapped emotion category"""
        sample = self.samples[idx]
        
        return {
            'waveform': torch.from_numpy(sample['audio']).float() if sample['audio'] is not None else torch.zeros(1),
            'sr': sample['sr'],  # FIXED: Encoder-specific
            'valence': torch.tensor(sample['valence'], dtype=torch.float32),
            'arousal': torch.tensor(sample['arousal'], dtype=torch.float32),
            'emotion': torch.tensor(sample['emotion'], dtype=torch.long),
            'track_id': sample['track_id'],
            'segment_idx': sample['segment_idx'],
        }


def get_deam_loaders(
    root_dir: str,
    batch_size: int = 32,
    sr: Optional[int] = None,  # If None, uses encoder default
    encoder: str = "hubert",  # 'hubert' (16kHz) or 'mert' (24kHz)
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_dir: Optional[str] = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train/test DataLoaders for DEAM with encoder-aware resampling and fixed chunking
    
    Args:
        encoder: 'hubert' (16kHz) or 'mert' (24kHz)
        sr: Override encoder sample rate (not recommended)
    
    Returns:
        train_loader, test_loader
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        sr_val = sr or DEAMDataset.ENCODER_SAMPLE_RATES.get(encoder.lower(), 16000)
        train_cache = os.path.join(cache_dir, f"deam_train_{encoder}_sr{sr_val}.pkl")
        test_cache = os.path.join(cache_dir, f"deam_test_{encoder}_sr{sr_val}.pkl")
    else:
        train_cache = test_cache = None
    
    train_dataset = DEAMDataset(
        root_dir=root_dir,
        split="train",
        sr=sr,
        encoder=encoder,
        cache_path=train_cache,
        seed=seed,
    )
    
    test_dataset = DEAMDataset(
        root_dir=root_dir,
        split="test",
        sr=sr,
        encoder=encoder,
        cache_path=test_cache,
        seed=seed,
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_audio_batch,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_audio_batch,
    )
    
    return train_loader, test_loader


def collate_audio_batch(batch: List[Dict]) -> Dict:
    """Collate function for audio batches with dynamic padding"""
    max_length = max(item['waveform'].shape[0] for item in batch)
    
    waveforms = []
    valences = []
    arousals = []
    emotions = []
    
    for item in batch:
        wav = item['waveform']
        if wav.shape[0] < max_length:
            wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[0]))
        else:
            wav = wav[:max_length]
        
        waveforms.append(wav)
        valences.append(item['valence'])
        arousals.append(item['arousal'])
        if 'emotion' in item:
            emotions.append(item['emotion'])
            
    res = {
        'waveform': torch.stack(waveforms),
        'valence': torch.stack(valences),
        'arousal': torch.stack(arousals),
        'sr': batch[0]['sr'],
    }
    if emotions:
        res['emotion'] = torch.stack(emotions)
    return res
