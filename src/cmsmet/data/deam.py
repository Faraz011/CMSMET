"""
DEAM Dataset Loader
Crowdsourced Emotional Annotation of Music (DEAM)
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
    """DEAM dataset loader for music emotion recognition"""
    
    CLIP_DURATION = 45  # seconds per clip in DEAM
    
    def __init__(
        self,
        root_dir: str,
        annotations_file: Optional[str] = None,
        split: str = "train",
        sr: int = 16000,
        segment_duration: float = 10.0,
        stride: float = 5.0,
        emotion_type: str = "arousal",  # "arousal", "valence", or "both"
        cache_path: Optional[str] = None,
    ):
        """
        Args:
            root_dir: Path to DEAM dataset root (containing audio and annotations)
            annotations_file: Path to DEAM annotations CSV file
            split: 'train', 'test', or 'all'
            sr: Sample rate
            segment_duration: Duration of each segment in seconds
            stride: Overlap stride for sliding window
            emotion_type: Type of emotion annotation to use
            cache_path: Path to cache processed data
        """
        self.root_dir = Path(root_dir)
        self.sr = sr
        self.segment_duration = segment_duration
        self.stride = stride
        self.emotion_type = emotion_type
        self.split = split
        
        self.samples = []
        self._load_dataset(annotations_file, cache_path)
    
    def _load_dataset(
        self,
        annotations_file: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """Load dataset from DEAM directory"""
        
        # Check cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            return
        
        # Default annotations file location - check multiple possible paths
        if annotations_file is None:
            possible_paths = [
                # DEAM structure
                self.root_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level" / "static_annotations_averaged_songs_1_2000.csv",
                self.root_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level" / "static_annotations_averaged_songs_2000_2058.csv",
                # Alternative structures
                self.root_dir / "annotations" / "annotations.csv",
                self.root_dir / "DEAM_Annotations.csv",
                self.root_dir / "annotations.csv",
            ]
            for p in possible_paths:
                if p.exists():
                    annotations_file = str(p)
                    break
        
        # Load annotations - combine multiple CSV files if needed
        annotations = {}
        annotation_files = []
        
        # Check if we found a single file
        if annotations_file and os.path.exists(annotations_file):
            annotation_files = [annotations_file]
        else:
            # Try to find all CSV files in song_level directory
            song_level_dir = self.root_dir / "DEAM_Annotations" / "annotations" / "annotations averaged per song" / "song_level"
            if song_level_dir.exists():
                annotation_files = sorted(song_level_dir.glob("*.csv"))
        
        # Load all annotation files
        for ann_file in annotation_files:
            if not os.path.exists(ann_file):
                continue
            with open(ann_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        # Handle different column name formats
                        track_id_col = [c for c in row.keys() if 'song_id' in c.lower() or 'track_id' in c.lower()]
                        valence_col = [c for c in row.keys() if 'valence' in c.lower() and 'mean' in c.lower()]
                        arousal_col = [c for c in row.keys() if 'arousal' in c.lower() and 'mean' in c.lower()]
                        
                        if not (track_id_col and valence_col and arousal_col):
                            continue
                        
                        track_id = int(row[track_id_col[0]].strip())
                        valence = float(row[valence_col[0]].strip())
                        arousal = float(row[arousal_col[0]].strip())
                        
                        annotations[track_id] = {
                            'valence': valence,
                            'arousal': arousal,
                        }
                    except (ValueError, KeyError, IndexError):
                        continue
        
        print(f"Loaded {len(annotations)} track annotations")
        
        # Find audio files - check multiple possible locations
        audio_dirs = [
            self.root_dir / "DEAM_audio" / "MEMD_audio",    # Main DEAM structure
            self.root_dir / "music_speech_files",
            self.root_dir / "audio",
            self.root_dir / "MEMD_audio",
            self.root_dir,
        ]
        
        audio_dir = None
        for d in audio_dirs:
            if d.exists():
                audio_dir = d
                break
        
        if audio_dir is None:
            print(f"Warning: Could not find audio directory in {self.root_dir}")
            return
        
        # Scan audio files
        mp3_files = list(audio_dir.glob("**/*.mp3"))
        wav_files = list(audio_dir.glob("**/*.wav"))
        audio_files = mp3_files + wav_files
        
        for audio_file in sorted(audio_files):
            try:
                # Extract track ID from filename
                # Filenames are like: 1.mp3, 2.mp3, 10.mp3, etc.
                filename = audio_file.stem
                try:
                    track_id = int(filename.split('_')[0])
                except ValueError:
                    # Try just the filename without extension
                    try:
                        track_id = int(filename)
                    except ValueError:
                        continue
                
                if track_id not in annotations:
                    continue
                
                # Determine split (use modulo for reproducibility)
                if self.split == "train" and (track_id % 10) >= 2:
                    continue
                elif self.split == "test" and (track_id % 10) < 2:
                    continue
                
                # Use sliding window to create segments
                try:
                    duration = librosa.get_duration(path=str(audio_file), sr=self.sr)
                except:
                    duration = self.CLIP_DURATION  # Assume standard length
                
                segment_samples = int(self.segment_duration * self.sr)
                stride_samples = int(self.stride * self.sr)
                
                start = 0
                while start + segment_samples <= int(duration * self.sr):
                    self.samples.append({
                        'audio_path': str(audio_file),
                        'track_id': track_id,
                        'start_sample': start,
                        'end_sample': start + segment_samples,
                        'valence': annotations[track_id]['valence'],
                        'arousal': annotations[track_id]['arousal'],
                    })
                    start += stride_samples
                
            except (ValueError, IndexError, Exception) as e:
                # Skip files that don't match naming convention
                continue
        
        print(f"Loaded {len(self.samples)} segments for split '{self.split}'")
        
        # Cache if path provided
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return audio segment and emotion annotations"""
        sample = self.samples[idx]
        
        # Load audio segment
        try:
            waveform, sr = librosa.load(
                sample['audio_path'],
                sr=self.sr,
                mono=True,
                offset=sample['start_sample'] / self.sr,
                duration=self.segment_duration,
            )
        except Exception as e:
            print(f"Error loading segment from {sample['audio_path']}: {e}")
            waveform = np.zeros(int(self.segment_duration * self.sr))
        
        # Normalize emotion dimensions to [0, 1] range (DEAM uses [1, 9])
        valence = (sample['valence'] - 1.0) / 8.0
        arousal = (sample['arousal'] - 1.0) / 8.0
        
        # Combine emotion dimensions if needed
        if self.emotion_type == "arousal":
            emotion = torch.tensor(arousal, dtype=torch.float32)
        elif self.emotion_type == "valence":
            emotion = torch.tensor(valence, dtype=torch.float32)
        else:  # both
            emotion = torch.tensor([valence, arousal], dtype=torch.float32)
        
        return {
            'waveform': torch.from_numpy(waveform).float(),
            'sr': self.sr,
            'emotion': emotion,
            'valence': valence,
            'arousal': arousal,
            'track_id': sample['track_id'],
        }


def get_deam_loaders(
    root_dir: str,
    batch_size: int = 32,
    sr: int = 16000,
    num_workers: int = 4,
    pin_memory: bool = True,
    emotion_type: str = "both",
    cache_dir: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train/test DataLoaders for DEAM
    
    Returns:
        train_loader, test_loader
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        train_cache = os.path.join(cache_dir, "deam_train.pkl")
        test_cache = os.path.join(cache_dir, "deam_test.pkl")
    else:
        train_cache = test_cache = None
    
    train_dataset = DEAMDataset(
        root_dir=root_dir,
        split="train",
        sr=sr,
        emotion_type=emotion_type,
        cache_path=train_cache,
    )
    
    test_dataset = DEAMDataset(
        root_dir=root_dir,
        split="test",
        sr=sr,
        emotion_type=emotion_type,
        cache_path=test_cache,
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
    emotions = []
    track_ids = []
    
    for item in batch:
        wav = item['waveform']
        # Pad to max_length
        if wav.shape[0] < max_length:
            wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[0]))
        else:
            wav = wav[:max_length]
        
        waveforms.append(wav)
        emotions.append(item['emotion'])
        track_ids.append(item['track_id'])
    
    return {
        'waveform': torch.stack(waveforms),
        'emotion': torch.stack(emotions),
        'sr': batch[0]['sr'],
        'track_ids': track_ids,
    }
