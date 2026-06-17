"""
IEMOCAP Dataset Loader with Speaker-Independent Splits
Interpersonal Emotional Speech Communication (IEMOCAP)

CRITICAL: Uses speaker-independent leave-2-speakers-out split to avoid train/test contamination
"""
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import json


class IEMOCAPDataset(Dataset):
    """IEMOCAP dataset loader with proper speaker-independent splits"""
    
    # IEMOCAP speaker IDs per session (M=Male, F=Female)
    # Format: Session1F = Female speaker 1, Session1M = Male speaker 1, etc.
    SPEAKERS_PER_SESSION = {
        1: ['M', 'F'],  # Ses01M, Ses01F
        2: ['M', 'F'],  # Ses02M, Ses02F
        3: ['M', 'F'],  # Ses03M, Ses03F
        4: ['M', 'F'],  # Ses04M, Ses04F
        5: ['M', 'F'],  # Ses05M, Ses05F
    }
    
    # CRITICAL: Use 4-class standard benchmark (matches published IEMOCAP results)
    # "excited" merged into "happy" per standard protocol
    EMOTION_MAP_4CLASS = {
        'happy': 0,
        'hap': 0,
        'sad': 1,
        'angry': 2,
        'ang': 2,
        'neutral': 3,
        'neu': 3,
        'excited': 0,
        'exc': 0,
    }
    
    # Text patterns to map evaluation labels to canonical emotions
    EMOTION_TEXT_MAP = {
        'neutral state': 'neutral',
        'happy': 'happy',
        'sad': 'sad',
        'anger': 'angry',
        'excited': 'happy',
        'excitement': 'happy',
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        sr: int = 16000,  # FIXED: HuBERT requires 16kHz
        min_duration: float = 0.5,
        max_duration: float = 30.0,
        sessions: List[int] = None,
        test_speakers: Optional[Set[str]] = None,
        emotion_labels: Optional[str] = None,
        cache_path: Optional[str] = None,
        use_4class: bool = True,  # CRITICAL: Standard benchmark uses 4 classes
        seed: int = 42,
    ):
        """
        Args:
            root_dir: Path to IEMOCAP dataset root
            split: 'train', 'test', or 'all'
            sr: Sample rate - FIXED to 16kHz for HuBERT
            min_duration: Minimum audio duration in seconds
            max_duration: Maximum audio duration in seconds
            sessions: List of session IDs to use (1-5)
            test_speakers: Set of speakers to use for testing (e.g., {'Ses01M', 'Ses02F'})
                          Leave as None for default 80/20 train/test split
            emotion_labels: Path to custom emotion label file
            cache_path: Path to cache processed data
            use_4class: Use 4-class standard (excited→happy merged). Set False for 5-class.
            seed: Random seed for reproducible splits
        """
        self.root_dir = Path(root_dir)
        self.sr = sr  # FIXED: Always 16kHz for HuBERT
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.split = split
        self.seed = seed
        self.use_4class = use_4class  # CRITICAL: Standard benchmark uses 4 classes
        
        if sessions is None:
            sessions = [1, 2, 3, 4, 5]
        self.sessions = sessions
        
        # Set up speaker-independent splits
        self.test_speakers = test_speakers or self._get_default_test_speakers()
        
        self.samples = []
        self._load_dataset(emotion_labels, cache_path)
    
    def _get_default_test_speakers(self) -> Set[str]:
        """Default: leave-2-speakers-out (e.g., Ses01M, Ses01F for test)"""
        # Use first session speakers for testing (reproducible)
        return {'Ses01M', 'Ses01F'}
    
    def _extract_speaker_id(self, sample_name: str) -> str:
        """Extract speaker ID from sample name (e.g., 'Ses01F_impro01' -> 'Ses01F')"""
        # Format: Ses##X_... where X is M or F
        parts = sample_name.split('_')
        if len(parts) >= 1:
            speaker_prefix = parts[0]  # e.g., 'Ses01F'
            return speaker_prefix
        return ""
    
    def _load_dataset(
        self,
        emotion_labels: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """Load dataset with speaker-independent splits"""
        
        # Check cache first
        if cache_path and os.path.exists(cache_path):
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                self.samples = pickle.load(f)
            return
        
        # Load custom labels if provided
        custom_labels = {}
        if emotion_labels and os.path.exists(emotion_labels):
            with open(emotion_labels, 'r') as f:
                if emotion_labels.endswith('.json'):
                    custom_labels = json.load(f)
                else:
                    custom_labels = pickle.load(f)
        
        print(f"Loading IEMOCAP with split: {self.split}")
        print(f"  Test speakers (held out): {self.test_speakers}")
        print(f"  Train speakers: {self._get_train_speakers()}")
        print(f"  Classes: {'4-class (standard benchmark)' if self.use_4class else '5-class (full)'}")
        
        # CRITICAL: 4-class standard benchmark (excited→happy merged)
        emotion_map = self.EMOTION_MAP_4CLASS if self.use_4class else self.EMOTION_MAP_4CLASS
        emotion_text_map_fn = lambda text: self.EMOTION_TEXT_MAP.get(text.lower().strip(), 'neutral')
        
        # Scan IEMOCAP sessions
        for session_id in self.sessions:
            session_dir = self.root_dir / f"Session{session_id}"
            if not session_dir.exists():
                print(f"Warning: {session_dir} not found")
                continue
            
            wav_dir = session_dir / "dialog" / "wav"
            eval_base_dir = session_dir / "dialog" / "EmoEvaluation"
            
            if not wav_dir.exists() or not eval_base_dir.exists():
                print(f"Warning: Audio or eval directory not found in {session_dir}")
                continue
            
            # Load all transcript evaluation files for this session
            eval_files = sorted(eval_base_dir.glob("*.txt"))
            
            for eval_file in eval_files:
                with open(eval_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                
                for line in lines:
                    line = line.strip()
                    if not line or line.startswith('%'):
                        continue
                    # Match lines like: [6.2901 - 8.2357] Ses01F_impro01_F000 neu [2.5000, 2.5000, 2.5000]
                    match = re.match(r"^\s*\[([0-9.]+)\s*-\s*([0-9.]+)\]\s+(\S+)\s+(\S+)", line)
                    if not match:
                        continue
                    start_time = float(match.group(1))
                    end_time = float(match.group(2))
                    sample_name = match.group(3)
                    emotion_str = match.group(4).lower()
                    
                    speaker_id = self._extract_speaker_id(sample_name)
                    
                    if self.split == "train" and speaker_id in self.test_speakers:
                        continue
                    elif self.split == "test" and speaker_id not in self.test_speakers:
                        continue
                    
                    # Check custom labels first
                    if sample_name in custom_labels:
                        emotion_category = custom_labels[sample_name]
                        emotion_name = str(emotion_category)
                    elif emotion_str in self.EMOTION_MAP_4CLASS:
                        emotion_category = self.EMOTION_MAP_4CLASS[emotion_str]
                        emotion_name = emotion_str
                    elif emotion_str in self.EMOTION_TEXT_MAP:
                        emotion_name = self.EMOTION_TEXT_MAP[emotion_str]
                        if emotion_name in self.EMOTION_MAP_4CLASS:
                            emotion_category = self.EMOTION_MAP_4CLASS[emotion_name]
                        else:
                            continue
                    else:
                        continue
                    
                    turn_parts = sample_name.split('_')
                    if len(turn_parts) >= 2:
                        dialog_id = '_'.join(turn_parts[:-1])
                    else:
                        dialog_id = sample_name
                    
                    wav_path = wav_dir / f"{dialog_id}.wav"
                    if not wav_path.exists():
                        wav_paths = list(wav_dir.glob(f"{dialog_id}*.wav"))
                        wav_path = wav_paths[0] if wav_paths else None
                    if wav_path is None:
                        continue
                    
                    segment_duration = end_time - start_time
                    if not (self.min_duration <= segment_duration <= self.max_duration):
                        continue
                    
                    self.samples.append({
                        'wav_path': str(wav_path),
                        'start_time': start_time,
                        'end_time': end_time,
                        'emotion': emotion_category,
                        'emotion_str': emotion_name,
                        'speaker': speaker_id,
                        'session': session_id,
                        'duration': segment_duration,
                        'sr': self.sr,
                    })
        
        print(f"Loaded {len(self.samples)} samples for split '{self.split}'")
        
        # Cache
        if cache_path:
            os.makedirs(os.path.dirname(cache_path) or '.', exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(self.samples, f)
    
    def _get_train_speakers(self) -> Set[str]:
        """Get train speakers (complement of test speakers)"""
        all_speakers = set()
        for session_id in self.sessions:
            for gender in self.SPEAKERS_PER_SESSION.get(session_id, ['M', 'F']):
                all_speakers.add(f"Ses{session_id:02d}{gender}")
        return all_speakers - self.test_speakers
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict:
        """Return audio tensor and emotion label"""
        sample = self.samples[idx]
        
        try:
            if 'start_time' in sample and 'end_time' in sample:
                waveform, sr = librosa.load(
                    sample['wav_path'],
                    sr=self.sr,
                    mono=True,
                    offset=sample['start_time'],
                    duration=sample['end_time'] - sample['start_time'],
                )
            else:
                waveform, sr = librosa.load(
                    sample['wav_path'],
                    sr=self.sr,
                    mono=True,
                )
            assert sr == self.sr, f"Resampling failed! Got {sr} instead of {self.sr}"
        except Exception as e:
            print(f"Error loading {sample['wav_path']}: {e}")
            waveform = np.zeros(self.sr)
        
        return {
            'waveform': torch.from_numpy(waveform).float(),
            'sr': self.sr,
            'emotion': torch.tensor(sample['emotion'], dtype=torch.long),
            'emotion_str': sample['emotion_str'],
            'speaker': sample['speaker'],
            'session': sample['session'],
            'duration': sample['duration'],
        }


def get_iemocap_loaders(
    root_dir: str,
    batch_size: int = 32,
    sr: int = 16000,  # FIXED: HuBERT requires 16kHz
    num_workers: int = 4,
    pin_memory: bool = True,
    cache_dir: Optional[str] = None,
    test_speakers: Optional[Set[str]] = None,
    use_4class: bool = True,  # CRITICAL: Standard benchmark uses 4 classes
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader]:
    """
    Get train/test DataLoaders for IEMOCAP with speaker-independent splits
    
    Args:
        test_speakers: Set of speaker IDs for testing (e.g., {'Ses01M', 'Ses01F'})
                      If None, uses default leave-2-speakers-out
        use_4class: Use 4-class standard (excited→happy merged). Affects num_classes.
    
    Returns:
        train_loader, test_loader
    """
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        suffix = "_4class" if use_4class else "_5class"
        train_cache = os.path.join(cache_dir, f"iemocap_train_sr{sr}{suffix}.pkl")
        test_cache = os.path.join(cache_dir, f"iemocap_test_sr{sr}{suffix}.pkl")
    else:
        train_cache = test_cache = None
    
    train_dataset = IEMOCAPDataset(
        root_dir=root_dir,
        split="train",
        sr=sr,  # FIXED: 16kHz
        test_speakers=test_speakers,
        use_4class=use_4class,
        cache_path=train_cache,
        seed=seed,
    )
    
    test_dataset = IEMOCAPDataset(
        root_dir=root_dir,
        split="test",
        sr=sr,  # FIXED: 16kHz
        test_speakers=test_speakers,
        use_4class=use_4class,
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
    emotions = []
    speakers = []
    sessions = []
    
    for item in batch:
        wav = item['waveform']
        if wav.shape[0] < max_length:
            wav = torch.nn.functional.pad(wav, (0, max_length - wav.shape[0]))
        else:
            wav = wav[:max_length]
        
        waveforms.append(wav)
        emotions.append(item['emotion'])
        speakers.append(item['speaker'])
        sessions.append(item['session'])
    
    return {
        'waveform': torch.stack(waveforms),
        'emotion': torch.stack(emotions),
        'sr': batch[0]['sr'],
        'speakers': speakers,
        'sessions': sessions,
    }
