"""
Extract speech embeddings from IEMOCAP once using frozen HuBERT-Base
Saves embeddings to disk for reuse during training
"""
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import HubertModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cmsmet.data.iemocap import IEMOCAPDataset
from cmsmet.config import ExperimentConfig


def extract_embeddings():
    """Extract HuBERT embeddings from IEMOCAP"""
    
    print("\n" + "="*80)
    print("EXTRACTING SPEECH EMBEDDINGS FROM IEMOCAP")
    print("="*80 + "\n")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    # Load config
    config = ExperimentConfig()
    
    # Load HuBERT model
    print("Loading HuBERT-Base...")
    hubert = HubertModel.from_pretrained("facebook/hubert-base-ls960")
    hubert = hubert.to(device)
    hubert.eval()
    print("[OK] HuBERT loaded\n")
    
    # Load IEMOCAP dataset
    print("Loading IEMOCAP dataset...")
    dataset = IEMOCAPDataset(
        root_dir=config.data.iemocap_root,
        split="train",
        sr=config.data.speech_sr,
        min_duration=config.data.min_duration,
        max_duration=config.data.max_duration,
    )
    print(f"[OK] IEMOCAP: {len(dataset)} samples\n")
    
    # Extract embeddings
    embeddings = {}  # {index: embedding_array}
    
    print("Extracting embeddings...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="IEMOCAP"):
            sample = dataset[idx]
            
            # sample is a dict with keys: waveform, sr, emotion, emotion_str, speaker, duration
            audio = sample['waveform']  # [length]
            
            # Add batch dimension and move to device
            audio = audio.unsqueeze(0).to(device)  # [1, length]
            
            # Extract HuBERT features
            outputs = hubert(audio)
            last_hidden = outputs.last_hidden_state  # [1, time_steps, 768]
            
            # Pool over time: mean pooling
            embedding = last_hidden.mean(dim=1)  # [1, 768]
            embedding = embedding.squeeze(0).cpu().numpy()  # [768]
            
            embeddings[idx] = embedding
    
    # Save embeddings
    output_dir = Path(config.output_root) / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_file = output_dir / "iemocap_embeddings.npy"
    np.save(embeddings_file, embeddings)
    
    print(f"\n[OK] Extracted {len(embeddings)} speech embeddings")
    print(f"[OK] Saved to {embeddings_file}")
    print(f"     Shape per embedding: {embeddings[0].shape}")
    print(f"     Total file size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB\n")
    
    return embeddings_file


if __name__ == "__main__":
    embeddings_file = extract_embeddings()
    print("="*80)
    print("SPEECH EMBEDDING EXTRACTION COMPLETE")
    print("="*80 + "\n")
