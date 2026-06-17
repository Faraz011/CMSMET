"""
Extract music embeddings from DEAM using MERT-v1-95M
MERT is specifically trained for music understanding (24kHz sample rate)
Saves embeddings to disk as deam_mert_embeddings.npy

Uses deam_fixed.py with encoder='mert' for proper 24kHz resampling.
"""
import os
import sys
from pathlib import Path
import numpy as np
from tqdm import tqdm

import torch
import torch.nn.functional as F
from transformers import AutoModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from cmsmet.data.deam_fixed import DEAMDataset


def extract_embeddings_mert():
    """Extract MERT embeddings from DEAM using deam_fixed.py with encoder='mert'"""
    
    print("\n" + "="*80)
    print("EXTRACTING MUSIC EMBEDDINGS FROM DEAM USING MERT-v1-95M")
    print("="*80 + "\n")
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")
    
    project_root = Path(__file__).parent.parent
    deam_root = project_root / "deam_data"
    output_dir = project_root / "outputs" / "embeddings"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load MERT model
    print("Loading MERT-v1-95M...")
    try:
        mert = AutoModel.from_pretrained(
            "m-a-p/MERT-v1-95M",
            trust_remote_code=True,
            output_hidden_states=True,
        )
        mert = mert.to(device)
        mert.eval()
        mert_dim = mert.config.hidden_size
        print(f"[OK] MERT loaded (hidden_size={mert_dim})\n")
    except Exception as e:
        print(f"ERROR: Failed to load MERT-v1-95M: {e}")
        print("Please ensure you have internet connection and transformers library updated.")
        sys.exit(1)
    
    # Load DEAM dataset with MERT encoder (24kHz resampling)
    print("Loading DEAM dataset with encoder='mert' (24kHz)...")
    dataset = DEAMDataset(
        root_dir=str(deam_root),
        split="train",
        encoder="mert",       # <-- Key: uses 24kHz for MERT
        metadata_only=False,   # We need actual audio
        seed=42,
    )
    print(f"[OK] DEAM: {len(dataset)} segments\n")
    
    if len(dataset) == 0:
        print("ERROR: No segments found. Check DEAM data directory.")
        sys.exit(1)
    
    # Extract embeddings
    embeddings = {}  # {index: embedding_array}
    
    print("Extracting MERT embeddings...")
    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="DEAM-MERT"):
            try:
                sample = dataset[idx]
                
                audio = sample['waveform']  # [length] at 24kHz
                
                if audio.ndim == 0 or audio.shape[0] < 100:
                    # Skip invalid segments
                    continue
                
                # Add batch dimension and move to device
                audio = audio.unsqueeze(0).to(device)  # [1, length]
                
                # Extract MERT features
                outputs = mert(audio)
                last_hidden = outputs.last_hidden_state  # [1, time_steps, hidden_size]
                
                # Pool over time: mean pooling
                embedding = last_hidden.mean(dim=1)  # [1, hidden_size]
                embedding = embedding.squeeze(0).cpu().numpy()  # [hidden_size]
                
                embeddings[idx] = embedding
                
            except Exception as e:
                print(f"\nWarning: Error processing segment {idx}: {e}")
                continue
    
    print(f"\nExtracted {len(embeddings)} embeddings out of {len(dataset)} segments")
    
    if len(embeddings) == 0:
        print("ERROR: No embeddings extracted!")
        sys.exit(1)
    
    # Save as deam_mert_embeddings.npy (separate from HuBERT embeddings)
    embeddings_file = output_dir / "deam_mert_embeddings.npy"
    np.save(embeddings_file, embeddings)
    
    print(f"\n[OK] Extracted {len(embeddings)} music embeddings using MERT-v1-95M")
    print(f"[OK] Saved to {embeddings_file}")
    print(f"     Encoder: MERT-v1-95M")
    print(f"     Sample rate: 24kHz")
    print(f"     Shape per embedding: {embeddings[0].shape}")
    print(f"     Total file size: {embeddings_file.stat().st_size / 1024 / 1024:.1f} MB\n")
    
    # Verification
    print("VERIFICATION:")
    print(f"  Total embeddings: {len(embeddings)}")
    print(f"  Embedding dimension: {embeddings[0].shape[0]}")
    print(f"  Min value: {np.min([e.min() for e in embeddings.values()]):.4f}")
    print(f"  Max value: {np.max([e.max() for e in embeddings.values()]):.4f}")
    print(f"  Mean value: {np.mean([e.mean() for e in embeddings.values()]):.4f}")
    
    # Verify it differs from HuBERT embeddings
    hubert_file = output_dir / "deam_embeddings.npy"
    if hubert_file.exists():
        hubert_embs = np.load(str(hubert_file), allow_pickle=True).item()
        if 0 in hubert_embs and 0 in embeddings:
            if np.array_equal(hubert_embs[0], embeddings[0]):
                print("\n  [WARNING] MERT embeddings are IDENTICAL to HuBERT — something is wrong!")
            else:
                print(f"\n  [OK] MERT embeddings differ from HuBERT (cos_sim of first: "
                      f"{np.dot(hubert_embs[0], embeddings[0]) / (np.linalg.norm(hubert_embs[0]) * np.linalg.norm(embeddings[0])):.4f})")
    
    return embeddings_file


if __name__ == "__main__":
    embeddings_file = extract_embeddings_mert()
    print("="*80)
    print("MUSIC EMBEDDING EXTRACTION COMPLETE (MERT-v1-95M)")
    print("="*80 + "\n")
