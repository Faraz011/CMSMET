# CMSMET: Cross-Modal Speech-Music Emotion Transfer

A PyTorch framework for learning shared emotion embeddings between speech and music using contrastive learning.

## Overview

**Core Idea**: Speech and music both express emotion, but no system learns a shared latent emotion space between them. This project trains a model where speech saying "I am happy" and a happy-sounding piece of music are pulled close together in embedding space using contrastive learning (like CLIP, but for audio emotion).

**Novel Contribution**: We leverage existing music emotion modeling (MMAS with r=0.89) and ask: can a shared embedding space improve speech emotion recognition by borrowing signal from music? This is **cross-modal transfer** with minimal published work in audio.

## Architecture

```
Speech Audio → HuBERT encoder → Speech Emotion Embedding
                                    ↕ Contrastive Loss
Music Audio  → MERT encoder   → Music Emotion Embedding
                                (shared space training)
→ Shared Emotion Space → Fine-tune on speech labels
```

## Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv
source venv/Scripts/activate  # Windows
# or source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Data

Ensure you have:

- **IEMOCAP**: `C:\Users\Faraz\CMSMET\IEMOCAP_full_release`
- **DEAM**: `C:\Users\Faraz\CMSMET\deam_data`

### 3. Run Training

```bash
# Single experiment
python scripts/train.py --experiment-name cmsmet_v1 --evaluate

# Run baseline comparison (Speech-Only vs Music-Only vs Contrastive)
python scripts/compare_baselines.py --output-dir ./baseline_results
```

## Project Structure

```
CMSMET/
├── src/cmsmet/
│   ├── config.py              # Configuration management
│   ├── data/
│   │   ├── iemocap.py        # IEMOCAP dataset loader
│   │   └── deam.py           # DEAM dataset loader
│   ├── models/
│   │   ├── encoders.py       # HuBERT & MERT encoders + shared space
│   │   └── losses.py         # Contrastive learning losses (NT-Xent, Triplet, SupCon)
│   ├── training/
│   │   └── trainer.py        # Main training loop
│   └── evaluation/
│       └── evaluator.py      # Cross-modal transfer evaluation
├── scripts/
│   ├── train.py              # Main training script
│   └── compare_baselines.py  # Baseline comparison script
├── configs/                  # Experiment configurations
├── requirements.txt
└── README.md
```

## Key Components

### Data Loading

- **IEMOCAPDataset**: Speech emotion (5 sessions, 4 emotions + variations)
- **DEAMDataset**: Music emotion (45s clips, valence & arousal annotations)
- Dynamic padding + efficient caching

### Models

- **HuBERTEncoder**: Speech feature extraction (768 → 128 dim embedding)
- **MERTEncoder**: Music feature extraction (256 → 128 dim embedding)
- **SharedEmotionSpace**: Joint embedding space with normalized embeddings

### Losses

- **NT-Xent Loss**: Normalized temperature-scaled cross-entropy (CLIP-style)
- **Triplet Loss**: Hard negative mining with emotion labels
- **SupCon Loss**: Supervised contrastive learning

### Evaluation

- **Single-modal baselines**: Speech-only and music-only classifiers
- **Cross-modal transfer**: Train on one modality, test on the other
- **Embedding similarity**: Cosine similarity analysis
- **Emotion correlation**: Per-emotion embedding alignment

## Configuration

Edit `configs/` or modify default config in `cmsmet/config.py`:

```python
config = ExperimentConfig(
    experiment_name="cmsmet_v1",
    data=DataConfig(
        iemocap_root="...",
        deam_root="...",
        batch_size=32,
    ),
    encoder=EncoderConfig(
        embedding_dim=128,
        speech_encoder_model="facebook/hubert-base-ls960",
        music_encoder_model="m-a-p/MERT-v1-95M",
    ),
    contrastive=ContrastiveLossConfig(
        temperature=0.07,
        loss_type="nt_xent",
    ),
    training=TrainingConfig(
        num_epochs=50,
        learning_rate=1e-4,
        device="cuda",
    ),
)
```

## Citation

If you use this code, please cite:

```bibtex
@project{cmsmet2024,
  title={Cross-Modal Speech-Music Emotion Transfer via Contrastive Learning},
  author={Your Name},
  year={2024}
}
```

## References

- [HuBERT](https://arxiv.org/abs/2106.07522) - Self-Supervised Representation Learning for Speech
- [MERT](https://arxiv.org/abs/2306.00107) - Acoustic Music Understanding Model
- [CLIP](https://arxiv.org/abs/2103.14030) - Contrastive Language-Image Pretraining
- [IEMOCAP](https://sail.usc.edu/iemocap/) - Interactive Emotional Dyadic Motion Capture Database
- [DEAM](https://www.mediaeval.eu/mediaeval2014/) - Crowdsourced Emotional Annotation of Music

## License

MIT
