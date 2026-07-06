# Multimodal Deepfake Detection Framework

A reproducible deepfake detection system with standardized evaluation and explainability artifacts, designed as the foundation for cross-channel fraud defense infrastructure.

## Overview

This framework implements Phase I of a multimodal fraud detection system: single-modality video deepfake detection with:

- **XceptionNet-based detector** fine-tuned on the 140K Real and Fake Faces dataset
- **Reproducible evaluation harness** with fixed splits, seeds, and versioned metrics (AUC, EER, F1)
- **GradCAM explainability** showing which facial regions trigger detection
- **Model card** documenting performance, limitations, and demographic considerations

## Architecture

```
Video Input → Frame Sampling → MTCNN Face Extraction → XceptionNet Classification → Risk Score
                                                                ↓
                                              GradCAM Heatmap + Confidence Interval
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Extract faces from DFDC data (assumes data in data/raw/)
python -m src.face_extraction

# Train the model
python -m src.train

# Evaluate on test set
python -m src.evaluate
```

Or use the Jupyter notebook: `notebooks/01_full_pipeline.ipynb`

## Results

| Metric | Value |
|--------|-------|
| AUC | See results/metrics/evaluation_report.json |
| EER | See results/metrics/evaluation_report.json |
| F1 | See results/metrics/evaluation_report.json |

*Update this table after training with your actual results.*

## Project Structure

```
├── src/
│   ├── config.py           # All hyperparameters and paths
│   ├── face_extraction.py  # MTCNN face extraction from video
│   ├── dataset.py          # PyTorch dataset with stratified splits
│   ├── model.py            # XceptionNet and EfficientNet architectures
│   ├── train.py            # Training loop with early stopping
│   ├── evaluate.py         # Metrics, ROC curves, confusion matrix
│   └── gradcam.py          # GradCAM explainability heatmaps
├── notebooks/
│   └── 01_full_pipeline.ipynb  # End-to-end notebook
├── results/
│   ├── metrics/            # Evaluation reports, plots
│   ├── heatmaps/           # GradCAM visualizations
│   └── checkpoints/        # Model weights
├── MODEL_CARD.md           # Model documentation
├── requirements.txt        # Python dependencies
└── README.md               # This file
```

## Reproducibility

This project is designed for full reproducibility:
- Fixed random seed (42) across all operations
- Deterministic PyTorch operations
- Versioned data splits (stratified, documented)
- All hyperparameters in a single config file
- Same input + same checkpoint = same output

## Roadmap

This repository is Phase I of a larger multimodal fraud detection system:

- [x] **Phase I.1**: Frame-level deepfake detection (XceptionNet)
- [ ] **Phase I.2**: Temporal modeling (TimeSformer/SlowFast)
- [ ] **Phase I.3**: Adversarial hardening (PGD training, frequency-domain analysis)
- [ ] **Phase II**: Audio clone detection (Wav2Vec2-based)
- [ ] **Phase II**: Text phishing detection (RoBERTa-based)
- [ ] **Phase II**: Cross-modal transformer fusion layer
- [ ] **Phase III**: Real-time streaming SDK

## Dataset

Training uses the [140K Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces) dataset (~70K real faces from Flickr/CelebA, ~70K StyleGAN-generated fake faces). Due to size and licensing, raw data is not included in this repository.

## License

Apache 2.0

## Author

Affan Ahmed Kazim
