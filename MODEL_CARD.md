# Model Card: XceptionNet Deepfake Detector

## Model Details

| Field | Value |
|-------|-------|
| Model Name | XceptionNet Deepfake Detector v0.1.0 |
| Architecture | XceptionNet (pretrained ImageNet) + custom classification head |
| Task | Binary classification (Real vs. Fake) |
| Input | RGB face crop, 299x299 pixels |
| Output | Probability of manipulation [0.0, 1.0] |
| Parameters | ~23M total |
| Framework | PyTorch + timm |
| License | Apache 2.0 |

## Intended Use

- **Primary use**: Detecting AI-generated or manipulated facial imagery in video frames
- **Intended users**: Fraud analysts, compliance teams, forensic investigators, identity verification systems
- **Out of scope**: Real-time production deployment without additional hardening; legal evidence without human review

## Training Data

- **Dataset**: Deepfake Detection Challenge (DFDC) — parts 00–03
- **Source**: Facebook AI / Kaggle
- **Preprocessing**: MTCNN face extraction, alignment, resize to 299x299
- **Split**: 70% train / 15% validation / 15% test (stratified)
- **Augmentation**: Random horizontal flip, rotation (±10°), color jitter, Gaussian blur, random resized crop

## Evaluation Results

*Results are populated after training. See `results/metrics/evaluation_report.json` for exact numbers.*

| Metric | Value |
|--------|-------|
| AUC | See evaluation_report.json |
| EER | See evaluation_report.json |
| F1 Score | See evaluation_report.json |
| Precision | See evaluation_report.json |
| Recall | See evaluation_report.json |

## Limitations and Bias

- **Known limitations**:
  - Trained only on DFDC dataset — may not generalize to all deepfake generation methods
  - Performance degrades on heavily compressed video (CRF > 30)
  - Frame-level only — does not use temporal signals across frames
  - Not tested against adversarial attacks designed to evade detection

- **Demographic considerations**:
  - DFDC dataset has demographic imbalances
  - Performance may vary across skin tones, ages, and lighting conditions
  - Per-demographic evaluation is a priority for future versions

- **Ethical considerations**:
  - False positives can wrongly accuse individuals of fraud
  - Should always be used with human review, never as sole decision-maker
  - Detection scores are probabilities, not verdicts

## Reproducibility

All training parameters are fixed:
- Random seed: 42
- Deterministic PyTorch operations enabled
- All hyperparameters documented in `src/config.py`
- Same input + same checkpoint = same output (verified)

To reproduce:
```bash
pip install -r requirements.txt
python -m src.train
python -m src.evaluate
```

## Explainability

GradCAM heatmaps are generated for all test predictions, showing which facial regions most influenced the model's decision. These serve as:
- Audit artifacts for human reviewers
- Debugging tools to identify model failure modes
- Transparency evidence for stakeholders

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | 2025-XX | Initial release: XceptionNet on DFDC |

## Citation

If you use this work, please cite:
```
@software{kazim2025deepfake,
  author = {Kazim, Affan Ahmed},
  title = {Multimodal Deepfake Detection Framework},
  year = {2025},
  url = {https://github.com/YOUR_USERNAME/multimodal-deepfake-detection}
}
```

## Contact

Affan Ahmed Kazim — [GitHub](https://github.com/YOUR_USERNAME)
