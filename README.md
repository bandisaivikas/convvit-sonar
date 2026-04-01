# ConvViT — Passive Sonar Threat Classification

> CNN-Transformer hybrid with Dynamic-Scale Attention for passive sonar signal classification · 95.21% accuracy · FPGA-deployed via Vitis HLS

## Classes

| CID | Class | Defense |
|---|---|---|
| 0 | Motorboat | Hostile |
| 1 | Fishing | Monitor |
| 2 | Cargo/Tanker | Non-Threat |
| 3 | Tugboat | Monitor |
| 4 | Environment | Clear |

## Architecture
```
Input (Mel Spectrogram)
       │
  CNN Backbone          ← local feature extraction
       │
Dynamic-Scale Attention ← adaptive multi-scale receptive field
       │
Transformer Encoder     ← global context modeling
       │
Classification Head     ← 5-class threat output
```

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **95.21%** |
| Post-quantization accuracy drop | **0%** |
| Deployment | Xilinx FPGA (Vitis HLS) |
| Total WAV files | 2,223 |

## Files

| File | Description |
|---|---|
| `convvit.ipynb` | Full training pipeline |
| `export_weights.py` | FPGA weight export |
| `best_sonar_convvit.pt` | Best model checkpoint |
| `confusion_matrix.png` | Confusion matrix |
| `training_curves.png` | Training curves |
| `sonar_spectrograms.png` | Sample spectrograms |

## Setup
```bash
git clone https://github.com/bandisaivikas/convvit-sonar.git
cd convvit-sonar
pip install torch torchaudio librosa numpy matplotlib scikit-learn
```

## Author

**Saivikas Bandi** · IIIT Naya Raipur · B.Tech Data Science & AI · Batch 2027
