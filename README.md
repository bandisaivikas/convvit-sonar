# Passive Sonar Threat Classification (ConvViT)

> Real-time underwater acoustic threat detection using a Hybrid ConvViT architecture—optimized for Apple Silicon (M2/MPS) and edge-deployable FPGA targets.

## Motivation

Underwater acoustic environments are characterized by high noise floors and complex temporal signatures. While standard CNNs excel at local feature extraction in Mel Spectrograms, they often struggle with global temporal dependencies. This project implements a **ConvViT** (CNN + Transformer) hybrid that leverages a CNN backbone for local spatial features and a Dynamic-Scale Transformer for long-range context, ensuring high accuracy even in "Clear Water" or hostile environments.

## Key Results (ShipsEar / DS3500)

| Metric | Value |
|---|---|
| FP32 Accuracy | **91.61%** |
| INT8 Accuracy (Quantized) | **90.84%** |
| Model Parameters | **633,253** |
| FPGA Latency | **12.44 ms/frame** |
| Compression Ratio | **3.5×** (2.54 MB → 0.73 MB) |
| DSP Usage (Zynq UltraScale+) | **7.3%** (184/2520) |

## Implementation Phases (Completed)

| Phase | Description |
|---|---|
| **Data Pipeline** | WAV to Mel Spectrogram conversion with global Z-score normalization and .npz caching. |
| **Hybrid Arch** | CNN Backbone + Feature Tokenization + Dynamic-Scale Transformer head. |
| **Optimization** | Native Apple Metal (MPS) training backend and Dynamic INT8 quantization. |
| **Robustness** | Acoustic Jamming simulation via noise-sweep robustness analysis. |
| **Edge Readiness** | DSP48 usage estimation and latency profiling for Zynq UltraScale+ FPGAs. |

## Files

| File | Description |
|---|---|
| `convvit.ipynb` | Main project notebook containing the full pipeline. |
| `cache/spectrograms.npz` | Pre-processed Mel tokens (cached for speed). |
| `best_sonar_convvit.pt` | Trained model weights (FP32). |
| `cache/eval_results.npz` | FP32 vs INT8 performance comparison data. |

## Setup
```bash
# Recommended for Apple M2/M3 Mac
pip install torch torchvision torchaudio
pip install librosa soundfile numpy matplotlib scikit-learn seaborn tqdm

Author
Saivikas Bandi· IIIT Naya Raipur · B.Tech Data Science & AI · Batch 2027
EOF
