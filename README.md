# AMSR2-Enhanced-SR: Satellite Thermal Imagery Super-Resolution with Spatial Attention

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> **Deep learning framework for 2×/4×/8× super-resolution of AMSR2 satellite thermal imagery using attention-enhanced U-Net architecture**

This repository implements an enhanced U-Net with spatial attention mechanisms for super-resolution of brightness temperature data from the Advanced Microwave Scanning Radiometer 2 (AMSR2). The model achieves high-fidelity upsampling through residual learning, skip connections, and CBAM-style attention modules, designed specifically for the physical constraints of passive microwave radiometry.

---

## Overview

Passive microwave satellite observations provide critical data for climate monitoring, but often suffer from coarse spatial resolution. This work addresses the spatial resolution limitation through deep learning-based super-resolution, reconstructing high-resolution brightness temperature fields while preserving physical consistency.

### Key Features

- **Attention-Enhanced Architecture**: Spatial and channel attention modules (CBAM-style) for adaptive feature refinement
- **Multi-Scale Processing**: U-Net encoder-decoder with ResNet blocks for hierarchical feature extraction
- **Physical Consistency**: Custom loss function incorporating gradient preservation and thermodynamic constraints
- **Cascaded Inference**: Sequential 2× upsampling stages for 4× and 8× super-resolution
- **Patch-Based Processing**: Gaussian-weighted blending for seamless reconstruction of arbitrary-sized inputs

---

## Architecture

### Enhanced U-Net with Spatial Attention

The model consists of three primary components: encoder, decoder, and upsampling blocks.

#### **Encoder (Feature Extraction)**

```
Input (1×2048×208) 
    ↓
Conv 7×7, stride 2 → BN → ReLU → MaxPool
    ↓
ResNet Layer 1: 3 blocks, 64 channels
    ↓
ResNet Layer 2: 4 blocks, 128 channels, stride 2
    ↓
ResNet Layer 3: 6 blocks, 256 channels, stride 2
    ↓
ResNet Layer 4: 3 blocks, 512 channels, stride 2
    ↓
Global CBAM Attention (512 channels)
```

Each ResNet block incorporates:
- **Convolutional Backbone**: 3×3 conv → BN → ReLU → Dropout(0.15) → 3×3 conv → BN
- **Residual Connection**: Identity shortcut or 1×1 conv projection
- **CBAM Attention**: Sequential channel and spatial attention refinement

#### **Attention Mechanisms**

**Channel Attention Module**:
```
Input Features (C×H×W)
    ↓
Global Average Pool ────┐
                        ├─→ Shared FC (C→C/16→C) → Sigmoid → Scale
Global Max Pool ────────┘
```

**Spatial Attention Module**:
```
Input Features (C×H×W)
    ↓
Channel-wise Max ────┐
                     ├─→ Concat → Conv 7×7 → Sigmoid → Scale
Channel-wise Avg ────┘
```

#### **Decoder (Feature Reconstruction)**

```
Encoded Features (512 channels, H/32×W/32)
    ↓
TransConv 2×2 → BN → ReLU → Conv 3×3 → BN → ReLU  [256 channels]
    ↑ (Spatial Attention on skip)
Skip Connection from Encoder Layer 3
    ↓
TransConv 2×2 → BN → ReLU → Conv 3×3 → BN → ReLU  [128 channels]
    ↑ (Spatial Attention on skip)
Skip Connection from Encoder Layer 2
    ↓
TransConv 2×2 → BN → ReLU → Conv 3×3 → BN → ReLU  [64 channels]
    ↑ (Spatial Attention on skip)
Skip Connection from Encoder Layer 1
    ↓
TransConv 2×2 [32 channels] → Final CBAM → Conv 3×3 → Conv 1×1
    ↓
Output (1×H×W)
```

#### **Upsampling Head (2× Resolution)**

```
Decoder Output (1×2048×208)
    ↓
TransConv 4×4, stride 2 → ReLU
    ↓
Conv 3×3 → ReLU → Conv 3×3 → ReLU
    ↓
Conv 1×1 → Residual Addition with Bicubic Upsampled Input
    ↓
Clamp[-1.5, 1.5]
    ↓
Super-Resolved Output (1×4096×416)
```

**Model Statistics**:
- Input Resolution: 2048×208 (height×width)
- Output Resolution: 4096×416 (2× upsampling)
- Memory Footprint: ~114 MB (FP32)

---

## Loss Function

The training objective combines multiple terms to ensure both perceptual quality and physical plausibility:

**Total Loss**:
```
L_total = α·L_L1 + β·L_grad + γ·L_phys + ε·L_SSIM
```

### Loss Components

**1. L1 Reconstruction Loss** (α=1.0):
```
L_L1 = (1/N) Σ |y_pred - y_true|
```
Primary reconstruction objective for pixel-wise fidelity.

**2. Gradient Loss** (β=0.15):
```
L_grad = (1/N) Σ |∇x(y_pred) - ∇x(y_true)| + |∇y(y_pred) - ∇y(y_true)|
```
Preserves edge sharpness and spatial structure through gradient matching.

**3. Physical Consistency Loss** (γ=0.05):
```
L_phys = ||μ(y_pred) - μ(y_true)||² + 0.5·||σ(y_pred) - σ(y_true)||² + 0.1·ReLU(|y_pred| - 1.0)
```
Enforces energy conservation (mean), distribution preservation (standard deviation), and valid temperature range constraints.

**4. SSIM Loss** (ε=0.1):
```
L_SSIM = 1 - SSIM(y_pred, y_true)
```
Structural similarity metric for perceptual quality alignment.

---

## Cascaded Multi-Stage Super-Resolution

### 4× Upsampling Strategy
```
Original (H×W) → Model 2× → (2H×2W) → Model 2× → (4H×4W)
```

### 8× Upsampling Variants

**Variant 1 (Bicubic-Model-Model)**:
```
Original (H×W) → Bicubic 2× → Model 2× → Model 2× → (8H×8W)
```

**Variant 2 (Triple-Model)**:
```
Original (H×W) → Model 2× → Model 2× → Model 2× → (8H×8W)
```

### Patch-Based Inference

For large-scale imagery, the framework implements overlapping patch processing:

1. **Patch Extraction**: Sliding window with 75% overlap (stride = patch_size/4)
2. **Gaussian Weighting**: 2D Gaussian weights (σ = 0.3 × patch_size) for smooth blending
3. **Accumulation**: Weighted sum of overlapping patch predictions
4. **Normalization**: Division by accumulated weights for final reconstruction

**Configuration**:
- Patch Size: 1024×104 (height×width)
- Overlap Ratio: 75%
- Blending: Gaussian kernel (σ_ratio=0.3)

This approach eliminates boundary artifacts while enabling processing of arbitrary input dimensions.

---

## Training Details

### Optimization

- **Optimizer**: AdamW (β₁=0.9, β₂=0.999, ε=1e-8)
- **Learning Rate**: 5×10⁻⁵ with warm restarts
- **Scheduler**: CosineAnnealingWarmRestarts (T₀=10, T_mult=2, η_min=1e-6)
- **Weight Decay**: 1×10⁻³
- **Gradient Clipping**: Max norm 0.5
- **Mixed Precision**: AMP enabled for memory efficiency

### Training Strategy

- **Batch Size**: 8 per GPU
- **Gradient Accumulation**: 4 steps (effective batch size: 32)
- **Epochs**: 100-150 with early stopping
- **Validation**: Every 2 epochs on reserved files
- **Data Augmentation**: Horizontal/vertical flips (30% probability)

### Preprocessing

**Normalization**:
```python
T_norm = (T_brightness - 200) / 150  # Maps [50, 350]K → [-1, 1]
```

**Degradation Model** (Training):
```python
# 2× downsampling via block averaging
low_res = high_res.reshape(H//2, 2, W//2, 2).mean(axis=(1,3))
# Additive Gaussian noise (σ=0.01)
low_res += np.random.randn(H//2, W//2) * 0.01
```

---

## Installation

### Requirements

```bash
Python >= 3.11
PyTorch == 2.1.0
torchvision == 0.16.0
numpy >= 1.21.0, < 1.24.0
opencv-python-headless
matplotlib >= 3.5.0
scikit-learn >= 1.0.0
tqdm >= 4.60.0
Pillow >= 9.0.0
psutil >= 5.8.0
```

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/amsr2-enhanced-sr.git
cd amsr2-enhanced-sr

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### Training

```bash
python enhanced_amsr2_model.py \
    --npz-dir /path/to/data \
    --max-files 100 \
    --epochs 150 \
    --batch-size 8 \
    --lr 5e-5 \
    --gradient-accumulation 4 \
    --num-workers 4 \
    --files-per-batch 5 \
    --max-swaths-per-file 500 \
    --save-dir ./models \
    --validate-every 2 \
    --use-amp
```

### Inference (Single 2× Upsampling)

```bash
python patch_based_inference.py \
    --npz-dir /path/to/data \
    --model-path ./models/best_model.pth \
    --num-samples 20 \
    --save-dir ./results \
    --overlap-ratio 0.75
```

### Cascaded 4× Super-Resolution

```bash
python patch_based_inference_4x_cascade.py \
    --npz-dir /path/to/data \
    --model-path ./models/best_model.pth \
    --num-samples 5 \
    --save-dir ./cascaded_4x_results
```

### Cascaded 8× Super-Resolution

```bash
python cascaded_8x_inference.py \
    --npz-dir /path/to/data \
    --model-path ./models/best_model.pth \
    --num-samples 5 \
    --save-dir ./cascaded_8x_results
```

---

## Repository Structure

```
amsr2-enhanced-sr/
├── enhanced_amsr2_model.py          # Main training script with attention-enhanced U-Net
├── gpu_sequential_amsr2_optimized.py # Optimized dataset loader and base model
├── patch_based_inference.py         # 2× super-resolution inference
├── patch_based_inference_4x_cascade.py  # Cascaded 4× upsampling
├── cascaded_8x_inference.py         # Cascaded 8× upsampling (two variants)
├── test_enhanced_amsr2.py           # Model evaluation on test set
├── utils/
│   ├── __init__.py
│   └── util_calculate_psnr_ssim.py  # PSNR/SSIM calculation utilities
├── run/
│   ├── SR_run.sbatch                # SLURM training script
│   ├── inference/
│   │   ├── patch_inference.sbatch   # 2× inference job
│   │   ├── patch_inference_4x.sbatch    # 4× inference job
│   │   └── patch_inference_8x.sbatch    # 8× inference job
│   └── test/
│       └── metrcis_test.sbatch      # Metrics evaluation job
├── requirements.txt                 # Python dependencies
└── README.md
```

---

## Model Performance

The enhanced architecture achieves state-of-the-art performance on AMSR2 brightness temperature super-resolution:

- **PSNR**: 41-42 dB (2× upsampling)
- **SSIM**: 0.97-0.975
- **Mean Temperature Error**: <1.5 K
- **Inference Speed**: 50-100 images/second (GPU)

Results demonstrate significant improvements over bicubic interpolation, preserving fine-scale thermal features and maintaining physical consistency across cascaded upsampling stages.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{amsr2-enhanced-sr,
  author = {Volodymyr Didur},
  title = {Advanced Deep Learning Models for Generating Super-resolution AMSR2 Imagery in Support of Sea Ice Forecasting and Analysis},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/amsr2-enhanced-sr}
}
```

---

## Acknowledgments

This work builds upon:
- SwinIR for transformer-based image restoration
- Real-ESRGAN for practical super-resolution algorithms
- BasicSR framework for training and evaluation utilities

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---

## Contact

For questions or collaboration inquiries, please open an issue or contact [volodymyr.didur@stonybrook.edu](mailto:volodymyr.didur@stonybrook.edu).
