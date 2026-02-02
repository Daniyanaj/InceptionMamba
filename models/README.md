# InceptionMamba Model Architecture

This directory contains the implementation of InceptionMamba for medical image segmentation.

## Architecture Overview

InceptionMamba consists of three main components:

### 1. Backbone Network
- **ResNet50** (default) or **PVT-V2-B2**
- Extracts multi-scale features from first 3 stages
- Pre-trained on ImageNet

### 2. Bottleneck Block
Contains two key modules:

#### Feature Calibration Module (FCM)
- Highlights fine details for overlapping structures
- Uses down-sampling followed by up-sampling for smoothness
- Combines subtraction and multiplication operations
- Critical for handling blurred boundaries

#### Inception Mamba Module (IMM)
- Combines depth-wise convolutions with different kernel sizes:
  - `3×3` for local features
  - `1×11` and `11×1` for elongated structures
- Integrates Mamba block for global context
- Identity branch for residual connection
- Efficient multi-contextual feature extraction

### 3. Decoder
- Lightweight design without dense skip connections
- Uses IMM for multi-contextual refinement
- Combines low and high-level semantics

## Model Files

### `inception_mamba.py`
Main model implementation containing:
- `InceptionMamba`: Complete model class
- `InceptionMambaModule`: IMM implementation
- `FeatureCalibrationModule`: FCM implementation
- `Decoder`: Decoder implementation

## Usage

### Basic Usage

```python
import torch
from models.inception_mamba import InceptionMamba

# Create model
model = InceptionMamba(
    num_classes=2,  # Binary segmentation
    backbone='resnet50',  # or 'pvt_v2_b2'
    pretrained=True
)

# Forward pass
input = torch.randn(1, 3, 224, 224)
output = model(input)  # Shape: (1, 2, 224, 224)
```

### With Custom Backbone

```python
model = InceptionMamba(
    num_classes=2,
    backbone='pvt_v2_b2',  # PVT-V2-B2 backbone
    pretrained=True
)
```

### Load Pre-trained Weights

```python
checkpoint = torch.load('checkpoints/inceptionmamba_segpc21.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
```

## Model Configurations

### ResNet50 Backbone (Default)
- Parameters: 11.92M
- GFLOPs: 6.72
- Input size: 224×224
- Best for: Balanced performance and efficiency

### PVT-V2-B2 Backbone
- Parameters: 27.27M
- GFLOPs: 4.86
- Input size: 224×224
- Best for: Maximum efficiency with good performance

## Key Features

1. **Efficient Design**: 5× fewer GFLOPs than previous SOTA
2. **Multi-Scale Features**: Captures features at multiple scales
3. **Boundary Enhancement**: FCM for better boundary segmentation
4. **Global Context**: Mamba block for long-range dependencies
5. **No Dense Connections**: Simpler and more efficient decoder

## Architecture Diagram

```
Input (3, 224, 224)
       ↓
   Backbone (ResNet50 or PVT-V2-B2)
       ↓
   [Stage 1, Stage 2, Stage 3 Features]
       ↓
 Bottleneck Block
   ├─ Feature Calibration Module (FCM)
   └─ Inception Mamba Module (IMM)
       ↓
   Decoder (with IMM)
       ↓
   Segmentation Head
       ↓
Output (2, 224, 224)
```

## Performance Comparison

| Component | Params (M) | GFLOPs | Contribution |
|-----------|------------|---------|--------------|
| Baseline U-Net | 11.42 | 5.27 | - |
| + FCM | 11.53 | 5.29 | +1.35% Dice |
| + IMM | 11.54 | 5.30 | +2.45% Dice |
| + Decoder with IMM | 11.67 | 6.09 | +3.00% Dice |
| + Stem skip | 11.92 | 6.72 | +3.51% Dice |

## Citation

```bibtex
@article{kareem2026inceptionmamba,
  title={InceptionMamba: Efficient Multi-Stage Feature Enhancement with Selective State Space Model for 2D Medical Image Segmentation},
  author={Kareem, Daniya Najiha Abdul Kareem and others},
  journal={MIDL},
  year={2026}
}
```

## References

- [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752)
- [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)
- [Inception v3](https://arxiv.org/abs/1512.00567)
