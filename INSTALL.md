# Installation Guide

Complete installation instructions for InceptionMamba.

## System Requirements

### Hardware
- **GPU**: NVIDIA GPU with CUDA support (>= 16GB VRAM recommended)
- **RAM**: At least 32GB
- **Storage**: At least 100GB free space (for datasets + models)

### Software
- **OS**: Linux (Ubuntu 20.04+), Windows 10+, or macOS
- **Python**: 3.8, 3.9, or 3.10
- **CUDA**: 11.8 or 12.1 (for GPU)
- **cuDNN**: Compatible version with CUDA

## Step-by-Step Installation

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/InceptionMamba.git
cd InceptionMamba
```

### 2. Create Conda Environment (Recommended)

```bash
# Create new environment
conda create -n inceptionmamba python=3.10
conda activate inceptionmamba
```

### 3. Install PyTorch

**For CUDA 11.8:**
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

**For CUDA 12.1:**
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```

**For CPU only:**
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 cpuonly -c pytorch
```

### 4. Install Other Dependencies

```bash
pip install -r requirements.txt
```

### 5. Install Mamba SSM

```bash
pip install mamba-ssm causal-conv1d
```

**Note**: Mamba requires compilation. If you encounter issues:

```bash
# Install build tools
pip install packaging ninja

# Try building from source
git clone https://github.com/state-spaces/mamba.git
cd mamba
pip install -e .
cd ..
```

## Verification

Verify your installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "from mamba_ssm import Mamba; print('Mamba installed successfully')"
python -c "import timm; print(f'timm: {timm.__version__}')"
```

Expected output:
```
PyTorch: 2.1.1+cu118
CUDA available: True
Mamba installed successfully
timm: 0.9.12
```

## Alternative: Docker Installation (Coming Soon)

```bash
# Pull image
docker pull YOUR_USERNAME/inceptionmamba:latest

# Run container
docker run --gpus all -it -v $(pwd):/workspace YOUR_USERNAME/inceptionmamba:latest
```

## Troubleshooting

### CUDA Out of Memory
Reduce batch size in config files:
```yaml
training:
  batch_size: 8  # Reduce from 16
```

### Mamba Installation Issues

**Problem**: `error: Microsoft Visual C++ 14.0 or greater is required` (Windows)

**Solution**:
1. Install [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/)
2. Restart terminal and retry

**Problem**: `ModuleNotFoundError: No module named 'mamba_ssm'`

**Solution**:
```bash
pip uninstall mamba-ssm causal-conv1d
pip install mamba-ssm causal-conv1d --no-cache-dir
```

### Import Errors

```bash
pip install -r requirements.txt --upgrade
```

## Next Steps

After successful installation:

1. **Prepare Datasets**: See [datasets/README.md](datasets/README.md)
2. **Download Pre-trained Models**: See README.md Model Zoo section
3. **Start Training**: See training instructions in README.md

## Getting Help

If issues persist:
1. Check [GitHub Issues](https://github.com/YOUR_USERNAME/InceptionMamba/issues)
2. Open new issue with:
   - Error message
   - System info (`python --version`, `nvidia-smi`)
   - Installation method used
