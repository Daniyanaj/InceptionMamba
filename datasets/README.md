# Dataset Preparation

This directory contains dataset loaders and preparation scripts for InceptionMamba.

## Supported Datasets

### 1. SegPC21 (Plasma Cell Segmentation)
- **Task**: Seg

mentation of Multiple Myeloma Plasma Cells
- **Training samples**: 775
- **Validation samples**: 194  
- **Test samples**: 194
- **Download**: [IEEE Dataport](https://ieee-dataport.org/competitions/segmentation-multiple-myeloma-plasma-cells-microscopic-images)

### 2. GlaS (Gland Segmentation)
- **Task**: Gland Segmentation in Colon Histology Images
- **Training samples**: 85
- **Test samples**: 80
- **Evaluation**: 3×5-fold cross-validation
- **Download**: [Warwick Dataset](https://warwick.ac.uk/fac/cross_fac/tia/data/glascontest/)

### 3. ISIC2017 (Skin Lesion Segmentation)
- **Task**: Skin Lesion Boundary Segmentation
- **Training samples**: 2000
- **Validation samples**: 150
- **Test samples**: 600
- **Download**: [ISIC 2017](https://challenge.isic-archive.com/data/#2017)

### 4. ISIC2018 (Skin Lesion Segmentation)
- **Task**: Skin Lesion Boundary Segmentation
- **Training samples**: 2594
- **Validation samples**: 100
- **Test samples**: 1000
- **Download**: [ISIC 2018](https://challenge.isic-archive.com/data/#2018)

## Preparation Instructions

### SegPC21

1. Download the dataset from IEEE Dataport
2. Extract to a directory (e.g., `/data/raw/segpc21`)
3. Run preparation script:

```bash
python prepare_segpc.py \
    --data_path /data/raw/segpc21 \
    --output_path ../data/segpc21 \
    --input_size 224
```

Expected directory structure after preparation:
```
data/segpc21/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

### GlaS

1. Download training and test data
2. Extract to a directory (e.g., `/data/raw/glas`)
3. Run preparation script:

```bash
python prepare_glas.py \
    --data_path /data/raw/glas \
    --output_path ../data/glas \
    --n_folds 5
```

### ISIC2017

1. Download:
   - ISIC2017_Task1-2_Training_Input
   - ISIC2017_Task1_Training_GroundTruth
   - ISIC2017_Task1_Validation_Input
   - ISIC2017_Task1_Validation_GroundTruth
   - ISIC2017_Task1_Test_Input
   - ISIC2017_Task1_Test_GroundTruth

2. Run preparation:

```bash
python prepare_isic.py \
    --data_path /data/raw/isic2017 \
    --output_path ../data/isic2017 \
    --year 2017 \
    --input_size 224
```

### ISIC2018

1. Download:
   - ISIC2018_Task1-2_Training_Input
   - ISIC2018_Task1_Training_GroundTruth
   - ISIC2018_Task1_Validation_Input
   - ISIC2018_Task1_Validation_GroundTruth
   - ISIC2018_Task1_Test_Input
   - ISIC2018_Task1_Test_GroundTruth

2. Run preparation:

```bash
python prepare_isic.py \
    --data_path /data/raw/isic2018 \
    --output_path ../data/isic2018 \
    --year 2018 \
    --input_size 224
```

## Dataset Loaders

### Using in Code

```python
from datasets.isic import ISIC2018DatasetFast
from datasets.segpc import SegPC2021Dataset
from torch.utils.data import DataLoader

# ISIC2018
train_dataset = ISIC2018DatasetFast(
    data_path='./data/isic2018/train',
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# SegPC21
train_dataset = SegPC2021Dataset(
    data_path='./data/segpc21/train',
    transform=train_transform
)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
```

## Data Augmentation

During training, we apply:
- Random rotation (±180°)
- Random horizontal/vertical flip
- Random color jittering
- Normalization (ImageNet statistics)

## Citation

If you use these datasets, please cite the original papers:

```bibtex
@article{gupta2021segpc,
  title={SegPC-2021: Segmentation of Multiple Myeloma Plasma Cells in Microscopic Images},
  author={Gupta, Anubha and Gupta, Ritu and Gehlot, Shiv and Goswami, Shubham},
  journal={IEEE Dataport},
  year={2021}
}

@article{sirinukunwattana2017glas,
  title={Gland segmentation in colon histology images: The glas challenge contest},
  author={Sirinukunwattana, Korsuk and others},
  journal={Medical Image Analysis},
  volume={35},
  pages={489--502},
  year={2017}
}

@inproceedings{codella2018isic2017,
  title={Skin lesion analysis toward melanoma detection: A challenge at the 2017 international symposium on biomedical imaging},
  author={Codella, Noel CF and others},
  booktitle={IEEE ISBI 2018},
  pages={168--172},
  year={2018}
}

@article{codella2019isic2018,
  title={Skin lesion analysis toward melanoma detection 2018: A challenge hosted by the international skin imaging collaboration},
  author={Codella, Noel and others},
  journal={arXiv preprint arXiv:1902.03368},
  year={2019}
}
```
