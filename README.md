# Incomplete Ring PET Reconstruction Model

This repository contains code for a deep learning-based approach to reconstructing complete Positron Emission Tomography (PET) sinograms from incomplete ring PET data using a Swin Transformer-based U-Net architecture.

## Overview

In Positron Emission Tomography (PET), incomplete detector rings can occur due to hardware failures, cost constraints, or specific clinical needs. This model aims to reconstruct complete sinogram data from incomplete detector ring configurations to enable high-quality image reconstruction.

## Features

- Swin Transformer U-Net architecture for sinogram reconstruction
- Robust training pipeline with checkpoint saving and resumption
- Evaluation metrics including PSNR, SSIM, and MAE
- Visualization tools for sinogram inspection
- Support for both batch testing and single-file inference

## Requirements

```
torch>=1.9.0
torchvision>=0.10.0
numpy>=1.20.0
matplotlib>=3.4.0
tqdm>=4.61.0
tensorboard>=2.6.0
```

## Usage

### Data Format

The data directory should be structured as follows:

```
data_dir/
├── train/
│   ├── incomplete_1_1.npy
│   ├── complete_1_1.npy
│   ├── incomplete_1_2.npy
│   ├── complete_1_2.npy
│   └── ...
└── test/
    ├── incomplete_1_1.npy
    ├── complete_1_1.npy
    ├── incomplete_1_2.npy
    ├── complete_1_2.npy
    └── ...
```

Where each `.npy` file contains a sinogram representation.

### Training

To train a model:

```bash
python main.py --mode train --data_dir path/to/data --output_dir path/to/output --img_size 224 --batch_size 16 --epochs 100
```

### Testing

To test a trained model on a test dataset:

```bash
python main.py --mode test --data_dir path/to/data --model_path path/to/model.pth --output_dir path/to/output
```

### Single Example Testing

To test a trained model on a single file:

```bash
python main.py --mode test_single --test_file path/to/incomplete_sinogram.npy --model_path path/to/model.pth --output_dir path/to/output
```

## Model Architecture

The model is based on the Swin Transformer U-Net architecture, which combines:
- Swin Transformer blocks for efficient self-attention on image data
- U-Net skip connections for preserving spatial information
- Hierarchical feature representation for capturing multi-scale patterns

## Results

The model outputs visualizations of:
- Input incomplete sinograms
- Ground truth complete sinograms
- Predicted complete sinograms
- Difference maps between ground truth and predictions

Performance metrics (PSNR, SSIM, MAE) are calculated and logged during both training and testing.

## Files Structure

- `main.py`: Entry point script for training or testing
- `train.py`: Training logic implementation
- `test.py`: Testing and evaluation implementation
- `util.py`: Utility functions for data loading, metrics, and visualization
- `SwinUnet.py`: Swin Transformer U-Net model implementation

## Citations

The implementation is based on the following papers:

1. Liu, Z., Lin, Y., Cao, Y., Hu, H., Wei, Y., Zhang, Z., ... & Guo, B. (2021). Swin transformer: Hierarchical vision transformer using shifted windows. In Proceedings of the IEEE/CVF International Conference on Computer Vision (pp. 10012-10022).

2. Cao, H., Wang, Y., Chen, J., Jiang, D., Zhang, X., Tian, Q., & Wang, M. (2021). Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation. In International Conference on Medical Image Computing and Computer Assisted Intervention.