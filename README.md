# Handshape Training Data Generation

Generate synthetic training data for hand gesture recognition. The framework includes support for pose-conditioned image generation, which allows controlling the hand pose in generated images.

## Overview

The main focus of this repository is to use conditional GANs to generate synthetic training data for hand gesture recognition tasks. The framework supports various GAN architectures and conditioning methods, with a particular emphasis on pose-conditioned generation using methods like SPADE (Spatially-Adaptive Normalization) and labe-conditioned generation using ReACGAN (Regularized and Adversarial Contrastive GAN).

## Features

- **Pose-conditioned image generation**: Generate images based on keypoint/pose information
- **Label-conditioned image generation**: Generate images based on sign label
- **Multiple GAN architectures**: Support for various backbones including ResNet, BigResNet, StyleGAN2, and StyleGAN3
- **Conditioning methods**: Supports SPADE, cBN (conditional BatchNorm), cAdaIN, and more
- **Data augmentation**: Includes differentiable augmentation techniques
- **Evaluation metrics**: Built-in support for FID, Inception Score, and Precision/Recall metrics
- **Dataset integration**: Ready-to-use with hand gesture datasets (RWTH, Hagrid, FreiHAND)
- **Training data generation utilities**: Tools for generating synthetic datasets

## Datasets

The repository includes configuration files for several hand gesture datasets:
- **RWTH**
- **Hagrid**
- **FreiHAND**

## Usage

### Training a GAN

```bash
python src/main.py -t -hdf5 -batch_stat -metrics is fid prdc -ref "valid" \
    -data /path/to/dataset/ -cfg ./src/configs/DATASET/CONFIG.yaml \
    -save ./samples/DATASET/MODEL/ --pose
```

### Generating synthetic data

```bash
python src/main.py -sd -sd_num 1000 -ref "valid" -data /path/to/dataset/ \
    -cfg ./src/configs/DATASET/CONFIG.yaml -save /path/to/output/ \
    -ckpt /path/to/checkpoint/ --num_workers 4 --truncation_factor 0.5 --pose
```

### Evaluating classifier performance

```bash
python src/classify.py --train_mode real --dset1 /path/to/real_dataset/ \
    --batch_size 8 -save /path/to/results/ --num_workers 4 --epochs 100 \
    --n_classes 18 --dims 64 --dset_used 180
```

## Configuration

The repository includes various configuration files for different datasets and GAN architectures. Key parameters include:

- **Backbone**: Model architecture (`big_resnet`, `stylegan2`, etc.)
- **Conditioning method**: How the pose information is incorporated (`SPADE`, `cBN`, etc.)
- **Batch size**: Training batch size (varies by configuration)
- **Augmentation**: Type of differentiable augmentation applied

## Training Modes

When training classifiers, several modes are available:
- `real`: Train only on real data
- `gen`: Pretrain on generated data
- `mixgen`: real and generated data with mixup
- `reggen`: Train with a regularized loss using both real and generated data

## License

This project is licensed under the MIT License - see the LICENSE file for details. Components derived from NVIDIA StyleGAN are covered by the NVIDIA Source Code License-NC.

## Acknowledgements

This project builds upon [PyTorch StudioGAN](https://github.com/POSTECH-CVLab/PyTorch-StudioGAN) by MinGuk Kang and includes some components adapted from other projects. The original PyTorch StudioGAN is also available under the MIT License.
