#!/bin/bash
# Download pretrained diffusion checkpoints and test datasets.
# Checkpoints and datasets are the same public ones used by DPS and DAPS.

mkdir -p models datasets

# FFHQ 256 DDPM (from DPS: https://github.com/DPS2022/diffusion-posterior-sampling)
echo "Downloading FFHQ 256 DDPM model..."
gdown https://drive.google.com/uc?id=1BGwhRWUoguF-D8wlZ65tf227gp3cDUDh -O models/ffhq_10m.pt

# ImageNet 256 DDPM (from guided-diffusion: https://github.com/openai/guided-diffusion)
echo "Downloading ImageNet 256 DDPM model..."
gdown https://drive.google.com/uc?id=1HAy7P19PckQLczVNXmVF-e_CRxq098uW -O models/imagenet256.pt

# Test datasets (100-image FFHQ / ImageNet subsets provided by DAPS)
echo "Downloading FFHQ test dataset..."
gdown https://drive.google.com/uc?id=1IzbnLWPpuIw6Z2E4IKrRByh6ihDE5QLO -O datasets/test-ffhq.zip
unzip -q datasets/test-ffhq.zip -d ./datasets
rm datasets/test-ffhq.zip

echo "Downloading ImageNet test dataset..."
gdown https://drive.google.com/uc?id=1pqVO-LYrRRL4bVxUidvy-Eb22edpuFCs -O datasets/test-imagenet.zip
unzip -q datasets/test-imagenet.zip -d ./datasets
rm datasets/test-imagenet.zip

# (Optional) nonlinear blur model, only needed for the nonlinear_blur task.
# Requires the bkse repo to be cloned first (see README).
if [ -d "bkse" ]; then
    echo "Downloading nonlinear blur model (GOPRO_wVAE)..."
    mkdir -p bkse/experiments/pretrained
    gdown https://drive.google.com/uc?id=1vRoDpIsrTRYZKsOMPNbPcMtFDpCT6Foy -O bkse/experiments/pretrained/GOPRO_wVAE.pth
else
    echo "Skipping nonlinear blur model (bkse/ not found; see README for the nonlinear_blur task setup)."
fi
