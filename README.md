# SI-N2V

Official implementation of **SI-N2V**, including the **PolyMambaUNet** backbone, for self-supervised fluorescence microscopy image denoising.

## Overview

SI-N2V is a self-supervised denoising framework designed for fluorescence microscopy images. It integrates two complementary priors into a Noise2Void-style training strategy:

- **Structure-aware probabilistic masking (SAPM):** adapts the masking probability according to local structural complexity.
- **Intensity-aware loss (IA-Loss):** assigns greater optimization weight to bright, informative fluorescence signals.
- **PolyMambaUNet:** serves as the reconstruction backbone used in the reported experiments.

The method is trained directly from noisy fluorescence images and does not require paired clean targets.

## Repository structure

```text
SI-N2V/
├── config.yaml          # Training and model configuration
├── dataset.py           # Dataset loading, patch extraction, and augmentation
├── inference.py         # Inference script
├── model.py             # PolyMambaUNet model definition
├── requirements.txt     # Python dependencies
├── train.py             # Training entry point
├── trainer.py           # Training, validation, and checkpoint logic
└── utils.py             # Masking, normalization, and loss utilities
```

## Requirements

The main dependencies are listed in `requirements.txt`:

- Python 3.9 or a compatible version
- PyTorch
- torchvision
- NumPy
- Pillow
- tifffile
- PyYAML
- tqdm

A CUDA-capable GPU is recommended for training.

## Installation

Clone the repository:

```bash
git clone https://github.com/mu9478492-tech/SI-N2V.git
cd SI-N2V
```

Create and activate a virtual environment, then install the dependencies:

```bash
python -m venv .venv
```

On Linux or macOS:

```bash
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Data preparation

By default, the code expects the following directory structure:

```text
data/
├── train/
│   └── raw/
│       ├── image_001.tif
│       ├── image_002.tif
│       └── ...
└── test/
    └── raw/
        ├── image_001.tif
        └── ...
```

Supported image formats include PNG, JPG, JPEG, TIFF, and BMP.

The repository does not redistribute the public datasets used in the manuscript. Download each dataset from its original source and organize the images locally before training or inference.

## Configuration

Edit `config.yaml` before training. The main fields include:

```yaml
train_raw_folder:
test_raw_folder:
save_path:

patch_size:
stride:

batch_size:
accum_iter:
num_workers:
pin_memory:
epochs:
lr:
weight_decay:

model:
  type: "PolyMambaUNet"
  in_ch: 1
  out_ch: 1
  width: 64
```

If the path fields are left empty, the code uses the default locations under `./data/`.

## Training

Run:

```bash
python train.py --exp_name example_run
```

Training outputs are written under:

```text
experiments/example_run/
├── checkpoints/
└── visualizations/
```

The script saves the latest model and periodic checkpoints during training.

## Inference

Before running inference, edit the following variables near the top of `inference.py`:

```python
RAW_IMAGE_PATH = "./test_data/sample_noisy.tif"
CHECKPOINT_PATH = "./checkpoints/model_weights.pth"
OUTPUT_IMAGE_NAME = "comparison_result.png"
CONFIG_PATH = "config.yaml"
```

Then run:

```bash
python inference.py
```

The inference script saves a side-by-side comparison of the input and restored image.

## Reproducibility notes

- The default random seed in `train.py` is `42`.
- Training parameters should be recorded in `config.yaml`.
- For publication-related experiments, retain the exact configuration file and checkpoint associated with each reported result.

## Code availability

The source code for SI-N2V, including the PolyMambaUNet backbone, is publicly available at:

https://github.com/mu9478492-tech/SI-N2V

## Citation

Please cite the associated iScience article when using this repository. Full citation information will be added after publication.

## Patent and license notice

A patent application related to aspects of this work is pending.

No open-source license is currently included in this repository. Public availability of the source code does not by itself grant permission to reproduce, modify, redistribute, or commercially use the code. Please contact the authors regarding reuse or licensing.
