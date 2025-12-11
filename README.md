# Re-implementation and Evaluation of RepViT: Revisiting Mobile CNN from ViT Perspective

Course: EC523 Deep Learning (Fall 2025)  
Project: Re-implementation and evaluation of **RepViT** on CIFAR-100

## Group Members

- Boyang Zhang (theostnc@bu.edu)  
- Jiatong Guo (gjt@bu.edu)
- Ziqi Tang  (ztang@bu.edu)
- Youwei Chen  (ychen143@bu.edu)

## 1. Overview

This repository contains our **custom re-implementation** of the **RepViT** architecture (CVPR 2024) in PyTorch and our experiments on the **CIFAR-100** dataset.  

**All code in this repository is written by our team members.** We implemented the RepViT model architecture, training pipeline, data processing modules, and evaluation framework based on the paper's methodology. The implementation is specifically designed for image classification tasks and adapted for training on the CIFAR-100 dataset.

The main goals are:

- **Implement** the RepViT model structure from scratch and verify it on a smaller dataset.  
- **Train** RepViT on CIFAR-100 and observe the training dynamics (Top-1 / Top-5 accuracy, loss curves).  
- **Experiments** on **model size reduction**, **architecture variation** and **hyper-parameter tuning**.

## 2. Repository Structure

Below is the structure of our cleaned and organized project repository:

.
├── main.py               # Training / evaluation entry (Boyang, 351 lines)
├── engine.py             # train_one_epoch, evaluate (Ziqi, Youwei, 100 lines)
├── utils.py              # EMA, logging, checkpoint utilities (Ziqi, Youwei, 228 lines)
├── plot.py               # Script for plotting training curves (Jiatong, 48 lines)
├── requirements.txt      # Python dependencies
│
├── model/                # RepViT model definitions
│   ├── __init__.py       # Model package initialization (Jiatong, 1 line)
│   └── repvit.py         # RepViT architecture implementation (Jiatong, 483 lines)
│
├── data/                 # Dataset building & augmentations
│   ├── datasets.py       # Dataset building and data transformation utilities (Boyang, Jiatong, 134 lines)
│   ├── samplers.py       # Data samplers for distributed training (Ziqi, Youwei, 46 lines)
│   └── threeaugment.py   # 3Augment data augmentation implementation (Ziqi, Youwei, 108 lines)
│
├── logs/                 # Training logs & summaries
│   ├── log.txt
│   └── model.txt
│
└── figures/              # Plots generated from logs
    ├── Loss_run1.png     # Training / validation loss curves
    └── Top1&5_r1.png     # Top-1 / Top-5 accuracy curves

---

## 3. Installation Instructions

### Prerequisites

- Python 3.7+
- CUDA 12.1+ (for GPU training)
- pip or conda

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Code
```

### Step 2: Create Virtual Environment (Recommended)

Using conda:
```bash
conda create -n repvit python=3.8
conda activate repvit
```

Or using venv:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

Install PyTorch with CUDA support:
```bash
pip install torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

Install other dependencies:
```bash
pip install timm==0.5.4 fvcore wandb matplotlib numpy
```

Or install all dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```

**Note:** The `requirements.txt` includes PyTorch installation with CUDA 12.1. If you need a different CUDA version or CPU-only version, modify the PyTorch installation command accordingly.

### Step 4: Verify Installation

Verify that PyTorch can detect your GPU:
```python
python -c "import torch; print(torch.cuda.is_available()); print(torch.__version__)"
```

### Step 5: Setup Weights & Biases (Optional)

If you want to use WandB for experiment tracking:
```bash
wandb login
```

## 4. Dataset

We do **not** include the CIFAR-100 dataset in this repository (due to size limitations).

Please download CIFAR-100 from the official website or via [torchvision.datasets](https://www.cs.toronto.edu/~kriz/cifar.html)
,  
and place it in the following structure:

data/  
└── cifar100/  
  └── cifar-100-python/  
    ├── train  
    ├── test  
    └── meta

## 5. Usage

### Training

Train RepViT on CIFAR-100:
```bash
python main.py \
    --data-set CIFAR \
    --data-path ./data/cifar100 \
    --model repvit_m0_9 \
    --batch-size 256 \
    --epochs 300 \
    --lr 1e-3 \
    --input-size 224 \
    --output_dir checkpoints \
    --project repvit
```

### Evaluation

Evaluate a trained model:
```bash
python main.py \
    --data-set CIFAR \
    --data-path ./data/cifar100 \
    --model repvit_m0_9 \
    --eval \
    --resume checkpoints/repvit_m0_9/checkpoint_best.pth
```

### Resume Training

Resume training from a checkpoint:
```bash
python main.py \
    --data-set CIFAR \
    --data-path ./data/cifar100 \
    --model repvit_m0_9 \
    --resume checkpoints/repvit_m0_9/checkpoint_10.pth \
    --batch-size 256 \
    --epochs 300
```

### Plotting Training Curves

Generate plots from training logs:
```python
python plot.py
```

**Note:** Update the `log_file` and `save_dir` paths in `plot.py` before running.

### Key Arguments

- `--data-set`: Dataset type (`CIFAR`, `IMNET`)
- `--data-path`: Path to dataset root directory
- `--model`: Model variant (`repvit_m0_9`, `repvit_m0_9_w088`, etc.)
- `--batch-size`: Batch size for training (default: 256)
- `--epochs`: Number of training epochs (default: 300)
- `--lr`: Learning rate (default: 1e-3)
- `--input-size`: Input image size (default: 224)
- `--output_dir`: Directory to save checkpoints and logs
- `--eval`: Evaluation mode only
- `--resume`: Path to checkpoint to resume from

For BU SCC users, we recommend you set the data path to:
```bash
--data-path /projectnb/ec523bn/students/<user>/Datasets/cifar100
```