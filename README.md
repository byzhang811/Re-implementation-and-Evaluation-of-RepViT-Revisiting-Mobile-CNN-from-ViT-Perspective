# Re-implementation and Evaluation of RepViT: Revisiting Mobile CNN from ViT Perspective

Course: EC523 Deep Learning (Fall 2025)  
Project: Re-implementation and evaluation of **RepViT** on CIFAR-100

## Group Members

- Boyang Zhang (theostnc@bu.edu)  
- Jiatong Guo (gjt@bu.edu)
- Ziqi Tang  (ztang@bu.edu)
- Youwei Chen  (ychen143@bu.edu

## 1. Overview

This repository contains our re-implementation of the **RepViT** architecture (CVPR 2024) in PyTorch and our experiments on the **CIFAR-100** dataset.  
We reuse the official RepViT code base, remove modules that are unrelated to image classification, and adapt the training pipeline from ImageNet to CIFAR-100.

The main goals are:

- Rebuild and verify the RepViT model structure on a smaller dataset.  
- Train RepViT on CIFAR-100 and observe the training dynamics (Top-1 / Top-5 accuracy, loss curves).  
- Prepare for later experiments on **hyper-parameter tuning** and **model size reduction** for small datasets.

## 2. Repository Structure

Below is the structure of our cleaned and organized project repository:

.
├── main.py               # Training / evaluation entry
├── engine.py             # train_one_epoch, evaluate
├── losses.py             # Cross-entropy + optional distillation loss
├── utils.py              # EMA, logging, checkpoint utilities
├── requirements.txt      # Python dependencies
│
├── model/                # RepViT model definitions
│   └── ...
│
├── data/                 # Dataset building & augmentations
│   └── ...
│
├── logs/                 # Training logs & summaries
│   ├── run1_log.txt
│   └── run1_model.txt
│
└── figures/              # Plots generated from logs
    ├── acc_curve.png     # Top-1 / Top-5 accuracy curves
    └── loss_curve.png    # Training / validation loss curves

---

## 3. Dataset

We do **not** include the CIFAR-100 dataset in this repository (due to size limitations).

Please download CIFAR-100 from the official website or via `torchvision.datasets`,  
and place it in the following structure:

data/  
└── cifar100/  
  └── cifar-100-python/  
    ├── train  
    ├── test  
    └── meta

When running main.py, set --data-path to the CIFAR-100 root directory. For example, if your path is data/cifar100, use:

--data-set CIFAR --data-path ./data/cifar100


On the BU SCC, we used a shared path similar to:

--data-path /projectnb/ec523bn/students/<user>/Datasets/cifar100
