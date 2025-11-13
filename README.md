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

```text
.
├── main.py            # Training / evaluation entry
├── engine.py          # train_one_epoch, evaluate
├── losses.py          # Cross-entropy + distillation loss
├── utils.py           # EMA, logging, checkpoint utilities
├── requirements.txt   # Python package dependencies
│
├── model/             # RepViT model definitions
│   └── ...
│
├── data/              # Dataset building and augmentations
│   └── ...
│
├── logs/              # Training logs and model summaries
│   ├── run1_log.txt
│   └── run1_model.txt
│
└── figures/           # Plots generated from logs
    ├── acc_curve.png  # Top-1 / Top-5 vs. epoch
    └── loss_curve.png # Train / test loss vs. epoch
