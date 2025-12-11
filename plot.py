"""
Filename: plot.py
Author: Jiatong
Date: 2025-11-18
Lines: 48
Description: Script for plotting training curves from log files.
"""

import json
import matplotlib.pyplot as plt
from pathlib import Path

log_file = "/Users/jiatongguo/Desktop/523-Project/logs/log_w125.txt"        
save_dir = "/Users/jiatongguo/Desktop/523-Project/figs"     
       
log_stem = Path(log_file).stem         # "train"
out_dir = Path(save_dir)
out_dir.mkdir(parents=True, exist_ok=True)

epochs = []
train_loss = []
test_loss = []
test_acc1 = []
test_acc5 = []

with open(log_file, "r") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        record = json.loads(line)
        epochs.append(record["epoch"])
        train_loss.append(record["train_loss"])
        test_loss.append(record["test_loss"])
        test_acc1.append(record["test_acc1"])
        test_acc5.append(record["test_acc5"])

# fig 1：train_loss & test_loss
plt.figure()
plt.plot(epochs, train_loss, label="train_loss")
plt.plot(epochs, test_loss, label="test_loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Train/Test Loss vs Epoch ({log_stem})")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = out_dir / f"loss_curve_{log_stem}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"saved: {out_path}")

# fig 2：test_acc1 & test_acc5
plt.figure()
plt.plot(epochs, test_acc1, label="test_acc1")
plt.plot(epochs, test_acc5, label="test_acc5")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title(f"Test Acc@1 and Acc@5 vs Epoch ({log_stem})")
plt.legend()
plt.grid(True)
plt.tight_layout()
out_path = out_dir / f"accuracy_curve_{log_stem}.png"
plt.savefig(out_path, dpi=300, bbox_inches="tight")
print(f"saved: {out_path}")

# plt.show()