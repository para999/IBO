Official PyTorch implementation of **Dual Representation Aggregation Network for Blind Image Super-Resolution via Iterative Bi-level Optimization**, published in **IEEE Transactions on Multimedia (TMM), 2026**.

**Xianhong Wen, Sheng Ren, Bin Hu, Xiangyuan Zhu, Zehui Xiao, and Kehua Guo**

[Paper](https://doi.org/10.1109/TMM.2026.3684305) | [Code](https://github.com/para999/IBO)

## Overview

DRANet combines low-resolution priors and super-resolution posteriors through a degradation-aware branch and a deviation-correction branch for blind image super-resolution. Iterative Bi-level Optimization (IBO) decouples the learning process into staged subproblems and iteratively optimizes the two branches to stabilize training.

## Requirements

The original environment uses the following packages:

```text
torch==1.12.1
torchvision==0.13.1
opencv-python
einops
scikit-image
numpy
```

The code can also run with PyTorch 2.x, provided that a matching torchvision version and compatible versions of the other dependencies are installed.
