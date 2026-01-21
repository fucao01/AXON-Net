# AXON-Net: Lightweight Axial Context Network for Unstructured Road Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-v1.10%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-orange)]()

**AXON-Net** 是一种面向非结构化道路语义分割的轻量化 U-Net 风格网络。本项目引入了 **CASAB**（通道-空间注意力）、**LightPCT**（轻量化部分上下文变换器）与 **DPCF-TSE**（细结构增强）模块，旨在低计算成本下兼顾全局上下文建模与边界细节恢复。

> ⚠️ **重要说明 / Note**：
> 本项目目前处于**活跃研究阶段**（正推进针对低光照/极端场景的扩展研究）。
> 因此，本仓库仅开源**核心模型结构定义**与**损失函数实现**以供学术参考。完整训练脚本、预训练权重及清洗后的数据集**暂不公开**。

## 📂 仓库内容 (Repository Contents)

```text
├── src/
│   └── AXONNet.py             # 网络模型定义 (Output: {"out": logits})
├── train_utils/
│   └── dice_coefficient_loss.py # 复合损失函数集合 (WCE + Dice + Boundary)
├── my_dataset.py              # 数据集加载器 (适配 TP-Dataset 格式)
├── transforms.py              # 数据增强与预处理变换
└── README.md                  # 项目说明


