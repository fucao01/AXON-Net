# AXON-Net: Lightweight Axial Context Network for Unstructured Road Segmentation

[![PyTorch](https://img.shields.io/badge/PyTorch-v1.10%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-orange)]()

**AXON-Net** 是一种面向非结构化道路语义分割的轻量化 U-Net 风格网络。本项目引入了 **CASAB**（通道-空间注意力）、**LightPCT**（轻量化部分上下文变换器）与 **DPCF-TSE**（细结构增强）模块，旨在低计算成本下兼顾全局上下文建模与边界细节恢复。

> ⚠️ **重要说明 / Note**：
> 本项目目前处于**活跃研究阶段**（正推进针对低光照/极端场景的扩展研究）。
> 因此，本仓库仅开源**核心模型结构定义**与**损失函数实现**以供学术参考。

## 📂 仓库内容 (Repository Contents)

```text
├── src/
│   └── AXONNet.py             # 网络模型定义 
├── train_utils/
│   └── dice_coefficient_loss.py # 复合损失函数集合 
├── my_dataset.py              # 数据集加载器 (适配 TP-Dataset 格式)
├── transforms.py              # 数据增强与预处理变换
└── README.md                  # 项目说明
```

### 2. 数据预处理与清洗流程 (Data Cleaning & Preprocessing Pipeline)

为了消除原始数据中的噪声并确保模型专注于非结构化道路特征，我们对印度的开源非结构化场景的全景分割数据集IDD　partI 部分的数据集和越野数据集ORDF进行清洗
形成匹配研究目标的数据集IDD-UR和ORFD-AV
数据集当前暂不公开，因为还有推进未来的研究正在推进





## 🖼️ 结果可视化 (Results Visualization)

为了直观展示 AXON-Net 的性能及后续研究潜力，我们提供了以下推理结果示例。

### 1. AXON-Net 实验对比推理图

以下展示了 AXON-Net 在 IDD-UR 、ORDF-AV非结构化道路数据集上的推理对比效果。模型在保持轻量化的同时，有效捕捉了复杂的道路边界与细微结构。

<p align="center">
  <img src="figures/当前研究推理图/推理图.png" alt="AXON-Net Current Inference" width="85%" />
  <br>
  <em>图 1：AXON-Net 在复杂非结构化场景下的分割表现</em>
</p>

### 🚀2. 后续研究规划：低光照/夜间场景 (Future Research: Low-light Scenarios)

本项目展示了我们在非结构化道路分割领域的阶段性成果。在此研究基础之上，我们正致力于推进更高效、更鲁棒的分割架构，


**延续非结构化道路分割这一研究主线**，我们正在积极推进面向低光照、夜间等极端场景的**新型算法研究**。以下展示了我们在该方向上取得的最新进展，探索结合 Mamba / State Space Models (SSM) 等线性复杂度技术，突破现有 CNN/Transformer 的性能瓶颈。（部分测试结果基于我们正在探索推进的新模型）。

<p align="center">
  <strong>极端低光照输入 (Input) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; vs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 新模型初步推理 (Inference)</strong>
  <br>
  <img src="figures/未来研究进展持续推进中/01-dark.jpg" width="45%" />
  <img src="figures/未来研究进展持续推进中/01inference.jpg" width="45%" />
  <br>
  <em>样本 01: Dark Scenario</em>
</p>

<br>

<p align="center">
  <strong>真值 (Ground Truth) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; vs &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 预测掩膜 (Prediction)</strong>
  <br>
  <img src="figures/未来研究进展持续推进中/02-GT.png" width="45%" />
  <img src="figures/未来研究进展持续推进中/02mask.png" width="45%" />
  <br>
  <em>样本 02: Mask Comparison</em>
</p>



