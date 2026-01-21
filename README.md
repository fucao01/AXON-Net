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


## 🖼️ 结果可视化 (Results Visualization)

为了直观展示 AXON-Net 的性能及后续研究潜力，我们提供了以下推理结果示例。

### 1. AXON-Net 当前研究成果 (Current Performance)

以下展示了 AXON-Net 在 IDD-UR 非结构化道路数据集上的推理效果。模型在保持轻量化的同时，有效捕捉了复杂的道路边界与细微结构。

![AXON-Net Current Inference](figures/当前研究推理图/推理图.png)
*图 1：AXON-Net 在复杂非结构化场景下的分割表现*

### 2. 后续研究进展：低光照/夜间场景 (Future Research: Low-light Scenarios)

**延续非结构化道路分割这一研究主线**，我们正在积极推进面向低光照、夜间等极端场景的**新型算法研究**。以下展示了我们在该方向上取得的最新进展（部分测试结果基于我们正在研发的新模型）。

![Input 01](figures/未来研究进展持续推进中/01-dark.jpg) ![Inference 01](figures/未来研究进展持续推进中/01inference.jpg)
*样本 01：极端低光照输入 (左) vs 新模型初步推理结果 (右)*

![GT 02](figures/未来研究进展持续推进中/02-GT.png) ![Mask 02](figures/未来研究进展持续推进中/02mask.png)
*样本 02：真值 GT (左) vs 新模型预测 Mask (右)*
