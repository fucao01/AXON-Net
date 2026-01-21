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
```

### 2. 数据预处理与清洗流程 (Data Cleaning & Preprocessing Pipeline)

为了消除原始数据中的噪声并确保模型专注于非结构化道路特征，我们执行了严格的数据清洗与标准化流程（与论文第 III-A 章一致）。

#### 2.1 语义标签重映射 (Semantic Label Remapping)
原始 IDD 数据集包含 30+ 类精细语义标注。针对本研究任务，我们执行了类别聚合与二值化操作：
* **前景提取 (Foreground)**：将 `Drivable Area` (可行驶区域) 及相关子类映射为 **Road (255)**。
* **背景抑制 (Background)**：将车辆、行人、植被、天空及障碍物统一视为 **Background (0)**。
* **忽略区域 (Ignore)**：原始标注中的 `void` 或模糊边界区域被设为 **Ignore Index (255)**，在 Loss 计算中不予考虑（如果是二分类通常设为255或特定值，具体视你代码而定，若无忽略类可删除此条）。

#### 2.2 人工质量清洗 (Manual Quality Control)
我们对训练集与验证集进行了逐张人工核查，剔除了约 **5-10%** 的低质量样本，剔除标准如下：
1.  **标注偏移 (Misalignment)**：GT 掩膜与 RGB 图像存在明显空间错位（>10 pixels）。
2.  **标签缺失 (Missing Labels)**：图像包含明显道路但 GT 全黑或大面积缺失的样本。
3.  **场景不符 (Out-of-Scope)**：剔除了包含大量城市高架桥或完全结构化道路的样本，确保数据集专注于**非结构化/乡村道路**场景。

#### 2.3 格式统一化 (Standardization)
* **图像格式**：统一转为 `.jpg` (RGB, 8-bit)。
* **标签格式**：统一转为单通道 `.png` (8-bit, 索引模式)。
* **分辨率对齐**：所有输入图像在训练前均执行了统一的宽高比适配与缩放处理。




## 🖼️ 结果可视化 (Results Visualization)

为了直观展示 AXON-Net 的性能及后续研究潜力，我们提供了以下推理结果示例。

### 1. AXON-Net 当前研究成果 (Current Performance)

以下展示了 AXON-Net 在 IDD-UR 非结构化道路数据集上的推理效果。模型在保持轻量化的同时，有效捕捉了复杂的道路边界与细微结构。

<p align="center">
  <img src="figures/当前研究推理图/推理图.png" alt="AXON-Net Current Inference" width="85%" />
  <br>
  <em>图 1：AXON-Net 在复杂非结构化场景下的分割表现</em>
</p>

### 2. 后续研究进展：低光照/夜间场景 (Future Research: Low-light Scenarios)

**延续非结构化道路分割这一研究主线**，我们正在积极推进面向低光照、夜间等极端场景的**新型算法研究**。以下展示了我们在该方向上取得的最新进展（部分测试结果基于我们正在研发的新模型）。

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
