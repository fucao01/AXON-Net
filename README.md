# AXON-Net：面向非结构化道路分割的轻量化轴向上下文网络

[![PyTorch](https://img.shields.io/badge/PyTorch-v1.10%2B-EE4C2C?logo=pytorch)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-blue)](./LICENSE)
[![Status](https://img.shields.io/badge/Status-Active_Research-orange)]()

**AXON-Net** 是一种面向**非结构化道路语义分割**的轻量化 U-Net 风格网络。项目引入了：
- **CASAB**：通道-空间注意力模块  
- **LightPCT**：轻量化部分上下文变换器  
- **DPCF-TSE**：细结构增强模块  

旨在在低计算成本下兼顾**全局上下文建模**与**边界细节恢复**。

> ⚠️ **重要说明**
> - 本项目目前处于**活跃研究阶段**（正在推进低光照/夜间等极端场景的扩展研究）。
> - 本仓库仅开源**核心模型结构定义**与**损失函数实现**，供学术参考与复现思路借鉴。
> - 完整训练脚本、预训练权重及清洗后的数据集**暂不公开**。

---

## 📂 仓库内容

```text
├── src/
│   └── AXONNet.py                 # 网络模型定义（forward 输出 {"out": logits}）
├── train_utils/
│   └── dice_coefficient_loss.py   # 复合损失函数集合（WCE + Dice + Boundary）
├── my_dataset.py                  # 数据集加载器（适配 TP-Dataset 格式）
├── transforms.py                  # 数据增强与预处理变换
└── README.md                      # 项目说明

🖼️ 结果可视化
1）当前研究成果（IDD-UR）

下图展示了 AXON-Net 在 IDD-UR 非结构化道路数据集上的推理效果。模型在保持轻量化的同时，可有效捕捉复杂道路边界与细微结构。

<p align="center"> <img src="figures/current/inference.png" alt="AXON-Net 当前推理结果" width="80%"/> </p> <p align="center"><em>图 1：AXON-Net 在复杂非结构化场景下的分割表现</em></p>
2）后续研究进展：低光照/夜间场景

我们正在推进面向低光照、夜间等极端场景的鲁棒分割研究。以下为正在研发的新模型的部分测试示例（阶段性结果）。

<p align="center"> <img src="figures/future/01-dark.jpg" alt="低光照输入 01" width="45%"/> <img src="figures/future/01-inference.jpg" alt="推理结果 01" width="45%"/> </p> <p align="center"><em>样本 01：极端低光照输入（左）vs 新模型初步推理结果（右）</em></p> <p align="center"> <img src="figures/future/02-gt.png" alt="GT 02" width="45%"/> <img src="figures/future/02-mask.png" alt="Mask 02" width="45%"/> </p> <p align="center"><em>样本 02：真值 GT（左）vs 新模型预测 Mask（右）</em></p>

💾 数据集构建协议（TP-Dataset）

尽管数据集暂不分发，为保证实验过程透明性，公开论文中使用的数据集清洗与构建标准如下。

1）目录结构规范

数据加载器（my_dataset.py）预期的目录结构
<root>/
└── TP-Dataset/
    ├── JPEGImages/        # 输入图像（.jpg）
    ├── GroundTruth/       # 标签掩膜（.png，8-bit，0=Background，255=Road）
    └── Index/
        ├── train.txt      # 索引文件（无扩展名，如 000123）
        └── val.txt
2）清洗与处理流程（与论文一致）

IDD-UR（主实验）：提取 “Drivable Area” 类别，二值化处理（Road=255，BG=0），并人工剔除标签缺失及非典型场景样本。

ORFD-AV（跨域验证）：筛选适用于越野道路分割的样本，执行统一的二值化与格式检查，用于评估泛化能力。

🚀 后续研究规划（Roadmap）

 AXON-Net 模型验证（已完成）

 新一代架构设计：探索结合 Mamba / State Space Models（SSM）等线性复杂度技术，突破现有 CNN/Transformer 的性能瓶颈

 低光照/极端场景适配：针对夜间成像退化问题，设计全新的特征增强模块与域适应策略

 跨模态感知：研究融合热成像或深度信息的轻量化多模态方案
