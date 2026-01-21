AXONNet
=======

AXONNet（Axial Context Network）是一种面向语义分割的轻量化 U-Net 风格网络，引入轻量化上下文建模与细结构增强模块，以兼顾效率与边界细节。本仓库仅提供模型结构与损失函数，供参考使用。

内容说明
--------

- `src/AXONNet.py`：网络结构定义，前向输出为 `{"out": logits}`。
- `train_utils/dice_coefficient_loss.py`：训练中使用的损失函数集合。
- `my_dataset.py`：数据集加载器（要求 TP-Dataset 目录结构）。
- `transforms.py`：数据增强与预处理变换。

说明
----

- 训练脚本、数据集与权重文件未公开。
- 项目仍处于进一步研究阶段（含极端场景研究）。

数据集结构（未包含）
-------------------

数据加载器期望的目录结构如下：

```
<root>/
  TP-Dataset/
    JPEGImages/        # 输入图像 (.jpg)
    GroundTruth/       # 标签掩膜 (.png, 0/255)
    Index/
      train.txt
      val.txt
```

索引文件中每一行是文件名（不含扩展名，例如 `000123`）。
