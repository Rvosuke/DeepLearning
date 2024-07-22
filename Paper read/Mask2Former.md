# Mask2Former

图像分割，这个研究像素分组问题的领域，在近些年得到了飞速的发展。各种算法和架构层出不穷，大致可以分为三个子任务：

1. 场景分割（Panoptic Segmentation）
2. 实例分割（Instance Segmentation）
3. 语义分割（Semantic Segmentation）

许多框架都是专门针对某一个子任务进行设计的，例如基于FCN的模型就专门针对实例分割。那么，有没有一种模型能够同时处理这三种分割任务呢？

DETR等通用架构（Universal Architectural）在不修改损失、架构、训练过程的情况下，确实能够应用于这三种不同的任务。然而，一方面，DETR等需要对不同的数据集、任务进行重新训练；另一方面，DETR等的性能并未达到面面俱到，无法在三种任务中都取得优异的结果。

这时，Mask2Former应运而生。它也是一种图像分割的通用架构，其元架构与MaskFormer保持一致，主要采用像素解码器（Pixel Decoder）和Transformer解码器（Transformer Decoder）进行综合解码。与之前不同的是，Mask2Former在Transformer解码器中添加了掩膜注意力（Masked Attention Operator）。

## 交叉注意力（Cross-Attention）

掩膜注意力是交叉注意力的一种变体。我们先简单说一下交叉注意力。标准的交叉注意力可以通过下式计算：
$$
X_l = softmax(Q_lK_l^T)V_l+X_{l−1}
$$

## 掩膜注意力（Masked Attention）

随着研究的深入，现在的工作越来越趋近多模态，在进行多种类任务使用，需要输入文本（content features）来确认具体进行的任务。

然而，一般来说，引入这种全局的文本引导，会使得基于Transformer架构的模型收敛的速度很慢。 这里，作者假设自注意力机制能够自行发掘文本特征。掩膜注意力通过下式计算：
$$
X_l = softmax{(M_{l-1}+Q_LK^T_l)V_l+X_{l-1}}
$$
其中 $M_{l-1}$ 是上一层的mask预测。

