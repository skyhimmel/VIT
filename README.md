# mnist-vit

vision transformer on mnist dataset

基于mnist手写数字集训练的vision transformer模型，用作学习用途，只能预测0~9

## 模型

1x28x28图片输入，对每个1x4x4区域做conv转成16宽向量，整个图片变为7x7=49个16宽patch向量.

* 所有patch向量做linear转patch embedding
* cls embeeding可学习，直接拼到patch embedding序列头部

# 25.8.5更新 Swin Transformer for MNIST Classification

包含了基于Swin Transformer的手写数字识别实现，与原有的ViT实现相对应。

## 文件说明

### 核心文件
- `swin.py`: Swin Transformer的完整实现
  - 包含窗口注意力机制(Window Attention)
  - 移位窗口机制(Shifted Window)
  - 层次化特征表示
  - 相对位置编码
- `train_swin.py`: Swin Transformer训练脚本
- `inference_swin.py`: Swin Transformer推理脚本