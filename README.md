# SELF-CS336

从零开始构建大语言模型（LLM）的学习实践项目，基于 CS336 课程内容，参考开源教程逐步实现 Transformer 模型的核心组件。

## 参考项目

本项目的学习过程主要参考了以下两个优秀的开源仓库：

- 📚 [easy-llm](https://github.com/ddkeeper/easy-llm) — 通俗易懂的 LLM 从零实现教程，包含详细的 Jupyter Notebook 讲解
- 🐋 [diy-llm](https://github.com/datawhalechina/diy-llm) — Datawhale 出品的大语言模型入门实践项目

## 项目结构

```
SELF-CS336/
└── 第一章_构建一个Transformer模型/
    ├── 1_transformer语言模型/        # Transformer 核心组件实现
    │   ├── 1.2_tensor基本运算.ipynb
    │   ├── 1.3_基础模块.ipynb
    │   ├── 1.4_前馈网络.ipynb
    │   ├── 1.4_多头注意力.ipynb
    │   ├── 1.4_层归一化&旋转编码.ipynb
    │   └── 1.5_transformer模型.ipynb
    ├── 2_模型训练/                    # 训练相关组件
    │   ├── 2.1_训练损失.ipynb
    │   ├── 2.2_优化器.ipynb
    │   └── 2.3_训练循环.ipynb
    ├── 3_实验/                        # 实验与应用
    │   ├── 3.0_训练资源估算.ipynb
    │   └── 3.3_文本生成.ipynb
    ├── 补充1_BPE分词器/               # BPE 分词器补充内容
    │   ├── 1.1_Unicode与UTF-8编码.ipynb
    │   └── 1.2_BPE训练.ipynb
    └── assignment1-basics/            # Assignment 1 作业实现
        ├── CS336_Assignment1_BPE.ipynb
        ├── CS336_Assignment1_Transformer.ipynb
        ├── model.py                   # Transformer 模型定义
        ├── train.py                   # 训练脚本
        ├── get_train_data.py          # 数据处理
        └── Assignment1_Ablations/     # 消融实验
```

## 学习内容

### 第一章：构建一个 Transformer 模型

#### 1. Transformer 语言模型
- **Tensor 基本运算** — PyTorch 张量操作基础
- **基础模块** — Embedding、Linear 等基础组件
- **多头注意力** — Multi-Head Attention 机制实现
- **前馈网络** — FFN / SwiGLU 激活函数
- **层归一化 & 旋转编码** — RMSNorm 与 RoPE 位置编码
- **Transformer 模型** — 完整的 Transformer 架构组装

#### 2. 模型训练
- **训练损失** — 交叉熵损失函数
- **优化器** — 自定义 AdamW 优化器，权重衰减
- **训练循环** — 完整训练流程，含验证与 checkpoint

#### 3. 实验
- **训练资源估算** — 计算参数量与显存占用
- **文本生成** — Top-K / Top-P 采样策略

#### 补充：BPE 分词器
- **Unicode 与 UTF-8 编码** — 字符编码基础
- **BPE 训练** — Byte-Pair Encoding 分词算法实现

### Assignment 1: Basics

基于 PyTorch 从零实现的 mini Transformer 语言模型，包含：

- 完整的 Transformer Decoder 架构
- RoPE 旋转位置编码
- SwiGLU 激活函数
- RMSNorm 归一化
- Flash Attention 支持
- 自定义 AdamW 优化器 + 余弦学习率调度
- 三个消融实验：移除归一化 / Post-Norm / SiLU 激活

详细使用说明请见 [assignment1-basics/README.md](./第一章_构建一个Transformer模型/assignment1-basics/README.md)。

## 环境要求

```bash
pip install torch numpy transformers tokenizers tqdm psutil matplotlib
```

- Python >= 3.8
- PyTorch >= 2.0（支持 Flash Attention）
- 支持 CUDA / MPS / CPU 训练

## 快速开始

1. **克隆仓库**
   ```bash
   git clone https://github.com/ZOUMDA28/SELF-CS336.git
   cd SELF-CS336
   ```

2. **安装依赖**
   ```bash
   pip install torch numpy transformers tokenizers tqdm psutil matplotlib
   ```

3. **浏览学习笔记**
   
   推荐按顺序阅读 `第一章_构建一个Transformer模型/` 下的 Jupyter Notebook，从基础组件到完整模型逐步学习。

4. **运行 Assignment 1**
   
   进入 `assignment1-basics/` 目录，按照 README 说明训练自己的 mini Transformer 模型。

## 模型特性

| 特性 | 说明 |
|------|------|
| 🔄 RoPE 位置编码 | 旋转位置嵌入，支持相对位置信息 |
| ⚡ SwiGLU 激活 | 改进的 FFN 层，提升模型表现 |
| 📏 RMSNorm | 高效的层归一化方案 |
| 🚀 Flash Attention | PyTorch 原生优化注意力实现 |
| 📉 AdamW + 余弦退火 | 带权重衰减和预热的学习率调度 |
| 💻 多设备支持 | CUDA / MPS / CPU 自动检测 |

## 进度

- ✅ 第一章：Transformer 模型基础组件
- ✅ Assignment 1: Basics（Transformer 实现与消融实验）
- ⏳ 持续更新中...

## License

MIT
