# 对论文 QBB: Quantization with Binary Bases for LLMs 的复现和改进

## 1、对论文QBB算法的呈现
QBB (Quantization with Bit-wise Branches) 的核心思想是将一个高精度的 FP16 权重矩阵 $W$ 近似分解为 $K$ 个$1bit$ 分支的线性组合作为初始化， 即：
$$W \approx \hat{W} = \sum_{i=1}^{K} \alpha_i \odot B_i$$
### 1.1 初始化核心流程
* **第一层 (First Branch) 初始化**：
$$\mathbf{B}_1 = \text{sign}(\mathbf{W})$$
    $$\alpha_1 = \frac{1}{C_{out}} \|\mathbf{W}\|_{\ell_{1, \text{col}}}$$
    $\alpha_1 \in \mathbb{R}^{C_{out} \times 1}$
* **后续层 (Subsequent Branches) 递归迭代**：
$$\Delta_i = \mathbf{W} - \sum_{j=1}^{i-1} \alpha_j \odot \mathbf{B}_j$$
$$\mathbf{B}_i = \text{sign}(\Delta_i)$$
    $$\alpha_i = \frac{1}{C_{out}} \|\Delta_i\|_{\ell_{1, \text{col}}}$$
### 1.2 对初始化的进一步迭代优化
初始化之后的量化模型和原始模型相比误差很大，因而进一步引入权重和缩放因子的迭代优化（(Iterative weights and scales optimization)，论文的目标是最小化如下的目标函数：
$$\mathcal{L}_2 = \left\| \mathbf{W} - \sum_{i=1}^{N} \alpha_i \mathbf{B}_i \right\|_2^2$$
其中$\alpha_i$ 和$\mathbf{B}_i$ 均为可训练参数，但是直接对所有参数进行联合优化的**朴素方法**会导致训练的不稳定，论文提出类似于块坐标下降的迭代优化方法，具体来说，对于$N$个二值矩阵设计$N$个训练步骤，每次训练一个矩阵，其他的矩阵保持冻结，缩放因子在所有步骤中都会参与训练。
### 1.3 数据无关的全局二值化蒸馏 (Data-free Holistic Binarization)

为了在全局层面进一步减小量化误差，论文引入了知识蒸馏（Knowledge Distillation）技术，利用全精度教师模型的输出（Logits）指导学生（量化）模型的训练。

#### 1.3.1 损失函数定义 (Loss Functions)

蒸馏过程结合了模型输出层的输出对齐和隐藏层的特征对齐：

1. **输出层 MSE 损失 (Logit-level Loss)**：
   使用教师模型和学生模型输出 Logits 之间的均方误差：
   $$\mathcal{L}_{\text{MSE}} = \sum_{v=1}^{V} \sum_{i=1}^{n} \|p_{i,v}^T - p_{i,v}^S\|_2^2$$
   其中 $V$ 为词表大小，$n$ 为 Token 数量。

2. **特征层 MSE 损失 (Layer-wise Feature Loss)**：
   为了对齐不同 Transformer 模块内部的特征分布，在每个 Block 末端计算特征损失：
   $$\mathcal{L}_{\text{feat}} = \sum_{l=1}^{L} \sum_{i=1}^{n} \|f_{i,l}^T - f_{i,l}^S\|_2^2$$
   其中 $f_{i,l}$ 表示第 $l$ 个模块输出的特征张量。

3. **最终总损失 (Total Distillation Loss)**：
   $$\mathcal{L}_{\text{distill}} = s_1 \mathcal{L}_{\text{MSE}} + s_2 \mathcal{L}_{\text{feat}}$$
   通过平衡系数 $s_1$ 和 $s_2$ 对两项损失进行加权。

#### 1.3.2 核心训练策略

* **数据生成 (Data-free Strategy)**：不依赖外部数据集，而是利用全精度模型随机选择起始 Token 自动生成合成数据（Synthetic Data）进行训练。
* **高效校准 (Efficient Calibration)**：为提升效率，训练过程中通常保持二值权重 $\mathbf{B}_i$ 冻结，仅微调（Fine-tune）缩放向量 $\alpha_i$。
* **样本筛选 (Simple Filtering)**：通过对生成的序列进行评分，仅保留教师与学生模型差异最大的前 $K$ 个样本进行训练，以进一步缩短校准时间。
* **模块替换**：为了减少蒸馏时student模型与teacher模型之间的巨大差距，论文提出可以以一定的概率用student模型的部分模块替换teacher中的相应模块，使得teacher的模型由弱变强，保证蒸馏可以收敛。

## 2、我的工作
原始论文主要在 Llama-7B 等大规模模型上进行验证，受限于硬件，我探索了将本文的方法应用于小型模型**TinyLlama/TinyLlama-1.1B-Chat-v1.0**
### 2.1 1-bit ($N=4$) 实验中遇到的失效问题
* 初始重构误差的累积与放大。
  初始化阶段，每一层对残差 $\Delta_i$ 的近似仍然存在偏差。对于参数量仅为 1.1B 的小模型，这种微小的层级误差在经过数十层 Transformer Block 的叠加后被显著放大，导致模型初始状态下的语言建模能力（PPL）出现毁灭性打击。
* 梯度不稳定性与爆炸。
  在量化后的微调（Fine-tuning）阶段，观测到了剧烈的梯度爆炸现象。由于二值化函数 $\text{sign}(\cdot)$ 的非连续性，当模型尝试通过梯度下降修正较大的初始量化偏离时，更新步长极易陷入不稳定区域。在小模型的低冗余度参数空间内，这种不稳定的梯度回传导致权重迅速失效，最终引发 PPL 彻底失去意义（3000+）。
**总结：$1.1B$ 模型的数值冗余度较小，论文中提到的较为极端的量化方法对该模型的性能造成极大的影响。**
### 2.2 对2-bit和4-bit的尝试
我选择将原始模型中的二值矩阵改为2-bit或4bit并修改了初始化部分，其他的逻辑基本不变，在该模型的量化上收到很好的效果，例如使用2-bit量化的PPL达到了**24.8433**（全精度模型PPL为**21.6145**）