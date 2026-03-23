# 对论文 QBB: Quantization with Binary Bases for LLMs 的复现和改进

## 1、对论文 QBB 算法的呈现

QBB (Quantization with Bit-wise Branches) 的核心思想是将一个高精度的 FP16 权重矩阵 $\mathbf{W}$ 近似分解为 $K$ 个 1-bit 分支的线性组合作为初始化，即：

$$
\mathbf{W} \approx \hat{\mathbf{W}} = \sum_{i=1}^{K} \alpha_i \odot \mathbf{B}_i
$$

### 1.1 初始化核心流程

* **第一层 (First Branch) 初始化**：

$$
\mathbf{B}_1 = \text{sign}(\mathbf{W})
$$

$$
\alpha_1 = \frac{1}{C_{out}} \|\mathbf{W}\|_{\ell_{1, \text{col}}}
$$

其中 $\alpha_1 \in \mathbb{R}^{C_{out} \times 1}$。

* **后续层 (Subsequent Branches) 递归迭代**：

$$
\Delta_i = \mathbf{W} - \sum_{j=1}^{i-1} \alpha_j \odot \mathbf{B}_j
$$

$$
\mathbf{B}_i = \text{sign}(\Delta_i)
$$

$$
\alpha_i = \frac{1}{C_{out}} \|\Delta_i\|_{\ell_{1, \text{col}}}
$$

### 1.2 对初始化的进一步迭代优化

初始化之后的量化模型和原始模型相比误差很大，因而进一步引入权重和缩放因子的迭代优化（Iterative weights and scales optimization）。论文的目标是最小化如下目标函数：

$$
\mathcal{L}_2 = \left\| \mathbf{W} - \sum_{i=1}^{N} \alpha_i \mathbf{B}_i \right\|_2^2
$$

其中 $\alpha_i$ 和 $\mathbf{B}_i$ 均为可训练参数。论文提出类似于块坐标下降的迭代优化方法：对于 $N$ 个二值矩阵设计 $N$ 个训练步骤，每次训练一个矩阵，其他矩阵保持冻结，缩放因子在所有步骤中都会参与训练。

### 1.3 数据无关的全局二值化蒸馏 (Data-free Holistic Binarization)

利用全精度教师模型的输出（Logits）指导学生（量化）模型的训练。

#### 1.3.1 损失函数定义 (Loss Functions)

1. **输出层 MSE 损失 (Logit-level Loss)**：

$$
\mathcal{L}_{\text{MSE}} = \sum_{v=1}^{V} \sum_{i=1}^{n} \|p_{i,v}^T - p_{i,v}^S\|_2^2
$$

2. **特征层 MSE 损失 (Layer-wise Feature Loss)**：

$$
\mathcal{L}_{\text{feat}} = \sum_{l=1}^{L} \sum_{i=1}^{n} \|f_{i,l}^T - f_{i,l}^S\|_2^2
$$

3. **最终总损失 (Total Distillation Loss)**：

$$
\mathcal{L}_{\text{distill}} = s_1 \mathcal{L}_{\text{MSE}} + s_2 \mathcal{L}_{\text{feat}}
$$

#### 1.3.2 核心训练策略

* **数据生成**：利用全精度模型随机选择起始 Token 自动生成合成数据。
* **高效校准**：训练过程中通常保持二值权重 $\mathbf{B}_i$ 冻结，仅微调缩放向量 $\alpha_i$。
* **样本筛选**：仅保留教师与学生模型差异最大的前 $K$ 个样本。
* **模块替换**：以一定概率用 Student 模块替换 Teacher 相应模块，确保蒸馏收敛。

## 2、我的工作

本项目探索了将该方法应用于小型模型 **TinyLlama-1.1B-Chat-v1.0**。

### 2.1 基于 $N=4$ 矩阵叠加的失效分析

在尝试使用 $N=4$（4个矩阵叠加）进行 1-bit 量化尝试时，观测到了严重的失效：

* **初始重构误差的累积与放大**：由于 1.1B 小模型的**数值冗余度较小**，初始化的微小偏差在数十层 Transformer Block 的叠加后被显著放大，导致初始状态 PPL 出现毁灭性打击。
* **梯度不稳定性与爆炸**：二值化函数 $\text{sign}(\cdot)$ 的非连续性导致模型在修正误差时，步长易陷入不稳定区域。在低冗余度参数空间内，这种梯度回传引发了严重的梯度爆炸，最终 PPL 飙升至 3000+。

**总结**：$1.1B$ 模型数值冗余度低，极端量化策略对其性能影响极大。

### 2.2 对 2-bit 与 4-bit 的成功尝试

通过将二值矩阵基底改为支持更高位宽的 2-bit 或 4-bit，并优化初始化逻辑，取得了显著效果：

| 模型配置 | WikiText-2 PPL | 状态 |
| :--- | :--- | :--- |
| **FP16 基准** | **21.6145** | Baseline |
| **2-bit 量化 (本项目改进)** | **24.8433** | **成功收敛** |
| **1-bit 量化** | 3000+ | 梯度爆炸 |

利用 2-bit 量化，模型在极低位宽下成功保留了优秀的语言生成能力。