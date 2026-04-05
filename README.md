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
\alpha_1 = \frac{1}{C_{out}} \Vert \mathbf{W} \Vert_{\ell_{1, \text{col}}}
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
\alpha_i = \frac{1}{C_{out}} \Vert \Delta_i \Vert_{\ell_{1, \text{col}}}
$$

### 1.2 对初始化的进一步迭代优化

初始化之后的量化模型和原始模型相比误差很大，因而进一步引入权重和缩放因子的迭代优化（Iterative weights and scales optimization）。论文的目标是最小化如下目标函数：

$$
\mathcal{L}_2 = \left\Vert \mathbf{W} - \sum_{i=1}^{N} \alpha_i \mathbf{B}_i \right\Vert_2^2
$$

其中 $\alpha_i$ 和 $\mathbf{B}_i$ 均为可训练参数。论文提出类似于块坐标下降的迭代优化方法：对于 $N$ 个二值矩阵设计 $N$ 个训练步骤，每次训练一个矩阵，其他矩阵保持冻结，缩放因子在所有步骤中都会参与训练。

### 1.3 数据无关的全局二值化蒸馏 (Data-free Holistic Binarization)

为了在全局层面进一步减小量化误差，论文引入了知识蒸馏（Knowledge Distillation）技术，利用全精度教师模型的输出（Logits）指导学生（量化）模型的训练。

#### 1.3.1 损失函数定义 (Loss Functions)

蒸馏过程结合了模型输出层的输出对齐和隐藏层的特征对齐：

1. **输出层 MSE 损失 (Logit-level Loss)**：
   使用教师模型和学生模型输出 Logits 之间的均方误差：
   
$$
\mathcal{L}_{\text{MSE}} = \sum_{v=1}^{V} \sum_{i=1}^{n} \Vert p_{i,v}^T - p_{i,v}^S \Vert_2^2
$$

2. **特征层 MSE 损失 (Layer-wise Feature Loss)**：
   为了对齐不同 Transformer 模块内部的特征分布，在每个 Block 末端计算特征损失：
   
$$
\mathcal{L}_{\text{feat}} = \sum_{l=1}^{L} \sum_{i=1}^{n} \Vert f_{i,l}^T - f_{i,l}^S \Vert_2^2
$$

3. **最终总损失 (Total Distillation Loss)**：

$$
\mathcal{L}_{\text{distill}} = s_1 \mathcal{L}_{\text{MSE}} + s_2 \mathcal{L}_{\text{feat}}
$$

#### 1.3.2 核心训练策略（核心解释）

* **数据生成 (Data-free Strategy)**：不依赖外部数据集，而是利用全精度模型随机选择起始 Token 自动生成合成数据（Synthetic Data）进行训练。
* **高效校准 (Efficient Calibration)**：为提升效率，训练过程中通常保持二值权重 $\mathbf{B}_i$ 冻结，仅微调（Fine-tune）缩放向量 $\alpha_i$。
* **样本筛选 (Simple Filtering)**：通过对生成的序列进行评分，仅保留教师与学生模型差异最大的前 $K$ 个样本进行训练，以进一步缩短校准时间。
* **模块替换**：为了减少蒸馏时 Student 模型与 Teacher 模型之间的巨大差距，论文提出可以以一定的概率用 Student 模型的部分模块替换 Teacher 中的相应模块，使得 Teacher 的模型由强变弱，保证蒸馏可以收敛。

## 2、我的工作

由于原始论文没有公开代码，受限于硬件，本项目探索了将该方法应用于小型模型 **TinyLlama-1.1B-Chat-v1.0**。

### 2.1 基于 1-bit 矩阵叠加的失效分析

在尝试使用 $N=4$（4个矩阵叠加）进行 1-bit 量化尝试时，观测到了严重的失效：

* **初始重构误差的累积与放大**：由于 1.1B 小模型的**数值冗余度较小**，初始化的微小偏差在数十层 Transformer Block 的叠加后被显著放大，导致模型初始状态下的语言建模能力（PPL）出现毁灭性打击。
* **梯度不稳定性与爆炸**：由于1-bit量化过于极端导致量化模型和全精度模型的误差较大，蒸馏过程易发生梯度爆炸，同时，由于初始化的误差过大，蒸馏操作很难降低量化模型PPL至正常水平（**$PPL=3000+$**)。

**总结：$1.1B$ 模型数值冗余度低，极端量化策略对其性能影响极大。**

### 2.2 对 2-bit 与 4-bit 的成功尝试

通过将二值矩阵基底改为支持更高位宽的 2-bit 或 4-bit，并优化初始化逻辑，经过迭代优化之后，2-bit量化模型的PPL达到90左右，4-bit量化模型的PPL达到60左右（全精度模型的PPL为21.6145），但是经过蒸馏之后发现PPL上升到3000+，经过分析推测是因为原论文中全精度模型的生成数据不能支持量化模型的进一步蒸馏，因而引入真实数据集**wikitext-2-raw-v1**，并进一步去掉teacher模型与student模型的模块替换操作，量化取得很好的效果。
代码运行结果如下：
| 模型配置 | WikiText-2 PPL | 状态 |
| :--- | :--- | :--- |
| **FP16 基准** | **21.6145** | Baseline |
| **2-bit 量化($N=4$)** | **24.8433** | **成功收敛** |
| **4-bit 量化($N=3$)** | **22.7354** | **成功收敛** |
| **1-bit 量化** | 3000+ | 梯度爆炸 |
## 3、消融实验结果
* 4-bit 基底消融实验 (k=3 Layers)

| 实验组别 | 初始化 (Init) | 迭代优化 (UPD) | 知识蒸馏 (Distill) | PPL (Wikitext-2) |
| :--- | :---: | :---: | :---: | :--- |
| **Teacher (FP16)** | - | - | - | **21.6145** |
| **QBB 4-bit (Full)** | **Yes** | **Yes** | **Yes** | **22.7354** |
| **消融-无优化** | **Yes** | **No** | **Yes** | **37.4682** |
| **消融-不蒸馏** | **Yes** | **Yes** | **No** | **60.0711** |
| **消融-随机初始化** | **No** | **Yes** | **Yes** | **NAN** |

* 2-bit 基底消融实验 (k=4 Layers)

| 实验组别 | 初始化 (Init) | 迭代优化 (UPD) | 知识蒸馏 (Distill) | PPL (Wikitext-2) |
| :--- | :---: | :---: | :---: | :--- |
| **QBB 2-bit (Full)** | **Yes** | **Yes** | **Yes** | **24.8433** |
| **消融-无优化** | **Yes** | **No** | **Yes** | **29.6147** |
| **消融-不蒸馏** | **Yes** | **Yes** | **No** | **91.7793** |
| **消融-随机初始化** | **No** | **Yes** | **Yes** | **NAN** |
## 4、分支说明
main：实现4-bit模型量化

qbb_2bit：修改4-bit模型的初始化部分，实现2-bit量化