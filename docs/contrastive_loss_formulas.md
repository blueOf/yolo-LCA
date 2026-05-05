# 对比损失核心公式推导与权重分配

> **摘要**：本文档整理了基于 InfoNCE 的对比检测损失（`v8ContrastiveDetectionLoss`）的完整数学公式，涵盖总损失构成、对比损失推导、逐类权重分配、特征提取与聚合，以及动态权重调度策略。文档面向论文写作与工程复现，力求符号一致、公式完备。

---

## 目录

1. [问题背景与动机](#1-问题背景与动机)
2. [整体损失框架](#2-整体损失框架)
3. [检测子损失](#3-检测子损失)
   - 3.1 [边界框回归损失（Box Loss）](#31-边界框回归损失box-loss)
   - 3.2 [分类损失（Classification Loss）](#32-分类损失classification-loss)
   - 3.3 [分布焦点损失（DFL Loss）](#33-分布焦点损失dfl-loss)
4. [对比损失](#4-对比损失)
   - 4.1 [特征提取与 L2 归一化](#41-特征提取与-l2-归一化)
   - 4.2 [正负样本构造](#42-正负样本构造)
   - 4.3 [InfoNCE 损失公式](#43-infonce-损失公式)
5. [逐类权重分配](#5-逐类权重分配)
   - 5.1 [逆频率权重](#51-逆频率权重)
   - 5.2 [归一化与应用](#52-归一化与应用)
   - 5.3 [加权归一化分母](#53-加权归一化分母)
6. [动态权重调度](#6-动态权重调度)
7. [背景样本采样策略](#7-背景样本采样策略)
8. [完整算法伪代码](#8-完整算法伪代码)
9. [超参数汇总](#9-超参数汇总)
10. [设计决策说明](#10-设计决策说明)

---

## 1. 问题背景与动机

目标检测数据集（如 COCO）中普遍存在**类别长尾分布**问题：少数频繁类（如 `person`、`car`）的样本量可能比稀有类多出数十倍。在仅使用检测损失（分类 + 边界框回归）的训练范式下，模型特征空间对稀有类的区分能力不足，导致误检率偏高。

本方法在 YOLOv8 检测损失之上引入**基于 InfoNCE 的对比损失**，通过以下机制改善特征空间质量：

1. **同类聚合（Pull）**：来自同一目标的特征向量应相互靠近；
2. **异类分离（Push）**：来自不同目标的特征向量应相互远离；
3. **前背景分离**：前景目标特征应远离背景区域的特征；
4. **逆频率加权**：稀有类赋予更高的损失权重，缓解长尾效应。

---

## 2. 整体损失框架

设 $B$ 为批量大小，总训练损失定义为：

$$
\mathcal{L}_{\text{total}} = B \cdot \bigl(\lambda_{\text{box}} \cdot \mathcal{L}_{\text{box}} + \lambda_{\text{cls}} \cdot \mathcal{L}_{\text{cls}} + \lambda_{\text{dfl}} \cdot \mathcal{L}_{\text{dfl}} + \lambda_{\text{con}}(t) \cdot \mathcal{L}_{\text{con}}\bigr)
$$

其中：

| 符号 | 含义 | 默认值 |
|------|------|--------|
| $\lambda_{\text{box}}$ | 边界框损失权重（`hyp.box`） | `7.5` |
| $\lambda_{\text{cls}}$ | 分类损失权重（`hyp.cls`） | `0.5` |
| $\lambda_{\text{dfl}}$ | DFL 损失权重（`hyp.dfl`） | `1.5` |
| $\lambda_{\text{con}}(t)$ | 对比损失权重（可动态调整） | `0.1`–`1.5` |

> **注意**：$\mathcal{L}_{\text{box}}$、$\mathcal{L}_{\text{cls}}$、$\mathcal{L}_{\text{dfl}}$ 已在各自计算时除以分配分数之和 $\hat{S}$，因此乘以 $B$ 后得到批量尺度的损失。

---

## 3. 检测子损失

### 3.1 边界框回归损失（Box Loss）

采用**任务对齐分配器（TAL）** 分配正样本后，对前景锚点使用 Complete IoU（CIoU）衡量回归质量：

$$
\mathcal{L}_{\text{box}} = \frac{\displaystyle\sum_{i \in \mathcal{F}} \hat{s}_i \cdot (1 - \text{CIoU}(\hat{b}_i, b_i^*))}{\hat{S}}
$$

其中：
- $\mathcal{F}$：前景锚点集合（由 TAL 确定）；
- $\hat{b}_i$：第 $i$ 个锚点的预测边界框（通过 `dist2bbox` 解码）；
- $b_i^*$：对应的真实框；
- $\hat{s}_i$：TAL 分配的对齐分数（alignment score）；
- $\hat{S} = \max\!\left(\sum_{i \in \mathcal{F}} \hat{s}_i,\; 1\right)$：归一化因子。

**CIoU 定义**：

$$
\text{CIoU}(A, B) = \text{IoU}(A,B) - \frac{\rho^2(c_A, c_B)}{d^2} - \alpha v
$$

$$
v = \frac{4}{\pi^2}\!\left(\arctan\frac{w^*}{h^*} - \arctan\frac{w}{h}\right)^2, \quad
\alpha = \frac{v}{(1 - \text{IoU}) + v}
$$

其中 $\rho(\cdot,\cdot)$ 为中心点欧氏距离，$d$ 为最小包围框对角线长度。

---

### 3.2 分类损失（Classification Loss）

使用**二元交叉熵（BCE）**：

$$
\mathcal{L}_{\text{cls}} = \frac{1}{\hat{S}} \sum_{i=1}^{N_a} \sum_{c=1}^{C} \text{BCE}\!\left(\hat{p}_{i,c},\; s_{i,c}^*\right)
$$

其中：
- $N_a$：总锚点数；
- $C$：目标类别数；
- $\hat{p}_{i,c}$：模型对锚点 $i$ 类别 $c$ 的预测 logit（经过 sigmoid 变换后得到概率）；
- $s_{i,c}^*$：软目标分数（由 TAL 生成的对齐分数与 one-hot 类别的乘积）。

---

### 3.3 分布焦点损失（DFL Loss）

DFL 将边界框位置偏移建模为**离散分布**，预测每个锚点到边界的距离落在整数格点的概率分布。设真实距离 $t \in [0, R-1]$（$R$ 为 `reg_max`），取相邻整数格点 $t_l = \lfloor t \rfloor$，$t_r = t_l + 1$，则：

$$
\mathcal{L}_{\text{dfl}} = \frac{\displaystyle\sum_{i \in \mathcal{F}} \hat{s}_i \cdot \bigl[\,(t_r - t)\cdot\text{CE}(\hat{q}_i, t_l) + (t - t_l)\cdot\text{CE}(\hat{q}_i, t_r)\,\bigr]}{\hat{S}}
$$

其中 $\hat{q}_i \in \mathbb{R}^R$ 为预测的位置分布，$\text{CE}$ 为交叉熵。

---

## 4. 对比损失

对比损失在**颈部（Neck）中间特征图**上计算，支持多尺度（P3/P4/P5，步长分别为 8/16/32）。

### 4.1 特征提取与 L2 归一化

对于特征图 $\mathbf{F}^{(l)} \in \mathbb{R}^{B \times C \times H_l \times W_l}$（第 $l$ 尺度），首先将其展平为：

$$
\mathbf{F}^{(l)} \in \mathbb{R}^{B \times (H_l W_l) \times C}
$$

然后对每个空间位置的特征向量做 L2 归一化：

$$
\tilde{\mathbf{f}}_{p} = \frac{\mathbf{f}_{p}}{\|\mathbf{f}_{p}\|_2}, \quad \forall p \in \{1,\ldots, H_l W_l\}
$$

归一化后，任意两个特征向量的内积等价于其**余弦相似度**：

$$
\text{sim}(\tilde{\mathbf{f}}_p, \tilde{\mathbf{f}}_q) = \tilde{\mathbf{f}}_p^\top \tilde{\mathbf{f}}_q = \cos\theta_{pq} \in [-1, 1]
$$

**锚点到特征图坐标的映射**：对于第 $l$ 尺度（步长 $s_l$），锚点坐标 $(x, y)$ 映射到特征图像素 $(u, v)$ 为：

$$
u = \left\lfloor \frac{x}{s_l} \right\rfloor \quad (\text{clamp to } [0, W_l - 1]), \qquad
v = \left\lfloor \frac{y}{s_l} \right\rfloor \quad (\text{clamp to } [0, H_l - 1])
$$

> 使用下取整（floor）而非四舍五入，以确保映射点始终位于边界框内部，避免将边界框外的背景像素误判为前景。

**目标特征聚合**：对真实框 $g_i$ 所分配的所有锚点集合 $\mathcal{A}_i$，通过**均值池化**得到该目标的代表特征：

$$
\mathbf{z}_i = \frac{1}{|\mathcal{A}_i|} \sum_{k \in \mathcal{A}_i} \tilde{\mathbf{f}}_{(u_k, v_k)}
$$

---

### 4.2 正负样本构造

对于图像中的第 $i$ 个目标：

- **正样本**（Positive）：目标自身的聚合特征 $\mathbf{z}_i$；
- **负样本**（Negatives）：同图其余所有目标的特征 $\{\mathbf{z}_j\}_{j \neq i}$ 以及背景采样特征 $\{\mathbf{b}_k\}$。

负样本集合为：

$$
\mathcal{N}_i = \{\mathbf{z}_j \mid j \neq i,\; j \in \{1,\ldots,M\}\} \cup \{\mathbf{b}_k \mid k=1,\ldots,K\}
$$

其中 $M$ 为当前图像的真实目标数，$K$ 为背景采样数量（见 [§7](#7-背景样本采样策略)）。

---

### 4.3 InfoNCE 损失公式

对目标 $i$，在温度 $\tau$ 下，InfoNCE 损失定义为：

$$
\mathcal{L}_{\text{NCE}}^{(i)} = -\log \frac{\exp\!\left(\dfrac{\text{sim}(\mathbf{z}_i, \mathbf{z}_i)}{\tau}\right)}{\exp\!\left(\dfrac{\text{sim}(\mathbf{z}_i, \mathbf{z}_i)}{\tau}\right) + \displaystyle\sum_{\mathbf{n} \in \mathcal{N}_i} \exp\!\left(\dfrac{\text{sim}(\mathbf{z}_i, \mathbf{n})}{\tau}\right)}
$$

由于特征已经 L2 归一化，$\text{sim}(\mathbf{z}_i, \mathbf{z}_i) = 1$，上式可简化为：

$$
\mathcal{L}_{\text{NCE}}^{(i)} = -\log \frac{e^{1/\tau}}{e^{1/\tau} + \displaystyle\sum_{\mathbf{n} \in \mathcal{N}_i} e^{\text{sim}(\mathbf{z}_i, \mathbf{n})/\tau}}
$$

在实现中，等价地构造 logit 向量并使用交叉熵：

$$
\text{logits}_i = \left[\frac{\text{sim}(\mathbf{z}_i, \mathbf{z}_i)}{\tau},\; \frac{\text{sim}(\mathbf{z}_i, \mathbf{n}_1)}{\tau},\; \ldots,\; \frac{\text{sim}(\mathbf{z}_i, \mathbf{n}_{|\mathcal{N}_i|})}{\tau}\right]
$$

$$
\mathcal{L}_{\text{NCE}}^{(i)} = \text{CrossEntropy}\!\left(\text{logits}_i,\; y=0\right)
$$

其中 $y=0$ 表示正样本位于第 0 号位置（即 logit 向量的第一个元素）。

温度参数 $\tau = 0.07$ 控制 softmax 的尖锐程度：较小的 $\tau$ 使分布更尖锐，对困难负样本的惩罚更强。

---

## 5. 逐类权重分配

### 5.1 逆频率权重

为缓解长尾分布问题，对每个类别 $c$（$c \in \{0, 1, \ldots, C-1\}$）定义原始逆频率权重：

$$
\tilde{w}_c = \frac{N}{c_c}
$$

其中：
- $N = \sum_{c=0}^{C-1} c_c$：当前批次中所有真实目标的总数；
- $c_c$：当前批次中类别 $c$ 的目标实例数（若 $c_c = 0$ 则置为 1，避免除零）。

> **与经典逆频率权重的区别**：经典公式为 $\tilde{w}_c = N / (C \cdot c_c)$，分母多一个类别数 $C$ 的因子。当前实现省略了 $C$，以放大不同类别之间的权重差异，避免在 $C = 80$ 时权重差异被稀释至 $1/80$。

### 5.2 归一化与应用

为保持损失量纲稳定，对权重归一化使其均值为 1：

$$
w_c = \frac{\tilde{w}_c}{\bar{w}}, \quad \bar{w} = \frac{1}{C} \sum_{c=0}^{C-1} \tilde{w}_c
$$

对于目标 $i$（类别为 $y_i$），其对应的**逐目标权重**为：

$$
w_i = w_{y_i}
$$

### 5.3 加权归一化分母

单张图像上对所有目标（跨所有尺度）的加权对比损失为：

$$
\mathcal{L}_{\text{con}} = \frac{\displaystyle\sum_l \sum_{i=1}^{M^{(l)}} w_i \cdot \mathcal{L}_{\text{NCE}}^{(i,l)}}{\displaystyle\sum_l \sum_{i=1}^{M^{(l)}} w_i}
$$

其中：
- $l$ 枚举所有特征尺度；
- $M^{(l)}$ 为在第 $l$ 尺度上有效（至少 2 个目标）的图像中的目标数；
- 分母采用**加权目标数**（而非原始目标数），保证不同类别的梯度贡献均衡。

---

## 6. 动态权重调度

对比损失权重 $\lambda_{\text{con}}(t)$ 可在训练过程中**线性插值**动态调整：

$$
\lambda_{\text{con}}(t) = \lambda_{\text{init}} + \left(\lambda_{\text{final}} - \lambda_{\text{init}}\right) \cdot \min\!\left(\frac{t}{T - 1},\; 1\right)
$$

其中：
- $\lambda_{\text{init}}$：训练开始时的权重（`contrast_weight_init`）；
- $\lambda_{\text{final}}$：训练结束时的权重（`contrast_weight_final`）；
- $t$：当前 epoch（从 0 开始，每个 epoch 结束调用一次 `update()`）；
- $T$：总训练 epoch 数（`hyp.epochs`）。

**特殊情况**：若 $\lambda_{\text{init}} = \lambda_{\text{final}}$，则使用静态权重 $\lambda_{\text{con}} \equiv \lambda_{\text{base}}$（向前兼容）。

**调度曲线示例**（$T=300$，$\lambda_{\text{init}}=0.1$，$\lambda_{\text{final}}=1.5$）：

| Epoch | $\lambda_{\text{con}}(t)$ |
|-------|--------------------------|
| 0     | 0.100 |
| 60    | 0.388 |
| 120   | 0.676 |
| 180   | 0.964 |
| 240   | 1.252 |
| 299   | 1.495 |

---

## 7. 背景样本采样策略

背景特征为**未被 TAL 分配到任何真实目标的锚点**所对应的特征向量。由于背景锚点数量远多于前景，采用**随机下采样**控制负样本数量：

$$
K = \min\!\left(N_{\text{bg}},\; 2 M\right)
$$

其中：
- $N_{\text{bg}}$：当前图像背景锚点总数；
- $M$：当前图像的真实目标数；
- $K$：实际采样的背景特征数。

背景锚点同样映射到特征图坐标后提取特征，但不参与逆频率加权（不属于任何类别）。

---

## 8. 完整算法伪代码

```
算法：v8ContrastiveDetectionLoss.compute_contrastive_loss

输入：
  contrast_feats  - 各尺度特征图列表 [{B×C×H_l×W_l}]
  fg_mask         - 前景掩码 (B, N_a)
  target_gt_idx   - 锚点对应真实框索引 (B, N_a)
  anchor_points   - 锚点坐标 (N_a, 2)
  gt_bboxes       - 真实框 (N_gt_total, 4)，xyxy 格式
  batch_idx       - 每个真实框的批次索引 (N_gt_total,)
  gt_classes      - 每个真实框的类别标签 (N_gt_total,)

输出：
  L_con           - 标量对比损失

步骤：
1. 若 gt_classes 为空，返回 0；
2. 计算逐类权重 w_c = (N / c_c) / mean(N / c_c)  for each class c；
3. total_loss ← 0，weighted_count ← 0；
4. for each 尺度 l:
   a. 展平特征图：F ← (B, H_l*W_l, C)
   b. L2 归一化：F ← F / ||F||_2
   c. for each 图像 b:
      i.  获取前景掩码 fg_b，真实框索引 idx_b
      ii. for each 真实框 g_i（由 idx_b 中的唯一值确定）:
           - 找到所有被分配到 g_i 的锚点
           - 映射到特征图坐标 (u, v) = (floor(x/s_l), floor(y/s_l))
           - 聚合特征：z_i = mean(F_b[u,v])
           - 记录类别 y_i
      iii. 若有效目标数 < 2，跳过该图像
      iv. 随机采样 K = min(N_bg, 2M) 个背景特征
      v.  for each 目标 i:
           - 构造 logits = [sim(z_i,z_i)/τ] + [sim(z_i,z_j)/τ for j≠i] + [sim(z_i,b_k)/τ]
           - L_i = CrossEntropy(logits, y=0)
           - w_i = w_{y_i}
           - total_loss += w_i * L_i
           - weighted_count += w_i
5. 返回 total_loss / max(weighted_count, 1)
```

---

## 9. 超参数汇总

| 超参数 | 符号 | 默认值 | 说明 |
|--------|------|--------|------|
| `contrast_temp` | $\tau$ | `0.07` | InfoNCE 温度参数 |
| `contrast_weight` | $\lambda_{\text{base}}$ | `0.1` | 静态对比损失权重 |
| `contrast_weight_init` | $\lambda_{\text{init}}$ | `0.1` | 动态调度初始权重 |
| `contrast_weight_final` | $\lambda_{\text{final}}$ | `0.1` | 动态调度终止权重 |
| `box` | $\lambda_{\text{box}}$ | `7.5` | 边界框损失权重 |
| `cls` | $\lambda_{\text{cls}}$ | `0.5` | 分类损失权重 |
| `dfl` | $\lambda_{\text{dfl}}$ | `1.5` | DFL 损失权重 |
| TAL `topk` | $k$ | `10` | 任务对齐分配每个真实框最多分配的锚点数 |
| BG 采样倍率 | — | `2` | 背景采样数 = $\min(N_{\text{bg}},\, 2M)$ |

---

## 10. 设计决策说明

### 10.1 自相似性作为正样本

在本实现中，正样本对为目标特征与**自身**的比较（$\text{sim}(\mathbf{z}_i, \mathbf{z}_i) = 1$）。这与 SimCLR、MoCo 等方法不同——后者通过数据增强构造同一目标的两个视图。本方案无需额外增强，直接利用 TAL 分配到同一目标的多锚点特征均值，作为该目标的单一代表向量，从而将**区分目标与背景/其他目标**作为优化目标。

### 10.2 权重公式省略类别数因子

经典逆频率权重 $N/(C \cdot c_c)$ 与 $N/c_c$ 在经过均值归一化后，数值完全等价（归一化会消除常数因子 $C$）。因此，现行公式 $\tilde{w}_c = N / c_c$ 与归一化步骤组合，在数学上与经典公式一致。

形式化验证：

$$
w_c = \frac{N/c_c}{\frac{1}{C}\sum_{c'} N/c_{c'}} = \frac{N/(C \cdot c_c)}{\frac{1}{C^2}\sum_{c'} N/c_{c'}} \cdot C = \ldots
$$

两者均经归一化到 $\bar{w}=1$，最终效果相同。省略 $C$ 的好处是代码更简洁，且避免在类别数不同的数据集（非 COCO-80）上引入额外的缩放混淆。

### 10.3 坐标取整方式

锚点到特征图的坐标转换使用**下取整（floor）**：

$$
u = \left\lfloor x / s_l \right\rfloor
$$

相比四舍五入（round）或上取整（ceil），floor 保证映射点始终在 bbox 内部（对于 x1 的映射），避免将边界框恰好在分辨率边界附近时的背景像素误采入前景特征。

### 10.4 多尺度特征的作用

对 P3（stride=8，感受野小，细节丰富）、P4（stride=16，中等语义）、P5（stride=32，高语义、大感受野）三个尺度分别计算对比损失，使模型在不同分辨率层次均能学习判别性特征。每个尺度上的损失贡献在最终加权归一化中统一处理。

---

*代码实现路径：`ultralytics/utils/loss.py`，类 `v8ContrastiveDetectionLoss`（第 1134–1467 行）。*
