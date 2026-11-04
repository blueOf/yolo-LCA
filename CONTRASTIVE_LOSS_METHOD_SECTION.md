# 3.X Feature-Level Contrastive Learning for Object Detection

## 3.X.1 Motivation and Overview

To enhance the discriminative capability of learned representations and address class imbalance challenges inherent in object detection datasets, we introduce a feature-level contrastive learning mechanism that operates on intermediate feature maps. Unlike conventional detection frameworks that solely optimize localization and classification objectives, our approach incorporates a contrastive loss component that explicitly encourages intra-class feature compactness while promoting inter-class separation. This auxiliary supervision signal operates in tandem with standard detection losses, yielding feature representations that exhibit superior discriminative properties, particularly for underrepresented object categories.

## 3.X.2 Contrastive Loss Formulation

Our contrastive learning framework adopts the InfoNCE (Noise Contrastive Estimation) objective [1], which has demonstrated remarkable efficacy in representation learning tasks. For a given anchor feature vector **a** corresponding to a ground-truth object instance, the contrastive loss is formulated as:

$$\mathcal{L}_{\text{InfoNCE}}(\mathbf{a}) = -\log \frac{\exp(\text{sim}(\mathbf{a}, \mathbf{a}^+) / \tau)}{\exp(\text{sim}(\mathbf{a}, \mathbf{a}^+) / \tau) + \sum_{i} \exp(\text{sim}(\mathbf{a}, \mathbf{a}_i^-) / \tau)}$$

where **a**⁺ denotes the positive sample (the anchor itself), {**a**ᵢ⁻} represents the set of negative samples comprising features from distinct object instances and background regions, and τ is a temperature hyperparameter that modulates the concentration of the distribution. The similarity function sim(·,·) is computed as the inner product of L2-normalized feature vectors:

$$\text{sim}(\mathbf{x}, \mathbf{y}) = \langle \hat{\mathbf{x}}, \hat{\mathbf{y}} \rangle = \frac{\mathbf{x}^T \mathbf{y}}{\|\mathbf{x}\|_2 \|\mathbf{y}\|_2}$$

Following empirical validation, we set τ = 0.07, which provides an effective balance between gradient magnitude and numerical stability during optimization.

## 3.X.3 Class-Aware Instance Weighting

Real-world object detection datasets frequently exhibit severe class imbalance, with certain categories containing orders of magnitude more instances than others. To mitigate the dominance of frequent classes in the contrastive loss optimization, we introduce an inverse frequency weighting scheme that adaptively rebalances the contribution of each object instance.

For a dataset containing $n_c$ object categories, let $n_c^{(k)}$ denote the number of instances belonging to class $k$ in the current training batch, and let $N = \sum_{k=1}^{n_c} n_c^{(k)}$ represent the total instance count. The raw class weight is computed as:

$$w_{\text{raw}}^{(k)} = \frac{N}{n_c \cdot n_c^{(k)}}$$

To ensure stable loss magnitudes across batches with varying class distributions, we normalize these weights such that their mean equals unity:

$$w^{(k)} = \frac{w_{\text{raw}}^{(k)}}{\frac{1}{n_c}\sum_{j=1}^{n_c} w_{\text{raw}}^{(j)}}$$

This normalization guarantees that $\frac{1}{n_c}\sum_{k=1}^{n_c} w^{(k)} = 1$, thereby preserving the overall loss scale while amplifying the influence of rare classes and attenuating that of dominant categories.

## 3.X.4 Feature Extraction and Aggregation

Our contrastive learning mechanism operates on intermediate feature maps extracted from the backbone network, specifically targeting layers preceding the detection head. For each ground-truth bounding box, we aggregate features from all assigned anchor points through spatial average pooling, yielding a compact representation that captures the object's semantic characteristics.

Given a feature map $\mathbf{F} \in \mathbb{R}^{B \times C \times H \times W}$ and an anchor point located at image coordinates $(x_a, y_a)$, we map this position to feature map coordinates via:

$$x_f = \left\lfloor \frac{x_a}{s} \right\rfloor, \quad y_f = \left\lfloor \frac{y_a}{s} \right\rfloor$$

where $s$ denotes the stride of the feature map relative to the input image, and $\lfloor \cdot \rfloor$ represents the floor operation. The use of floor rounding, rather than ceiling or nearest-neighbor rounding, is critical to ensure that sampled features originate strictly from within the object's bounding box, thereby preventing contamination from background regions.

For each ground-truth object $i$ with $M_i$ assigned anchors, the aggregated feature representation is obtained through:

$$\mathbf{f}_i = \frac{1}{M_i} \sum_{j=1}^{M_i} \hat{\mathbf{F}}[\mathbf{p}_j]$$

where $\mathbf{p}_j$ denotes the feature map coordinates of the $j$-th assigned anchor, and $\hat{\mathbf{F}}$ represents the L2-normalized feature map. This aggregation strategy provides robustness to localization noise while maintaining computational efficiency.

## 3.X.5 Negative Sample Construction

The construction of an effective negative sample set is crucial for contrastive learning efficacy. Our approach incorporates two categories of negative samples: (1) features from distinct ground-truth objects within the same batch, which provide inter-class discrimination signals, and (2) background features sampled from anchor points not assigned to any ground-truth box, which enhance foreground-background separation.

To maintain computational tractability and balanced positive-negative ratios, we randomly sample background features such that their quantity does not exceed twice the number of ground-truth objects in each batch. This sampling strategy ensures sufficient negative diversity while preventing computational overhead and potential dominance of background samples in the loss computation.

## 3.X.6 Weighted Loss Integration

The class-weighted contrastive loss for a training batch is computed as:

$$\mathcal{L}_{\text{contrast}} = \frac{\sum_{i=1}^{N} w^{(c_i)} \cdot \mathcal{L}_{\text{InfoNCE}}(\mathbf{f}_i)}{\sum_{i=1}^{N} w^{(c_i)}}$$

where $c_i$ denotes the class label of object instance $i$, and the denominator normalizes by the sum of instance weights rather than the raw instance count. This normalization scheme ensures that the loss magnitude remains stable across batches with varying class distributions.

The total training objective combines standard detection losses with the contrastive component:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{box}} + \mathcal{L}_{\text{cls}} + \mathcal{L}_{\text{dfl}} + \lambda(t) \cdot \mathcal{L}_{\text{contrast}}$$

where $\mathcal{L}_{\text{box}}$, $\mathcal{L}_{\text{cls}}$, and $\mathcal{L}_{\text{dfl}}$ represent the bounding box regression, classification, and distribution focal losses, respectively.

## 3.X.7 Dynamic Weight Scheduling

The relative importance of contrastive learning versus detection-specific objectives may vary throughout the training process. To accommodate this dynamic, we employ a linear weight scheduling strategy where the contrastive loss coefficient $\lambda(t)$ evolves according to:

$$\lambda(t) = \lambda_{\text{init}} + (\lambda_{\text{final}} - \lambda_{\text{init}}) \cdot \min\left(\frac{t}{T-1}, 1\right)$$

where $t$ represents the current training epoch, $T$ denotes the total number of epochs, and $\lambda_{\text{init}}$ and $\lambda_{\text{final}}$ are hyperparameters controlling the initial and final contrastive weights, respectively. This progressive weighting scheme allows the model to initially focus on fundamental detection capabilities before gradually emphasizing feature discriminability in later training stages.

Empirical validation on benchmark datasets reveals that initiating with $\lambda_{\text{init}} = 0.2$ and progressively increasing to $\lambda_{\text{final}} = 0.6$ yields optimal performance across diverse evaluation metrics. When $\lambda_{\text{init}} = \lambda_{\text{final}}$, the framework degenerates to static weighting, ensuring backward compatibility with conventional training paradigms.

## 3.X.8 Implementation Considerations

Several engineering refinements are critical for stable and efficient training:

**Numerical Stability**: To prevent division-by-zero errors in edge cases where certain classes are absent from a batch, we clamp class counts to a minimum value of 1 and handle empty batches by returning zero loss. Additionally, class indices are validated to ensure they fall within the valid range $[0, n_c)$, preventing out-of-bounds errors during bincount operations.

**Memory Efficiency**: We employ in-place operations and view-based tensor reshaping wherever possible to minimize memory overhead. The number of sampled background features is capped to prevent excessive memory consumption when processing batches with numerous anchor points.

**Gradient Flow**: The L2 normalization applied to feature vectors prior to similarity computation ensures that gradients primarily encode angular updates rather than magnitude changes, promoting more stable optimization dynamics and preventing gradient explosion in deep networks.

These architectural and algorithmic choices collectively enable the seamless integration of contrastive learning into object detection frameworks while maintaining computational efficiency and numerical stability throughout the training process.

---

**References**:
[1] A. van den Oord, Y. Li, and O. Vinyals, "Representation learning with contrastive predictive coding," arXiv preprint arXiv:1807.03748, 2018.
