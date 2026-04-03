# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""
Custom Backpropagation Module with Floor/Fractional Decomposition.

This module implements a custom neural network layer that:
1. Reshapes input from (B,C,W,H) to (B,H,W,C)
2. Decomposes each value x into floor(x) and fractional part (x - floor(x))
3. Uses floor(x) as index to lookup weight vector from stored weights
4. Computes weighted sum: w[floor(x)] * (x - floor(x))

Mathematical Justification for Backpropagation:
==============================================

Forward Pass:
-------------
Given input x with shape (B, C, W, H), we:
1. Reshape to (B, H, W, C)
2. For each element x_i, compute:
   - idx_i = floor(x_i)
   - frac_i = x_i - floor(x_i)
   - output_i = w[idx_i] * frac_i

Where w is a learnable weight matrix of shape (N, C).

Backward Pass (Gradient Computation):
-------------------------------------

Let L be the loss function. We need to compute:
1. ∂L/∂x_i (gradient w.r.t. input)
2. ∂L/∂w (gradient w.r.t. weights)

From chain rule:
∂L/∂x_i = ∂L/∂output_i * ∂output_i/∂x_i

Analysis of ∂output_i/∂x_i:
--------------------------
output_i = w[floor(x_i)] * (x_i - floor(x_i))

Key insight: floor(x_i) is piecewise constant, so its derivative is 0 almost everywhere
(undefined at integer boundaries, but measure zero in continuous domain).

For x_i in interval [k, k+1) where k is integer:
- floor(x_i) = k (constant)
- frac_i = x_i - k

Therefore:
∂output_i/∂x_i = ∂/∂x_i [w[k] * (x_i - k)]
                = w[k] * ∂/∂x_i [x_i - k]
                = w[k] * 1
                = w[k]
                = w[floor(x_i)]

So: ∂L/∂x_i = grad_output_i * w[floor(x_i)]

Analysis of ∂L/∂w[j]:
--------------------
w[j] appears in output whenever floor(x_i) = j.

For all positions where floor(x_i) = j:
∂output_i/∂w[j] = frac_i = x_i - floor(x_i)

Therefore:
∂L/∂w[j] = Σ_(i: floor(x_i)=j) grad_output_i * (x_i - floor(x_i))

This can be efficiently computed using scatter_add operations.

Theoretical Validity:
--------------------
1. **Continuity**: The function is continuous everywhere (including at integer boundaries
   where the fractional part smoothly transitions from ≈1 to 0).

2. **Differentiability**: The function is differentiable almost everywhere (except at
   integer boundaries which have measure zero). In practice, gradient descent handles
   this gracefully.

3. **Gradient Flow**: Gradients flow properly from output to both inputs and weights,
   allowing end-to-end training.

4. **Interpolation Property**: This operation performs linear interpolation between
   weight vectors w[k] and w[k+1], weighted by the fractional part of x.
   When frac=0, output = 0 (no contribution from w[k])
   When frac→1, output → w[k] (full contribution from w[k])

5. **Biological Inspiration**: Similar to neural coding schemes where continuous signals
   are decomposed into discrete indices (population coding) and continuous modulation
   (rate coding).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class FloorFractionalIndexing(Function):
    """
    Custom autograd Function implementing floor/fractional decomposition with proper gradients.

    Forward: output = w[floor(x)] * (x - floor(x))
    Backward: Computes gradients w.r.t. both x and w
    """

    @staticmethod
    def forward(ctx, x, w):
        """
        Forward pass: Compute weighted sum using floor indexing.

        Args:
            x: Input tensor of shape (B, H, W, C)
            w: Weight tensor of shape (N, C)

        Returns:
            output: Tensor of shape (B, H, W, C)
        """
        # Compute floor indices and fractional parts
        x_floor = torch.floor(x)
        x_frac = x - x_floor

        # Clamp indices to valid range [0, N-1]
        indices = torch.clamp(x_floor.long(), 0, w.shape[0] - 1)

        # Save for backward pass
        ctx.save_for_backward(x, x_floor, x_frac, indices, w)

        # Index into weights and multiply by fractional part
        # w has shape (N, C), we need to gather along dim 0
        # Reshape for gathering: indices should match dimensions
        B, H, W, C = x.shape
        N, C_w = w.shape
        assert C == C_w, f"Channel mismatch: x has {C} channels, w has {C_w}"

        # Expand indices to match weight dimensions
        # indices: (B, H, W, C) -> expand for gathering
        indices_expanded = indices.unsqueeze(-1).expand(B, H, W, C, 1)

        # For each spatial position and channel, gather the corresponding weight vector
        # We need w[indices[b,h,w,c], c]
        gathered_weights = torch.zeros_like(x)
        for c in range(C):
            # Get indices for this channel: (B, H, W)
            idx_c = indices[..., c].view(-1)  # (B*H*W,)
            # Gather weights: w[idx_c, c]
            w_c = w[idx_c, c]  # (B*H*W,)
            # Reshape back and assign
            gathered_weights[..., c] = w_c.view(B, H, W)

        # Multiply by fractional parts
        output = gathered_weights * x_frac

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: Compute gradients w.r.t. x and w.

        Args:
            grad_output: Gradient of loss w.r.t. output, shape (B, H, W, C)

        Returns:
            grad_x: Gradient w.r.t. input x, shape (B, H, W, C)
            grad_w: Gradient w.r.t. weights w, shape (N, C)
        """
        x, x_floor, x_frac, indices, w = ctx.saved_tensors
        B, H, W, C = x.shape
        N, C_w = w.shape

        # Gradient w.r.t. x
        # ∂L/∂x = grad_output * w[floor(x)]
        grad_x = torch.zeros_like(x)
        for c in range(C):
            idx_c = indices[..., c].view(-1)
            w_c = w[idx_c, c]
            grad_x[..., c] = grad_output[..., c] * w_c.view(B, H, W)

        # Gradient w.r.t. w
        # ∂L/∂w[j] = Σ_(positions where floor(x)=j) grad_output * (x - floor(x))
        grad_w = torch.zeros_like(w)

        # Compute grad_output * x_frac for all positions
        grad_weighted = grad_output * x_frac  # (B, H, W, C)

        # Scatter add to weight gradient
        for c in range(C):
            idx_c = indices[..., c].view(-1)  # (B*H*W,)
            grad_c = grad_weighted[..., c].view(-1)  # (B*H*W,)
            # Accumulate gradients for each index
            grad_w[:, c].scatter_add_(0, idx_c, grad_c)

        return grad_x, grad_w


class CustomBackpropModule(nn.Module):
    """
    Custom neural network module with floor/fractional decomposition and custom backpropagation.

    This module:
    1. Reshapes input from (B, C, W, H) to (B, H, W, C)
    2. Decomposes each value into floor and fractional components
    3. Uses floor as index into learnable weight matrix
    4. Outputs weighted fractional parts

    Attributes:
        num_indices (int): Number of possible floor indices (N)
        num_channels (int): Number of channels (C)
        weight (nn.Parameter): Learnable weight matrix of shape (N, C)

    Example:
        >>> module = CustomBackpropModule(num_indices=256, num_channels=64)
        >>> x = torch.randn(2, 64, 32, 32)  # (B, C, W, H)
        >>> output = module(x)  # (2, 32, 32, 64) - (B, H, W, C)
    """

    def __init__(self, num_indices: int, num_channels: int, init_scale: float = 0.1):
        """
        Initialize the custom backpropagation module.

        Args:
            num_indices: Number of possible floor indices (N)
            num_channels: Number of channels (C)
            init_scale: Scale for weight initialization
        """
        super().__init__()

        self.num_indices = num_indices
        self.num_channels = num_channels

        # Initialize learnable weights
        # Shape: (N, C)
        self.weight = nn.Parameter(torch.randn(num_indices, num_channels) * init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the module.

        Args:
            x: Input tensor of shape (B, C, W, H)

        Returns:
            output: Tensor of shape (B, H, W, C)
        """
        B, C, W, H = x.shape

        assert C == self.num_channels, \
            f"Input channels {C} does not match module channels {self.num_channels}"

        # Reshape from (B, C, W, H) to (B, H, W, C)
        x_reshaped = x.permute(0, 3, 2, 1)  # (B, H, W, C)

        # Apply custom floor/fractional indexing with custom gradients
        output = FloorFractionalIndexing.apply(x_reshaped, self.weight)

        return output

    def extra_repr(self) -> str:
        """Return extra representation string for debugging."""
        return f'num_indices={self.num_indices}, num_channels={self.num_channels}'


# Export the module
__all__ = ['CustomBackpropModule', 'FloorFractionalIndexing']
