# """
# Adapted from:
# https://github.com/kyegomez/Griffin/tree/main

# Modification Summary:
# 1. class Griffin: forward() method was updated to support both
# token IDs (x: LongTensor) and precomputed embeddings (inputs_embeds: FloatTensor).
# 2. class GriffinResidualBlock: decouples structural definition from parameter learning
# by introducing non-learnable, randomly initialized linear and
# convolutional transformations within the forward() pass.
# (e.g., Linear, Conv1d, LayerNorm, GELU)


# """

# import numpy as np
# import torch
# import torch.nn.functional as F
# from torch import Tensor, nn
# from zeta.nn import FeedForward


# class RMSNorm(nn.Module):
#     def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
#         """
#             Root Mean Square Layer Normalization
#         :param d: model size
#         :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
#         :param eps:  epsilon value, default 1e-8
#         :param bias: whether use bias term for RMSNorm, disabled by
#             default because RMSNorm doesn't enforce re-centering invariance.
#         """
#         super(RMSNorm, self).__init__()

#         self.eps = eps
#         self.d = d
#         self.p = p
#         self.bias = bias

#         self.scale = nn.Parameter(torch.ones(d))
#         self.register_parameter("scale", self.scale)

#         if self.bias:
#             self.offset = nn.Parameter(torch.zeros(d))
#             self.register_parameter("offset", self.offset)

#     def forward(self, x):
#         if self.p < 0.0 or self.p > 1.0:
#             norm_x = x.norm(2, dim=-1, keepdim=True)
#             d_x = self.d
#         else:
#             partial_size = int(self.d * self.p)
#             partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

#             norm_x = partial_x.norm(2, dim=-1, keepdim=True)
#             d_x = partial_size

#         rms_x = norm_x * d_x ** (-1.0 / 2)
#         x_normed = x / (rms_x + self.eps)

#         if self.bias:
#             return self.scale * x_normed + self.offset

#         return self.scale * x_normed


# class RMSNorm_ori(nn.Module):
#     """
#     Root Mean Square Normalization (RMSNorm) module.

#     Args:
#         dim (int): The dimension of the input tensor.

#     Attributes:
#         scale (float): The scaling factor for the normalized output.
#         g (nn.Parameter): The learnable parameter used for scaling.

#     """

#     def __init__(self, dim):
#         super().__init__()
#         self.scale = dim**0.5
#         self.g = nn.Parameter(torch.ones(dim))

#     def forward(self, x):
#         """
#         Forward pass of the RMSNorm module.

#         Args:
#             x (torch.Tensor): The input tensor.

#         Returns:
#             torch.Tensor: The normalized output tensor.

#         """
#         return F.normalize(x, dim=-1) * self.scale * self.g


# def random_linear(x: Tensor) -> Tensor:
#     """Apply a random linear transform to x, without tracking gradients."""
#     d = x.shape[-1]
#     weight = torch.randn(d, d, device=x.device).detach() * 0.02
#     bias = torch.zeros(d, device=x.device).detach()
#     return F.linear(x, weight, bias)


# def random_conv1d(x: Tensor) -> Tensor:
#     """Apply a random conv1d to x, without tracking gradients."""
#     d = x.shape[-1]
#     x = x.transpose(1, 2)
#     weight = torch.randn(d, d, 3, device=x.device).detach() * 0.02  # (out_c, in_c, k)
#     bias = torch.zeros(d, device=x.device).detach()
#     x = F.conv1d(x, weight, bias, padding=1)
#     return x.transpose(1, 2)


# class GriffinResidualBlock(nn.Module):
#     """
#     GriffinResidualBlock is a residual block used in the Griffin model.

#     Args:
#         dim (int): The input dimension.
#         mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
#         dropout (float, optional): The dropout rate. Defaults to 0.1.

#     Attributes:
#         dim (int): The input dimension.
#         mlp_mult (int): The multiplier for the hidden dimension in the feedforward network.
#         dropout (float): The dropout rate.
#         norm (RMSNorm): The normalization layer.
#         mlp (FeedForward): The feedforward network.

#     """

#     def __init__(
#         self,
#         dim: int,
#         mlp_mult: int,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_mult = mlp_mult
#         self.dropout = dropout

#         # Norm
#         self.norm = RMSNorm(dim)

#         # Feedforward
#         self.mlp = FeedForward(
#             dim,
#             dim,
#             mlp_mult,
#             post_act_ln=True,
#             dropout=dropout,
#         )
#         self.gelu = nn.GELU()
#         # RG-LRU
#         # self.lru = RGLRU(
#         #     dim,
#         #     mult=4,
#         # )

#     def forward(self, x: Tensor):
#         """
#         Forward pass of the GriffinResidualBlock.

#         Args:
#             x (Tensor): The input tensor.

#         Returns:
#             Tensor: The output tensor.

#         """
#         skip1 = x

#         # Norm
#         x = self.norm(x)

#         # Temporal Mixing Block
#         linear_1 = random_linear(x)
#         linear_1 = random_conv1d(linear_1)
#         # RG-LRU
#         # linear_1 = self.lru(linear_1)

#         linear_2 = random_linear(x)
#         linear_2 = self.gelu(linear_2)

#         # Element wise multiplication to merge the paths
#         x = linear_1 * linear_2 + skip1

#         # Skip2
#         skip2 = x

#         # Norm
#         x = self.norm(x)

#         # Feedforward
#         x = self.mlp(x)

#         return x + skip2


# class Griffin(nn.Module):
#     """
#     Griffin module for performing Griffin Residual Network operations.

#     Args:
#         dim (int): Dimension of the input tensor.
#         mlp_mult (int): Multiplier for the hidden dimension of the MLP layers.
#         dropout (float, optional): Dropout probability. Defaults to 0.1.

#     Attributes:
#         dim (int): Dimension of the input tensor.
#         mlp_mult (int): Multiplier for the hidden dimension of the MLP layers.
#         dropout (float): Dropout probability.
#         layers (nn.ModuleList): List of GriffinResidualBlock layers.

#     """

#     def __init__(
#         self,
#         dim: int,
#         mlp_mult: int = 4,
#         dropout: float = 0.1,
#     ):
#         super().__init__()
#         self.dim = dim
#         self.mlp_mult = mlp_mult
#         self.dropout = dropout

#         # Layers
#         self.layers = GriffinResidualBlock(dim, mlp_mult, dropout)

#         # Rmsnorm
#         self.norm = RMSNorm(dim)
#         self.output_head = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim), nn.GELU())

#     def forward(self, x: Tensor = None) -> Tensor:
#         """Forward pass of the Griffin module."""
#         # Normilize
#         x = self.norm(x)
#         x = self.layers(x) + x
#         # return x

#         return self.output_head(x)


"""
Adapted from:
https://github.com/knotgrass/Griffin/blob/main/griffin/griffin.py

Modification Summary:
RGLRU -> GELU
Temporal_Conv1D -> random_conv1d
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, nn


class RMSNorm(nn.Module):
    def __init__(self, d, p=-1.0, eps=1e-8, bias=False):
        """
            Root Mean Square Layer Normalization
        :param d: model size
        :param p: partial RMSNorm, valid value [0, 1], default -1.0 (disabled)
        :param eps:  epsilon value, default 1e-8
        :param bias: whether use bias term for RMSNorm, disabled by
            default because RMSNorm doesn't enforce re-centering invariance.
        """
        super(RMSNorm, self).__init__()

        self.eps = eps
        self.d = d
        self.p = p
        self.bias = bias

        self.scale = nn.Parameter(torch.ones(d))
        self.register_parameter("scale", self.scale)

        if self.bias:
            self.offset = nn.Parameter(torch.zeros(d))
            self.register_parameter("offset", self.offset)

    def forward(self, x):
        if self.p < 0.0 or self.p > 1.0:
            norm_x = x.norm(2, dim=-1, keepdim=True)
            d_x = self.d
        else:
            partial_size = int(self.d * self.p)
            partial_x, _ = torch.split(x, [partial_size, self.d - partial_size], dim=-1)

            norm_x = partial_x.norm(2, dim=-1, keepdim=True)
            d_x = partial_size

        rms_x = norm_x * d_x ** (-1.0 / 2)
        x_normed = x / (rms_x + self.eps)

        if self.bias:
            return self.scale * x_normed + self.offset

        return self.scale * x_normed


class Gated_MLP_block(nn.Module):
    def __init__(
        self,
        D: int,
        expansion_factor: int = 3,
        approximate: str = "none",
    ) -> None:
        super().__init__()
        self.D = D
        self.M = expansion_factor
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D * self.M)
        self.p2 = nn.Linear(in_features=D, out_features=D * self.M)
        self.p3 = nn.Linear(in_features=D * self.M, out_features=D)

    def forward(self, x: Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)

        return y


class Temporal_Conv1D(nn.Module):
    def __init__(self, D: int, kernel_size: int = 4):
        super().__init__()
        # A separable 1D convolution:
        # - Input channels = output channels = D
        # - groups = D makes it depthwise (channel-wise) convolution.
        self.conv = nn.Conv1d(
            in_channels=D,
            out_channels=D,
            kernel_size=kernel_size,
            groups=D,
            bias=False,
            padding=kernel_size // 2,  # optional, to preserve sequence length
        )

    def forward(self, x: Tensor) -> Tensor:
        # https://chatgpt.com/share/67692a55-5224-8005-a271-80067aa3bcbb
        # B = Batch size
        # T = Sequence length
        # D = Feature dimension
        # x: (B, T, D)
        # Transpose to (B, D, T) for Conv1d
        x = x.transpose(1, 2)
        x = self.conv(x)  # (B, D, T)
        # Transpose back to (B, T, D)
        x = x.transpose(1, 2)
        return x


def random_conv1d(x: Tensor) -> Tensor:
    """Apply a random conv1d to x, without tracking gradients."""
    d = x.shape[-1]
    x = x.transpose(1, 2)
    weight = torch.randn(d, d, 3, device=x.device).detach() * 0.02  # (out_c, in_c, k)
    bias = torch.zeros(d, device=x.device).detach()
    x = F.conv1d(x, weight, bias, padding=1)
    return x.transpose(1, 2)


class Recurrent_block(nn.Module):
    def __init__(self, D: int, D_rnn: int = ..., approximate: str = "none"):
        super().__init__()
        self.D = D
        self.D_rnn = D_rnn
        self.gelu = nn.GELU(approximate)
        self.p1 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p2 = nn.Linear(in_features=D, out_features=D_rnn)
        self.p3 = nn.Linear(in_features=D_rnn, out_features=D)
        # self.separableConv1D = Temporal_Conv1D(
        #     D_rnn, kernel_size=4)  ########### Temporal_Conv1D(D, kernel_size=4)
        # self.rglru = RGLRU(D_rnn)  ########### RGLRU(self.D)

    def forward(self, x: Tensor) -> Tensor:
        # left branch
        x1 = self.p1(x)
        x1 = self.gelu(x1)

        # right branch
        x2 = self.p2(x)
        x2 = random_conv1d(x2)
        x2 = self.gelu(x2)
        # x2 = self.separableConv1D(x2)
        # x2 = self.rglru(x2)

        y = x1 * x2  # element-wise multiplication
        y = self.p3(y)
        return y


class Griffin(nn.Module):
    def __init__(self, dim: int, **kwargs):
        super().__init__()
        self.mlp = Gated_MLP_block(dim, expansion_factor=3)
        self.tmb = Recurrent_block(dim, D_rnn=int(4 * dim / 3))
        self.rmsnorm = RMSNorm(d=dim)  # ?

    def forward(self, x: Tensor) -> Tensor:
        x1 = self.rmsnorm(x)
        x1 = self.tmb(x1)

        y1 = x + x1

        x2 = self.rmsnorm(y1)
        x2 = self.mlp(x2)

        y2 = y1 + x2

        return y2
