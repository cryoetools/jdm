# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from functools import partial
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba
from monai.networks.blocks import PatchEmbed
from monai.networks.blocks.convolutions import Convolution
from monai.networks.blocks.dynunet_block import (
    UnetBasicBlock,
    UnetOutBlock,
    UnetResBlock,
    get_conv_layer,
)
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock
from monai.networks.layers.factories import Norm
from monai.utils import ensure_tuple_rep, optional_import
from torch.nn import LayerNorm
from zeta.nn.modules.simple_mamba import MambaBlock
from zeta.nn.modules.ssm import SSM

# third party
from .griffin import Griffin

rearrange, _ = optional_import("einops", name="rearrange")


class AxisWiseLayer(nn.Module):
    def __init__(
        self,
        pos_shape,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba=True,
        use_pos=True,
        use_res=True,
    ):
        super().__init__()

        self.mamba_x, self.mamba_y, self.mamba_z = [
            MambaLayer(pos_shape, d_model, d_state, d_conv, expand, use_mamba, use_pos)
            for _ in range(3)
        ]

        self.fwd = self.res_block if use_res else self.base_block

    def base_block(self, x):
        x = (
            self.mamba_z(x.permute(0, 1, 3, 4, 2).contiguous())
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        x = (
            self.mamba_y(x.permute(0, 1, 4, 2, 3).contiguous())
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        return self.mamba_x(x)

    def res_block(self, x):
        # Z: [ 0,  9, 18]
        fz = (
            self.mamba_z(x.permute(0, 1, 3, 4, 2).contiguous())
            .permute(0, 1, 4, 2, 3)
            .contiguous()
        )
        # Y: [ 0,  3,  6]
        fy = (
            self.mamba_y((fz + x).permute(0, 1, 4, 2, 3).contiguous())
            .permute(0, 1, 3, 4, 2)
            .contiguous()
        )
        # X: [ 0,  1,  2]
        return self.mamba_x(fz + fy + x)

    def forward(self, x):
        return self.fwd(x)


class MambaLayer(nn.Module):
    def __init__(
        self,
        pos_shape,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        use_mamba=True,
        use_pos=True,
    ):
        super().__init__()
        self.pos_shape = pos_shape
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.norm = nn.LayerNorm(d_model)
        self.pos_embed = torch.zeros(1, d_model, *pos_shape)
        if use_pos:
            self.pos_embed = nn.Parameter(self.pos_embed)
            # nn.init.trunc_normal_(self.pos_embed, std=0.02)
        # self.core_block = self.griffin_block() if use_mamba else self.base_block()
        if not use_mamba:
            self.core_block = self.base_block()
        elif use_mamba == 1:
            self.core_block = self.mamba_block()
        elif use_mamba == 2:
            self.core_block = self.ssm_block()
        else:
            self.core_block = self.griffin_block()

    def base_block(self):
        return nn.Sequential(
            nn.Linear(self.d_model, self.d_model * self.expand),
            nn.ReLU(),
            nn.Linear(self.d_model * self.expand, self.d_model),
        )

    def mamba_block(self):
        return Mamba(
            d_model=self.d_model,  # Model dimension d_model
            d_state=self.d_state,  # SSM state expansion factor
            d_conv=self.d_conv,  # Local convolution width
            expand=self.expand,  # Block expansion factor
        )

    def ssm_block(self):
        return nn.Sequential(
            SSM(
                in_features=self.d_model,
                dim_inner=self.d_model,
                dt_rank=32,  # 32,
                d_state=6,  # self.d_state,
            ),
            nn.SiLU(),
        )

    def griffin_block(self):
        return Griffin(
            dim=self.d_model,  # Dimension of the model
            mlp_mult=4,  # Multiplier for the hidden dimension in the MLPs
            dropout=0.0,  # Dropout rate
        )

    def forward(self, x):
        x = x + self.pos_embed.to(x.device)
        # print(f"x.shape:{x.shape}")
        out = self.norm(x.flatten(2).transpose(-1, -2))
        # print(f"out.shape:{out.shape}")
        out = self.core_block(out).transpose(-1, -2).reshape_as(x)
        return out + x


class MSSA(nn.Module):
    """MultiScaleSelfAttention
    Note:
        Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/attentionunet.py
    """

    def __init__(self, in_channels, spatial_dims=3, dropout=0.0):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dropout = dropout

        # self.conv1 = self._create_block(in_channels, in_channels, 1, 0)
        self.conv3 = self._create_block(in_channels, in_channels, 3, 1)
        self.conv5 = self._create_block(in_channels, in_channels, 5, 2)
        self.relu = nn.ReLU()
        self.psi = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid(),
        )

    def _create_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                dropout=self.dropout,
                conv_only=True,
            ),
            nn.InstanceNorm3d(in_channels),
            # Norm[Norm.BATCH, self.spatial_dims](out_channels),
        )

    def forward(self, x) -> torch.Tensor:
        # x1 = self.conv1(x)
        x3 = self.conv3(x)
        x5 = self.conv5(x)

        # psi = self.psi(self.relu(x1 + x3 + x5))
        psi = self.psi(self.relu(x3 + x5))
        return x + x * psi


class MSAF(nn.Module):
    """
    MultiScaleAttentionFusion
    Note:
        Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/attentionunet.py
    """

    def __init__(self, in_channels, spatial_dims=3, dropout=0.0):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dropout = dropout

        self.conv1 = self._create_block(in_channels, in_channels, 1, 0)
        self.conv3 = self._create_block(in_channels, in_channels, 3, 1)
        self.conv5 = self._create_block(in_channels, in_channels, 5, 2)

        self.psi3 = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid(),
        )
        self.psi5 = nn.Sequential(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=in_channels,
                out_channels=1,
                kernel_size=1,
                strides=1,
                padding=0,
                dropout=dropout,
                conv_only=True,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.Sigmoid(),
        )

    def _create_block(self, in_channels, out_channels, kernel_size, padding):
        return nn.Sequential(
            Convolution(
                spatial_dims=self.spatial_dims,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=padding,
                dropout=self.dropout,
                conv_only=True,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
        feat1 = self.conv1(x)
        feat3 = self.conv3(x)
        feat5 = self.conv5(x)

        att3 = self.psi3(feat3)
        att5 = self.psi5(feat5)
        return x + feat1 + feat3 * att3 + feat5 * att5


class GSC(nn.Module):
    def __init__(self, in_channels) -> None:
        super().__init__()

        self.block1_1 = self._create_block(in_channels, 3, 1)
        self.block1_2 = self._create_block(in_channels, 3, 1)
        self.block2 = self._create_block(in_channels, 1, 0)
        self.block3 = self._create_block(in_channels, 1, 0)

    def _create_block(self, in_channels, kernel_size, padding, stride=1):
        return nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size, stride, padding),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        x1 = self.block1_1(x)
        x1 = self.block1_2(x1)
        x2 = self.block2(x)
        return x + self.block3(x1 + x2)


class RMSA(nn.Module):
    """ResidualMultiScaleAttention
    A Block combining residual filtering, multi-scale feature fusion, and light attention.
    """

    def __init__(self, in_channels: int, spatial_dims=3, dropout=0.0) -> None:
        super().__init__()

        self._conv = partial(
            Convolution,
            spatial_dims=spatial_dims,
            strides=1,
            dropout=dropout,
            conv_only=True,
        )

        # Main residual path
        # self.block_main = self._res_block(in_channels)

        # Multi-scale feature paths
        self.conv3 = self._conv_block(in_channels, 3, 1)
        self.conv5 = self._conv_block(in_channels, 5, 2)

        # Lightweight attention
        self.attention = nn.Sequential(
            self._conv(
                in_channels=in_channels, out_channels=in_channels // 2, kernel_size=1
            ),
            nn.InstanceNorm3d(in_channels // 2),
            nn.ReLU(inplace=True),
            self._conv(
                in_channels=in_channels // 2, out_channels=in_channels, kernel_size=1
            ),
            nn.Sigmoid(),
        )

    def _res_block(self, in_channels: int) -> nn.Sequential:
        """Creates a basic residual block using 3x3x3 convolutions."""
        return nn.Sequential(
            self._conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            self._conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=3,
                padding=1,
            ),
            nn.InstanceNorm3d(in_channels),
        )

    def _conv_block(self, in_channels: int, k: int, p: int) -> nn.Sequential:
        """Multi-scale convolution block."""
        return nn.Sequential(
            self._conv(
                in_channels=in_channels,
                out_channels=in_channels,
                kernel_size=k,
                padding=p,
            ),
            nn.InstanceNorm3d(in_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        # Output from the residual block
        # residual = self.block_main(x)

        # Compute attention map based Multi-scale features
        attn = self.attention(self.conv3(x) + self.conv5(x))

        # Final output: input + residual weighted by attention
        # return x + residual * attn
        return x + x * attn


class MlpChannel(nn.Module):
    def __init__(self, hidden_size, mlp_dim):
        super().__init__()
        self.fc1 = nn.Conv3d(hidden_size, mlp_dim, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv3d(mlp_dim, hidden_size, 1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


class PatchMergingV2Custom(nn.Module):
    """
    Patch merging layer based on: "Liu et al.,
    Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
    <https://arxiv.org/abs/2103.14030>"

    Note:
        Adapted from: https://github.com/microsoft/Swin-Transformer
    """

    def __init__(
        self,
        dim: int,
        norm_layer: type[LayerNorm] = nn.LayerNorm,
        spatial_dims: int = 3,
        data_format: str = "channels_last",
    ) -> None:
        """
        Args:
            dim: number of feature channels.
            norm_layer: normalization layer.
            spatial_dims: number of spatial dims.
        """

        super().__init__()
        self.dim = dim
        if spatial_dims == 3:
            self.reduction = nn.Linear(8 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(8 * dim)
        elif spatial_dims == 2:
            self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
            self.norm = norm_layer(4 * dim)
        self.fwd = (
            self.channels_last
            if data_format == "channels_last"
            else self.channels_first
        )

    def channels_last(self, x):
        x_shape = x.size()
        if len(x_shape) == 5:
            b, d, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1) or (d % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2, 0, d % 2))
            x = torch.cat(
                [
                    x[:, i::2, j::2, k::2, :]
                    for i, j, k in itertools.product(range(2), range(2), range(2))
                ],
                -1,
            )

        elif len(x_shape) == 4:
            b, h, w, c = x_shape
            pad_input = (h % 2 == 1) or (w % 2 == 1)
            if pad_input:
                x = F.pad(x, (0, 0, 0, w % 2, 0, h % 2))
            x = torch.cat(
                [x[:, j::2, i::2, :] for i, j in itertools.product(range(2), range(2))],
                -1,
            )

        x = self.norm(x)
        x = self.reduction(x)
        return x

    def channels_first(self, x):
        """Rearranges the input tensor from channels_first to channels_last format."""
        x = rearrange(x, "b c d h w -> b d h w c")
        x = self.channels_last(x)
        return rearrange(x, "b d h w c -> b c d h w")

    def forward(self, x):
        return self.fwd(x)


class HybridPatchDownsample(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, spatial_dims: int = 3):
        """Hybrid downsampling module combining convolutional and patch-merging features."""
        super().__init__()
        self.spatial_dims = spatial_dims

        # Convolutional downsampling branch
        if spatial_dims == 3:
            self.conv_down = nn.Sequential(
                nn.Conv3d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm3d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.fuse = nn.Conv3d(out_channels, out_channels, kernel_size=1)
        elif spatial_dims == 2:
            self.conv_down = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, stride=2, padding=1
                ),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
            self.fuse = nn.Conv2d(out_channels, out_channels, kernel_size=1)
        else:
            raise ValueError("Only 2D or 3D spatial dims are supported.")

        # Patch merging branch using the prebuilt module
        self.patch_merging = PatchMergingV2Custom(
            dim=in_channels,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            data_format="channels_first",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        # Convolutional branch
        x_conv = self.conv_down(x)

        # Patch merging branch
        x_patch = self.patch_merging(x)

        # Fuse two branches via element-wise addition (you may also try concat + Conv)
        x_fused = self.fuse(x_conv + x_patch)

        return x_fused


class MambaEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        depths: Optional[Sequence[int]] = (3, 3, 3, 3),
        feature_sizes: Optional[Sequence[int]] = (12, 24, 48, 96),
        downsample: Optional[str] = "conv",
        out_indices: Optional[List] = None,
        img_size: Optional[int] = None,
        spatial_dims: Optional[int] = 3,
        use_mamba: Optional[bool] = True,
        use_pos: Optional[bool] = True,
        use_ax: Optional[bool] = True,
        use_attn: Optional[bool] = True,
    ):
        super().__init__()
        self.use_mamba = use_mamba
        self.use_pos = use_pos
        self.use_ax = use_ax
        self.use_attn = use_attn

        self.img_size = img_size
        self.spatial_dims = spatial_dims
        self.stage_num = len(depths)
        self.out_indices = range(self.stage_num) if out_indices is None else out_indices

        self.downsample_layers = (
            self._vit_downsample(in_channels, feature_sizes)
            if "conv" not in downsample
            else self._conv_downsample(in_channels, feature_sizes)
        )
        self.attns, self.cores, self.norms, self.mlps = self._create_layers(
            depths, feature_sizes
        )

    def _conv_downsample(self, in_channels, feature_sizes):
        downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_channels, feature_sizes[0], kernel_size=7, stride=2, padding=3)
        )
        downsample_layers.append(stem)

        for i in range(self.stage_num - 1):
            downsample_layers.append(
                nn.Sequential(
                    nn.InstanceNorm3d(feature_sizes[i]),
                    nn.Conv3d(
                        feature_sizes[i], feature_sizes[i + 1], kernel_size=2, stride=2
                    ),
                )
            )
        return downsample_layers

    def _vit_downsample(
        self,
        in_channels,
        feature_sizes,
        patch_size=2,
        spatial_dims=3,
        patch_norm=False,
    ):

        patch_sizes = ensure_tuple_rep(patch_size, spatial_dims)

        downsample = partial(
            PatchMergingV2Custom,
            norm_layer=nn.LayerNorm,
            spatial_dims=spatial_dims,
            data_format="channels_first",
        )

        downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            PatchEmbed(
                patch_size=patch_sizes,
                in_chans=in_channels,
                embed_dim=feature_sizes[0],
                norm_layer=nn.LayerNorm if patch_norm else None,
                spatial_dims=spatial_dims,
            ),
        )

        downsample_layers.append(stem)

        for i in range(self.stage_num - 1):
            downsample_layers.append(downsample(dim=feature_sizes[i]))
        return downsample_layers

    def _hybrid_downsample(self, in_channels, feature_sizes, spatial_dims=3):
        downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv3d(in_channels, feature_sizes[0], kernel_size=7, stride=2, padding=3)
        )
        downsample_layers.append(stem)

        for i in range(self.stage_num - 1):
            downsample_layers.append(
                HybridPatchDownsample(
                    in_channels=feature_sizes[i],
                    out_channels=feature_sizes[i + 1],
                    spatial_dims=spatial_dims,
                )
            )

    def _create_layers(self, depths, feature_sizes):
        attns = nn.ModuleList()
        cores = nn.ModuleList()
        norms = nn.ModuleList()
        mlps = nn.ModuleList()

        for i in self.out_indices:
            hidden_dim = feature_sizes[i]
            # attn = GSC(hidden_dim) if self.use_attn else nn.Identity()
            # print("GSC")

            if not self.use_attn:
                attn = nn.Identity()
                print("use Identity")
            elif int(self.use_attn) == 1:
                attn = RMSA(hidden_dim)
                print("use RMSA")
            elif int(self.use_attn) == 2:
                attn = GSC(hidden_dim)
                print("use GSC")
            elif int(self.use_attn) == 3:
                attn = MSAF(hidden_dim)
                print("use MSAF")
            else:
                attn = MSSA(hidden_dim)
                print("use MSSA")

            # attn = MSAF(hidden_dim) if self.use_attn else nn.Identity()
            attns.append(attn)

            factor = hidden_dim * 2 // feature_sizes[0]
            pos_shape = (self.img_size // factor,) * self.spatial_dims

            core = (
                AxisWiseLayer(
                    pos_shape=pos_shape,
                    d_model=hidden_dim,
                    use_mamba=self.use_mamba,
                    use_pos=self.use_pos,
                )
                if self.use_ax
                else nn.Sequential(
                    *[
                        MambaLayer(
                            pos_shape=pos_shape,
                            d_model=hidden_dim,
                            use_mamba=self.use_mamba,
                            use_pos=self.use_pos,
                        )
                        for _ in range(depths[i])  # for _ in range(1)  #
                    ]
                )
            )
            cores.append(core)

            norms.append(nn.InstanceNorm3d(hidden_dim))
            mlps.append(MlpChannel(hidden_dim, 2 * hidden_dim))

        return attns, cores, norms, mlps

    def forward(self, x):
        outs = []
        for i in range(len(self.downsample_layers)):
            x = self.downsample_layers[i](x)
            x = self.attns[i](x)
            x = self.cores[i](x)

            if i in self.out_indices:
                x_out = self.norms[i](x)
                x_out = self.mlps[i](x_out)
                outs.append(x_out)
        return tuple(outs)


class JDMNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        spatial_dims: Optional[int] = 3,
        depths: Optional[Sequence[int]] = (3, 3, 3, 3),
        feature_size: Optional[int] = 12,
        norm_name: Optional[tuple | str] = "instance",
        res_block: Optional[bool] = True,
        downsample: Optional[str] = "conv",
        img_size: Optional[int] = None,
        use_mamba: Optional[bool] = True,
        use_pos: Optional[bool] = True,
        use_ax: Optional[bool] = True,
        use_attn: Optional[bool] = True,
        use_upf: Optional[bool] = True,
    ) -> None:
        super().__init__()
        print(
            f"use_mamba:{use_mamba}, use_pos:{use_pos}, use_ax:{use_ax}, use_attn:{use_attn}, use_upf:{use_upf}"
        )
        stage_num = len(depths)
        if stage_num not in [2, 3, 4]:
            raise ValueError(
                "Unsupported stage number, length of depths should be 2, 3 or 4."
            )
        if spatial_dims not in [2, 3]:
            raise ValueError(
                "Unsupported spatial_dims, spatial dimension should be 2 or 3."
            )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_dims = spatial_dims
        self.depths = depths
        self.feature_sizes = [feature_size * pow(2, i) for i in range(len(depths))]

        self.feature_encoder = MambaEncoder(
            in_channels,
            depths=depths,
            feature_sizes=self.feature_sizes,
            downsample=downsample,
            img_size=img_size,
            spatial_dims=spatial_dims,
            use_mamba=use_mamba,
            use_pos=use_pos,
            use_ax=use_ax,
            use_attn=use_attn,
        )
        self.basic_blk = partial(
            UnetrBasicBlock,
            spatial_dims=spatial_dims,
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.up_blk = (
            partial(
                UnetrUpBlockF,
                img_size=img_size,
                feature_size=feature_size,
                spatial_dims=spatial_dims,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
            if use_upf
            else partial(
                UnetrUpBlock,
                spatial_dims=spatial_dims,
                kernel_size=3,
                upsample_kernel_size=2,
                norm_name=norm_name,
                res_block=res_block,
            )
        )
        # self.up_blk = partial(
        #     UnetrUpBlockF,
        #     img_size=img_size,
        #     feature_size=feature_size,
        #     spatial_dims=spatial_dims,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )

        # self.upf_blk = partial(
        #     UnetrUpBlock,
        #     spatial_dims=spatial_dims,
        #     kernel_size=3,
        #     upsample_kernel_size=2,
        #     norm_name=norm_name,
        #     res_block=res_block,
        # )

        self.out = UnetOutBlock(
            spatial_dims=spatial_dims,
            in_channels=feature_size,
            out_channels=out_channels,
        )

        self.encoder1 = self.basic_blk(
            in_channels=in_channels, out_channels=feature_size
        )
        self.encoder2 = self.basic_blk(
            in_channels=feature_size, out_channels=feature_size * 2
        )
        self.encoder3 = self.basic_blk(
            in_channels=feature_size * 2, out_channels=feature_size * 4
        )
        self.decoder3 = self.up_blk(
            in_channels=feature_size * 4, out_channels=feature_size * 2
        )
        self.decoder2 = self.up_blk(
            in_channels=feature_size * 2, out_channels=feature_size
        )
        self.decoder1 = self.basic_blk(
            in_channels=feature_size, out_channels=feature_size
        )

        if stage_num == 2:
            self.fwd = self.stage2
        else:
            self.encoder4 = self.basic_blk(
                in_channels=feature_size * 4, out_channels=feature_size * 8
            )
            self.decoder4 = self.up_blk(
                in_channels=feature_size * 8, out_channels=feature_size * 4
            )
            if stage_num == 3:
                self.fwd = self.stage3
            else:
                self.encoder5 = self.basic_blk(
                    in_channels=feature_size * 8, out_channels=feature_size * 16
                )
                self.decoder5 = self.up_blk(
                    in_channels=feature_size * 16, out_channels=feature_size * 8
                )
                self.fwd = self.stage4

    def stage2(self, x, hidden_states_out):
        enc1 = self.encoder1(x)  # 1 x S -> F x S
        enc2 = self.encoder2(hidden_states_out[0])  # F x S/2 -> 2F x S/2
        enc3 = self.encoder3(hidden_states_out[1])  # 2F x S/4 -> 4F x S/4

        dec1 = self.decoder3(enc3, enc2)  # 4F x S/4, 2F x S/2 -> 2F x S/2
        dec0 = self.decoder2(dec1, enc1)  # 2F x S/2, F x S -> F x S
        out = self.decoder1(dec0)  # F x S -> F x S
        return out

    def stage3(self, x, hidden_states_out):
        enc1 = self.encoder1(x)  # 1 x S -> F x S
        enc2 = self.encoder2(hidden_states_out[0])  # F x S/2 -> 2F x S/2
        enc3 = self.encoder3(hidden_states_out[1])  # 2F x S/4 -> 4F x S/4
        enc4 = self.encoder4(hidden_states_out[2])  # 4F x S/8 -> 8F x S/8

        dec2 = self.decoder4(enc4, enc3)  # 8F x S/8, 4F x S/4 -> 4F x S/4
        dec1 = self.decoder3(dec2, enc2)  # 4F x S/4, 2F x S/2 -> 2F x S/2
        dec0 = self.decoder2(dec1, enc1)  # 2F x S/2, F x S -> F x S
        out = self.decoder1(dec0)  # F x S -> F x S
        return out

    def stage4(self, x, hidden_states_out):
        enc1 = self.encoder1(x)  # 1 x S -> F x S
        enc2 = self.encoder2(hidden_states_out[0])  # F x S/2 -> 2F x S/2
        enc3 = self.encoder3(hidden_states_out[1])  # 2F x S/4 -> 4F x S/4
        enc4 = self.encoder4(hidden_states_out[2])  # 4F x S/8 -> 8F x S/8
        enc5 = self.encoder5(hidden_states_out[3])  # 8F x S/16 -> 16F x S/16

        dec3 = self.decoder5(enc5, enc4)  # 16F x S/16, 8F x S/8 -> 8F x S/8
        dec2 = self.decoder4(dec3, enc3)  # 8F x S/8, 4F x S/4 -> 4F x S/4
        dec1 = self.decoder3(dec2, enc2)  # 4F x S/4, 2F x S/2 -> 2F x S/2
        dec0 = self.decoder2(dec1, enc1)  # 2F x S/2, F x S -> F x S
        out = self.decoder1(dec0)  # F x S -> F x S
        return out

    def forward(self, x):
        hidden_states_out = self.feature_encoder(x)
        out = self.fwd(x, hidden_states_out)
        return self.out(out)


class WSLayer(nn.Module):
    """Weighted Spectrum"""

    def __init__(self, dim=1, h=128, w=128, d=65, std_factor=0.02):
        super().__init__()
        self.complex_weight = nn.Parameter(
            torch.randn(dim, h, w, d, 2, dtype=torch.float32) * std_factor
        )

    def forward(self, x):
        H, W, D = x.shape[2:]
        weight = torch.view_as_complex(self.complex_weight)

        x = torch.fft.rfftn(x, dim=(-3, -2, -1), norm="ortho")
        x = x * weight
        x = torch.fft.irfftn(x, s=(H, W, D), dim=(-3, -2, -1), norm="ortho")
        return x


class SpectralLayer(nn.Module):
    def __init__(self, dim, img_shape, alpha_init: float = 0.05):
        super().__init__()
        self.dim = dim

        h, w, d = img_shape
        d = d // 2 + 1
        self.ffp_module = WSLayer(dim, h, w, d)
        self.alpha = nn.Parameter(
            torch.full((dim, 1, 1, 1), alpha_init, dtype=torch.float32)
        )

    def forward(self, x):
        assert x.shape[1] == self.dim
        x_spec = self.ffp_module(x)
        return x + self.alpha * x_spec


class UnetrUpBlockF(nn.Module):
    """
    An upsampling module that can be used for UNETR: "Hatamizadeh et al.,
    UNETR: Transformers for 3D Medical Image Segmentation <https://arxiv.org/abs/2103.10504>"

    Note:
        Adapted from:
        https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/blocks/unetr_block.py#L22
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Sequence[int] | int,
        upsample_kernel_size: Sequence[int] | int,
        norm_name: tuple | str,
        img_size: Sequence[int] | int,
        feature_size: int,
        res_block: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            res_block: bool argument to determine if residual block is used.

        """

        super().__init__()
        factor = out_channels // feature_size
        img_shape = (img_size // factor,) * spatial_dims
        self.spectral_layer = SpectralLayer(out_channels, img_shape)

        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        if res_block:
            self.conv_block = UnetResBlock(
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

        else:
            self.conv_block = UnetBasicBlock(  # type: ignore
                spatial_dims,
                out_channels * 2,
                out_channels,
                kernel_size=kernel_size,
                stride=1,
                norm_name=norm_name,
            )

    def forward(self, inp, skip):
        # number of channels for skip should equals to out_channels
        out = self.transp_conv(inp)
        out = self.spectral_layer(out)
        out = torch.cat((out, skip), dim=1)
        out = self.conv_block(out)
        return out
