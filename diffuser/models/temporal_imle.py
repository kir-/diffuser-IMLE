import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    Downsample1d,
    Upsample1d,
    Conv1dBlock,
    Residual,
    PreNorm,
    LinearAttention,
)

class AdaptiveMLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim * 2)  # Outputs scale and shift
        )

    def forward(self, x):
        return self.net(x)

class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_channels, cond_dim):
        super().__init__()
        self.norm = nn.GroupNorm(32, num_channels)  # Standard GroupNorm
        self.mlp = nn.Sequential(
            nn.Linear(cond_dim, num_channels * 2),  # Output scale and shift
            nn.SiLU(),
            nn.Linear(num_channels * 2, num_channels * 2)
        )

    def forward(self, x, scale, shift):
        normalized_x = self.norm(x)

        # Reshape scale and shift for broadcasting
        scale = scale.view(scale.shape[0], -1, *([1] * (normalized_x.dim() - 2)))
        shift = shift.view(shift.shape[0], -1, *([1] * (normalized_x.dim() - 2)))

        return normalized_x * (1 + scale) + shift


class ResidualTemporalBlock(nn.Module):
    def __init__(self, inp_channels, out_channels, cond_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        # Adaptive MLP to generate scale and shift
        self.adaptive_mlp = AdaptiveMLP(input_dim=cond_dim, output_dim=out_channels)

        # Adaptive normalization
        self.adaptive_norm = AdaptiveGroupNorm(out_channels, cond_dim)

        # Residual connection
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, cond):
        """
        x    : [batch_size x inp_channels x horizon]
        cond : [batch_size x cond_dim]
        cond : [batch_size x cond_dim x 2]
        """
        # Generate scale and shift using `AdaptiveMLP`
        scale_shift = self.adaptive_mlp(cond)  # [batch_size, out_channels * 2]
        scale, shift = torch.chunk(scale_shift, 2, dim=-1)  # Split into scale and shift

        # Process the input through the convolutional blocks
        out = self.blocks[0](x)
        out = self.adaptive_norm(out, scale, shift)  # Apply adaptive normalization

        out = self.blocks[1](out)
        out = self.adaptive_norm(out, scale, shift)  # Apply adaptive normalization again

        # Add residual connection
        return out + self.residual_conv(x)

class TemporalUnetIMLE(nn.Module):

    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, cond_dim=cond_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim=cond_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim=cond_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, cond_dim=cond_dim, horizon=horizon)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, cond_dim=cond_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, cond_dim=cond_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))

            if not is_last:
                horizon = horizon * 2

        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, transition_dim, 1),
        )

    def forward(self, x, cond):
        """
        x    : [batch x horizon x transition_dim]
        cond : [batch x cond_dim]
        """

        x = einops.rearrange(x, 'b h t -> b t h')
        h = []

        for resnet, resnet2, attn, downsample in self.downs:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = attn(x)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_attn(x)
        x = self.mid_block2(x, cond)

        for resnet, resnet2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1) 
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = attn(x)
            x = upsample(x)

        x = self.final_conv(x)
        x = einops.rearrange(x, 'b t h -> b h t') 
        return x


class ValueFunctionIMLE(nn.Module):
    def __init__(
        self,
        horizon,
        transition_dim,
        cond_dim,
        dim=32,
        dim_mults=(1, 2, 4, 8),
        out_dim=1,
    ):
        super().__init__()

        dims = [transition_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, cond_dim=cond_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, cond_dim=cond_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4

        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, cond_dim=cond_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2

        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, cond_dim=cond_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2

        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + cond_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, *args):
        """
        x    : [batch x horizon x transition]
        cond : [batch x cond_dim]
        """

        x = einops.rearrange(x, 'b h t -> b t h')

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, cond)
            x = resnet2(x, cond)
            x = downsample(x)

        x = self.mid_block1(x, cond)
        x = self.mid_down1(x)

        x = self.mid_block2(x, cond)
        x = self.mid_down2(x)

        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, cond], dim=-1))
        return out
