import torch
import torch.nn as nn
import einops
from einops.layers.torch import Rearrange
import pdb

from .helpers import (
    SinusoidalPosEmb,
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
    def __init__(self, num_features):
        super().__init__()
        self.num_features = num_features
        self.norm = nn.LayerNorm(num_features)

    def forward(self, x, scale, shift):
        x = self.norm(x)
        return x * scale + shift

class ResidualTemporalBlock(nn.Module):

    def __init__(self, inp_channels, out_channels, embed_dim, horizon, kernel_size=5):
        super().__init__()

        self.blocks = nn.ModuleList([
            Conv1dBlock(inp_channels, out_channels, kernel_size),
            Conv1dBlock(out_channels, out_channels, kernel_size),
        ])

        self.time_mlp = nn.Sequential(
            nn.Mish(),
            nn.Linear(embed_dim, out_channels),
            Rearrange('batch t -> batch t 1'),
        )

        self.adaptive_norm = AdaptiveGroupNorm(out_channels)
        self.residual_conv = nn.Conv1d(inp_channels, out_channels, 1) \
            if inp_channels != out_channels else nn.Identity()

    def forward(self, x, t, scale, shift):
        '''
            x : [ batch_size x inp_channels x horizon ]
            t : [ batch_size x embed_dim ]
            scale : [batch_size x out_channels x horizon]
            shift : [batch_size x out_channels x horizon]
            returns:
            out : [ batch_size x out_channels x horizon ]
        '''
        out = self.blocks[0](x) + self.time_mlp(t)
        out = self.blocks[1](out)
        out = self.adaptive_norm(out, scale, shift)
        return out + self.residual_conv(x)

class TemporalUnetIMLE(nn.Module):
    def __init__(
        self,
        horizon,
        input_dim,       # Should match z_dim
        output_dim,      # Should match observation_dim + action_dim
        cond_dim,        # Dimension of s_t
        dim=32,
        dim_mults=(1, 2, 4, 8),
        attention=False,
        state_normalizer=None,
    ):
        super().__init__()

        self.horizon = horizon
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.cond_dim = cond_dim
        self.state_normalizer = state_normalizer

        # Calculate channel dimensions for each level of the U-Net
        dims = [input_dim + cond_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        print(f'[ models/temporal ] Channel dimensions: {in_out}')

        # Adaptive MLP to generate scale and shift parameters for each block
        total_blocks = len(in_out) * 2 + 2  # Two blocks per down/up level + two mid blocks
        self.adaptive_mlp = AdaptiveMLP(input_dim=cond_dim, output_dim=2 * sum(d[1] for d in in_out) * 2 + 2 * dims[-1] * 2)

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # Downsampling blocks
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind == (len(in_out) - 1)
            self.downs.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, embed_dim=self.time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, embed_dim=self.time_dim, horizon=horizon),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))) if attention else nn.Identity(),
                Downsample1d(dim_out) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon // 2

        # Middle blocks
        mid_dim = dims[-1]
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=self.time_dim, horizon=horizon)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim))) if attention else nn.Identity()
        self.mid_block2 = ResidualTemporalBlock(mid_dim, mid_dim, embed_dim=self.time_dim, horizon=horizon)

        # Upsampling blocks
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[:-1])):
            is_last = ind == (len(in_out) - 1)
            self.ups.append(nn.ModuleList([
                ResidualTemporalBlock(dim_out * 2, dim_in, embed_dim=self.time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_in, dim_in, horizon=horizon, embed_dim=self.time_dim, horizon=horizon),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))) if attention else nn.Identity(),
                Upsample1d(dim_in) if not is_last else nn.Identity()
            ]))
            if not is_last:
                horizon = horizon * 2

        # Final convolution layer
        self.final_conv = nn.Sequential(
            Conv1dBlock(dim, dim, kernel_size=5),
            nn.Conv1d(dim, output_dim, 1),
        )

    def forward(self, x, cond):
        '''
        x    : [batch_size, horizon, input_dim]
        cond : [batch_size, cond_dim]
        '''

        batch_size = x.size(0)

        # Normalize conditioning variable if state_normalizer is provided
        if self.state_normalizer is not None:
            cond = self.state_normalizer.normalize(cond.cpu().numpy())
            cond = torch.tensor(cond, dtype=torch.float32, device=x.device)

        # Expand cond along horizon dimension for concatenation
        cond_expanded = cond.unsqueeze(1).expand(-1, self.horizon, -1)  # [batch_size, horizon, cond_dim]

        # Concatenate x and cond along feature dimension
        x = torch.cat([x, cond_expanded], dim=-1)  # [batch_size, horizon, input_dim + cond_dim]

        # Permute x for Conv1d: [batch_size, channels, horizon]
        x = x.permute(0, 2, 1)  # [batch_size, channels, horizon]

        # Generate scale and shift parameters from adaptive MLP
        scale_shift = self.adaptive_mlp(cond)  # [batch_size, total_scale_shift_dim]
        scale_shift_idx = 0

        # Prepare list to store intermediate activations
        h = []

        # Function to extract scale and shift for each block
        def get_scale_shift(dim_out):
            nonlocal scale_shift_idx
            scale_shift_block = scale_shift[:, scale_shift_idx:scale_shift_idx + 2 * dim_out]
            scale_shift_idx += 2 * dim_out
            scale, shift = torch.chunk(scale_shift_block, 2, dim=-1)
            scale = scale.unsqueeze(-1)  # [batch_size, dim_out, 1]
            shift = shift.unsqueeze(-1)  # [batch_size, dim_out, 1]
            return scale, shift

        # Downsampling blocks
        for i, (resnet1, resnet2, attn, downsample) in enumerate(self.downs):
            dim_out = resnet1.out_channels

            # Get scale and shift for resnet1
            scale, shift = get_scale_shift(dim_out)
            x = resnet1(x, scale, shift)

            # Get scale and shift for resnet2
            scale, shift = get_scale_shift(dim_out)
            x = resnet2(x, scale, shift)

            x = attn(x)
            h.append(x)
            x = downsample(x)

        # Middle blocks
        mid_dim = self.mid_block1.out_channels

        # Get scale and shift for mid_block1
        scale, shift = get_scale_shift(mid_dim)
        x = self.mid_block1(x, scale, shift)

        x = self.mid_attn(x)

        # Get scale and shift for mid_block2
        scale, shift = get_scale_shift(mid_dim)
        x = self.mid_block2(x, scale, shift)

        # Upsampling blocks
        for i, (resnet1, resnet2, attn, upsample) in enumerate(self.ups):
            x = torch.cat((x, h.pop()), dim=1)  # Concatenate with skip connection

            dim_in = resnet1.out_channels

            # Get scale and shift for resnet1
            scale, shift = get_scale_shift(dim_in)
            x = resnet1(x, scale, shift)

            # Get scale and shift for resnet2
            scale, shift = get_scale_shift(dim_in)
            x = resnet2(x, scale, shift)

            x = attn(x)
            x = upsample(x)

        # Final convolution
        x = self.final_conv(x)  # [batch_size, output_dim, horizon]

        # Permute x back to [batch_size, horizon, output_dim]
        x = x.permute(0, 2, 1)  # [batch_size, horizon, output_dim]

        return x


class ValueFunction(nn.Module):

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

        time_dim = dim
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

        self.blocks = nn.ModuleList([])
        num_resolutions = len(in_out)

        print(in_out)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.blocks.append(nn.ModuleList([
                ResidualTemporalBlock(dim_in, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                ResidualTemporalBlock(dim_out, dim_out, kernel_size=5, embed_dim=time_dim, horizon=horizon),
                Downsample1d(dim_out)
            ]))

            if not is_last:
                horizon = horizon // 2

        mid_dim = dims[-1]
        mid_dim_2 = mid_dim // 2
        mid_dim_3 = mid_dim // 4
        ##
        self.mid_block1 = ResidualTemporalBlock(mid_dim, mid_dim_2, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down1 = Downsample1d(mid_dim_2)
        horizon = horizon // 2
        ##
        self.mid_block2 = ResidualTemporalBlock(mid_dim_2, mid_dim_3, kernel_size=5, embed_dim=time_dim, horizon=horizon)
        self.mid_down2 = Downsample1d(mid_dim_3)
        horizon = horizon // 2
        ##
        fc_dim = mid_dim_3 * max(horizon, 1)

        self.final_block = nn.Sequential(
            nn.Linear(fc_dim + time_dim, fc_dim // 2),
            nn.Mish(),
            nn.Linear(fc_dim // 2, out_dim),
        )

    def forward(self, x, cond, time, *args):
        '''
            x : [ batch x horizon x transition ]
        '''

        x = einops.rearrange(x, 'b h t -> b t h')

        ## mask out first conditioning timestep, since this is not sampled by the model
        # x[:, :, 0] = 0

        t = self.time_mlp(time)

        for resnet, resnet2, downsample in self.blocks:
            x = resnet(x, t)
            x = resnet2(x, t)
            x = downsample(x)

        ##
        x = self.mid_block1(x, t)
        x = self.mid_down1(x)
        ##
        x = self.mid_block2(x, t)
        x = self.mid_down2(x)
        ##
        x = x.view(len(x), -1)
        out = self.final_block(torch.cat([x, t], dim=-1))
        return out
