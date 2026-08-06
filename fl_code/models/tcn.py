import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class CausalConv1d(nn.Module):
    """1D causal convolution: output[t] depends only on input[≤t]."""

    def __init__(self, in_channels, out_channels, kernel_size, dilation, **conv_kwargs):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            dilation=dilation, padding=self.pad, **conv_kwargs,
        )

    def forward(self, x):
        # x: (N, C, L)
        x = self.conv(x)
        if self.pad > 0:
            x = x[..., : -self.pad]
        return x


class TemporalBlock(nn.Module):
    """Single TCN block: two causal conv layers with residual connection."""

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        dilation,
        dropout,
        use_weight_norm,
        use_batch_norm,
    ):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)

        if use_weight_norm:
            self.conv1.conv = weight_norm(self.conv1.conv)
            self.conv2.conv = weight_norm(self.conv2.conv)

        self.norm1 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
        self.norm2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        out = self.relu(self.norm1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.norm2(self.conv2(out)))
        out = self.dropout(out)
        residual = self.downsample(x)
        return self.relu(out + residual)


class TCNEncoder(nn.Module):
    """Temporal Convolutional Network encoder.

    Parameters
    ----------
    in_channels : int
        Number of input features.
    hidden_channels : int | list[int]
        Channel size per layer. If int, the same size is used for all layers.
    out_channels : int
        Output feature dimension (per time step).
    num_layers : int
        Number of TemporalBlock layers. Each layer contains two causal convs.
    kernel_size : int
        Convolution kernel size.
    dilation_base : int
        Dilation factor at layer i is dilation_base ** i (e.g. 2 → 1,2,4,8,...).
    dropout : float
        Dropout probability.
    use_weight_norm : bool
        Apply weight normalisation to conv layers.
    use_batch_norm : bool
        Apply batch normalisation after each conv.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels=64,
        out_channels=64,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        use_weight_norm=True,
        use_batch_norm=True,
    ):
        super().__init__()
        if isinstance(hidden_channels, int):
            hidden_channels = [hidden_channels] * num_layers
        if len(hidden_channels) != num_layers:
            raise ValueError(
                f"hidden_channels length ({len(hidden_channels)}) must match "
                f"num_layers ({num_layers})"
            )

        ch_in = in_channels
        self.blocks = nn.ModuleList()
        for i, ch_out in enumerate(hidden_channels):
            self.blocks.append(
                TemporalBlock(
                    in_channels=ch_in,
                    out_channels=ch_out,
                    kernel_size=kernel_size,
                    dilation=dilation_base ** i,
                    dropout=dropout,
                    use_weight_norm=use_weight_norm,
                    use_batch_norm=use_batch_norm,
                )
            )
            ch_in = ch_out

        self.proj = nn.Conv1d(ch_in, out_channels, 1)

    @property
    def receptive_field(self):
        """Theoretical receptive field size of the full encoder."""
        rf = 1
        for i, block in enumerate(self.blocks):
            dilation = block.conv1.conv.dilation[0]
            k = block.conv1.conv.kernel_size[0]
            rf += 2 * (k - 1) * dilation  # two causal convs per block
        return rf

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Shape (N, in_channels, L) — batch, features, time steps.

        Returns
        -------
        Tensor of shape (N, out_channels, L).
        """
        for block in self.blocks:
            x = block(x)
        return self.proj(x)
