import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    """TCN with FFN output head: encoder -> last time step -> FFN.

    Canonical architecture from Bai et al. (2018).  Maps a full-length
    input sequence to a fixed-size output via the last time step's
    representation, then a small FFN (Linear + LeakyReLU + Linear) projects
    it to the output dimension.

    Parameters
    ----------
    input_size : int
        Number of input channels.
    output_size : int
        Output dimension (e.g. ``pred_len`` for forecasting).
    num_channels : list[int]
        Output channels per TCN layer.  Length = number of TemporalBlocks.
        Must satisfy ``receptive_field >= input_sequence_length``.
    kernel_size : int
        Conv kernel size (paper default: 2).
    dropout : float
        Dropout rate in TCN blocks.
    head_hidden : int
        FFN head hidden size ``[num_channels[-1], head_hidden, output_size]``.
    """

    def __init__(self, input_size, output_size, num_channels, kernel_size=2,
                 dropout=0.2, head_hidden=32):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout)
        self.head = nn.Sequential(
            nn.Linear(num_channels[-1], head_hidden),
            nn.LeakyReLU(),
            nn.Linear(head_hidden, output_size),
        )
        self.init_weights()

    def init_weights(self):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x: (B, input_size, L)
        y1 = self.tcn(x)                # (B, num_channels[-1], L)
        return self.head(y1[:, :, -1])  # (B, output_size)
