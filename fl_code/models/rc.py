"""Residual Corrector — per-client personalised model, never shared.

Three implementations share the same input layout::

    per-step input  = [y_pre (1), window context (window_ctx_dim), prev residual (1)]
    x_window        = (B, 11 + D_local, input_steps)   # Global-TCN 窗口 + 本地特征

The window context is produced by a small conv encoder over the full
Global-TCN input window (public + load + local channels), then broadcast
to every predicted step — so all three models see the historical pattern
of the current input window:

  - :class:`MLPRC`  — per-step MLP, no temporal interaction
  - :class:`LSTMRC` — FFN-compress → lightweight LSTM → FFN head
  - :class:`TCNRC`  — FFN-compress → causal TCN (same-length output) → FFN head

All FFN / MLP layers use LeakyReLU (default slope 0.01); the TCN TemporalBlock
keeps the canonical ReLU from Bai et al. (2018).
"""

import torch
import torch.nn as nn

from .tcn import TemporalConvNet


def _monotone_quantiles(raw: torch.Tensor) -> torch.Tensor:
    """Monotone quantile parameterisation of the residual corrections.

    ``raw`` holds 3 free values per step; the first is unbounded (corrections
    may be positive or negative) and the next two are non-negative increments
    (softplus), so the output strictly satisfies::

        e_lo <= e_mid <= e_hi   =>   Y_pre + e  is quantile-monotone

    Replaces the plain head output whose quantiles were not guaranteed
    monotone under pinball training.

    Parameters
    ----------
    raw : Tensor, shape ``(..., 3)``

    Returns
    -------
    Tensor of shape ``(..., 3)`` = [e_lo, e_mid, e_hi].
    """
    d = torch.nn.functional.softplus(raw[..., 1:])            # >= 0
    e_lo = raw[..., :1]
    return torch.cat([e_lo, e_lo + torch.cumsum(d, dim=-1)], dim=-1)


def _ffn(dims: list[int], dropout: float = 0.1) -> nn.Sequential:
    """Feed-forward stack: Linear + LeakyReLU + Dropout (no activation on last)."""
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.LeakyReLU())
            layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class _WindowContext(nn.Module):
    """Global-TCN input window ``(B, C, T)`` -> context vector ``(B, ctx_dim)``.

    Two 1-D convs with LeakyReLU + global average pooling over time.  The
    context summarises the historical load / time / local-feature pattern of
    the current input window and is broadcast to every predicted step.
    """

    def __init__(self, in_channels: int, ctx_dim: int = 512):
        super().__init__()
        self.ctx_dim = ctx_dim
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 256, kernel_size=3, padding=1),
            nn.LeakyReLU(),
            nn.Conv1d(256, ctx_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(),
        )

    def forward(self, x_window: torch.Tensor) -> torch.Tensor:
        return self.conv(x_window).mean(dim=2)   # (B, ctx_dim)


def _window_input(y_pre: torch.Tensor, residual_history: torch.Tensor,
                  x_window: torch.Tensor, window_ctx: _WindowContext,
                  pred_len: int) -> torch.Tensor:
    """Build the per-step input ``(B, pred_len, 2 + ctx_dim)``."""
    ctx = window_ctx(x_window)                          # (B, ctx_dim)
    ctx = ctx.unsqueeze(1).expand(-1, pred_len, -1)     # broadcast to all steps
    return torch.cat([
        y_pre.unsqueeze(-1),                            # (B, pred_len, 1)
        ctx,                                            # (B, pred_len, ctx_dim)
        residual_history.unsqueeze(-1),                 # (B, pred_len, 1)
    ], dim=-1)


# ============================================================================
# MLP-based residual corrector
# ============================================================================

class MLPRC(nn.Module):
    """MLP residual corrector — point-wise across time steps.

    Concatenates ``[Y_pre, window context, prev residual]`` along the feature
    dimension, then applies the same MLP independently at each of the
    ``pred_len`` time steps.  No temporal interaction between steps — simple,
    fast, and a strong baseline.

    Parameters
    ----------
    pred_len : int
        Forecast horizon.
    local_feat_dim : int
        Number of local feature channels (included in the window channels).
    quantiles : tuple[float]
        Quantile levels.
    hidden_dims : tuple[int, ...]
        Hidden sizes of the MLP.
    dropout : float
    window_in_channels : int
        Global TCN input channels (10 public + 1 load); window channels are
        ``window_in_channels + local_feat_dim``.
    window_ctx_dim : int
        Dimension of the window context vector (configurable, default 512).
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=6,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        hidden_dims=(512, 256, 128, 64, 32),
        dropout=0.1,
        window_in_channels=11,
        window_ctx_dim=512,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)
        self.window_ctx_dim = window_ctx_dim

        self.window_ctx = _WindowContext(
            window_in_channels + local_feat_dim, window_ctx_dim)
        in_dim = 2 + window_ctx_dim                     # y_pre + ctx + residual
        dims = [in_dim] + list(hidden_dims) + [self.num_quantiles]
        self.mlp = _ffn(dims, dropout)

    def forward(self, y_pre, residual_history, x_window):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
        residual_history : Tensor, shape ``(B, pred_len)``
        x_window : Tensor, shape ``(B, C, input_steps)``

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)``.
        """
        x = _window_input(y_pre, residual_history, x_window,
                          self.window_ctx, self.pred_len)
        raw = self.mlp(x)                                # (B, pred_len, num_quantiles)
        return _monotone_quantiles(raw)                  # e_lo <= e_mid <= e_hi


# ============================================================================
# LSTM-based residual corrector
# ============================================================================

class LSTMRC(nn.Module):
    """LSTM residual corrector — lightweight sequential model.

    Per-step inputs ``[Y_pre, window context, prev residual]`` are compressed
    by a shared FFN (default ``[in, 256, 128]``), fed through a single-layer
    LSTM, and a small FFN head (default ``[128, 32, 3]``) maps each hidden
    state to quantile corrections.

    Parameters
    ----------
    pred_len : int
        Forecast horizon.
    local_feat_dim : int
        Number of local feature channels (included in the window channels).
    quantiles : tuple[float]
        Quantile levels.
    hidden_size : int
        LSTM hidden size (also the FFN head input dimension).
    num_layers : int
        LSTM layers (default 1 for lightweight).
    dropout : float
    window_in_channels : int
        Global TCN input channels (10 public + 1 load).
    window_ctx_dim : int
        Dimension of the window context vector (default 512).
    ffn_hidden : tuple[int, ...]
        Per-step compress FFN hidden sizes ``[in, 256, 128]``.
    head_hidden : int
        Output FFN hidden size ``[seq_out, head_hidden, 3]``.
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=6,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        hidden_size=128,
        num_layers=1,
        dropout=0.1,
        window_in_channels=11,
        window_ctx_dim=512,
        ffn_hidden=(256, 128),
        head_hidden=32,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)
        self.window_ctx_dim = window_ctx_dim

        self.window_ctx = _WindowContext(
            window_in_channels + local_feat_dim, window_ctx_dim)
        in_dim = 2 + window_ctx_dim                     # y_pre + ctx + residual
        self.compress = _ffn([in_dim] + list(ffn_hidden), dropout)
        self.lstm = nn.LSTM(
            input_size=ffn_hidden[-1],
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _ffn([hidden_size, head_hidden, self.num_quantiles], dropout)

    def forward(self, y_pre, residual_history, x_window):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
        residual_history : Tensor, shape ``(B, pred_len)``
        x_window : Tensor, shape ``(B, C, input_steps)``

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)``.
        """
        x = _window_input(y_pre, residual_history, x_window,
                          self.window_ctx, self.pred_len)   # (B, pred_len, in_dim)
        x = self.compress(x)                                # (B, pred_len, 128)
        out, _ = self.lstm(x)                               # (B, pred_len, hidden_size)
        raw = self.head(out)                                # (B, pred_len, num_quantiles)
        return _monotone_quantiles(raw)                     # e_lo <= e_mid <= e_hi


# ============================================================================
# TCN-based residual corrector
# ============================================================================

class TCNRC(nn.Module):
    """TCN Residual Corrector — causal TCN with same-length output.

    Per-step inputs ``[Y_pre, window context, prev residual]`` are compressed
    by a shared FFN (default ``[in, 256, 128]``), modelled by a lightweight
    :class:`TemporalConvNet` across the ``pred_len`` window, and a small FFN
    head (default ``[128, 32, 3]``) maps each step to quantile corrections.

    Parameters
    ----------
    pred_len : int
        Forecast horizon.
    local_feat_dim : int
        Number of local feature channels (included in the window channels).
    quantiles : tuple[float]
        Quantile levels.
    num_channels : tuple[int, ...]
        Internal TCN channel sizes (also the FFN head input dimension).
    kernel_size : int
    dropout : float
    window_in_channels : int
        Global TCN input channels (10 public + 1 load).
    window_ctx_dim : int
        Dimension of the window context vector (default 512).
    ffn_hidden : tuple[int, ...]
        Per-step compress FFN hidden sizes ``[in, 256, 128]``.
    head_hidden : int
        Output FFN hidden size ``[seq_out, head_hidden, 3]``.
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=6,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        num_channels=(128,) * 4,
        kernel_size=2,
        dropout=0.1,
        window_in_channels=11,
        window_ctx_dim=512,
        ffn_hidden=(256, 128),
        head_hidden=32,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)
        self.window_ctx_dim = window_ctx_dim

        self.window_ctx = _WindowContext(
            window_in_channels + local_feat_dim, window_ctx_dim)
        in_dim = 2 + window_ctx_dim                     # y_pre + ctx + residual
        self.compress = _ffn([in_dim] + list(ffn_hidden), dropout)
        self.corrector = TemporalConvNet(
            num_inputs=ffn_hidden[-1],
            num_channels=list(num_channels),
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.head = _ffn([num_channels[-1], head_hidden, self.num_quantiles],
                         dropout)

    def forward(self, y_pre, residual_history, x_window):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
        residual_history : Tensor, shape ``(B, pred_len)``
        x_window : Tensor, shape ``(B, C, input_steps)``

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)``.
        """
        x = _window_input(y_pre, residual_history, x_window,
                          self.window_ctx, self.pred_len)   # (B, pred_len, in_dim)
        x = self.compress(x)                                # (B, pred_len, 128)
        x = x.transpose(1, 2)                               # (B, 128, pred_len)
        out = self.corrector(x)                             # (B, 128, pred_len)
        raw = self.head(out.transpose(1, 2))                # (B, pred_len, num_quantiles)
        return _monotone_quantiles(raw)                     # e_lo <= e_mid <= e_hi


# ============================================================================
# Loss
# ============================================================================

def quantile_loss(y_true, y_pred_quantiles, quantiles=(0.1, 0.5, 0.9)):
    """Pinball / quantile loss for residual correction training.

    Parameters
    ----------
    y_true : Tensor, shape ``(B, pred_len)``
    y_pred_quantiles : Tensor, shape ``(B, pred_len, num_quantiles)``
    quantiles : tuple[float]

    Returns
    -------
    Scalar loss.
    """
    errors = y_true.unsqueeze(-1) - y_pred_quantiles
    q = torch.tensor(quantiles, device=y_pred_quantiles.device, dtype=y_pred_quantiles.dtype)
    loss = torch.max((q - 1) * errors, q * errors)
    return loss.mean()
