"""Residual Corrector — per-client personalised model, never shared.

Two implementations:
  - :class:`MLPRC` — lightweight MLP, per time-step independent
  - :class:`TCNRC` — TCN with same-length output (lighter than Global TCN)
"""

import torch
import torch.nn as nn

from .tcn import TemporalConvNet


# ============================================================================
# MLP-based residual corrector
# ============================================================================

class MLPRC(nn.Module):
    """MLP residual corrector — point-wise across time steps.

    Concatenates ``Y_pre`` + ``residual_history`` + ``x_local_dynamic``
    along the feature dimension, then applies the same MLP independently
    at each of the ``pred_len`` time steps.  No temporal interaction
    between steps — simple, fast, and a strong baseline.

    Parameters
    ----------
    pred_len : int
        Forecast horizon.
    local_feat_dim : int
        Number of local dynamic feature channels.
    quantiles : tuple[float]
        Quantile levels.
    hidden_dims : tuple[int, ...]
        Hidden sizes of the MLP.
    dropout : float
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=336,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        hidden_dims=(64, 32),
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)

        in_dim = 2 + local_feat_dim
        dims = [in_dim] + list(hidden_dims) + [self.num_quantiles]
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
        self.mlp = nn.Sequential(*layers)

    def forward(self, y_pre, residual_history, x_local_dynamic=None):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
        residual_history : Tensor, shape ``(B, pred_len)``
        x_local_dynamic : Tensor or None, shape ``(B, pred_len, D_local)``

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)``.
        """
        parts = [
            y_pre.unsqueeze(-1),                    # (B, pred_len, 1)
            residual_history.unsqueeze(-1),         # (B, pred_len, 1)
        ]
        if self.local_feat_dim > 0 and x_local_dynamic is not None:
            parts.append(x_local_dynamic)           # (B, pred_len, D_local)

        x = torch.cat(parts, dim=-1)                # (B, pred_len, in_dim)
        return self.mlp(x)                           # (B, pred_len, num_quantiles)


# ============================================================================
# TCN-based residual corrector
# ============================================================================

class TCNRC(nn.Module):
    """TCN Residual Corrector — causal TCN with same-length output.

    Uses a lightweight :class:`TemporalConvNet` to model temporal
    interactions across the ``pred_len`` window.  Inputs are concatenated
    along the channel dimension.

    Parameters
    ----------
    pred_len : int
        Forecast horizon.
    local_feat_dim : int
        Number of local dynamic feature channels.
    quantiles : tuple[float]
        Quantile levels.
    num_channels : tuple[int, ...]
        Internal TCN channel sizes.  Quantile output channels appended.
    kernel_size : int
    dropout : float
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=336,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        num_channels=(16,) * 8,
        kernel_size=2,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)

        in_ch = 2 + local_feat_dim
        self.corrector = TemporalConvNet(
            num_inputs=in_ch,
            num_channels=list(num_channels) + [self.num_quantiles],
            kernel_size=kernel_size,
            dropout=dropout,
        )

    def forward(self, y_pre, residual_history, x_local_dynamic=None):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
        residual_history : Tensor, shape ``(B, pred_len)``
        x_local_dynamic : Tensor or None, shape ``(B, pred_len, D_local)``

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)``.
        """
        parts = [
            y_pre.unsqueeze(1),                           # (B, 1, pred_len)
            residual_history.unsqueeze(1),                # (B, 1, pred_len)
        ]
        if self.local_feat_dim > 0 and x_local_dynamic is not None:
            parts.append(x_local_dynamic.transpose(1, 2)) # (B, D_local, pred_len)

        x = torch.cat(parts, dim=1)                       # (B, in_ch, pred_len)
        out = self.corrector(x)                            # (B, num_quantiles, pred_len)
        return out.transpose(1, 2)                         # (B, pred_len, num_quantiles)


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
