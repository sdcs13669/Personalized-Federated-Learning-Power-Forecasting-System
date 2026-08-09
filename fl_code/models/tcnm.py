"""Two-stage federated TCN model: global point forecast + local residual correction.

Stage 1 — :class:`GlobalTCN` (FedAvg aggregated):
    Public features (time-derived + category) → TCN encoder → MLP decoder
    → 336-step point forecast ``Y_pre``.

Stage 2 — :class:`ResidualCorrector` (per-client, never shared):
    ``Y_pre`` + recent history + local dynamic features → TCN
    → 3-quantile residual corrections ``E_corr``.

:class:`FedTCN` composes both stages so that
    ``Y_final = Y_pre + E_corr``  (shape ``[B, 336, 3]``).
"""

import torch
import torch.nn as nn

from .tcn import TCNEncoder


# ============================================================================
# Stage 1: Global TCN main model
# ============================================================================

class GlobalTCN(nn.Module):
    """TCN-based global model producing a single point forecast.

    Input ``(B, in_channels, input_steps)`` → output ``(B, pred_len)``.

    Designed to be the federated encoder: only the public features flow
    through this network, and its weights are aggregated via FedAvg.

    Parameters
    ----------
    in_channels : int
        Number of public feature channels (e.g. 8: 7 time-derived + 1 category).
    pred_len : int
        Forecast horizon in steps (e.g. 336 for 7 days @ 30 min).
    input_steps : int
        Input window length in steps (e.g. 1440 for 30 days @ 30 min).
    hidden_channels : int | list[int]
        Per-layer channel count for the TCN encoder.
    out_channels : int
        TCN encoder output channels.
    num_layers : int
        Number of TemporalBlocks.
    kernel_size : int
        Conv kernel size.
    dilation_base : int
        Dilation factor base.
    dropout : float
        Dropout rate in TCN blocks.
    use_weight_norm : bool
    use_batch_norm : bool
    decoder_hidden : list[int]
        Hidden sizes of the MLP that maps encoder summary → ``pred_len``.
    """

    def __init__(
        self,
        in_channels=8,
        pred_len=336,
        input_steps=1440,
        hidden_channels=64,
        out_channels=64,
        num_layers=4,
        kernel_size=3,
        dilation_base=2,
        dropout=0.2,
        use_weight_norm=True,
        use_batch_norm=True,
        decoder_hidden=(256,),
    ):
        super().__init__()
        self.in_channels = in_channels
        self.pred_len = pred_len
        self.input_steps = input_steps

        self.encoder = TCNEncoder(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_base=dilation_base,
            dropout=dropout,
            use_weight_norm=use_weight_norm,
            use_batch_norm=use_batch_norm,
        )

        # Decoder: encoder summary → point forecast
        dec_dims = [out_channels] + list(decoder_hidden) + [pred_len]
        dec_layers = []
        for i in range(len(dec_dims) - 1):
            dec_layers.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
            if i < len(dec_dims) - 2:
                dec_layers.append(nn.ReLU())
                dec_layers.append(nn.Dropout(dropout))
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x_public):
        """Forward pass.

        Parameters
        ----------
        x_public : Tensor, shape ``(B, in_channels, input_steps)``
            Public features (time-derived + category_id).

        Returns
        -------
        Tensor of shape ``(B, pred_len)`` — point forecast ``Y_pre``.
        """
        enc = self.encoder(x_public)                # (B, out_channels, input_steps)
        summary = enc.mean(dim=-1)                   # (B, out_channels)  global pooling
        y_pre = self.decoder(summary)                # (B, pred_len)
        return y_pre

    def state_dict_for_aggregation(self):
        """Return state dict for federated aggregation (same as encoder)."""
        return self.state_dict()


# ============================================================================
# Stage 2: Local residual corrector
# ============================================================================

class ResidualCorrector(nn.Module):
    """Per-client residual correction network — **never shared**.

    Takes three sources concatenated along the **feature** dimension:

    - ``Y_pre``: global model point forecast       ``(B, pred_len, 1)``
    - ``recent_history``: true target before forecast start ``(B, pred_len, 1)``
    - ``x_local_dynamic``: local exogenous features ``(B, pred_len, D_local)``

    and outputs 3-quantile residual corrections ``(B, pred_len, 3)``.

    A lightweight TCN encoder maps the combined channels to three output
    channels (one per quantile).

    Parameters
    ----------
    pred_len : int
        Forecast horizon (same as :class:`GlobalTCN` — 336).
    local_feat_dim : int
        Number of dynamic local feature channels *per dataset*
        (0 for eld_ind, 1 for lcl_res, 5 for tetouan, 6 for steel_ind).
    quantiles : tuple[float]
        Quantile levels, default ``(0.1, 0.5, 0.9)``.
    hidden_channels : int
        Channels in the correction TCN.
    num_layers : int
        Correction TCN depth (keep shallow to avoid overfitting on small clients).
    kernel_size : int
    dropout : float
    """

    quantiles: tuple[float, ...]

    def __init__(
        self,
        pred_len=336,
        local_feat_dim=0,
        quantiles=(0.1, 0.5, 0.9),
        hidden_channels=32,
        num_layers=3,
        kernel_size=5,
        dropout=0.1,
    ):
        super().__init__()
        self.pred_len = pred_len
        self.local_feat_dim = local_feat_dim
        self.quantiles = tuple(quantiles)
        self.num_quantiles = len(quantiles)

        in_ch = 2 + local_feat_dim                       # Y_pre + history + local
        self.corrector = TCNEncoder(
            in_channels=in_ch,
            hidden_channels=hidden_channels,
            out_channels=self.num_quantiles,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dilation_base=2,
            dropout=dropout,
            use_weight_norm=False,
            use_batch_norm=True,
        )

    def forward(self, y_pre, recent_history, x_local_dynamic=None):
        """Forward pass.

        Parameters
        ----------
        y_pre : Tensor, shape ``(B, pred_len)``
            Global model point forecast.
        recent_history : Tensor, shape ``(B, pred_len)``
            True target values for ``pred_len`` steps *before* the forecast start.
        x_local_dynamic : Tensor or None, shape ``(B, pred_len, D_local)``
            Local dynamic features over the same ``pred_len`` window as
            ``recent_history``.

        Returns
        -------
        Tensor of shape ``(B, pred_len, num_quantiles)`` — residual corrections.
        """
        # Build (B, in_ch, pred_len) input
        parts = [
            y_pre.unsqueeze(1),                          # (B, 1, pred_len)
            recent_history.unsqueeze(1),                 # (B, 1, pred_len)
        ]
        if self.local_feat_dim > 0 and x_local_dynamic is not None:
            # x_local_dynamic: (B, pred_len, D_local) → (B, D_local, pred_len)
            parts.append(x_local_dynamic.transpose(1, 2))

        x = torch.cat(parts, dim=1)                      # (B, in_ch, pred_len)
        out = self.corrector(x)                           # (B, num_quantiles, pred_len)
        return out.transpose(1, 2)                        # (B, pred_len, num_quantiles)

    def state_dict_for_local(self):
        """Return state dict for local-only storage (never uploaded)."""
        return self.state_dict()


# ============================================================================
# Stage 1 + 2 combined
# ============================================================================

class FedTCN(nn.Module):
    """Complete federated TCN: global forecast + local residual correction.

    Usage::

        model = FedTCN(global_config, corrector_config)
        y_final, y_pre, e_corr = model(x_public, recent_history, x_local)

    ``get_global_state()`` / ``get_local_state()`` split the state for FL.
    """

    def __init__(self, global_config, corrector_config):
        super().__init__()
        self.global_model = GlobalTCN(**global_config)
        self.corrector = ResidualCorrector(**corrector_config)

    def forward(self, x_public, recent_history, x_local_dynamic=None):
        """Full two-stage forward pass.

        Parameters
        ----------
        x_public : Tensor, shape ``(B, in_channels, input_steps)``
            Public features for the global model.
        recent_history : Tensor, shape ``(B, pred_len)``
            True target history of ``pred_len`` steps before forecast.
        x_local_dynamic : Tensor or None, shape ``(B, pred_len, D_local)``
            Local dynamic features.

        Returns
        -------
        y_final : Tensor, shape ``(B, pred_len, num_quantiles)``
            Final quantile forecast = Y_pre + E_corr.
        y_pre : Tensor, shape ``(B, pred_len)``
            Global point forecast (for baseline evaluation).
        e_corr : Tensor, shape ``(B, pred_len, num_quantiles)``
            Residual corrections (for interpretability).
        """
        y_pre = self.global_model(x_public)                       # (B, pred_len)
        e_corr = self.corrector(y_pre, recent_history, x_local_dynamic)  # (B, pred_len, 3)
        y_final = y_pre.unsqueeze(-1) + e_corr                     # (B, pred_len, 3)
        return y_final, y_pre, e_corr

    def get_global_state(self):
        """State dict for the global model (FedAvg aggregated)."""
        return self.global_model.state_dict()

    def get_local_state(self):
        """State dict for the local corrector (never shared)."""
        return self.corrector.state_dict()


# ============================================================================
# Loss
# ============================================================================

def quantile_loss(y_true, y_pred_quantiles, quantiles=(0.1, 0.5, 0.9)):
    """Pinball / quantile loss.

    Parameters
    ----------
    y_true : Tensor, shape ``(B, pred_len)``
        Ground-truth target.
    y_pred_quantiles : Tensor, shape ``(B, pred_len, num_quantiles)``
        Predicted quantile values.
    quantiles : tuple[float]
        Quantile levels.

    Returns
    -------
    Scalar loss.
    """
    errors = y_true.unsqueeze(-1) - y_pred_quantiles            # (B, pred_len, nq)
    q = torch.tensor(quantiles, device=y_pred_quantiles.device, dtype=y_pred_quantiles.dtype)
    loss = torch.max((q - 1) * errors, q * errors)
    return loss.mean()
