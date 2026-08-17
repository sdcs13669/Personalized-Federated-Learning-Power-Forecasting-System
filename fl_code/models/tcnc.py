"""Two-stage federated model: Global TCN + Local Residual Corrector."""

import torch
import torch.nn as nn


class TCNC(nn.Module):
    """Two-stage federated model: Global TCN + Local Residual Corrector.

    Stage 1 (shared, FedAvg aggregated) → ``Y_pre`` (point forecast).
    Stage 2 (per-client, never shared)    → ``E_corr`` (quantile corrections).

    Final output: ``Y_final = Y_pre + E_corr``.

    Usage::

        global_model = TCN(...)
        corrector = TCNRC(...)
        model = TCNC(global_model, corrector)
        y_final, y_pre, e_corr = model(x_public, residual_history, x_window)
    """

    def __init__(self, global_model, corrector):
        super().__init__()
        self.global_model = global_model
        self.corrector = corrector

    def forward(self, x_public, residual_history, x_window):
        """Full two-stage forward pass.

        Parameters
        ----------
        x_public : Tensor, shape ``(B, in_channels, input_steps)``
            Public features + historical load for the global model.
        residual_history : Tensor, shape ``(B, pred_len)``
            Residual from previous forecast cycle.
        x_window : Tensor, shape ``(B, in_channels + D_local, input_steps)``
            Full input window (public + load + local features) for the
            Corrector's window encoder.

        Returns
        -------
        y_final : Tensor, shape ``(B, pred_len, num_quantiles)``
            Final quantile forecast.
        y_pre : Tensor, shape ``(B, pred_len)``
            Global point forecast (baseline).
        e_corr : Tensor, shape ``(B, pred_len, num_quantiles)``
            Residual corrections.
        """
        y_pre = self.global_model(x_public)
        e_corr = self.corrector(y_pre, residual_history, x_window)
        y_final = y_pre.unsqueeze(-1) + e_corr
        return y_final, y_pre, e_corr

    def get_global_state(self):
        """State dict for the global model (FedAvg aggregated)."""
        return self.global_model.state_dict()

    def get_local_state(self):
        """State dict for the local corrector (never shared)."""
        return self.corrector.state_dict()
