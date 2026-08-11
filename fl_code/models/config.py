"""Dataclass model configuration and factory functions.

Matches ``fl_code/models/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal


@dataclass
class TCNConfig:
    """Global TCN point-forecast model.

    Receptive field with defaults:
        rf = 1 + 2*(k-1)*(2^L - 1) = 1 + 2*(2-1)*(2^10 - 1) = 2047 >= 1440
    """

    in_channels: int = 9             # 8 public features + 1 historical load
    input_steps: int = 1440          # 30 days @ 30 min
    pred_len: int = 336              # 7 days @ 30 min

    num_channels: tuple[int, ...] = (64,) * 10  # 10 layers, 64ch each
    kernel_size: int = 2
    dropout: float = 0.2

    def to_dict(self):
        return asdict(self)


@dataclass
class CorrectorConfig:
    """Per-client residual corrector (never shared).

    Supports three architectures selected by ``rc_type``:

    - ``"mlp"``  — :class:`MLPRC`:  per-step MLP, no temporal interaction
    - ``"lstm"`` — :class:`LSTMRC`: lightweight LSTM, sequential modelling
    - ``"tcn"``  — :class:`TCNRC`:  causal TCN (rf=511 >= 336)
    """

    rc_type: Literal["mlp", "lstm", "tcn"] = "tcn"

    pred_len: int = 336
    local_feat_dim: int = 0          # varies per dataset
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    dropout: float = 0.1

    # MLP settings (used when rc_type="mlp")
    hidden_dims: tuple[int, ...] = (64, 32)

    # LSTM settings (used when rc_type="lstm")
    lstm_hidden_size: int = 32
    lstm_num_layers: int = 1

    # TCN settings (used when rc_type="tcn")
    num_channels: tuple[int, ...] = (16,) * 8  # 8 layers, rf=511
    kernel_size: int = 2

    def to_dict(self):
        return asdict(self)


@dataclass
class TCNCConfig:
    """Top-level two-stage federated config."""

    global_model: TCNConfig = field(default_factory=TCNConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)

    # Metadata
    time_step_minutes: int = 30
    input_window_steps: int = 1440
    output_window_steps: int = 336
    quantile_loss_quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    def to_dict(self):
        d = asdict(self)
        d["global_model"] = self.global_model.to_dict()
        d["corrector"] = self.corrector.to_dict()
        return d


# ============================================================================
# Factory functions
# ============================================================================

def build_tcn(config: TCNConfig):
    """Build a :class:`TCN` for point forecasting (Phase 2 baseline)."""
    from .tcn import TCN

    return TCN(
        input_size=config.in_channels,
        output_size=config.pred_len,
        num_channels=list(config.num_channels),
        kernel_size=config.kernel_size,
        dropout=config.dropout,
    )


def build_corrector(config: CorrectorConfig):
    """Build a :class:`MLPRC` or :class:`TCNRC` based on ``rc_type`` (Phase 3)."""
    from .rc import MLPRC, LSTMRC, TCNRC

    if config.rc_type == "mlp":
        return MLPRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        )
    elif config.rc_type == "lstm":
        return LSTMRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout,
        )
    else:
        return TCNRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            num_channels=config.num_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
        )


def build_fed_tcn(config: TCNCConfig):
    """Build the full two-stage :class:`TCNC` (Phase 3)."""
    from .tcn import TCN
    from .tcnc import TCNC

    global_model = TCN(
        input_size=config.global_model.in_channels,
        output_size=config.global_model.pred_len,
        num_channels=list(config.global_model.num_channels),
        kernel_size=config.global_model.kernel_size,
        dropout=config.global_model.dropout,
    )

    corrector = build_corrector(config.corrector)

    return TCNC(global_model, corrector)
