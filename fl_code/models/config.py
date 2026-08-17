"""Dataclass model configuration and factory functions.

Matches ``fl_code/models/config.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Literal

from ..config import INPUT_STEPS, PRED_LEN


@dataclass
class TCNConfig:
    """Global TCN point-forecast model.

    Receptive field with defaults:
        rf = 1 + 2*(k-1)*(2^L - 1) = 1 + 2*(2-1)*(2^7 - 1) = 255 >= 144
    """

    in_channels: int = 11            # 10 public (7 time + 3 one-hot) + 1 historical load
    input_steps: int = INPUT_STEPS   # from fl_code/config.py
    pred_len: int = PRED_LEN         # from fl_code/config.py

    num_channels: tuple[int, ...] = (64,) * 7  # 7 layers, 64ch each
    kernel_size: int = 2
    dropout: float = 0.2
    head_hidden: int = 32            # FFN head: [64, 32, pred_len] + LeakyReLU

    def to_dict(self):
        return asdict(self)


@dataclass
class CorrectorConfig:
    """Per-client residual corrector (never shared).

    All three architectures share the same per-step input layout:

        [y_pre (1), window context (window_ctx_dim), prev residual (1)]

    where the window context is a conv-encoded summary of the full Global-TCN
    input window (public + load + local channels).  Supports three
    architectures selected by ``rc_type``:

    - ``"mlp"``  — :class:`MLPRC`:  per-step MLP, no temporal interaction
    - ``"lstm"`` — :class:`LSTMRC`: FFN-compress + LSTM + FFN head
    - ``"tcn"``  — :class:`TCNRC`:  FFN-compress + causal TCN + FFN head
    """

    rc_type: Literal["mlp", "lstm", "tcn"] = "mlp"   # default: MLP (simplest, fastest)

    pred_len: int = PRED_LEN         # from fl_code/config.py
    local_feat_dim: int = 0          # varies per dataset
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)
    dropout: float = 0.1

    # Window context (shared by all rc types)
    window_in_channels: int = 11     # Global TCN input channels (10 public + 1 load)
    window_ctx_dim: int = 512        # window context vector dimension (configurable)

    # MLP settings (used when rc_type="mlp")
    hidden_dims: tuple[int, ...] = (512, 256, 128, 64, 32)

    # LSTM/TCN shared FFNs (used when rc_type in {"lstm", "tcn"})
    ffn_hidden: tuple[int, ...] = (256, 128)   # per-step compress FFN: [in, 256, 128]
    head_hidden: int = 32                      # output FFN: [seq_out, 32, 3]

    # LSTM settings (used when rc_type="lstm")
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1

    # TCN settings (used when rc_type="tcn")
    num_channels: tuple[int, ...] = (128,) * 4  # 4 layers, rf=31 >= 6
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
    input_window_steps: int = INPUT_STEPS   # from fl_code/config.py
    output_window_steps: int = PRED_LEN     # from fl_code/config.py
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
        head_hidden=config.head_hidden,
    )


def build_corrector(config: CorrectorConfig):
    """Build a :class:`MLPRC` / :class:`LSTMRC` / :class:`TCNRC` (Phase 3)."""
    from .rc import MLPRC, LSTMRC, TCNRC

    if config.rc_type == "mlp":
        return MLPRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
            window_in_channels=config.window_in_channels,
            window_ctx_dim=config.window_ctx_dim,
        )
    elif config.rc_type == "lstm":
        return LSTMRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            hidden_size=config.lstm_hidden_size,
            num_layers=config.lstm_num_layers,
            dropout=config.dropout,
            window_in_channels=config.window_in_channels,
            window_ctx_dim=config.window_ctx_dim,
            ffn_hidden=config.ffn_hidden,
            head_hidden=config.head_hidden,
        )
    else:
        return TCNRC(
            pred_len=config.pred_len,
            local_feat_dim=config.local_feat_dim,
            quantiles=config.quantiles,
            num_channels=config.num_channels,
            kernel_size=config.kernel_size,
            dropout=config.dropout,
            window_in_channels=config.window_in_channels,
            window_ctx_dim=config.window_ctx_dim,
            ffn_hidden=config.ffn_hidden,
            head_hidden=config.head_hidden,
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
