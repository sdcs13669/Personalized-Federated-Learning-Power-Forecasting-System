"""Dataclass-based model configuration and factory functions.

Matches the architecture defined in ``fl_code/models/config.yaml``:

    Global TCN (FedAvg)  →  Y_pre  (point forecast)
         +
    Local Residual Corrector (per-client)  →  E_corr  (quantile corrections)
         =
    Y_final = Y_pre + E_corr
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ============================================================================
# Config dataclasses
# ============================================================================

@dataclass
class GlobalTCNConfig:
    """Configuration for :class:`GlobalTCN` — the federated point-forecast model.

    Matches ``config.yaml §2 main_model``.
    """

    # Input / output
    in_channels: int = 8           # public features: 7 time-derived + 1 category
    input_steps: int = 1440        # 30 days @ 30 min
    pred_len: int = 336            # 7 days @ 30 min

    # TCN encoder
    hidden_channels: int | list[int] = 64
    out_channels: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.2
    use_weight_norm: bool = True
    use_batch_norm: bool = True

    # Decoder MLP (encoder summary → pred_len)
    decoder_hidden: tuple[int, ...] = (256,)

    def to_dict(self):
        return asdict(self)


@dataclass
class CorrectorConfig:
    """Configuration for :class:`ResidualCorrector` — per-client, never shared.

    Matches ``config.yaml §3 residual_corrector``.
    """

    pred_len: int = 336
    local_feat_dim: int = 0       # varies per dataset: 0..6
    quantiles: tuple[float, ...] = (0.1, 0.5, 0.9)

    # Correction TCN (lighter than the global encoder)
    hidden_channels: int = 32
    num_layers: int = 3
    kernel_size: int = 5
    dropout: float = 0.1

    def to_dict(self):
        return asdict(self)


@dataclass
class FedTCNConfig:
    """Top-level configuration combining global model + local corrector.

    Matches the complete ``config.yaml``.
    """

    global_model: GlobalTCNConfig = field(default_factory=GlobalTCNConfig)
    corrector: CorrectorConfig = field(default_factory=CorrectorConfig)

    # Metadata (from config.yaml §1)
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

def build_fed_tcn(config: FedTCNConfig):
    """Build the complete two-stage :class:`FedTCN`.

    Returns a :class:`FedTCN` instance ready for training.
    """
    from .tcnm import FedTCN

    global_kwargs = config.global_model.to_dict()
    corr_kwargs = config.corrector.to_dict()

    return FedTCN(
        global_config=global_kwargs,
        corrector_config=corr_kwargs,
    )


def build_global_model(config: FedTCNConfig):
    """Build only the global :class:`GlobalTCN` (for baseline evaluation).

    Returns a standalone :class:`GlobalTCN` that outputs ``Y_pre`` only,
    without the local corrector.
    """
    from .tcnm import GlobalTCN

    kwargs = config.global_model.to_dict()
    return GlobalTCN(**kwargs)


def build_local_corrector(config: FedTCNConfig, local_feat_dim: int):
    """Build only the :class:`ResidualCorrector` for a specific client.

    Parameters
    ----------
    config : FedTCNConfig
        Top-level config.
    local_feat_dim : int
        Number of local dynamic feature channels for *this* client
        (differs by dataset: 0 for eld_ind, 1 for lcl_res, 5 for tetouan, 6 for steel_ind).

    Returns
    -------
    :class:`ResidualCorrector`
    """
    from .tcnm import ResidualCorrector

    kwargs = config.corrector.to_dict()
    kwargs["local_feat_dim"] = local_feat_dim
    return ResidualCorrector(**kwargs)
