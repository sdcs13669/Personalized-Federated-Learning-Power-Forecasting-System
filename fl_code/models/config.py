"""Dataclass-based model configuration and factory functions.

Every model parameter lives in a dataclass so it can be serialised, validated,
and overridden without digging through constructor kwargs.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


# ---------------------------------------------------------------------------
# Config dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TCNConfig:
    """Configuration for :class:`TCNEncoder`."""

    in_channels: int
    hidden_channels: int | list[int] = 64
    out_channels: int = 64
    num_layers: int = 4
    kernel_size: int = 3
    dilation_base: int = 2
    dropout: float = 0.2
    use_weight_norm: bool = True
    use_batch_norm: bool = True

    def to_dict(self):
        return asdict(self)


@dataclass
class MLPConfig:
    """Configuration for :class:`MLP`."""

    in_features: int | None = None  # None → set at build time
    hidden_dims: list[int] = field(default_factory=lambda: [128, 64])
    out_features: int | None = None  # None → set at build time
    activation: str = "relu"
    dropout: float = 0.0
    use_batch_norm: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class EncoderHeadConfig:
    """Configuration for :class:`EncoderHeadModel` / :class:`EncoderHeadModelFixed`.

    ``seq_len`` is optional; when provided the head is built eagerly
    (:class:`EncoderHeadModelFixed`).  When ``None`` the head is built lazily
    on the first forward pass.
    """

    encoder: TCNConfig = field(default_factory=TCNConfig)
    head: MLPConfig = field(default_factory=MLPConfig)
    seq_len: int | None = None
    pred_len: int = 96
    num_quantiles: int = 3
    local_feat_dim: int = 0

    def to_dict(self):
        d = asdict(self)
        d["encoder"] = self.encoder.to_dict()
        d["head"] = self.head.to_dict()
        return d


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_encoder_head(config: EncoderHeadConfig):
    """Build an Encoder-Head model from a config dataclass.

    Returns :class:`EncoderHeadModelFixed` when ``config.seq_len`` is set,
    otherwise :class:`EncoderHeadModel`.
    """
    encoder_kwargs = _strip_none(config.encoder.to_dict())

    if config.seq_len is not None:
        from .tcnm import EncoderHeadModelFixed

        return EncoderHeadModelFixed(
            encoder_config=encoder_kwargs,
            seq_len=config.seq_len,
            head_hidden_dims=config.head.hidden_dims,
            pred_len=config.pred_len,
            num_quantiles=config.num_quantiles,
            local_feat_dim=config.local_feat_dim,
            head_activation=config.head.activation,
            head_dropout=config.head.dropout,
            head_use_batch_norm=config.head.use_batch_norm,
        )

    from .tcnm import EncoderHeadModel

    return EncoderHeadModel(
        encoder_config=encoder_kwargs,
        head_hidden_dims=config.head.hidden_dims,
        pred_len=config.pred_len,
        num_quantiles=config.num_quantiles,
        local_feat_dim=config.local_feat_dim,
        head_activation=config.head.activation,
        head_dropout=config.head.dropout,
        head_use_batch_norm=config.head.use_batch_norm,
    )


def build_local_model(config: EncoderHeadConfig):
    """Build a standalone local model (TCN + head, no federation split).

    Same architecture as the encoder-head model but intended as the baseline:
    each client trains this entirely on its own data.  The model is identical
    to the federated version so comparisons are fair — the only difference is
    whether the encoder weights come from federation or from local SGD.
    """
    return build_encoder_head(config)


def _strip_none(d: dict) -> dict:
    return {k: v for k, v in d.items() if v is not None}
