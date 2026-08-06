import torch
import torch.nn as nn

from .tcn import TCNEncoder
from .mlp import MLP


class TCNM(nn.Module):
    """Shared TCN encoder + local MLP prediction head.

    The encoder extracts temporal features from public input features.  Its
    output is flattened and concatenated with local features (if any) before
    the head produces quantile forecasts.

    The prediction head is built lazily on the first forward pass so the
    encoder's output sequence length does not need to be known up front.

    Parameters
    ----------
    encoder_config : dict
        Keyword arguments forwarded to :class:`TCNEncoder`.
    head_hidden_dims : list[int]
        Hidden layer sizes for the prediction head MLP.
    pred_len : int
        Number of future time steps to predict (e.g. 96 for 24 h @ 15 min).
    num_quantiles : int
        Number of quantiles to output (default 3 → P10, P50, P90).
    local_feat_dim : int
        Dimensionality of local-only features appended before the head.
        Set to 0 when no local features are available.
    head_activation : str
        Activation for the head MLP (``relu``, ``gelu``, etc.).
    head_dropout : float
        Dropout rate for the head MLP.
    head_use_batch_norm : bool
        Whether the head MLP uses batch normalisation.
    """

    def __init__(
        self,
        encoder_config,
        head_hidden_dims,
        pred_len,
        num_quantiles=3,
        local_feat_dim=0,
        head_activation="relu",
        head_dropout=0.0,
        head_use_batch_norm=False,
    ):
        super().__init__()
        self.encoder = TCNEncoder(**encoder_config)
        self.pred_len = pred_len
        self.num_quantiles = num_quantiles
        self.local_feat_dim = local_feat_dim
        self.encoder_out_channels = encoder_config.get("out_channels", 64)

        # Store head config; the actual MLP is built on first forward.
        self.head_config = dict(
            hidden_dims=head_hidden_dims,
            out_features=num_quantiles * pred_len,
            activation=head_activation,
            dropout=head_dropout,
            use_batch_norm=head_use_batch_norm,
        )
        self.head = None

    def _build_head(self, seq_len, device):
        head_in_dim = self.encoder_out_channels * seq_len + self.local_feat_dim
        self.head = MLP(in_features=head_in_dim, **self.head_config).to(device)

    def forward(self, x_public, x_local=None):
        """Forward pass.

        Parameters
        ----------
        x_public : Tensor
            Public features, shape (N, C_public, L).
        x_local : Tensor or None
            Local features, shape (N, C_local).  Ignored when
            ``local_feat_dim == 0``.

        Returns
        -------
        Tensor of shape (N, num_quantiles, pred_len).
        """
        enc_out = self.encoder(x_public)  # (N, C_enc, L)
        enc_flat = enc_out.flatten(1)      # (N, C_enc * L)

        if self.head is None:
            self._build_head(enc_out.size(-1), enc_flat.device)

        if self.local_feat_dim > 0 and x_local is not None:
            combined = torch.cat([enc_flat, x_local], dim=1)
        else:
            combined = enc_flat

        out = self.head(combined)  # (N, num_quantiles * pred_len)
        return out.view(out.size(0), self.num_quantiles, self.pred_len)

    def get_encoder_state(self):
        """Return encoder state dict for federated aggregation."""
        return self.encoder.state_dict()

    def get_head_state(self):
        """Return local head state dict (never shared)."""
        if self.head is None:
            raise RuntimeError("Head not built yet; call forward first.")
        return self.head.state_dict()


class TCNMFixed(nn.Module):
    """Like :class:`TCNM` but ``seq_len`` is required at
    construction time so the head is built eagerly.  Prefer this when the
    input sequence length is fixed and known up front."""

    def __init__(
        self,
        encoder_config,
        seq_len,
        head_hidden_dims,
        pred_len,
        num_quantiles=3,
        local_feat_dim=0,
        head_activation="relu",
        head_dropout=0.0,
        head_use_batch_norm=False,
    ):
        super().__init__()
        self.encoder = TCNEncoder(**encoder_config)
        self.pred_len = pred_len
        self.num_quantiles = num_quantiles
        self.local_feat_dim = local_feat_dim

        enc_out_channels = encoder_config.get("out_channels", 64)
        head_in_dim = enc_out_channels * seq_len + local_feat_dim

        self.head = MLP(
            in_features=head_in_dim,
            hidden_dims=head_hidden_dims,
            out_features=num_quantiles * pred_len,
            activation=head_activation,
            dropout=head_dropout,
            use_batch_norm=head_use_batch_norm,
        )

    def forward(self, x_public, x_local=None):
        enc_out = self.encoder(x_public)  # (N, C_enc, L)
        enc_flat = enc_out.flatten(1)      # (N, C_enc * L)

        if x_local is not None:
            combined = torch.cat([enc_flat, x_local], dim=1)
        else:
            combined = enc_flat

        out = self.head(combined)
        return out.view(out.size(0), self.num_quantiles, self.pred_len)

    def get_encoder_state(self):
        return self.encoder.state_dict()

    def get_head_state(self):
        return self.head.state_dict()
