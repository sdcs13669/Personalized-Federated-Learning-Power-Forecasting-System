import torch.nn as nn

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
    "leaky_relu": nn.LeakyReLU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Configurable multi-layer perceptron.

    Parameters
    ----------
    in_features : int
        Input dimension.
    hidden_dims : list[int]
        Hidden layer sizes, e.g. [128, 64].  If empty the MLP is a single
        linear projection.
    out_features : int
        Output dimension.
    activation : str
        One of ``relu``, ``gelu``, ``silu``, ``leaky_relu``, ``tanh``.
    dropout : float
        Dropout probability applied after each hidden layer.
    use_batch_norm : bool
        Insert BatchNorm1d after each hidden linear layer (before activation).
    """

    def __init__(
        self,
        in_features,
        hidden_dims,
        out_features,
        activation="relu",
        dropout=0.0,
        use_batch_norm=False,
    ):
        super().__init__()
        act_cls = _ACTIVATIONS.get(activation)
        if act_cls is None:
            raise ValueError(
                f"Unknown activation '{activation}'. Choices: {list(_ACTIVATIONS)}"
            )

        self.hidden_dims = list(hidden_dims)
        self.out_features = out_features
        self.activation = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        layers = []
        dims = [in_features] + list(hidden_dims) + [out_features]

        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:  # hidden layers
                if use_batch_norm:
                    layers.append(nn.BatchNorm1d(dims[i + 1]))
                layers.append(act_cls())
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
