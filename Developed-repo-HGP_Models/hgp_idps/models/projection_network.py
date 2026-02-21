"""
Deep-Kernel Projection Network  (Section IV.A, Eq. 25-27)
=========================================================

Implements the neural-network feature extractor  π : X^(d) → Z  that maps
heterogeneous, domain-specific inputs into a shared latent space before the
GP layer.  Architecture: [D_d, 128, 64, 32] with SiLU activations, batch-
normalisation, and dropout (p = 0.1).
"""

import torch
import torch.nn as nn
from typing import List, Optional


class ProjectionNetwork(nn.Module):
    """Three-layer deep-kernel projection  π(x^(d)) → z ∈ R^{d_z}.

    Parameters
    ----------
    input_dim : int
        Raw feature dimensionality D_d (varies per domain).
    hidden_dims : list[int], default [128, 64, 32]
        Widths of the hidden layers.
    activation : str, default "silu"
        Non-linearity ("silu", "relu", "gelu").
    dropout : float, default 0.1
        Dropout rate applied after each hidden layer.
    batch_norm : bool, default True
        Whether to apply BatchNorm1d before each activation.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Optional[List[int]] = None,
        activation: str = "silu",
        dropout: float = 0.1,
        batch_norm: bool = True,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [128, 64, 32]
        self.output_dim = hidden_dims[-1]

        act_fn = {"silu": nn.SiLU, "relu": nn.ReLU, "gelu": nn.GELU}[activation]

        layers: list[nn.Module] = []
        prev = input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev, dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(act_fn())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = dim

        self.net = nn.Sequential(*layers)

        # Xavier initialisation
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw features to latent representation."""
        return self.net(x)
