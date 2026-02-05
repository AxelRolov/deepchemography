"""
GRU Encoder for Peptide WAE.
"""

import torch
import torch.nn as nn


class GRUEncoder(nn.Module):
    """
    Encoder is GRU with FC layers connected to last hidden unit.
    Outputs mu and logvar for reparameterization.
    """

    def __init__(
        self,
        emb_dim: int,
        h_dim: int,
        z_dim: int,
        biGRU: bool = True,
        layers: int = 1,
        p_dropout: float = 0.0,
    ):
        super(GRUEncoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=h_dim,
            num_layers=layers,
            dropout=p_dropout if layers > 1 else 0,
            bidirectional=biGRU,
            batch_first=True,
        )
        # Bidirectional GRU has 2*hidden_state
        self.biGRU_factor = 2 if biGRU else 1
        self.biGRU = biGRU

        # Reparameterization layers
        self.q_mu = nn.Linear(self.biGRU_factor * h_dim, z_dim)
        self.q_logvar = nn.Linear(self.biGRU_factor * h_dim, z_dim)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Embeddings of shape (mbsize x seq_len x emb_dim)

        Returns:
            mu: Mean of latent distribution (mbsize x z_dim)
            logvar: Log variance of latent distribution (mbsize x z_dim)
        """
        _, h = self.rnn(x, None)

        if self.biGRU:
            # Concatenates features from Forward and Backward
            # Uses the highest layer representation
            h = torch.cat((h[-2, :, :], h[-1, :, :]), 1)

        # Forward to latent
        h = h.view(-1, h.shape[-1])
        mu = self.q_mu(h)
        logvar = self.q_logvar(h)

        return mu, logvar
