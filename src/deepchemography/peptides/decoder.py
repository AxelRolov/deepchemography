"""
GRU Decoder for Peptide WAE.
"""

import numpy as np
import torch
import torch.nn as nn

from deepchemography.peptides.config import UNK_IDX
from deepchemography.peptides.utils import soft_embed


class WordDropout(nn.Module):
    """Word dropout layer that randomly replaces tokens with <unk>."""

    def __init__(self, p_word_dropout: float):
        super(WordDropout, self).__init__()
        self.p = p_word_dropout

    def forward(self, x):
        """
        Apply word dropout: with prob p_word_dropout, set the word to '<unk>'.
        """
        if not self.training or self.p == 0:
            return x

        data = x.clone().detach()

        # Sample masks: elements with val 1 will be set to <unk>
        mask = torch.from_numpy(
            np.random.binomial(1, p=self.p, size=tuple(data.size())).astype('uint8')
        ).to(x.device)

        mask = mask.bool()
        # Set to <unk>
        data[mask] = UNK_IDX

        return data


class GRUDecoder(nn.Module):
    """
    GRU Decoder with word dropout and optional skip connections.
    """

    def __init__(
        self,
        embedding: nn.Embedding,
        emb_dim: int,
        output_dim: int,
        h_dim: int,
        p_word_dropout: float = 0.3,
        p_out_dropout: float = 0.3,
        skip_connections: bool = False,
    ):
        super(GRUDecoder, self).__init__()

        # Reference to word embedding
        self.emb = embedding
        self.rnn = nn.GRU(emb_dim, h_dim, batch_first=True)

        self.fc = nn.Sequential(
            nn.Dropout(p_out_dropout),
            nn.Linear(h_dim, output_dim),
        )
        self.word_dropout = WordDropout(p_word_dropout)

        self.skip_connections = skip_connections
        if self.skip_connections:
            self.skip_weight_x = nn.Linear(h_dim, h_dim, bias=False)
            self.skip_weight_z = nn.Linear(h_dim, h_dim, bias=False)

    def init_hidden(self, z, c):
        """Initialize hidden state from z and c."""
        return torch.cat([z, c], dim=1)

    def forward(self, x, z, c):
        """
        Forward pass with teacher forcing.

        Args:
            x: Input sequences (mbsize x seq_len)
            z: Latent vectors (mbsize x z_dim)
            c: Condition vectors (mbsize x c_dim)

        Returns:
            Logits (mbsize x seq_len x vocab_size)
        """
        mbsize, seq_len = x.shape

        # Initialize hidden state
        init_h = self.init_hidden(z, c)

        # Apply word dropout and embed
        dec_inputs = self.emb(self.word_dropout(x))

        # Expand init_h for concatenation with embeddings
        expanded_init_h = init_h.unsqueeze(1).expand(-1, seq_len, -1)

        # Construct input to RNN: concat embeddings with z,c
        dec_inputs = torch.cat([dec_inputs, expanded_init_h], 2)

        # Compute outputs
        rnn_out, _ = self.rnn(dec_inputs, init_h.unsqueeze(0))

        # Apply skip connection if enabled
        if self.skip_connections:
            rnn_out = self.skip_weight_x(rnn_out) + self.skip_weight_z(expanded_init_h)

        y = self.fc(rnn_out)
        return y

    def forward_sample(self, sampleSoft, sampleHard, z, c, h):
        """
        Forward pass for a single timestep during sampling.

        Args:
            sampleSoft: Soft sample (mbsize x vocab_size) or None
            sampleHard: Hard sample indices (mbsize,)
            z: Latent vectors (mbsize x z_dim)
            c: Condition vectors (mbsize x c_dim)
            h: Hidden state (1 x mbsize x h_dim)

        Returns:
            logits: Output logits (mbsize x vocab_size)
            h: Updated hidden state
        """
        if sampleSoft is not None:
            # With soft indices, gradients pass through
            emb = soft_embed(self.emb, sampleSoft)
        else:
            # With hard indices, no gradients
            emb = self.emb(sampleHard)

        # Concatenate embedding with z and c
        emb = torch.cat([emb, z, c], 1)

        # Insert seqlen dimension
        emb = emb.unsqueeze(1)

        # Forward through RNN
        output, h = self.rnn(emb, h)
        output = output.squeeze(1)

        # Apply skip connection if enabled
        if self.skip_connections:
            latent_code = torch.cat([z, c], 1)
            output = self.skip_weight_x(output) + self.skip_weight_z(latent_code)

        logits = self.fc(output)
        return logits, h
