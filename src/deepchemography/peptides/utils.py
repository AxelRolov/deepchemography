"""
Utility functions for peptide WAE.
"""

import torch
import torch.nn as nn

from deepchemography.peptides.config import UNK_IDX, PAD_IDX, START_IDX, EOS_IDX

__all__ = ['UNK_IDX', 'PAD_IDX', 'START_IDX', 'EOS_IDX', 'soft_embed', 'onehot_embed']


def onehot_embed(hardIx, vocabSize):
    """Get tensor hardIx (mbsize), return its one hot embedding (mbsize x vocabSize)."""
    assert hardIx.dim() == 1, 'expecting 1D tensor: minibatch of indices.'
    softIx = torch.zeros(hardIx.size(0), vocabSize).to(hardIx.device)
    softIx.scatter_(1, hardIx.unsqueeze(1), 1.0)
    return softIx


def soft_embed(embed, softIx):
    """Soft embedding using matrix multiplication."""
    assert isinstance(embed, nn.Embedding), 'Expecting nn.Embedding'
    out = softIx @ embed.weight  # MMult: [mbsize x vocab] * [vocab x emb_dim]
    return out
