"""
Loss functions for Peptide WAE training.
"""

import torch
import torch.nn.functional as F

from deepchemography.peptides.config import PAD_IDX

# Re-export shared losses for backward compatibility
from deepchemography.shared.losses import (  # noqa: F401
    kl_gaussianprior,
    wae_mmd_gaussianprior,
    mmd_full_kernel,
    compute_mmd_kernel,
    mmd_random_features,
)


def reconstruction_loss(sequences, logits):
    """
    Compute reconstruction error (NLL of next-timestep predictions).

    Args:
        sequences: Input sequences (mbsize x seq_len)
        logits: Decoder outputs (mbsize x seq_len x vocab_size)

    Returns:
        Reconstruction loss
    """
    mbsize = sequences.size(0)
    pad_words = torch.LongTensor(mbsize, 1).fill_(PAD_IDX).to(sequences.device)
    dec_targets = torch.cat([sequences[:, 1:], pad_words], dim=1)

    recon_loss = F.cross_entropy(
        logits.view(-1, logits.size(2)),
        dec_targets.view(-1),
        reduction='mean',
        ignore_index=PAD_IDX,
    )
    return recon_loss
