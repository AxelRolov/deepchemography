"""
Loss functions for Peptide WAE training.
"""

import math

import torch
import torch.nn.functional as F

from deepchemography.peptides.config import PAD_IDX


def kl_gaussianprior(mu, logvar):
    """Analytically compute KL divergence with unit Gaussian."""
    return torch.mean(0.5 * torch.sum((logvar.exp() + mu ** 2 - 1 - logvar), 1))


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


def wae_mmd_gaussianprior(z, sigma=7.0, kernel='gaussian'):
    """
    Compute MMD with samples from unit Gaussian (WAE loss).

    Args:
        z: Latent vectors (mbsize x z_dim)
        sigma: Kernel width
        kernel: Kernel type ('gaussian', 'laplace', 'energy')

    Returns:
        MMD loss
    """
    z_prior = torch.randn_like(z)
    return mmd_full_kernel(z, z_prior, sigma=sigma, kernel=kernel)


def mmd_full_kernel(z1, z2, sigma=7.0, kernel='gaussian'):
    """
    Compute full kernel MMD between two distributions.

    Args:
        z1: Samples from first distribution (N x d)
        z2: Samples from second distribution (M x d)
        sigma: Kernel width
        kernel: Kernel type

    Returns:
        Unbiased MMD estimate
    """
    K11 = compute_mmd_kernel(z1, z1, sigma=sigma, kernel=kernel)
    K22 = compute_mmd_kernel(z2, z2, sigma=sigma, kernel=kernel)
    K12 = compute_mmd_kernel(z1, z2, sigma=sigma, kernel=kernel)

    N = z1.size(0)
    assert N == z2.size(0), 'Expected matching sizes z1 z2'

    # Gretton 2012 eq (4)
    H = K11 + K22 - K12 * 2

    # Unbiased: remove diagonal
    H = H - torch.diag(torch.diag(H))

    loss = 1.0 / (N * (N - 1)) * H.sum()
    return loss


def compute_mmd_kernel(x, y, sigma=7.0, kernel='gaussian'):
    """
    Compute kernel matrix between x and y.

    Args:
        x: First set of samples (N x d)
        y: Second set of samples (M x d)
        sigma: Kernel width
        kernel: Kernel type

    Returns:
        Kernel matrix (N x M)
    """
    x_i = x.unsqueeze(1)
    y_j = y.unsqueeze(0)
    xmy = ((x_i - y_j) ** 2).sum(2)

    if kernel == "gaussian":
        K = torch.exp(-xmy / sigma ** 2)
    elif kernel == "laplace":
        K = torch.exp(-torch.sqrt(xmy + sigma ** 2))
    elif kernel == "energy":
        K = torch.pow(xmy + sigma ** 2, -0.25)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    return K


# Random features MMD (more efficient for high dimensions)
_rf_cache = {}


def mmd_random_features(z1, z2, sigma=7.0, rf_dim=500, rf_resample=False):
    """
    Compute MMD using random features approximation.

    Args:
        z1: Samples from first distribution
        z2: Samples from second distribution
        sigma: Kernel width
        rf_dim: Number of random features
        rf_resample: Whether to resample random features each call

    Returns:
        MMD loss
    """
    global _rf_cache

    if 'gaussian' not in _rf_cache or rf_resample:
        rf_w = torch.randn((z1.shape[1], rf_dim), device=z1.device)
        rf_b = math.pi * 2 * torch.rand((rf_dim,), device=z1.device)
        _rf_cache['gaussian'] = (rf_w, rf_b)
    else:
        rf_w, rf_b = _rf_cache['gaussian']
        rf_w = rf_w.to(z1.device)
        rf_b = rf_b.to(z1.device)

    def compute_rf(z):
        z_emb = (z @ rf_w) / sigma + rf_b
        z_emb = torch.cos(z_emb) * (2.0 / rf_dim) ** 0.5
        return z_emb.mean(0)

    mu1 = compute_rf(z1)
    mu2 = compute_rf(z2)

    loss = ((mu1 - mu2) ** 2).sum()
    return loss
