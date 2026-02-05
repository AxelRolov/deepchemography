"""
High-level API for Peptide WAE model.

Provides functions for loading models, encoding, decoding, sampling,
and interpolation.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from deepchemography.peptides.model import PeptideWAE
from deepchemography.peptides.vocab import PeptideVocab
from deepchemography.peptides.config import get_default_config

logger = logging.getLogger(__name__)


def load_peptide_model(
    model_path: str,
    vocab_path: str = None,
    device: str = 'cpu',
    config: dict = None,
) -> Tuple[PeptideWAE, PeptideVocab]:
    """
    Load a trained Peptide WAE model.

    Args:
        model_path: Path to model checkpoint (.pt file)
        vocab_path: Path to vocabulary file (if None, looks for vocab.dict in same dir)
        device: Device to load model on ('cpu' or 'cuda')
        config: Model configuration (if None, uses defaults)

    Returns:
        model: Loaded PeptideWAE model
        vocab: PeptideVocab instance
    """
    model_path = Path(model_path)

    # Find vocab file
    if vocab_path is None:
        vocab_path = model_path.parent / 'vocab.dict'
        if not vocab_path.exists():
            logger.warning(f"Vocab file not found at {vocab_path}, using default vocab")
            vocab = PeptideVocab.from_default()
        else:
            vocab = PeptideVocab(str(vocab_path))
    else:
        vocab = PeptideVocab(vocab_path)

    # Get config
    if config is None:
        config = get_default_config()

    # Create model
    model = PeptideWAE(
        n_vocab=vocab.size(),
        max_seq_len=config['max_seq_len'],
        z_dim=config['z_dim'],
        c_dim=config['c_dim'],
        emb_dim=config['emb_dim'],
        encoder_config=config['encoder'],
        decoder_config=config['decoder'],
    )

    # Load weights
    state_dict = torch.load(model_path, map_location=device)

    # Filter out classifier weights if present (we don't use classifier)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('classifier')}

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from {model_path}")
    return model, vocab


def encode_peptide(
    model: PeptideWAE,
    vocab: PeptideVocab,
    sequence: str,
    sample_q: str = 'max',
) -> torch.Tensor:
    """
    Encode a peptide sequence to latent space.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        sequence: Space-separated amino acid sequence
        sample_q: 'max' to use mean, or int for number of samples

    Returns:
        Latent vector(s) (1 x z_dim) or (sample_q x z_dim)
    """
    enc_inputs = vocab.to_ix(sequence)
    enc_inputs = enc_inputs.to(model.device)

    with torch.no_grad():
        mu, logvar = model.forward_encoder(enc_inputs)

        if sample_q == 'max':
            z = mu
        else:
            z = torch.cat([model.sample_z(mu, logvar) for _ in range(sample_q)], dim=0)

    return z


def decode_latent(
    model: PeptideWAE,
    vocab: PeptideVocab,
    z: torch.Tensor,
    c: torch.Tensor = None,
    sample_mode: str = 'categorical',
    temp: float = 1.0,
    print_special_tokens: bool = False,
) -> List[str]:
    """
    Decode latent vectors to peptide sequences.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        z: Latent vectors (n x z_dim)
        c: Condition vectors (n x c_dim), or None to sample from prior
        sample_mode: 'categorical' or 'greedy'
        temp: Temperature for sampling
        print_special_tokens: Whether to include special tokens

    Returns:
        List of peptide sequences
    """
    n_samples = z.size(0)
    z = z.to(model.device)

    if c is None:
        c = model.sample_c_prior(n_samples)
    else:
        c = c.to(model.device)

    with torch.no_grad():
        samples, _, _ = model.generate_sentences(
            n_samples,
            z=z,
            c=c,
            sample_mode=sample_mode,
            temp=temp,
        )

    # Convert to strings
    predictions = []
    for sample in samples:
        seq_str = vocab.to_string(sample, print_special_tokens=print_special_tokens)
        predictions.append(seq_str)

    return predictions


def sample_peptides(
    model: PeptideWAE,
    vocab: PeptideVocab,
    n_samples: int = 10,
    latent_std: float = 1.0,
    sample_mode: str = 'categorical',
    temp: float = 1.0,
    print_special_tokens: bool = False,
) -> List[str]:
    """
    Sample peptides from the prior distribution.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        n_samples: Number of samples to generate
        latent_std: Standard deviation for latent sampling
        sample_mode: 'categorical' or 'greedy'
        temp: Temperature for sampling
        print_special_tokens: Whether to include special tokens

    Returns:
        List of generated peptide sequences
    """
    with torch.no_grad():
        z = torch.randn(n_samples, model.z_dim).to(model.device) * latent_std
        c = model.sample_c_prior(n_samples)

        samples, _, _ = model.generate_sentences(
            n_samples,
            z=z,
            c=c,
            sample_mode=sample_mode,
            temp=temp,
        )

    predictions = []
    for sample in samples:
        seq_str = vocab.to_string(sample, print_special_tokens=print_special_tokens)
        predictions.append(seq_str)

    return predictions


def interpolate_peptides(
    model: PeptideWAE,
    vocab: PeptideVocab,
    seq1: str,
    seq2: str,
    n_steps: int = 10,
    method: str = 'linear',
    sample_mode: str = 'categorical',
    temp: float = 1.0,
) -> Tuple[List[str], List[float]]:
    """
    Interpolate between two peptide sequences in latent space.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        seq1: First peptide sequence
        seq2: Second peptide sequence
        n_steps: Number of interpolation steps (excluding endpoints)
        method: 'linear', 'slerp', or 'tanh'
        sample_mode: 'categorical' or 'greedy'
        temp: Temperature for sampling

    Returns:
        sequences: List of interpolated sequences (including endpoints)
        weights: Interpolation weights
    """
    with torch.no_grad():
        z1 = encode_peptide(model, vocab, seq1, sample_q='max')
        z2 = encode_peptide(model, vocab, seq2, sample_q='max')

        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()

        # Compute interpolation weights
        weights = [0.0] + [1.0 / (n_steps + 1) * i for i in range(1, n_steps + 1)] + [1.0]

        # Interpolate
        z_list = [z1_np]
        for w in weights[1:-1]:
            if method == 'linear':
                z_interp = (1 - w) * z1_np + w * z2_np
            elif method == 'slerp':
                # Spherical linear interpolation
                z1_norm = z1_np / np.linalg.norm(z1_np)
                z2_norm = z2_np / np.linalg.norm(z2_np)
                omega = np.arccos(np.clip(np.dot(z1_norm.flatten(), z2_norm.flatten()), -1, 1))
                if np.abs(omega) < 1e-6:
                    z_interp = (1 - w) * z1_np + w * z2_np
                else:
                    z_interp = (np.sin((1 - w) * omega) * z1_np + np.sin(w * omega) * z2_np) / np.sin(omega)
            elif method == 'tanh':
                # Tanh interpolation for smoother transitions
                w_tanh = (np.tanh(w * 4 - 2) + 1) / 2
                z_interp = (1 - w_tanh) * z1_np + w_tanh * z2_np
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            z_list.append(z_interp)
        z_list.append(z2_np)

        # Decode
        z_tensor = torch.tensor(np.vstack(z_list), dtype=torch.float32).to(model.device)
        sequences = decode_latent(model, vocab, z_tensor, sample_mode=sample_mode, temp=temp)

    return sequences, weights


def reconstruct_peptide(
    model: PeptideWAE,
    vocab: PeptideVocab,
    sequence: str,
    sample_mode: str = 'greedy',
    temp: float = 0.1,
) -> str:
    """
    Reconstruct a peptide by encoding and decoding.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        sequence: Input peptide sequence
        sample_mode: 'categorical' or 'greedy'
        temp: Temperature for sampling

    Returns:
        Reconstructed peptide sequence
    """
    z = encode_peptide(model, vocab, sequence, sample_q='max')
    reconstructed = decode_latent(model, vocab, z, sample_mode=sample_mode, temp=temp)
    return reconstructed[0]


def explore_neighborhood(
    model: PeptideWAE,
    vocab: PeptideVocab,
    sequence: str,
    noise_scale: float = 0.1,
    n_neighbors: int = 10,
    sample_mode: str = 'categorical',
    temp: float = 1.0,
) -> List[str]:
    """
    Generate peptides similar to input by exploring latent neighborhood.

    Args:
        model: PeptideWAE model
        vocab: PeptideVocab instance
        sequence: Base peptide sequence
        noise_scale: Standard deviation of noise to add
        n_neighbors: Number of neighbors to generate
        sample_mode: 'categorical' or 'greedy'
        temp: Temperature for sampling

    Returns:
        List of neighbor peptide sequences
    """
    with torch.no_grad():
        z_base = encode_peptide(model, vocab, sequence, sample_q='max')

        # Add noise
        noise = torch.randn(n_neighbors, model.z_dim).to(model.device) * noise_scale
        z_neighbors = z_base.expand(n_neighbors, -1) + noise

        neighbors = decode_latent(model, vocab, z_neighbors, sample_mode=sample_mode, temp=temp)

    return neighbors
