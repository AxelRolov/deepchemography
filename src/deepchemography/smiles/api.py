"""
High-level API for SMILES WAE model.

Mirrors peptides/api.py with functions for loading models, encoding,
decoding, sampling, and interpolation.
"""

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch

from deepchemography.smiles.wae import SmilesWAE
from deepchemography.smiles.vocab import OneHotVocab
from deepchemography.smiles.wae_config import get_default_wae_config

logger = logging.getLogger(__name__)


def load_smiles_wae(
    model_dir: str,
    device: str = 'cpu',
    config: dict = None,
) -> SmilesWAE:
    """
    Load a trained SMILES WAE model from a checkpoint directory.

    Expects model_dir to contain:
    - config.pt (model config dict)
    - vocab.pt (OneHotVocab instance)
    - model_best.pt or model state dict

    Args:
        model_dir: Path to checkpoint directory
        device: Device to load model on ('cpu' or 'cuda')
        config: Override config (if None, loads from config.pt or uses defaults)

    Returns:
        Loaded SmilesWAE model (in eval mode, with vocab stored as model.vocabulary)
    """
    model_dir = Path(model_dir)

    # Load vocab
    vocab_path = model_dir / 'vocab.pt'
    if vocab_path.exists():
        vocab = torch.load(vocab_path, map_location='cpu')
    else:
        raise FileNotFoundError(f"Vocab file not found at {vocab_path}")

    # Load config
    if config is None:
        config_path = model_dir / 'config.pt'
        if config_path.exists():
            config = torch.load(config_path, map_location='cpu')
        else:
            logger.warning("config.pt not found, using default WAE config")
            config = get_default_wae_config()

    # Create and load model
    model = SmilesWAE(vocab, config)

    model_path = model_dir / 'model_best.pt'
    if not model_path.exists():
        # Try other common names
        for name in ['model.pt', 'state_dict.pt']:
            alt = model_dir / name
            if alt.exists():
                model_path = alt
                break

    if model_path.exists():
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded model from {model_path}")
    else:
        logger.warning(f"No model weights found in {model_dir}")

    model = model.to(device)
    model.eval()
    return model


def encode_smiles(
    model: SmilesWAE,
    smiles: str,
) -> torch.Tensor:
    """
    Encode a SMILES string to its latent representation (mu).

    Args:
        model: SmilesWAE model
        smiles: SMILES string

    Returns:
        Latent vector (1 x z_dim)
    """
    tensor = model.string2tensor(smiles)
    return model.encode([tensor])


def decode_latent(
    model: SmilesWAE,
    z: torch.Tensor,
    temp: float = 1.0,
    decode: str = 'sample',
) -> List[str]:
    """
    Decode latent vectors to SMILES strings.

    Args:
        model: SmilesWAE model
        z: Latent vectors (n x z_dim)
        temp: Temperature for sampling
        decode: 'greedy' or 'sample'

    Returns:
        List of SMILES strings
    """
    z = z.to(model.device)
    return model.sample(n_batch=z.size(0), z=z, temp=temp, decode=decode)


def sample_smiles(
    model: SmilesWAE,
    n_samples: int = 10,
    latent_std: float = 1.0,
    temp: float = 1.0,
    decode: str = 'sample',
) -> List[str]:
    """
    Sample SMILES from the prior distribution N(0, I).

    Args:
        model: SmilesWAE model
        n_samples: Number of samples
        latent_std: Standard deviation for latent sampling
        temp: Temperature for decoding
        decode: 'greedy' or 'sample'

    Returns:
        List of generated SMILES strings
    """
    with torch.no_grad():
        z = torch.randn(n_samples, model.z_dim).to(model.device) * latent_std
    return model.sample(n_batch=n_samples, z=z, temp=temp, decode=decode)


def interpolate_smiles(
    model: SmilesWAE,
    smi1: str,
    smi2: str,
    n_steps: int = 10,
    method: str = 'linear',
    temp: float = 1.0,
    decode: str = 'sample',
) -> Tuple[List[str], List[float]]:
    """
    Interpolate between two SMILES in latent space.

    Args:
        model: SmilesWAE model
        smi1: First SMILES string
        smi2: Second SMILES string
        n_steps: Number of interpolation steps (excluding endpoints)
        method: 'linear' or 'slerp'
        temp: Temperature for decoding
        decode: 'greedy' or 'sample'

    Returns:
        sequences: List of interpolated SMILES (including endpoints)
        weights: Interpolation weights
    """
    with torch.no_grad():
        z1 = encode_smiles(model, smi1)
        z2 = encode_smiles(model, smi2)

        z1_np = z1.cpu().numpy()
        z2_np = z2.cpu().numpy()

        weights = [0.0] + [1.0 / (n_steps + 1) * i for i in range(1, n_steps + 1)] + [1.0]

        z_list = [z1_np]
        for w in weights[1:-1]:
            if method == 'linear':
                z_interp = (1 - w) * z1_np + w * z2_np
            elif method == 'slerp':
                z1_norm = z1_np / np.linalg.norm(z1_np)
                z2_norm = z2_np / np.linalg.norm(z2_np)
                omega = np.arccos(np.clip(
                    np.dot(z1_norm.flatten(), z2_norm.flatten()), -1, 1
                ))
                if np.abs(omega) < 1e-6:
                    z_interp = (1 - w) * z1_np + w * z2_np
                else:
                    z_interp = (np.sin((1 - w) * omega) * z1_np +
                                np.sin(w * omega) * z2_np) / np.sin(omega)
            else:
                raise ValueError(f"Unknown interpolation method: {method}")
            z_list.append(z_interp)
        z_list.append(z2_np)

        z_tensor = torch.tensor(
            np.vstack(z_list), dtype=torch.float32
        ).to(model.device)
        sequences = decode_latent(model, z_tensor, temp=temp, decode=decode)

    return sequences, weights


def reconstruct_smiles(
    model: SmilesWAE,
    smiles: str,
    decode: str = 'greedy',
    temp: float = 0.1,
) -> str:
    """
    Reconstruct a SMILES string by encoding and decoding.

    Args:
        model: SmilesWAE model
        smiles: Input SMILES string
        decode: 'greedy' or 'sample'
        temp: Temperature for sampling

    Returns:
        Reconstructed SMILES string
    """
    z = encode_smiles(model, smiles)
    result = decode_latent(model, z, temp=temp, decode=decode)
    return result[0]


def explore_neighborhood(
    model: SmilesWAE,
    smiles: str,
    noise_scale: float = 0.1,
    n_neighbors: int = 10,
    temp: float = 1.0,
    decode: str = 'sample',
) -> List[str]:
    """
    Generate SMILES similar to input by exploring latent neighborhood.

    Args:
        model: SmilesWAE model
        smiles: Base SMILES string
        noise_scale: Standard deviation of noise to add to z
        n_neighbors: Number of neighbors to generate
        temp: Temperature for decoding
        decode: 'greedy' or 'sample'

    Returns:
        List of neighbor SMILES strings
    """
    with torch.no_grad():
        z_base = encode_smiles(model, smiles)
        noise = torch.randn(n_neighbors, model.z_dim).to(model.device) * noise_scale
        z_neighbors = z_base.expand(n_neighbors, -1) + noise

    return decode_latent(model, z_neighbors, temp=temp, decode=decode)
