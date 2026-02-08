"""SMILES autoencoder module for molecular representations."""

from deepchemography.smiles.model import LSTMAutoencoder
from deepchemography.smiles.trainer import AutoencoderTrainer
from deepchemography.smiles.config import get_parser
from deepchemography.smiles.vocab import OneHotVocab, CharVocab, SpecialTokens
from deepchemography.smiles.wae import SmilesWAE
from deepchemography.smiles.wae_config import get_default_wae_config
from deepchemography.smiles.wae_trainer import SmilesWAETrainer
from deepchemography.smiles.api import (
    load_smiles_wae,
    encode_smiles,
    decode_latent as decode_smiles_latent,
    sample_smiles,
    interpolate_smiles,
    reconstruct_smiles,
    explore_neighborhood as explore_smiles_neighborhood,
)

__all__ = [
    # Existing LSTM autoencoder
    'LSTMAutoencoder',
    'AutoencoderTrainer',
    'get_parser',
    'OneHotVocab',
    'CharVocab',
    'SpecialTokens',
    # SMILES WAE
    'SmilesWAE',
    'get_default_wae_config',
    'SmilesWAETrainer',
    'load_smiles_wae',
    'encode_smiles',
    'decode_smiles_latent',
    'sample_smiles',
    'interpolate_smiles',
    'reconstruct_smiles',
    'explore_smiles_neighborhood',
]
