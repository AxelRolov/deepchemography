"""
DeepChemography: Deep learning autoencoders for molecular sequences.

This package provides autoencoder models for:
- SMILES: Small molecule representations (LSTMAutoencoder)
- Peptides: Amino acid sequences (PeptideWAE)
"""

__version__ = "0.2.0"

# SMILES autoencoder
from deepchemography.smiles import (
    LSTMAutoencoder,
    AutoencoderTrainer,
    OneHotVocab,
    CharVocab,
    get_parser,
)

# Peptide WAE
from deepchemography.peptides import (
    PeptideWAE,
    PeptideVocab,
    load_peptide_model,
    encode_peptide,
    sample_peptides,
    decode_latent,
    interpolate_peptides,
    reconstruct_peptide,
)

# Shared utilities
from deepchemography.shared import (
    Logger,
    setup_logging,
    set_seed,
)

# Script utilities
from deepchemography.script_utils import (
    read_smiles_csv,
    add_common_args,
    add_train_args,
)

__all__ = [
    # SMILES
    'LSTMAutoencoder',
    'AutoencoderTrainer',
    'OneHotVocab',
    'CharVocab',
    'get_parser',
    # Peptides
    'PeptideWAE',
    'PeptideVocab',
    'load_peptide_model',
    'encode_peptide',
    'sample_peptides',
    'decode_latent',
    'interpolate_peptides',
    'reconstruct_peptide',
    # Shared
    'Logger',
    'setup_logging',
    'set_seed',
    # Script utils
    'read_smiles_csv',
    'add_common_args',
    'add_train_args',
]
