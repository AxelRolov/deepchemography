# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DeepChemography is a deep learning library for molecular sequence encoding and generation, extracted from the larger ChemEidos project. It contains three autoencoder models:

- **SMILES Autoencoder** (`src/deepchemography/smiles/`): Bidirectional LSTM autoencoder for small molecule SMILES strings. 99.71% reconstruction accuracy. Latent dim: 256.
- **SMILES WAE** (`src/deepchemography/smiles/`): Wasserstein Autoencoder with GRU encoder/decoder for SMILES strings. Uses MMD loss. Latent dim: 128.
- **Peptide WAE** (`src/deepchemography/peptides/`): Wasserstein Autoencoder with GRU encoder/decoder for amino acid sequences. Uses MMD loss. Latent dim: 100.

## Build & Development Commands

```bash
# Install dependencies
uv sync
uv sync --extra dev    # includes pytest, jupyter, matplotlib

# Run tests
uv run pytest

# Run training script
uv run python scripts/train_autoencoder.py --help

# Run a notebook
uv run jupyter notebook notebooks/autoencoder_example.ipynb
```

Python >=3.11 required. Build system uses `hatchling` with `uv` as the package manager.

## Architecture

### Package Layout

```
src/deepchemography/
├── __init__.py              # Re-exports all public API
├── utils.py                 # Backward compat shim (re-exports from smiles/shared)
├── script_utils.py          # CLI helpers: device parsing, argparse, CSV loading
├── shared/                  # Shared utilities
│   ├── logging.py           # Logger, setup_logging
│   ├── losses.py            # MMD and KL losses (shared by both WAEs)
│   └── utils.py             # set_seed
├── smiles/                  # SMILES autoencoders
│   ├── model.py             # LSTMAutoencoder (encoder/bottleneck/decoder)
│   ├── wae.py               # SmilesWAE (GRU encoder/decoder, WordDropout)
│   ├── wae_config.py        # Dict-based WAE config (get_default_wae_config)
│   ├── wae_trainer.py       # SmilesWAETrainer (recon + MMD loss)
│   ├── api.py               # High-level WAE API (load, encode, decode, sample, interpolate)
│   ├── vocab.py             # SpecialTokens, CharVocab, OneHotVocab
│   ├── config.py            # Argparse-based LSTM config (get_parser)
│   └── trainer.py           # AutoencoderTrainer, CircularBuffer
└── peptides/                # Peptide Wasserstein autoencoder
    ├── model.py             # PeptideWAE
    ├── api.py               # High-level functions (load, encode, sample, decode, interpolate)
    ├── vocab.py             # PeptideVocab (space-separated amino acid codes)
    ├── encoder.py           # Bidirectional GRU encoder
    ├── decoder.py           # GRU decoder with teacher forcing
    ├── config.py            # Default hyperparameters dict
    ├── losses.py            # Re-exports from shared/losses.py (backward compat)
    └── utils.py             # to_tensor helper
```

### Key Patterns

- **Each module** (smiles, peptides) has its own model, vocab, and config — they are independent and don't share code except losses via `shared/`.
- **Vocab classes** handle string-to-tensor conversion. SMILES uses character-level tokenization; peptides use space-separated amino acid tokens.
- **`utils.py`** at the package root exists solely for backward compatibility — it re-exports `OneHotVocab`, `CharVocab`, `SpecialTokens`, `set_seed` from their actual locations.
- **Both WAE modules** have high-level functional APIs in `api.py` while SMILES LSTM uses the model class methods directly.
- **Config**: SMILES LSTM uses argparse (`get_parser()`); both WAEs use plain dicts (`get_default_wae_config()`, `get_default_config()`).
- **Model checkpoints** are saved/loaded with `torch.save`/`torch.load` (config, vocab, and state_dict as separate `.pt` files).
- **Shared losses** in `shared/losses.py` provide MMD and KL loss functions used by both WAE modules. `peptides/losses.py` re-exports from `shared/losses.py` for backward compatibility.

### Inference Flow

1. Load config.pt, vocab.pt, model state dict from checkpoint directory
2. Instantiate model with vocab + config, load state dict, `.to(device)`, `.eval()`
3. Convert input string to tensor via vocab
4. Encode → latent vector → decode/sample

### Training Flow (SMILES)

Uses `AutoencoderTrainer.fit()` with Adam optimizer, `ReduceLROnPlateau` scheduler, gradient clipping, and early stopping. Checkpoints saved per epoch.

## Dependencies

Runtime: `torch`, `numpy`, `pandas`, `tqdm`, `rdkit`, `seaborn`, `plotly`, `scikit-learn`, `chemographykit`
Dev: `pytest`, `jupyter`, `matplotlib`, `ipykernel`

## Trained Models

`models/autoencoder_v1/` contains the SMILES autoencoder checkpoint (config.pt, vocab.pt, model_best.pt, epoch checkpoints, training.log).
