# CODEX.md

This file provides guidance to OpenAI Codex when working in this repository.

## Project Overview

DeepChemography is a deep learning library for molecular sequence encoding and generation, extracted from ChemEidos. It contains two autoencoder models:

- **SMILES Autoencoder** (`src/deepchemography/smiles/`): Bidirectional LSTM autoencoder for SMILES strings. Latent dim: 256.
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
├── __init__.py          # Re-exports all public API
├── utils.py             # Backward compat shim (re-exports from smiles/shared)
├── script_utils.py      # CLI helpers: device parsing, argparse, CSV loading
├── shared/              # Logging (Logger, setup_logging) and set_seed()
├── smiles/              # SMILES LSTM autoencoder
│   ├── model.py         # LSTMAutoencoder (encoder/bottleneck/decoder)
│   ├── vocab.py         # SpecialTokens, CharVocab, OneHotVocab
│   ├── config.py        # Argparse-based config (get_parser)
│   └── trainer.py       # AutoencoderTrainer, CircularBuffer
└── peptides/            # Peptide Wasserstein autoencoder
    ├── model.py         # PeptideWAE
    ├── api.py           # High-level functions (load, encode, sample, decode, interpolate)
    ├── vocab.py         # PeptideVocab (space-separated amino acid codes)
    ├── encoder.py       # Bidirectional GRU encoder
    ├── decoder.py       # GRU decoder with teacher forcing
    ├── config.py        # Default hyperparameters dict
    ├── losses.py        # KL, reconstruction, MMD losses (gaussian/laplace/energy kernels)
    └── utils.py         # to_tensor helper
```

### Key Patterns

- Each module (smiles, peptides) has its own model, vocab, and config — they are independent.
- Vocab classes handle string-to-tensor conversion. SMILES is character-level; peptides are space-separated amino acid tokens.
- Root `utils.py` is only for backward compatibility — it re-exports `OneHotVocab`, `CharVocab`, `SpecialTokens`, `set_seed`.
- Peptides module exposes a high-level functional API in `api.py`; SMILES uses model class methods directly.
- SMILES config uses argparse (`get_parser()`); peptides uses a plain dict (`get_default_config()`).
- Checkpoints use `torch.save`/`torch.load` with config, vocab, and state dict as separate `.pt` files.

### Inference Flow

1. Load `config.pt`, `vocab.pt`, and model state dict from checkpoint directory.
2. Instantiate model with vocab + config, load state dict, `.to(device)`, `.eval()`.
3. Convert input string to tensor via vocab.
4. Encode → latent vector → decode/sample.

### Training Flow (SMILES)

`AutoencoderTrainer.fit()` uses Adam optimizer, `ReduceLROnPlateau` scheduler, gradient clipping, and early stopping. Checkpoints saved per epoch.

## Dependencies

Runtime: `torch`, `numpy`, `pandas`, `tqdm`, `rdkit`
Dev: `pytest`, `jupyter`, `matplotlib`, `ipykernel`

## Trained Models

`models/autoencoder_v1/` contains the SMILES autoencoder checkpoint (config.pt, vocab.pt, model_best.pt, epoch checkpoints, training.log).
