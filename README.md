# DeepChemography

Deep learning autoencoders for molecular sequence encoding and generation.

![Python](https://img.shields.io/badge/python-%3E%3D3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![CI](https://github.com/AxelRolov/deepchemography/actions/workflows/ci.yml/badge.svg)

## Overview

DeepChemography provides three autoencoder architectures for learning latent representations of molecular sequences — small-molecule SMILES strings and peptide amino acid sequences. It supplies the generative deep learning components used in the [ChemSpace Copilot](https://doi.org/10.26434/chemrxiv.15000527/v1) agentic AI framework for interactive visualization and exploration of chemical space.

## Highlights

- **Three autoencoder models**: LSTM autoencoder, SMILES WAE, and Peptide WAE
- **99.71% reconstruction accuracy** on the SMILES LSTM autoencoder
- **High-level functional APIs** for encoding, decoding, sampling, interpolation, and neighborhood exploration
- **Shared loss functions** (MMD with gaussian/laplace/energy kernels, KL divergence)
- **Training infrastructure** with learning rate scheduling, gradient clipping, early stopping, and checkpointing
- **Pretrained SMILES LSTM checkpoint** included (`models/autoencoder_v1/`)

## Models

| Model | Architecture | Latent dim | Input | Loss |
|---|---|---|---|---|
| `LSTMAutoencoder` | Bidirectional LSTM | 256 | SMILES (char-level) | Reconstruction |
| `SmilesWAE` | Bidirectional GRU | 128 | SMILES (char-level) | Recon + MMD |
| `PeptideWAE` | Bidirectional GRU | 100 | Peptides (token-level) | Recon + MMD |

## Installation

```bash
# From source (recommended)
git clone https://github.com/AxelRolov/deepchemography.git
cd deepchemography
uv sync

# Or with pip
pip install -e .

# Dev extras (pytest, jupyter, matplotlib)
uv sync --extra dev
```

Requires Python >= 3.11.

## Quick Start

### SMILES LSTM Autoencoder

```python
import torch
from pathlib import Path
from deepchemography import LSTMAutoencoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_dir = Path("models/autoencoder_v1")

config = torch.load(model_dir / "config.pt", weights_only=False)
vocab = torch.load(model_dir / "vocab.pt", weights_only=False)
state = torch.load(model_dir / "model_best.pt", weights_only=False)

model = LSTMAutoencoder(vocab, config)
model.load_state_dict(state)
model.to(device).eval()

smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
tensor = model.string2tensor(smiles, device=device)
z = model.forward_encoder([tensor])
reconstructed = model.sample(n_batch=1, z=z, temp=0.1, decode="greedy")
print(reconstructed[0])
```

### SMILES WAE

```python
from deepchemography import load_smiles_wae, encode_smiles, sample_smiles

model, vocab = load_smiles_wae("path/to/checkpoint_dir")
z = encode_smiles(["CCO", "c1ccccc1"], model, vocab)
samples = sample_smiles(model, vocab, n_samples=10)
```

### Peptide WAE

```python
from deepchemography import load_peptide_model, encode_peptide, sample_peptides

model, vocab, config = load_peptide_model("path/to/checkpoint_dir")
z = encode_peptide(["A L G", "G P R K"], model, vocab, config)
samples = sample_peptides(model, vocab, config, n_samples=10)
```

## API Reference

### SMILES LSTM Autoencoder

| Function / Class | Description |
|---|---|
| `LSTMAutoencoder` | Bidirectional LSTM autoencoder model |
| `AutoencoderTrainer` | Training loop with LR scheduling and early stopping |
| `OneHotVocab` | One-hot character vocabulary for SMILES |
| `CharVocab` | Character-level vocabulary |
| `get_parser` | Argparse-based configuration |

### SMILES WAE

| Function / Class | Description |
|---|---|
| `SmilesWAE` | GRU-based Wasserstein autoencoder model |
| `SmilesWAETrainer` | Training loop with MMD loss |
| `get_default_wae_config` | Default hyperparameters dict |
| `load_smiles_wae` | Load trained model from checkpoint |
| `encode_smiles` | Encode SMILES strings to latent vectors |
| `decode_smiles_latent` | Decode latent vectors to SMILES |
| `sample_smiles` | Sample novel SMILES from prior |
| `interpolate_smiles` | Interpolate between two molecules |
| `reconstruct_smiles` | Encode and decode (round-trip) |
| `explore_smiles_neighborhood` | Generate neighbors by perturbing latent vectors |

### Peptide WAE

| Function / Class | Description |
|---|---|
| `PeptideWAE` | GRU-based Wasserstein autoencoder model |
| `PeptideVocab` | Space-separated amino acid vocabulary |
| `load_peptide_model` | Load trained model from checkpoint |
| `encode_peptide` | Encode peptide sequences to latent vectors |
| `sample_peptides` | Sample novel peptides from prior |
| `decode_latent` | Decode latent vectors to peptide sequences |
| `interpolate_peptides` | Interpolate between two peptides |
| `reconstruct_peptide` | Encode and decode (round-trip) |

### Shared Utilities

| Function / Class | Description |
|---|---|
| `Logger` | Training logger |
| `setup_logging` | Configure logging |
| `set_seed` | Set random seeds for reproducibility |
| `read_smiles_csv` | Load SMILES from CSV files |

## Project Structure

```
src/deepchemography/
├── __init__.py              # Re-exports all public API
├── utils.py                 # Backward compatibility shim
├── script_utils.py          # CLI helpers: device parsing, argparse, CSV loading
├── shared/
│   ├── logging.py           # Logger, setup_logging
│   ├── losses.py            # MMD and KL losses (shared by both WAEs)
│   └── utils.py             # set_seed
├── smiles/
│   ├── model.py             # LSTMAutoencoder
│   ├── wae.py               # SmilesWAE (GRU encoder/decoder, WordDropout)
│   ├── wae_config.py        # Dict-based WAE config
│   ├── wae_trainer.py       # SmilesWAETrainer
│   ├── api.py               # High-level API (load, encode, decode, sample, interpolate)
│   ├── vocab.py             # SpecialTokens, CharVocab, OneHotVocab
│   ├── config.py            # Argparse-based LSTM config
│   └── trainer.py           # AutoencoderTrainer, CircularBuffer
└── peptides/
    ├── model.py             # PeptideWAE
    ├── api.py               # High-level API (load, encode, sample, decode, interpolate)
    ├── vocab.py             # PeptideVocab
    ├── encoder.py           # Bidirectional GRU encoder
    ├── decoder.py           # GRU decoder with teacher forcing
    ├── config.py            # Default hyperparameters dict
    ├── losses.py            # Re-exports from shared/losses.py
    └── utils.py             # to_tensor helper
```

## Training

Training scripts and documentation are in `scripts/` and `docs/`. See [`docs/TRAINING.md`](docs/TRAINING.md) for detailed instructions on training each model, including dataset preparation, hyperparameter tuning, and checkpoint management.

## Citation

If you use DeepChemography in your research, please cite:

```bibtex
@article{orlov2026chemspace,
  title={ChemSpace Copilot: Agentic AI for Interactive Visualization and Exploration of Chemical Space},
  author={Orlov, Alexander A. and Volkov, Maxim and Milova, Ekaterina S. and Horvath, Dragos and Varnek, Alexandre},
  journal={ChemRxiv},
  year={2026},
  doi={10.26434/chemrxiv.15000527/v1}
}
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Developed at the [Laboratory of Chemoinformatics](https://infochim.u-strasbg.fr/), UMR 7140 CNRS, University of Strasbourg.
