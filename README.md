# DeepChemography

LSTM Autoencoder for SMILES molecular representations - extracted from ChemEidos project.

## Overview

This repository contains a trained LSTM autoencoder specifically designed for encoding and sampling SMILES (Simplified Molecular Input Line Entry System) molecular representations. The model achieves 99.71% reconstruction accuracy.

## Project Structure

```
deepchemography/
├── src/
│   └── deepchemography/
│       ├── __init__.py
│       ├── utils.py              # Vocabulary and tokenization utilities
│       └── autoencoder/
│           ├── __init__.py
│           ├── model.py          # LSTMAutoencoder implementation
│           └── config.py         # Configuration parser
├── models/
│   └── autoencoder_v1/           # Trained model checkpoints
│       ├── config.pt             # Model configuration
│       ├── vocab.pt              # Character vocabulary
│       ├── model_best.pt         # Best model weights
│       └── training.log          # Training metrics
├── notebooks/
│   └── autoencoder_example.ipynb # Example usage notebook
└── README.md
```

## Architecture

The LSTM Autoencoder follows the optimal architecture from research by Xu et al. (2017):

### Encoder
- **Type**: Bidirectional LSTM
- **Layers**: 2
- **Hidden units**: 128 per direction (256 total)
- **Output**: Concatenated hidden (h) and cell (c) states from all layers → 1024 units

### Bottleneck
- **Latent dimension**: 256 units
- **Compression**: 1024 → 256 (4× compression)

### Decoder
- **Type**: Unidirectional LSTM
- **Layers**: 2
- **Hidden units**: 256 per layer
- **Initialization**: Latent vector decoded by four parallel dense layers to form initial states

### Key Features
- **Batch Normalization**: Crucial for reaching high reconstruction accuracy
- **Teacher Forcing**: Used during training
- **Expected Performance**: 99.71% reconstruction accuracy

## Installation

```bash
# Clone the repository
cd /data/aorlov/deepchemography

# Install dependencies (using pdm)
pdm add torch numpy pandas tqdm rdkit

# Or install with pip
pip install torch numpy pandas tqdm rdkit
```

## Quick Start

```python
import torch
from pathlib import Path
from deepchemography import LSTMAutoencoder

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_dir = Path('models/autoencoder_v1')

config = torch.load(model_dir / 'config.pt', weights_only=False)
vocab = torch.load(model_dir / 'vocab.pt', weights_only=False)
model_state = torch.load(model_dir / 'model_best.pt', weights_only=False)

model = LSTMAutoencoder(vocab, config)
model.load_state_dict(model_state)
model = model.to(device)
model.eval()

# Encode SMILES to latent vector
smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
tensor = model.string2tensor(smiles, device=device)
z = model.forward_encoder([tensor])

# Decode latent vector back to SMILES
reconstructed = model.sample(n_batch=1, z=z, temp=0.1, decode='greedy')
print(f"Original:      {smiles}")
print(f"Reconstructed: {reconstructed[0]}")
```

## Usage Examples

See the [example notebook](notebooks/autoencoder_example.ipynb) for comprehensive demonstrations:

1. **Loading the model**: Load trained autoencoder with configuration and vocabulary
2. **Encoding SMILES**: Convert molecular structures to latent vectors
3. **Reconstruction**: Encode and decode molecules (encode → decode)
4. **Sampling**: Generate novel molecules from Gaussian prior
5. **Interpolation**: Smoothly transition between molecules in latent space
6. **Neighborhood exploration**: Generate similar molecules by adding noise

## Key Functions

### encode_smiles(smiles_list, model, batch_size=32)
Encode a list of SMILES strings to latent vectors.

```python
latent_vectors = encode_smiles(
    ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"],
    model,
    batch_size=32
)
# Returns: numpy array of shape (n_samples, 256)
```

### sample_from_latent(model, z=None, n_samples=10, temp=1.0, decode='greedy')
Sample SMILES from the autoencoder latent space.

```python
# Sample from Gaussian prior
samples = sample_from_latent(model, z=None, n_samples=10, latent_std=1.0, temp=1.0)

# Decode specific latent vectors
samples = sample_from_latent(model, z=latent_vectors, temp=0.1, decode='greedy')
```

## Model Details

### Trained Model Checkpoints

The `models/autoencoder_v1/` directory contains:
- `config.pt`: Model architecture configuration
- `vocab.pt`: One-hot vocabulary (30 SMILES characters)
- `model_best.pt`: Best model weights from training
- `model_000.pt` to `model_050.pt`: Epoch checkpoints
- `training.log`: Training metrics (loss, accuracy, learning rate)

### Training Details

- **Dataset**: ChEMBL 33 (standardized, unique SMILES)
- **Optimizer**: Adam
- **Learning Rate**: 0.001 with ReduceLROnPlateau scheduler
- **Batch Size**: 256
- **Gradient Clipping**: 5.0
- **Early Stopping**: 10 epochs patience

## Components Copied from ChemEidos

This project contains only the necessary components for LSTM autoencoder functionality:

### Core Modules
- `autoencoder/model.py`: LSTMAutoencoder implementation (344 lines)
- `autoencoder/config.py`: Argument parser for configuration (91 lines)
- `utils.py`: Vocabulary classes (OneHotVocab, CharVocab, SpecialTokens) (93 lines)

### Model Files
- All trained model checkpoints from `models/autoencoder_v1/`
- Configuration and vocabulary files

### Documentation
- Example notebook demonstrating all key functionalities
- This README with usage instructions

**Not included**: Training code, dataset utilities, evaluation metrics, other model types (VAE, AAE, etc.)

## Citation

If you use this autoencoder in your research, please cite:

```
Xu, Y., Lin, K., Wang, S., Wang, L., Cai, C., Song, C., ... & Pei, J. (2017).
Deep learning for molecular generation.
Future medicinal chemistry, 11(6), 567-597.
```

## License

See [LICENSE](LICENSE) file for details.

## Acknowledgments

This code is extracted from the [ChemEidos](https://github.com/aorlov/ChemEidos) project, which implements multiple molecular generation models. The LSTM autoencoder component has been isolated for focused use in molecular encoding and sampling tasks.
