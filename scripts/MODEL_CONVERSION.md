# Model Conversion Guide

This guide explains how to convert models trained with ChemEidos to the deepchemography format.

## Overview

The `convert_model.py` script allows you to re-save models that were trained with the ChemEidos codebase using the deepchemography classes, without any retraining.

## Usage

### Basic Usage

```bash
PYTHONPATH=src python3 scripts/convert_model.py \
    --source models/autoencoder_v1 \
    --output models/autoencoder_converted
```

### Arguments

- `--source`: Directory containing the ChemEidos model files (should contain `config.pt`, `vocab.pt`, and model checkpoint files like `model_best.pt` or `model.pt`)
- `--output`: Directory where the converted model will be saved
- `--model`: Which model checkpoint to convert (default: `model_best.pt`)

### Example

```bash
# Convert the best model checkpoint
PYTHONPATH=src python3 scripts/convert_model.py \
    --source models/autoencoder_v1 \
    --output models/autoencoder_converted

# Convert a specific checkpoint
PYTHONPATH=src python3 scripts/convert_model.py \
    --source models/autoencoder_v1 \
    --output models/autoencoder_converted \
    --model model_050.pt
```

## What Gets Converted

The script converts:
1. **Config** (`config.pt`): Model configuration/hyperparameters
2. **Vocabulary** (`vocab المُنة`) and class definitions 
3. **Model weights** (`.pt` files): The trained model parameters

All files are re-saved using deepchemography's classes, making them compatible with the deepchemography codebase.

## Loading Converted Models

After conversion, you can load the model using deepchemography:

```python
import torch
from deepchemography import LSTMAutoencoder

# Load the converted model
config = torch.load('models/autoencoder_converted/config.pt')
vocab = torch.load('models/autoencoder_converted/vocab.pt')
model_state = torch.load('models/autoencoder_converted/model.pt', map_location='cpu')

# Create and load model
model = LSTMAutoencoder(vocab, config)
model.load_state_dict(model_state)
model.eval()

# Use the model for inference
smiles = "CCO"  # ethanol
z = model.encode([model.string2tensor(smiles)])
reconstructed = model.sample(n_batch=1, z=z)
print(f"Original: {smiles}")
print(f"Reconstructed: {reconstructed[0]}")
```

## Technical Details

### How It Works

The script works by:
1. Creating fake `chemeidos.utils` module in Python's `sys.modules` with the deepchemography classes
2. Loading the ChemEidos model using `torch.load`, which uses pickle under the hood
3. The pickle system finds the classes it needs in the fake module (which are actually deepchemography classes)
4. Creating a new deepchemography model instance with the loaded config and vocab
5. Loading the state dict into the model
6. Saving everything using deepchemography's save format

This approach works because:
- The ChemEidos and deepchemography `LSTMAutoencoder` classes have the same architecture
- The vocabulary classes are compatible
- Only the module paths differ in the pickle files

## Notes

- The conversion does NOT retrain the model - all weights are preserved
- The architecture must be compatible between ChemEidos and deepchemography
- The vocabulary must use the same format (character-based or SELFIES)

