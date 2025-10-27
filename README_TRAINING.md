# How to Retrain the Autoencoder

This guide shows you how to train the LSTM autoencoder using a command-line script, following ideas from ChemEidos.

## Quick Start

The training script is located at `scripts/train_autoencoder.py`. Here's how to use it:

### Basic Usage

```bash
pdm run python scripts/train_autoencoder.py \
    --train_load data/train.csv \
    --val_load data/test.csv \
    --model_save models/my_model/model.pt \
    --log_file models/my_model/training.log \
    --vocab_save models/my_model/vocab.pt \
    --device cuda
```

### Required Data Format

Your CSV file needs a `SMILES` column (or only one column with SMILES):

```csv
SMILES
CCO
CC(=O)O
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

### Recommended Settings (High Accuracy)

```bash
pdm run python scripts/train_autoencoder.py \
    --train_load ../ChemEidos/data/train.csv \
    --val_load ../ChemEidos/data/test.csv \
    --model_save models/optimal_autoencoder/model.pt \
    --log_file models/optimal_autoencoder/training.log \
    --vocab_save models/optimal_autoencoder/vocab.pt \
    --device cuda \
    --q_cell lstm \
    --q_bidir \
    --q_n_layers 2 \
    --q_d_h 128 \
    --d_cell lstm \
    --d_n_layers 2 \
    --d_d_h 256 \
    --d_z 256 \
    --use_batch_norm \
    --n_batch 256 \
    --lr_start 0.005 \
    --lr_patience 2 \
    --early_stop_patience 5
```

Key settings for high accuracy:
- `--q_bidir`: Bidirectional encoder
- `--use_batch_norm`: Batch normalization (critical!)
- `--d_z 256`: Bottleneck dimension
- `--d_d_h 256`: Decoder hidden size

### What Gets Saved

After training, you'll find:
- `model.pt`: The final trained model
- `model_best.pt`: The best model (based on validation metrics)
- `model_NNN.pt`: Checkpoints saved every N epochs
- `vocab.pt`: The vocabulary
- `config.pt`: Training configuration
- `training.log`: Training metrics in CSV format

### Help

To see all available options:

```bash
pdm run python scripts/train_autoencoder.py --help
```

### Example: Training with Custom Settings

```bash
# Smaller model for faster training/testing
pdm run python scripts/train_autoencoder.py \
    --train_load data/train.csv \
    --val_load data/test.csv \
    --model_save models/test/model.pt \
    --log_file models/test/training.log \
    --vocab_save models/test/vocab.pt \
    --device cuda \
    --n_batch 128 \
    --d_z 128 \
    --q_d_h 64 \
    --d_d_h 128 \
    --n_epochs 50 \
    --save_frequency 5
```

## Programmatic Training

You can also train from Python code:

```python
from deepchemography import LSTMAutoencoder, AutoencoderTrainer, get_parser
from deepchemography.utils import OneHotVocab
from deepchemography.script_utils import read_smiles_csv, set_seed
import torch

# Set seed for reproducibility
set_seed(42)

# Parse configuration
parser = get_parser()
config = parser.parse_args([
    '--q_bidir',
    '--use_batch_norm',
    '--n_batch', '256',
    '--model_save', 'model.pt',
    '--log_file', 'training.log',
])

# Load data
train_data = read_smiles_csv('data/train.csv')
val_data = read_smiles_csv('data/test.csv')

# Create vocabulary and model
vocab = OneHotVocab.from_data(train_data)
model = LSTMAutoencoder(vocab, config)

# Move to GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

# Train
trainer = AutoencoderTrainer(config)
model = trainer.fit(model, train_data, val_data)

# Save
torch.save(model.state_dict(), config.model_save)
torch.save(vocab, 'vocab.pt')
torch.save(config, 'config.pt')
```

## Key Parameters

- **Architecture**: Use `--q_bidir` and `--use_batch_norm` for best results
- **Batch size**: `--n_batch 256` is standard (reduce if out of memory)
- **Learning rate**: Start with `--lr_start 0.005` (paper) or `0.001` (more stable)
- **Device**: Use `--device cuda` for GPU training (much faster)

## Troubleshooting

**GPU out of memory**: 
```bash
--n_batch 128  # or smaller
```

**Training is slow**:
```bash
--device cuda  # use GPU
--n_workers 4  # use multiple workers
```

**Poor reconstruction accuracy**:
```bash
--use_batch_norm  # critical for high accuracy
--q_bidir         # bidirectional encoder
```

For more details, see `docs/TRAINING.md`.

