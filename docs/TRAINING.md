# Training the LSTM Autoencoder

This guide explains how to train the LSTM autoencoder model on SMILES molecular representations.

## Quick Start

### Basic Training

```bash
# From the project root
uv run python scripts/train_autoencoder.py \
    --train_load data/train.csv \
    --val_load data/test.csv \
    --model_save models/autoencoder_v1/model.pt \
    --log_file models/autoencoder_v1/training.log \
    --vocab_save models/autoencoder_v1/vocab.pt \
    --device cuda
```

### Optimal Architecture (99.71% reconstruction accuracy)

```bash
uv run python scripts/train_autoencoder.py \
    --train_load data/train.csv \
    --val_load data/test.csv \
    --model_save models/autoencoder_v1/model.pt \
    --log_file models/autoencoder_v1/training.log \
    --vocab_save models/autoencoder_v1/vocab.pt \
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
    --lr_factor 0.5 \
    --early_stop_patience 5
```

## Data Format

Your CSV file should have a `SMILES` column:

```csv
SMILES
CC(=O)O
CCO
CN1C=NC2=C1C(=O)N(C(=O)N2C)C
```

## Command-Line Arguments

### Common Arguments
- `--device`: Device to use (cpu, cuda, or cuda:0)
- `--seed`: Random seed for reproducibility

### Data Arguments
- `--train_load`: Path to training data CSV file (required)
- `--val_load`: Path to validation data CSV file
- `--vocab_load`: Path to load existing vocabulary
- `--vocab_save`: Path to save vocabulary

### Model Architecture
- `--q_cell`: Encoder RNN type (default: 'lstm')
- `--q_bidir`: Use bidirectional encoder (recommended)
- `--q_n_layers`: Number of encoder layers (default: 2)
- `--q_d_h`: Encoder hidden dimension (default: 128)
- `--d_cell`: Decoder RNN type (default: 'lstm')
- `--d_n_layers`: Number of decoder layers (default: 2)
- `--d_d_h`: Decoder hidden dimension (default: 256)
- `--d_z`: Latent/bottleneck dimension (default: 256)
- `--use_batch_norm`: Enable batch normalization (strongly recommended)

### Training Arguments
- `--n_batch`: Batch size (default: 256)
- `--n_epochs`: Maximum epochs (default: 100)
- `--lr_start`: Initial learning rate (default: 0.001)
- `--lr_patience`: Epochs before reducing LR (default: 3)
- `--early_stop_patience`: Early stopping patience (default: 10)
- `--save_frequency`: Checkpoint save frequency (default: 10)

## Output Files

After training, you'll get:
- `model.pt`: Final model state
- `model_best.pt`: Best model based on validation metric
- `model_NNN.pt`: Periodic checkpoints
- `vocab.pt`: Vocabulary file
- `config.pt`: Training configuration
- `training.log`: Training metrics (CSV)

## Example: Training on ChEMBL Data

```bash
uv run python scripts/train_autoencoder.py \
    --train_load ../ChemEidos/data/train.csv \
    --val_load ../ChemEidos/data/test.csv \
    --model_save models/autoencoder_optimal/model.pt \
    --log_file models/autoencoder_optimal/training.log \
    --vocab_save models/autoencoder_optimal/vocab.pt \
    --device cuda \
    --q_bidir \
    --use_batch_norm \
    --n_batch 256 \
    --lr_start 0.005 \
    --seed 42
```

## Troubleshooting

**Out of Memory**: Reduce batch size with `--n_batch 128`

**Slow Training**: Use `--device cuda` to enable GPU acceleration

**Poor Reconstruction**: Enable batch normalization with `--use_batch_norm`
