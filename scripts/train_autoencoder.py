#!/usr/bin/env python3
"""
Train LSTM Autoencoder from scratch.

This script trains the LSTM autoencoder on SMILES strings with configurable
architecture and training parameters.

Usage:
    python scripts/train_autoencoder.py \\
        --train_load data/train.csv \\
        --val_load data/test.csv \\
        --model_save models/autoencoder_v1/model.pt \\
        --log_file models/autoencoder_v1/training.log \\
        --vocab_save models/autoencoder_v1/vocab.pt
"""

import os
import logging
import torch
from deepchemography.autoencoder import LSTMAutoencoder, AutoencoderTrainer, get_parser
from deepchemography.script_utils import read_smiles_csv, add_train_args, add_common_args, set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Main training function."""
    # Parse configuration
    parser = get_parser()
    add_common_args(parser)
    add_train_args(parser)
    
    config = parser.parse_args()
    
    # Set random seed
    set_seed(config.seed)
    
    # Create output directory
    if config.model_save is not None:
        model_dir = os.path.dirname(config.model_save)
        if model_dir and not os.path.exists(model_dir):
            os.makedirs(model_dir, exist_ok=True)
            logger.info(f"Created output directory: {model_dir}")
    
    # Load data
    if config.train_load is None:
        raise ValueError("--train_load is required")
    
    logger.info("Loading training data...")
    train_data = read_smiles_csv(config.train_load)
    logger.info(f"Loaded {len(train_data)} training samples")
    
    val_data = None
    if config.val_load:
        logger.info("Loading validation data...")
        val_data = read_smiles_csv(config.val_load)
        logger.info(f"Loaded {len(val_data)} validation samples")
    
    # Create trainer and vocabulary
    logger.info("Creating vocabulary...")
    trainer = AutoencoderTrainer(config)
    
    if config.vocab_load:
        logger.info(f"Loading vocabulary from {config.vocab_load}")
        vocab = torch.load(config.vocab_load)
    else:
        vocab = trainer.get_vocabulary(train_data)
    
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    # Save vocabulary and config
    if config.vocab_save:
        logger.info(f"Saving vocabulary to {config.vocab_save}")
        torch.save(vocab, config.vocab_save)
    
    if config.model_save:
        config_save = os.path.join(os.path.dirname(config.model_save), 'config.pt')
        logger.info(f"Saving config to {config_save}")
        torch.save(config, config_save)
    
    # Create model
    logger.info("Creating model...")
    logger.info("Architecture:")
    logger.info(f"  Encoder: {'Bidirectional' if config.q_bidir else 'Unidirectional'} "
                f"{config.q_cell.upper()}, {config.q_n_layers} layers, "
                f"{config.q_d_h} units/direction")
    logger.info(f"  Encoder output: {config.q_d_h * (2 if config.q_bidir else 1) * config.q_n_layers * 2} dims")
    logger.info(f"  Bottleneck: {config.d_z} dims")
    logger.info(f"  Decoder: {config.d_cell.upper()}, {config.d_n_layers} layers, "
                f"{config.d_d_h} units")
    logger.info(f"  Batch Normalization: {'Enabled' if config.use_batch_norm else 'Disabled'}")
    
    model = LSTMAutoencoder(vocab, config)
    
    # Move to device
    device = torch.device(config.device)
    logger.info(f"Using device: {device}")
    model = model.to(device)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Train
    logger.info("\n" + "=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)
    
    model = trainer.fit(model, train_data, val_data)
    
    # Save final model
    if config.model_save:
        logger.info("\nSaving final model...")
        model = model.to('cpu')
        torch.save(model.state_dict(), config.model_save)
    
    logger.info("\n" + "=" * 80)
    logger.info("Training complete!")
    logger.info("=" * 80)
    
    if config.model_save:
        logger.info(f"Best model saved to: {config.model_save[:-3]}_best.pt")
        logger.info(f"Final model saved to: {config.model_save}")
    if config.vocab_save:
        logger.info(f"Vocabulary saved to: {config.vocab_save}")
    if config.log_file:
        logger.info(f"Training log saved to: {config.log_file}")


if __name__ == '__main__':
    main()

