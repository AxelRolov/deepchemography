#!/usr/bin/env python3
"""
Convert ChemEidos models to deepchemography format.

This script loads models that were trained with ChemEidos and resaves them
using the deepchemography codebase without any retraining.

Usage:
    pdm run python scripts/convert_model.py --source models/autoencoder_v1 --output models/autoencoder_converted
"""

import argparse
import torch
import sys
from pathlib import Path
from deepchemography.utils import setup_logging, OneHotVocab, CharVocab, SpecialTokens
from deepchemography.autoencoder import LSTMAutoencoder, get_parser

logger = setup_logging()


# Create a fake chemeidos module structure to handle pickle imports
# This allows loading vocab files saved by ChemEidos
import types

# Create fake chemeidos module
chemeidos_module = types.ModuleType('chemeidos')
sys.modules['chemeidos'] = chemeidos_module

# Create fake chemeidos.utils module
utils_module = types.ModuleType('chemeidos.utils')
utils_module.OneHotVocab = OneHotVocab
utils_module.CharVocab = CharVocab
utils_module.SpecialTokens = SpecialTokens

# Register the utils module
chemeidos_module.utils = utils_module
sys.modules['chemeidos.utils'] = utils_module


def convert_model(source_dir, output_dir, model_filename='model_best.pt', chemeidos_path=None):
    """
    Convert a ChemEidos model to deepchemography format.
    
    Args:
        source_dir: Directory containing the ChemEidos model files
        output_dir: Directory to save the converted model
        model_filename: Which model checkpoint to convert (e.g., 'model_best.pt', 'model.pt')
        chemeidos_path: Path to ChemEidos src directory (optional, tries to find automatically)
    """
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Converting model from {source_path} to {output_path}")
    
    # Define file paths
    config_file = source_path / 'config.pt'
    vocab_file = source_path / 'vocab.pt'
    model_file = source_path / model_filename
    
    # Check if files exist
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_file}")
    if not vocab_file.exists():
        raise FileNotFoundError(f"Vocab file not found: {vocab_file}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_file}")
    
    logger.info("Loading ChemEidos model components...")
    
    # Load config and state dict normally
    logger.info("Loading config...")
    config = torch.load(config_file, weights_only=False)
    logger.info(f"  Config loaded: {len(vars(config))} parameters")
    
    # Load vocab using torch.load with the fake module mapping
    logger.info("Loading vocabulary...")
    vocab = torch.load(vocab_file, map_location='cpu', weights_only=False)
    logger.info(f"  Vocab loaded: {len(vocab)} tokens")
    
    # Load model state
    logger.info("Loading model state...")
    state_dict = torch.load(model_file, map_location='cpu', weights_only=False)
    logger.info(f"  Model state loaded: {len(state_dict)} layers")
    
    # Create model using deepchemography classes
    logger.info("Creating deepchemography model...")
    model = LSTMAutoencoder(vocab, config)
    model.load_state_dict(state_dict)
    
    logger.info("Model created successfully!")
    
    # Save in deepchemography format
    logger.info(f"Saving to {output_path}...")
    
    # Save config
    output_config = output_path / 'config.pt'
    torch.save(config, output_config)
    logger.info(f"  Saved config to {output_config}")
    
    # Save vocab
    output_vocab = output_path / 'vocab.pt'
    torch.save(vocab, output_vocab)
    logger.info(f"  Saved vocab to {output_vocab}")
    
    # Save model
    output_model = output_path / 'model.pt'
    model = model.to('cpu')
    torch.save(model.state_dict(), output_model)
    logger.info(f"  Saved model to {output_model}")
    
    # Also save a copy with the same name as input if different
    if model_filename != 'model.pt':
        output_model_copy = output_path / model_filename
        torch.save(model.state_dict(), output_model_copy)
        logger.info(f"  Saved model copy to {output_model_copy}")
    
    logger.info("✓ Conversion complete!")
    
    # Print model summary
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {total_params:,}")
    logger.info(f"Vocabulary size: {len(vocab)}")
    logger.info(f"Latent dimension: {config.d_z}")
    logger.info(f"Encoder: {config.q_cell.upper()} ({config.q_n_layers} layers, "
                f"{config.q_d_h} units, bidirectional={config.q_bidir})")
    logger.info(f"Decoder: {config.d_cell.upper()} ({config.d_n_layers} layers, "
                f"{config.d_d_h} units)")
    logger.info(f"Batch normalization: {config.use_batch_norm}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert ChemEidos models to deepchemography format'
    )
    parser.add_argument('--source', type=str, required=True,
                       help='Source directory with ChemEidos model files')
    parser.add_argument('--output', type=str, required=True,
                       help='Output directory for converted model')
    parser.add_argument('--model', type=str, default='model_best.pt',
                       help='Which model file to convert (default: model_best.pt)')
    
    args = parser.parse_args()
    
    convert_model(args.source, args.output, args.model)


if __name__ == '__main__':
    main()

