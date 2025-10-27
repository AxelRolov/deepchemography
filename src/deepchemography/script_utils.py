"""
Utility functions for training scripts.
"""
import argparse
import random
import re
import numpy as np
import pandas as pd
import torch
from deepchemography.utils import setup_logging

logger = setup_logging()


def torch_device(arg):
    """Validate and return torch device argument."""
    if re.match('^(cuda(:[0-9]+)?|cpu)$', arg) is None:
        raise argparse.ArgumentTypeError(
            'Wrong device format: {}'.format(arg)
        )

    if arg != 'cpu':
        splited_device = arg.split(':')

        if (not torch.cuda.is_available()) or \
                (len(splited_device) > 1 and
                 int(splited_device[1]) > torch.cuda.device_count()):
            raise argparse.ArgumentTypeError(
                'Wrong device: {} is not available'.format(arg)
            )

    return arg


def add_common_args(parser):
    """Add common arguments to argument parser."""
    parser.add_argument('--device',
                        type=torch_device, default='cuda',
                        help='Device to run: "cpu" or "cuda:<device number>"')
    parser.add_argument('--seed',
                        type=int, default=0,
                        help='Random seed for reproducibility')
    return parser


def add_train_args(parser):
    """Add training-specific arguments to argument parser."""
    parser.add_argument('--train_load',
                        type=str,
                        help='Input data in csv format to train')
    parser.add_argument('--val_load', 
                        type=str,
                        help='Input data in csv format for validation')
    parser.add_argument('--model_save',
                        type=str, default='model.pt',
                        help='Where to save the model')
    parser.add_argument('--save_frequency',
                        type=int, default=10,
                        help='How often to save checkpoints (every N epochs)')
    parser.add_argument('--log_file',
                        type=str,
                        help='Where to save the training log')
    parser.add_argument('--vocab_save',
                        type=str,
                        help='Where to save the vocabulary')
    parser.add_argument('--vocab_load',
                        type=str,
                        help='Where to load the vocabulary from')
    return parser


def read_smiles_csv(path):
    """
    Read SMILES strings from a CSV file.
    
    Args:
        path: path to CSV file with 'SMILES' column
        
    Returns:
        list of SMILES strings
    """
    df = pd.read_csv(path)
    if 'SMILES' in df.columns:
        smiles = df['SMILES'].squeeze().astype(str).tolist()
    elif len(df.columns) == 1:
        # If only one column, assume it's SMILES
        smiles = df.iloc[:, 0].squeeze().astype(str).tolist()
    else:
        raise ValueError(f"CSV file must have a 'SMILES' column or only one column")
    
    return smiles


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to: {seed}")


