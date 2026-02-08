"""
Default configuration for SMILES WAE model.

Dict-based config matching the peptides pattern (not argparse).
"""

import copy

DEFAULT_WAE_CONFIG = {
    'max_seq_len': 150,
    'z_dim': 128,
    'emb_dim': 128,

    'encoder': {
        'h_dim': 128,
        'biGRU': True,
        'layers': 2,
        'p_dropout': 0.0,
    },

    'decoder': {
        'p_word_dropout': 0.3,
        'p_out_dropout': 0.3,
    },

    'wae_mmd': {
        'sigma': 7.0,
        'kernel': 'gaussian',
        'lambda_mmd': 10.0,
    },

    'training': {
        'n_batch': 256,
        'lr': 0.001,
        'lr_patience': 3,
        'lr_factor': 0.5,
        'lr_min': 1e-6,
        'clip_grad': 5.0,
        'n_epochs': 100,
        'early_stop_patience': 10,
    },
}


def get_default_wae_config():
    """Return a copy of the default SMILES WAE configuration."""
    return copy.deepcopy(DEFAULT_WAE_CONFIG)
