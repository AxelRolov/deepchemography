"""
Default configuration for Peptide WAE model.
"""

# Token indices (must match vocab.dict)
UNK_IDX = 0
PAD_IDX = 1
START_IDX = 2
EOS_IDX = 3

# Default model configuration
DEFAULT_CONFIG = {
    'max_seq_len': 25,
    'z_dim': 100,
    'c_dim': 2,
    'emb_dim': 150,
    'n_vocab': 26,

    # Encoder config
    'encoder': {
        'h_dim': 80,
        'biGRU': True,
        'layers': 1,
        'p_dropout': 0.0,
    },

    # Decoder config
    'decoder': {
        'p_word_dropout': 0.3,
        'p_out_dropout': 0.3,
        'skip_connections': False,
    },

    # WAE MMD loss config
    'wae_mmd': {
        'sigma': 7.0,
        'kernel': 'gaussian',
        'rf_dim': 500,
        'rf_resample': False,
    },
}


def get_default_config():
    """Return a copy of the default configuration."""
    import copy
    return copy.deepcopy(DEFAULT_CONFIG)
