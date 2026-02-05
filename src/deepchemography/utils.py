"""
Backward compatibility alias for deepchemography.shared and smiles vocabs.

This module exists to allow unpickling vocab files that were
saved with the old module structure (deepchemography.utils).
"""

from deepchemography.shared import *
from deepchemography.shared import Logger, setup_logging, set_seed

# Import vocab classes for backward compatibility with pickled vocab files
from deepchemography.smiles import OneHotVocab, CharVocab

__all__ = ['Logger', 'setup_logging', 'set_seed', 'OneHotVocab', 'CharVocab']
