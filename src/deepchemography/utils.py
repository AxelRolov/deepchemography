"""
Backward compatibility alias for deepchemography.shared.

This module exists to allow unpickling vocab files that were
saved with the old module structure (deepchemography.utils).
"""

from deepchemography.shared import *
from deepchemography.shared import Logger, setup_logging, set_seed

__all__ = ['Logger', 'setup_logging', 'set_seed']
