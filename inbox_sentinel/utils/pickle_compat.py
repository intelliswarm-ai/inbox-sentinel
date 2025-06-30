"""
Pickle compatibility utilities for loading models with old import paths
"""

import pickle
import sys
from typing import Any
from pathlib import Path


class ImportRemapperUnpickler(pickle.Unpickler):
    """Custom unpickler that remaps old import paths to new ones"""
    
    def find_class(self, module: str, name: str) -> Any:
        """Override find_class to remap module paths"""
        
        # Define module remappings
        remappings = {
            'email_preprocessor': 'inbox_sentinel.ml.preprocessing.email_preprocessor',
            '__main__': 'inbox_sentinel.ml.preprocessing.email_preprocessor',  # For classes defined in __main__
        }
        
        # Check if module needs remapping
        if module in remappings:
            module = remappings[module]
        
        # Handle specific class remappings
        if module == '__main__' and 'EmailPreprocessor' in name:
            module = 'inbox_sentinel.ml.preprocessing.email_preprocessor'
        
        return super().find_class(module, name)


def load_model_compat(filepath: Path) -> dict:
    """
    Load a pickled model with import path compatibility
    
    Args:
        filepath: Path to the pickle file
        
    Returns:
        Dictionary containing model data
    """
    with open(filepath, 'rb') as f:
        return ImportRemapperUnpickler(f).load()