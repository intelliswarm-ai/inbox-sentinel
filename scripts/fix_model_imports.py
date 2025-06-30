#!/usr/bin/env python3
"""
Fix import paths in pickled models
"""

import pickle
import sys
from pathlib import Path

# Add the old module name mapping
class ModuleRenamer:
    def __init__(self):
        self.rename_map = {
            'email_preprocessor': 'inbox_sentinel.ml.preprocessing.email_preprocessor'
        }
    
    def __call__(self, module_name):
        return self.rename_map.get(module_name, module_name)

# Custom unpickler that handles module renames
class RenameUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # Handle the module rename
        if module == 'email_preprocessor':
            module = 'inbox_sentinel.ml.preprocessing.email_preprocessor'
        elif module.startswith('__main__'):
            # Handle classes that were defined in __main__
            if 'EmailPreprocessor' in name:
                module = 'inbox_sentinel.ml.preprocessing.email_preprocessor'
        
        return super().find_class(module, name)

def fix_model_file(filepath):
    """Fix import paths in a single model file"""
    print(f"Fixing {filepath.name}...")
    
    try:
        # Load with custom unpickler
        with open(filepath, 'rb') as f:
            model_data = RenameUnpickler(f).load()
        
        # Check if it has the old preprocessor
        if 'preprocessor' in model_data and hasattr(model_data['preprocessor'], '__module__'):
            old_module = model_data['preprocessor'].__module__
            if old_module in ['email_preprocessor', '__main__']:
                print(f"  Found old module: {old_module}")
                # We need to retrain or fix the preprocessor
                return False
        
        print(f"  Model appears to be OK")
        return True
        
    except Exception as e:
        print(f"  Error: {e}")
        return False

def main():
    models_dir = Path("data/models")
    
    if not models_dir.exists():
        print("Models directory not found!")
        return
    
    print("Checking model files...")
    print("-" * 40)
    
    issues_found = False
    for model_file in models_dir.glob("*.pkl"):
        if not fix_model_file(model_file):
            issues_found = True
    
    if issues_found:
        print("\n" + "=" * 40)
        print("Some models have import issues.")
        print("The models need to be retrained with the new package structure.")
        print("Run: inbox-sentinel models train")
    else:
        print("\n" + "=" * 40)
        print("All models appear to be compatible!")

if __name__ == "__main__":
    main()