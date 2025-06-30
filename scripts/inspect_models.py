#!/usr/bin/env python3
"""
Inspect saved model files to see what's stored
"""

import pickle
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from inbox_sentinel.utils.pickle_compat import load_model_compat


def inspect_model(filepath):
    """Inspect a model file"""
    print(f"\n{'='*60}")
    print(f"Inspecting: {filepath.name}")
    print('='*60)
    
    try:
        # Load with compatibility
        model_data = load_model_compat(filepath)
        
        print(f"Keys in model data: {list(model_data.keys())}")
        
        for key, value in model_data.items():
            if key == 'model':
                print(f"  {key}: {type(value).__name__}")
                if hasattr(value, 'classes_'):
                    print(f"    - Classes: {value.classes_}")
            elif key == 'preprocessor':
                print(f"  {key}: {type(value).__name__ if value else 'None'}")
            elif key == 'scaler':
                print(f"  {key}: {type(value).__name__}")
                if hasattr(value, 'mean_'):
                    print(f"    - Is fitted: True")
                else:
                    print(f"    - Is fitted: False")
            else:
                print(f"  {key}: {value}")
                
    except Exception as e:
        print(f"Error loading model: {e}")


def main():
    models_dir = Path("data/models")
    
    if not models_dir.exists():
        print("Models directory not found!")
        return
    
    for model_file in sorted(models_dir.glob("*.pkl")):
        inspect_model(model_file)


if __name__ == "__main__":
    main()