#!/usr/bin/env python
"""
Script to run the training without installing the package.
This is a workaround for the relative import issues.
"""
import sys
import os

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Now we can import from the package
from diffusion_transformer_fmnist.train import main

if __name__ == "__main__":
    main()