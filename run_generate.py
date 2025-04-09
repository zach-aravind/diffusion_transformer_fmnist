#!/usr/bin/env python
"""
Script to run the generation without installing the package.
This is a workaround for the relative import issues.
"""
import sys
import os
import argparse

# Add the root directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import configuration and generation function
from diffusion_transformer_fmnist.generate import main

if __name__ == "__main__":
    main()