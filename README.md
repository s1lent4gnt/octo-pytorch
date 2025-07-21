# Octo VLA Pytorch

This repository contains a PyTorch implementation of the Octo VLA model.

## Installation

To get started, follow these steps to set up the environment and install the required dependencies.

```bash
# Create a virtual environment using uv
uv venv --python 3.10

# Activate the virtual environment
source .venv/bin/activate

# Install the main project in editable mode
uv pip install -e .

# Navigate to the submodule directory
cd octo_jax

# Install the submodule and its specific dependencies
uv pip install -e .

# Return to the root directory
cd ..
```

## Usage

After installation, you can run the following scripts:

```bash
# Convert the model weights
python scripts/convert_weights.py

# Verify the weights conversion
python scripts/check_weights_conversion.py
