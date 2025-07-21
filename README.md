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
```

## Current result

The output of `scripts/check_weights_conversion.py`

```
Prefix groups
mean diff: 8.948925511731431e-08
max diff: 5.21540641784668e-07
Timestep group obs primary
mean diff: 2.529011680962867e-06
max diff: 2.396106719970703e-05
Timestep group obs wrist
mean diff: 1.2871533954239567e-06
max diff: 1.1831521987915039e-05
Timestep group obs task language
mean diff: 8.948925511731431e-08
max diff: 5.21540641784668e-07
TImestep group readout
mean diff: 0.0
max diff: 0.0
Transformer output readout
mean diff: 0.000528939941432327
max diff: 0.003875255584716797
Output action
mean diff: 0.010795089416205883
max diff: 0.07824158668518066
==================================================
Jax action: [[[-1.0992778e-03 -1.9052917e-04 -1.5777939e-03  1.1988581e-04
   -2.5188070e-04  8.9190584e-03  1.0005574e+00]
  [-1.0591345e-03  9.5277937e-06 -1.8663091e-03 -9.4582138e-05
    2.6988535e-05  7.6952660e-03  9.9499583e-01]
  [-1.0775316e-03 -2.2007209e-04 -1.9080547e-03  4.6080755e-04
   -4.4456596e-04  7.5108083e-03  1.0059277e+00]
  [-1.3929720e-03 -5.6658973e-05 -2.0325133e-03  3.1690890e-04
   -2.7163411e-04  7.8267800e-03  1.0000223e+00]]]
PyTorch action: [[-1.0929997e-03 -1.9189580e-04 -1.5653233e-03  1.2200765e-04
  -2.4986829e-04  8.8705095e-03  1.0759373e+00]
 [-1.0527084e-03  8.9068926e-06 -1.8505100e-03 -9.4457020e-05
   2.7643400e-05  7.6572932e-03  1.0732374e+00]
 [-1.0684899e-03 -2.1767950e-04 -1.8900186e-03  4.5751416e-04
  -4.4464995e-04  7.4724685e-03  1.0786003e+00]
 [-1.3842678e-03 -5.7903701e-05 -2.0171027e-03  3.1517821e-04
  -2.6703652e-04  7.7819237e-03  1.0757086e+00]]
```
