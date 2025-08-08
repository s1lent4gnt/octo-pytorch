# Octo VLA Pytorch

This repository contains a PyTorch implementation of the Octo VLA model.

## Installation

To get started, follow these steps to set up the environment and install the required dependencies.

```bash
# Clone the repo
git clone https://github.com/s1lent4gnt/octo-pytorch.git

# Add Octo JAX submodule
git submodule update --init --recursive

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
mean diff: 3.6370201996760443e-06
max diff: 1.8864870071411133e-05
Output action
mean diff: 1.0715926634929929e-07
max diff: 4.936009645462036e-07
==================================================
Jax action: [[[-1.0992778e-03 -1.9052917e-04 -1.5777939e-03  1.1988581e-04
   -2.5188070e-04  8.9190584e-03  1.0005574e+00]
  [-1.0591345e-03  9.5277937e-06 -1.8663091e-03 -9.4582138e-05
    2.6988535e-05  7.6952660e-03  9.9499583e-01]
  [-1.0775316e-03 -2.2007209e-04 -1.9080547e-03  4.6080755e-04
   -4.4456596e-04  7.5108083e-03  1.0059277e+00]
  [-1.3929720e-03 -5.6658973e-05 -2.0325133e-03  3.1690890e-04
   -2.7163411e-04  7.8267800e-03  1.0000223e+00]]]
PyTorch action: [[-1.0992435e-03 -1.9059055e-04 -1.5777745e-03  1.1989131e-04
  -2.5185541e-04  8.9187222e-03  1.0005577e+00]
 [-1.0590951e-03  9.4702336e-06 -1.8663026e-03 -9.4573530e-05
   2.7001981e-05  7.6948088e-03  9.9499589e-01]
 [-1.0774876e-03 -2.2013187e-04 -1.9080514e-03  4.6082804e-04
  -4.4454675e-04  7.5103147e-03  1.0059276e+00]
 [-1.3929410e-03 -5.6704972e-05 -2.0325142e-03  3.1692439e-04
  -2.7163568e-04  7.8263544e-03  1.0000226e+00]]
```
## Citation

```bibtex
@inproceedings{octo_2023,
    title={Octo: An Open-Source Generalist Robot Policy},
    author = {{Octo Model Team} and Dibya Ghosh and Homer Walke and Karl Pertsch and Kevin Black and Oier Mees and Sudeep Dasari and Joey Hejna and Charles Xu and Jianlan Luo and Tobias Kreiman and {You Liang} Tan and Pannag Sanketi and Quan Vuong and Ted Xiao and Dorsa Sadigh and Chelsea Finn and Sergey Levine},
    booktitle = {Proceedings of Robotics: Science and Systems},
    address  = {Delft, Netherlands},
    year = {2024},
}
```
