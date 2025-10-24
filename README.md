# finsimtorch

[![CI](https://github.com/simu-ai/finsimtorch/workflows/CI/badge.svg)](https://github.com/simu-ai/finsimtorch/actions)
[![Documentation Status](https://readthedocs.org/projects/finsimtorch/badge/?version=latest)](https://finsimtorch.readthedocs.io/en/latest/?badge=latest)
[![PyPI version](https://badge.fury.io/py/finsimtorch.svg)](https://badge.fury.io/py/finsimtorch)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A GPU-accelerated Monte Carlo simulation library for quantitative finance using PyTorch. finsimtorch provides efficient implementations of well-known stochastic models like GJR-GARCH, Rough Heston, and Rough Bergomi.

## Features

- **GPU Acceleration**: Leverages PyTorch for efficient GPU-based computations
- **Modern Stochastic Models**: Implementation of cutting-edge financial models
- **Easy to Use**: Clean, intuitive API for Monte Carlo simulation
- **Well Tested**: Comprehensive test suite and documentation

## Installation

### From PyPI

```bash
pip install finsimtorch
```

### With Examples

```bash
pip install finsimtorch[examples]
```

### From Source

```bash
git clone https://github.com/simu-ai/finsimtorch.git
cd finsimtorch
pip install -e .
```

## Quick Start

```python
import torch
from finsimtorch import GjrGarch

# Auto-detect best available device
if torch.cuda.is_available():
    device = 'cuda'
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = 'mps'  # MacBook GPU
else:
    device = 'cpu'

# Create a GJR-GARCH model (HansenSkewedT created internally)
model = GjrGarch(
    mu=0.0, omega=0.1, alpha=0.1, gamma=0.05, beta=0.8,
    initial_variance=1.0, device=device
)

# Simulate paths (supports any iterable!)
returns = model.paths(range(1, 21), 1000)  # Memory efficient with range()
```

## Device Support

finsimtorch supports multiple compute devices for GPU acceleration:

- **CUDA**: NVIDIA GPUs (`device='cuda'`)
- **MPS**: Apple Silicon MacBooks (`device='mps'`) 
- **CPU**: Fallback option (`device='cpu'`)

The library automatically detects the best available device, prioritizing GPU acceleration when available.

## Models

### GJR-GARCH

The Glosten-Jagannathan-Runkle GARCH model with asymmetric volatility:

```python
from finsimtorch import GjrGarch

model = GjrGarch(
    mu=0.0,           # Mean return
    omega=0.1,        # Variance intercept
    alpha=0.1,        # ARCH coefficient
    gamma=0.05,       # Asymmetry coefficient
    beta=0.8,         # GARCH coefficient
    initial_variance=1.0,    # Initial variance
    device='cuda'     # PyTorch device: 'cuda', 'mps' (MacBook), or 'cpu'
)

# Simulate 1000 paths (supports range, list, tuple, numpy arrays, etc.)
returns = model.paths(range(1, 51), 1000)
```

### Fractional Brownian Motion

For rough volatility models:

```python
from finsimtorch import FractionalBrownianMotion
import numpy as np

# Create fBM with Hurst parameter H = 0.1 (rough volatility)
fbm = FractionalBrownianMotion(hurst=0.1, device='cuda')

# Generate time points
time_points = np.linspace(0, 1, 100)

# Generate 1000 fBM paths
fbm._reset(1000)
paths = fbm.paths(time_points)
```

## Documentation

Full documentation is available at [finsimtorch.readthedocs.io](https://finsimtorch.readthedocs.io/).

## Examples

Check out the [examples](examples/) directory for Jupyter notebooks demonstrating various use cases.

## Development

### Setup

```bash
git clone https://github.com/simu-ai/finsimtorch.git
cd finsimtorch
poetry install --with dev
pre-commit install
```

### Running Tests

```bash
poetry run pytest
```

### Building Documentation

```bash
cd docs
make html
```

## Contributing

We welcome contributions! Please see our [Contributing Guide](docs/contributing.rst) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use finsimtorch in your research, please cite:

```bibtex
@software{finsimtorch,
  title={finsimtorch: GPU-accelerated Monte Carlo simulation for quantitative finance},
  author={simu.ai},
  year={2024},
  url={https://github.com/simu-ai/finsimtorch}
}
```

## Support

For questions and support, please open an issue on [GitHub](https://github.com/simu-ai/finsimtorch/issues).