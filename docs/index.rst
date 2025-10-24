Welcome to finsimtorch's documentation!
========================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   contributing

finsimtorch is a GPU-accelerated Monte Carlo simulation library for quantitative finance using PyTorch. 
It provides efficient implementations of well-known stochastic models like GJR-GARCH, Rough Heston, and Rough Bergomi.

Key Features
============

* **GPU Acceleration**: Leverages PyTorch for efficient GPU-based computations
* **Modern Stochastic Models**: Implementation of cutting-edge financial models
* **Easy to Use**: Clean, intuitive API for Monte Carlo simulation
* **Well Tested**: Comprehensive test suite and documentation

Quick Start
===========

.. code-block:: python

   import torch
   from finsimtorch import GJRGARCH_torch, HansenSkewedT
   
   # Create a GJR-GARCH model
   dist = HansenSkewedT(eta=10.0, lam=0.1)
   model = GJRGARCH_torch(
       mu=0.0, omega=0.1, alpha=0.1, gamma=0.05, beta=0.8,
       sigma0_sq=1.0, dist=dist
   )
   
   # Simulate paths
   model.reset(1000)  # 1000 paths
   returns = model.path([1, 5, 10, 20])  # Returns at different time points

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
