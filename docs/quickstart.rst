Quick Start Guide
=================

This guide will help you get started with finsimtorch quickly.

Installation
------------

.. code-block:: bash

   pip install finsimtorch

For examples: ``pip install finsimtorch[examples]``

Basic Usage
-----------

Let's start with a simple example using the GJR-GARCH model:

.. code-block:: python

   import torch
   from finsimtorch import GjrGarch, HansenSkewedT
   
   # Create a skewed Student-t distribution for innovations
   dist = HansenSkewedT(eta=10.0, lam=0.1, device='cuda')
   
   # Set up GJR-GARCH model parameters
   model = GjrGarch(
       mu=0.0,           # Mean return
       omega=0.1,        # Variance intercept
       alpha=0.1,        # ARCH coefficient
       gamma=0.05,       # Asymmetry coefficient
       beta=0.8,         # GARCH coefficient
       sigma0_sq=1.0,    # Initial variance
       dist=dist
   )
   
   # Simulate 1000 paths
   model.reset(1000)
   
   # Get cumulative returns at different time points
   time_points = [1, 5, 10, 20, 50]
   returns = model.path(time_points)
   
   print(f"Returns shape: {returns.shape}")  # (5, 1000)
   print(f"Mean returns: {returns.mean(dim=1)}")

Fractional Brownian Motion
--------------------------

For rough volatility models, you can use fractional Brownian motion:

.. code-block:: python

   from finsimtorch import FractionalBrownianMotion
   import numpy as np
   
   # Create fBM with Hurst parameter H = 0.1 (rough volatility)
   fbm = FractionalBrownianMotion(hurst=0.1, device='cuda')
   
   # Generate time points
   time_points = np.linspace(0, 1, 100)
   
   # Generate 1000 fBM paths
   paths = fbm.generate_paths(1000, time_points)
   
   print(f"fBM paths shape: {paths.shape}")  # (1000, 100)

Advanced Usage
--------------

You can also compute quantiles and other statistics:

.. code-block:: python

   # Compute quantiles of returns
   quantiles = model.quantiles(
       t=[10, 20, 50],
       lo=0.01,    # 1st percentile
       hi=0.99,    # 99th percentile
       size=100    # 100 quantile points
   )
   
   print(f"Quantiles shape: {quantiles.shape}")  # (3, 100)

Performance Tips
----------------

* Use GPU acceleration by setting `device='cuda'`
* For large simulations, consider using mixed precision
* Batch multiple simulations together for better GPU utilization

.. code-block:: python

   # Example with mixed precision
   with torch.cuda.amp.autocast():
       returns = model.path(time_points)
