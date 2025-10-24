GJR-GARCH Example
=================

This example demonstrates how to use the GJR-GARCH model for simulating financial returns.

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from finsimtorch import GjrGarch, HansenSkewedT
   
   # Set up the model
   dist = HansenSkewedT(eta=10.0, lam=0.1, device='cuda')
   model = GjrGarch(
       mu=0.0,
       omega=0.1,
       alpha=0.1,
       gamma=0.05,
       beta=0.8,
       sigma0_sq=1.0,
       dist=dist
   )
   
   # Simulate 1000 paths for 100 time steps
   model.reset(1000)
   time_points = list(range(1, 101))
   returns = model.path(time_points)
   
   # Plot some sample paths
   plt.figure(figsize=(12, 6))
   for i in range(10):
       plt.plot(time_points, returns[:, i].cpu().numpy(), alpha=0.7)
   plt.xlabel('Time')
   plt.ylabel('Cumulative Returns')
   plt.title('GJR-GARCH Sample Paths')
   plt.grid(True)
   plt.show()
   
   # Compute statistics
   final_returns = returns[-1, :]  # Returns at t=100
   print(f"Mean final return: {final_returns.mean():.4f}")
   print(f"Std final return: {final_returns.std():.4f}")
   print(f"Skewness: {final_returns.skew():.4f}")
   print(f"Kurtosis: {final_returns.kurtosis():.4f}")
