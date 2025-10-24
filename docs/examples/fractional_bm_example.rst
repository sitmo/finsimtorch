Fractional Brownian Motion Example
===================================

This example shows how to generate fractional Brownian motion paths for rough volatility models.

.. code-block:: python

   import torch
   import numpy as np
   import matplotlib.pyplot as plt
   from finsimtorch import FractionalBrownianMotion
   
   # Create fBM with different Hurst parameters
   hurst_values = [0.1, 0.3, 0.5, 0.7, 0.9]
   time_points = np.linspace(0, 1, 100)
   
   plt.figure(figsize=(15, 10))
   
   for i, hurst in enumerate(hurst_values):
       fbm = FractionalBrownianMotion(hurst=hurst, device='cuda')
       paths = fbm.generate_paths(5, time_points)
       
       plt.subplot(2, 3, i+1)
       for j in range(5):
           plt.plot(time_points, paths[j].cpu().numpy(), alpha=0.7)
       plt.title(f'Hurst = {hurst}')
       plt.xlabel('Time')
       plt.ylabel('fBM Value')
       plt.grid(True)
   
   plt.tight_layout()
   plt.show()
   
   # Compare covariance structures
   plt.figure(figsize=(10, 8))
   for i, hurst in enumerate(hurst_values):
       fbm = FractionalBrownianMotion(hurst=hurst, device='cpu')
       cov_matrix = fbm.get_covariance_matrix(time_points[::10])  # Every 10th point
       
       plt.subplot(2, 3, i+1)
       plt.imshow(cov_matrix, cmap='viridis')
       plt.colorbar()
       plt.title(f'Covariance Matrix (H = {hurst})')
   
   plt.tight_layout()
   plt.show()
