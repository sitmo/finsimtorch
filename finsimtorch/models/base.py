"""Base protocol for Monte Carlo simulators."""

from typing import Protocol, runtime_checkable, Iterable, Union
import torch
import numpy as np


@runtime_checkable
class MonteCarloSimulator(Protocol):
    """Protocol defining the interface for Monte Carlo simulators.
    
    All models must implement:
    - paths(time_points, num_paths): Generate simulation paths at specified time points
    - quantiles(time_points, num_paths, ...): Compute quantiles of paths at specified time points
    
    Attributes:
        device: PyTorch device for computations
        dtype: Data type for returned tensors
    """
    device: torch.device
    dtype: torch.dtype
    
    def paths(
        self, 
        time_points: Iterable[Union[int, float]],
        num_paths: int
    ) -> torch.Tensor:
        """Generate paths at specified time points.
        
        Parameters
        ----------
        time_points : array-like
            Time points (must be sorted, positive)
        num_paths : int
            Number of Monte Carlo paths to simulate
            
        Returns
        -------
        torch.Tensor
            Shape (len(time_points), num_paths) containing values at each time point
            with dtype matching self.dtype
        """
        ...
    
    def quantiles(
        self,
        time_points: Iterable[Union[int, float]],
        num_paths: int,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        center: bool = True,
    ) -> torch.Tensor:
        """Compute quantiles of paths at specified time points.
        
        Parameters
        ----------
        time_points : array-like
            Time points (must be sorted, positive)
        num_paths : int
            Number of Monte Carlo paths to simulate
        lo : float
            Lower quantile bound (default: 0.001)
        hi : float
            Upper quantile bound (default: 0.999)
        size : int
            Number of quantile points (default: 512)
        center : bool
            If True, center by subtracting mean (default: True)
            
        Returns
        -------
        torch.Tensor
            Shape (len(time_points), size) containing quantiles at each time point
            with dtype matching self.dtype
        """
        ...
