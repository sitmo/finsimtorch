"""Fractional Brownian Motion simulation using PyTorch."""

from typing import Iterable, Union
import numpy as np
import torch
from scipy.linalg import cholesky


def generate_fbm_increments(
    hurst: float,
    time_points: np.ndarray,
    num_paths: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate fractional Brownian motion increments.

    Parameters
    ----------
    hurst : float
        Hurst parameter (0 < H < 1)
    time_points : np.ndarray
        Time points for the fBM paths (must be sorted, positive)
    num_paths : int
        Number of Monte Carlo paths to generate
    device : torch.device
        PyTorch device for computations
    dtype : torch.dtype
        Data type for returned tensors

    Returns
    -------
    torch.Tensor
        Shape (len(time_points), num_paths) containing fBM paths
    """
    n = len(time_points)
    
    # Covariance matrix for fBM
    cov_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            cov_matrix[i, j] = 0.5 * (
                time_points[i] ** (2 * hurst)
                + time_points[j] ** (2 * hurst)
                - abs(time_points[i] - time_points[j]) ** (2 * hurst)
            )
    
    # Cholesky decomposition
    try:
        L = cholesky(cov_matrix, lower=True)
    except np.linalg.LinAlgError:
        # Fallback to approximate method if Cholesky fails
        return _generate_approximate_fbm(hurst, time_points, num_paths, device, dtype)
    
    # Generate standard normal random variables
    Z = torch.randn(n, num_paths, device=device, dtype=dtype)
    
    # Transform to fBM
    fbm_paths = torch.tensor(L @ Z.numpy(), device=device, dtype=dtype)
    
    return fbm_paths


def _generate_approximate_fbm(
    hurst: float,
    time_points: np.ndarray,
    num_paths: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate fBM using approximate method."""
    n = len(time_points)
    
    # Use Davies-Harte method approximation
    dt = np.diff(time_points, prepend=0)
    fbm_increments = torch.zeros((n, num_paths), device=device, dtype=dtype)
    
    for i in range(n):
        if i == 0:
            fbm_increments[i] = torch.zeros(num_paths, device=device, dtype=dtype)
        else:
            # Approximate increment
            dt_i = dt[i]
            increment = dt_i ** hurst * torch.randn(num_paths, device=device, dtype=dtype)
            fbm_increments[i] = fbm_increments[i-1] + increment
    
    return fbm_increments


class FractionalBrownianMotion:
    """
    Fractional Brownian Motion simulator.

    Parameters
    ----------
    hurst : float
        Hurst parameter (0 < H < 1)
    device : str | torch.device | None, optional
        PyTorch device for computations. Default 'cpu'.
    dtype : torch.dtype, optional
        Data type for returned tensors. Default torch.float32.
    """

    def __init__(
        self,
        hurst: float,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        if not (0 < hurst < 1):
            raise ValueError("Hurst parameter must be in (0, 1)")
        
        self.hurst = float(hurst)
        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype
        self._eps = 1e-8

    def paths(
        self,
        time_points: Iterable[Union[int, float]],
        num_paths: int,
        method: str = "cholesky",
    ) -> torch.Tensor:
        """
        Generate fBM paths for given time points.

        Parameters
        ----------
        time_points : array-like
            Time points for the fBM paths (must be sorted, positive)
        num_paths : int
            Number of Monte Carlo paths to simulate.
        method : str
            Generation method: "cholesky" (exact) or "approximate"

        Returns
        -------
        torch.Tensor
            Shape (len(time_points), num_paths) containing fBM paths
        """
        time_points = np.asarray(time_points)
        n = len(time_points)

        if method == "cholesky":
            return self._generate_cholesky(num_paths, time_points)
        elif method == "approximate":
            return self._generate_approximate(num_paths, time_points)
        else:
            raise ValueError(f"Unknown method: {method}")

    def _generate_cholesky(
        self, num_paths: int, time_points: np.ndarray
    ) -> torch.Tensor:
        """Generate fBM using exact Cholesky decomposition."""
        n = len(time_points)

        # Covariance matrix for fBM
        cov_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                cov_matrix[i, j] = 0.5 * (
                    time_points[i] ** (2 * self.hurst)
                    + time_points[j] ** (2 * self.hurst)
                    - abs(time_points[i] - time_points[j]) ** (2 * self.hurst)
                )

        # Cholesky decomposition
        try:
            L = cholesky(cov_matrix, lower=True)
        except np.linalg.LinAlgError:
            # Fallback to approximate method if Cholesky fails
            return self._generate_approximate(num_paths, time_points)

        # Generate standard normal random variables
        Z = torch.randn(n, num_paths, device=self.device, dtype=self.dtype)

        # Transform to fBM
        fbm_paths = torch.tensor(L @ Z.numpy(), device=self.device, dtype=self.dtype)

        return fbm_paths

    def _generate_approximate(
        self, num_paths: int, time_points: np.ndarray
    ) -> torch.Tensor:
        """Generate fBM using approximate method."""
        n = len(time_points)

        # Use Davies-Harte method approximation
        dt = np.diff(time_points, prepend=0)
        fbm_increments = torch.zeros((n, num_paths), device=self.device, dtype=self.dtype)

        for i in range(n):
            if i == 0:
                fbm_increments[i] = torch.zeros(num_paths, device=self.device, dtype=self.dtype)
            else:
                # Approximate increment
                dt_i = dt[i]
                increment = dt_i ** self.hurst * torch.randn(num_paths, device=self.device, dtype=self.dtype)
                fbm_increments[i] = fbm_increments[i-1] + increment

        return fbm_increments

    def quantiles(
        self,
        time_points: Iterable[Union[int, float]],
        num_paths: int,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Compute quantiles of fBM paths at specified time points.

        Parameters
        ----------
        time_points : array-like
            Time points for the fBM paths (must be sorted, positive)
        num_paths : int
            Number of Monte Carlo paths to simulate.
        lo : float, default 0.001
            Lower quantile bound (e.g., 0.001 for 0.1th percentile).
        hi : float, default 0.999
            Upper quantile bound (e.g., 0.999 for 99.9th percentile).
        size : int, default 512
            Number of quantiles to compute between lo and hi.
        center : bool, default True
            If True, subtract the average of paths before computing quantiles.

        Returns
        -------
        torch.Tensor
            Shape (len(time_points), size) containing quantiles at each specified time point.
            First row contains quantiles at time_points[0], etc.
        """
        import numpy as np

        t = np.asarray(time_points)
        if len(t) == 0:
            return torch.empty((0, size), device=self.device, dtype=self.dtype)

        # Validate time points
        if not np.all(t > 0):
            raise ValueError("All time points must be positive")
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted")

        # Validate quantile bounds
        if not (0 <= lo < hi <= 1):
            raise ValueError("Quantile bounds must satisfy 0 <= lo < hi <= 1")

        # Generate paths
        paths = self.paths(time_points, num_paths)

        # Compute quantiles
        quantile_levels = torch.linspace(lo, hi, size, device=self.device, dtype=self.dtype)
        result = torch.empty((len(t), size), device=self.device, dtype=self.dtype)

        for i in range(len(t)):
            path_values = paths[i, :]
            if center:
                path_values = path_values - path_values.mean()
            result[i] = torch.quantile(path_values, quantile_levels)

        return result