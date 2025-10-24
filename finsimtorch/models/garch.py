from typing import List, Tuple, Union, Iterable

import numpy as np
import torch

from ..distributions.skew_student_t import HansenSkewedT as HansenSkewedT_torch


class GjrGarchReduced:
    """
    Shape-only (normalized) GJR–GARCH(1,1) simulator.

    Model (normalized):
        r'_t = ε'_t,    ε'_t = σ'_t z_t,   z_t ~ S_t_{ν,λ}(0,1)
        (σ'_{t+1})^2 = ω' + (α + γ * I[ε'_t < 0]) * (ε'_t)^2 + β * (σ'_t)^2

    with:
        κ = α + β + γ * P0,      P0 = E[z^2 | z < 0]
        ω' = 1 - κ
        Initial variance (σ'_0)^2 = initial_variance  (typically 1.0 in the normalized model)

    Notes
    -----
    - This is the "reduced / shape-only" engine: mean = 0, long-run variance = 1.
    - It advances variance from t -> t+1 using the shock realized at time t.
    """

    def __init__(
        self,
        *,
        alpha: float,
        gamma: float,
        beta: float,
        initial_variance: float = 1.0,
        eta: float = 100.0,
        lam: float = 0.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """
        Parameters
        ----------
        alpha, gamma, beta : float
            GJR–GARCH coefficients.
        initial_variance : float, optional
            Initial normalized variance (σ'_0)^2. Default 1.0.
        eta : float, optional
            Degrees of freedom for Hansen's skewed Student-t distribution. Default 100.0.
        lam : float, optional
            Skewness parameter for Hansen's skewed Student-t distribution. Default 0.0.
        device : str | torch.device | None, optional
            PyTorch device for computations. Default 'cpu'.
        dtype : torch.dtype, optional
            Data type for returned tensors. Default torch.float32.
        """
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.sigma0_sq = float(initial_variance)

        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype
        self._eps = 1e-8

        # Create HansenSkewedT distribution internally
        self.dist = HansenSkewedT_torch(eta=eta, lam=lam, device=self.device)

        P0 = self.dist.second_moment_left()  # E[z^2 | z<0]
        self.P0 = P0
        self.kappa = self.alpha + self.beta + self.gamma * P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa} >= 1.")
        self.omega = 1.0 - self.kappa  # ω' in normalized model

        # Runtime state
        self._ti: int = 0
        self._var: torch.Tensor | None = None  # (σ'_t)^2
        self._cum_returns: torch.Tensor | None = None  # sum of r'_t

    def _reset(self, num_paths: int) -> None:
        """
        Initialize (or reinitialize) the simulator state.

        Parameters
        ----------
        num_paths : int
            Number of independent paths to simulate in parallel.
        """
        self._ti = 0
        sigma0_sq = max(self.sigma0_sq, self._eps)
        self._var = torch.full((num_paths,), sigma0_sq, device=self.device, dtype=self.dtype)
        self._cum_returns = torch.zeros((num_paths,), device=self.device, dtype=self.dtype)

    def _step(self) -> tuple[torch.Tensor, int]:
        """
        Advance the simulation by one time step.

        Returns
        -------
        cum_returns : torch.Tensor, shape (num_paths,)
            Cumulative normalized returns Σ_{s=1}^t r'_s after this step.
        t : int
            Current time index (1-based).
        """
        self._ti += 1

        # Draw standardized shocks z_t and form ε'_t
        num_paths = self._var.shape[0]
        z = self.dist.rvs(num_paths)  # shape (num_paths,)
        if self._var is None:
            raise ValueError("Variance not initialized")
        shock = z * torch.sqrt(self._var)  # ε'_t

        # Update cumulative return r'
        self._cum_returns += shock

        # Update variance to (t+1) using ε'_t
        is_negative = (shock < 0).to(self._var.dtype)
        self._var = (
            self.omega
            + (self.alpha + self.gamma * is_negative) * shock.pow(2)
            + self.beta * self._var
        )
        self._var = torch.clamp(self._var, min=self._eps)

        return self._cum_returns, self._ti

    def paths(
        self, 
        time_points: Iterable[Union[int, float]],
        num_paths: int
    ) -> torch.Tensor:
        """
        Simulate paths up to specified time points and return cumulative returns.

        Parameters
        ----------
        time_points : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return cumulative returns after 4, 10, and 12 steps.
        num_paths : int
            Number of Monte Carlo paths to simulate.

        Returns
        -------
        torch.Tensor, shape (len(time_points), num_paths)
            Cumulative returns at each specified time point.
            First row contains cum_returns after time_points[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(time_points)
        if len(t) == 0:
            return torch.empty((0, num_paths), device=self.device, dtype=self.dtype)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Reset to beginning
        self._reset(num_paths)

        # Initialize result tensor
        result = torch.empty((len(t), num_paths), device=self.device, dtype=self.dtype)

        # Simulate up to each time point
        for i, target_t in enumerate(t):
            # Simulate from current time to target time
            while self._ti < target_t:
                self._step()

            # Store cumulative returns at this time point
            if self._cum_returns is None:
                raise ValueError("Cumulative returns not initialized")
            result[i] = self._cum_returns

        return result

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
        Compute quantiles of cumulative returns at specified time points.

        Parameters
        ----------
        time_points : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return quantiles after 4, 10, and 12 steps.
        num_paths : int
            Number of Monte Carlo paths to simulate.
        lo : float, default 0.001
            Lower quantile bound (e.g., 0.001 for 0.1th percentile).
        hi : float, default 0.999
            Upper quantile bound (e.g., 0.999 for 99.9th percentile).
        size : int, default 512
            Number of quantiles to compute between lo and hi.
        center : bool, default True
            If True, subtract the average of cum_returns before computing quantiles.

        Returns
        -------
        torch.Tensor
            Shape (len(time_points), size) containing quantiles at each specified time point.
            First row contains quantiles after time_points[0] steps, etc.
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


class GjrGarch:
    """
    Full GJR–GARCH(1,1) simulator with mean and variance parameters.

    Model (raw):
        r_t = μ + ε_t,    ε_t = σ_t z_t,   z_t ~ S_t_{ν,λ}(0,1)
        σ_{t+1}^2 = ω + (α + γ * I[ε_t < 0]) * ε_t^2 + β * σ_t^2

    with:
        κ = α + β + γ * P0,      P0 = E[z^2 | z < 0]
        v = ω / (1 - κ)          (long-run variance)
        Initial variance σ_0^2 = initial_variance

    Notes
    -----
    - This is the "full" engine: includes mean μ and raw variance parameters.
    - It uses GjrGarchReduced internally for the normalized simulation.
    - The mapping is:
        α'    = α,    β'    = β,    γ'    = γ
        ω'    = 1 - κ
        r_t   = μ + √v * r'_t
        σ_t^2 = v * (σ'_t)^2
    """

    def __init__(
        self,
        *,
        mu: float,
        omega: float,
        alpha: float,
        gamma: float,
        beta: float,
        initial_variance: float,
        eta: float = 100.0,
        lam: float = 0.0,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        reduced_engine_cls: type = GjrGarchReduced,
    ):
        """
        Parameters
        ----------
        mu, omega, alpha, gamma, beta : float
            Raw GJR–GARCH parameters.
        initial_variance : float
            Initial raw variance σ_0^2.
        eta : float, optional
            Degrees of freedom for Hansen's skewed Student-t distribution. Default 100.0.
        lam : float, optional
            Skewness parameter for Hansen's skewed Student-t distribution. Default 0.0.
        device : str | torch.device | None, optional
            PyTorch device for computations. Default 'cpu'.
        dtype : torch.dtype, optional
            Data type for returned tensors. Default torch.float32.
        reduced_engine_cls : type
            Class implementing the reduced model (default: GjrGarchReduced).
        """
        self.mu = float(mu)
        self.omega = float(omega)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.beta = float(beta)
        self.sigma0_sq = float(initial_variance)

        self.device = torch.device(device) if device else torch.device("cpu")
        self.dtype = dtype
        self._eps = 1e-8

        # Build reduced engine first (it will create the HansenSkewedT distribution)
        self.reduced = reduced_engine_cls(
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            initial_variance=1.0,  # Will be normalized below
            eta=eta,
            lam=lam,
            device=self.device,
            dtype=self.dtype,
        )

        # Get P0 from the reduced engine's distribution
        P0 = self.reduced.dist.second_moment_left()
        self.kappa = self.alpha + self.beta + self.gamma * P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa} >= 1.")

        # Long-run variance v
        self.v = self.omega / (1.0 - self.kappa)
        if not (self.v > 0.0):
            raise ValueError(f"Long-run variance must be positive; got v={self.v}.")
        self.sqrt_v = float(self.v**0.5)

        # Update reduced engine with correct normalized initial variance
        sigma0_sq_reduced = max(self.sigma0_sq / self.v, self._eps)
        self.reduced.sigma0_sq = sigma0_sq_reduced
        self.reduced.omega = 1.0 - self.kappa  # Update omega in reduced engine

        # Raw state mirrors reduced state (but mapped back to raw units)
        self._ti: int = 0
        self._var: torch.Tensor | None = None  # raw σ_t^2

    def _reset(self, num_paths: int) -> None:
        """Initialize (or reinitialize) the simulator state."""
        self._ti = 0
        self.reduced._reset(num_paths)
        # Map reduced variance to raw variance: σ_t^2 = v * (σ'_t)^2
        if self.reduced._var is None:
            raise ValueError("Reduced variance not initialized")
        self._var = self.reduced._var * self.v

    def _step(self) -> tuple[torch.Tensor, int]:
        """
        Advance the simulation by one time step.

        Returns
        -------
        cum_returns : torch.Tensor, shape (num_paths,)
            Cumulative raw returns Σ_{s=1}^t r_s after this step.
        t : int
            Current time index (1-based).
        """
        self._ti += 1

        # Step the reduced model
        reduced_cum_returns, _ = self.reduced._step()

        # Map back to raw units: r_t = μ + √v * r'_t
        raw_cum_returns = self.mu * self._ti + self.sqrt_v * reduced_cum_returns

        # Update raw variance: σ_t^2 = v * (σ'_t)^2
        if self.reduced._var is None:
            raise ValueError("Reduced variance not initialized")
        self._var = self.v * self.reduced._var

        return raw_cum_returns, self._ti

    def paths(
        self, 
        time_points: Iterable[Union[int, float]],
        num_paths: int
    ) -> torch.Tensor:
        """
        Simulate paths up to specified time points and return cumulative returns.

        Parameters
        ----------
        time_points : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return cumulative returns after 4, 10, and 12 steps.
        num_paths : int
            Number of Monte Carlo paths to simulate.

        Returns
        -------
        torch.Tensor, shape (len(time_points), num_paths)
            Cumulative returns at each specified time point.
            First row contains cum_returns after time_points[0] steps, etc.
        """
        import numpy as np

        t = np.asarray(time_points)
        if len(t) == 0:
            return torch.empty((0, num_paths), device=self.device, dtype=self.dtype)

        # Ensure t is sorted
        if not np.all(np.diff(t) >= 0):
            raise ValueError("Time points must be sorted in ascending order")

        # Ensure all time points are positive
        if np.any(t <= 0):
            raise ValueError("All time points must be positive")

        # Reset to beginning
        self._reset(num_paths)

        # Initialize result tensor
        result = torch.empty((len(t), num_paths), device=self.device, dtype=self.dtype)

        # Simulate up to each time point
        for i, target_t in enumerate(t):
            # Simulate from current time to target time
            while self._ti < target_t:
                self._step()

            # Store cumulative returns at this time point
            if self._var is None:
                raise ValueError("Variance not initialized")
            result[i] = self.mu * self._ti + self.sqrt_v * self.reduced._cum_returns

        return result

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
        Compute quantiles of cumulative returns at specified time points.

        Parameters
        ----------
        time_points : array-like
            List or array of integers representing time steps (must be sorted).
            E.g., [4, 10, 12] will return quantiles after 4, 10, and 12 steps.
        num_paths : int
            Number of Monte Carlo paths to simulate.
        lo : float, default 0.001
            Lower quantile bound (e.g., 0.001 for 0.1th percentile).
        hi : float, default 0.999
            Upper quantile bound (e.g., 0.999 for 99.9th percentile).
        size : int, default 512
            Number of quantiles to compute between lo and hi.
        center : bool, default True
            If True, subtract the average of cum_returns before computing quantiles.

        Returns
        -------
        torch.Tensor
            Shape (len(time_points), size) containing quantiles at each specified time point.
            First row contains quantiles after time_points[0] steps, etc.
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