from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

from ..distributions.skew_student_t import HansenSkewedT


# -------- Utilities --------

def _validate_time_points(time_points: Iterable[int]) -> np.ndarray:
    t = np.asarray(list(time_points), dtype=int)
    if t.size == 0:
        return t
    if (t <= 0).any():
        raise ValueError("All time points must be positive integers.")
    if np.any(np.diff(t) < 0):
        raise ValueError("Time points must be sorted in ascending order.")
    return t


def _linspace_tensor(
    lo: float, hi: float, size: int, *, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    if not (0.0 <= lo < hi <= 1.0):
        raise ValueError("Quantile bounds must satisfy 0 <= lo < hi <= 1.")
    if size <= 0:
        raise ValueError("`size` must be a positive integer.")
    return torch.linspace(lo, hi, size, device=device, dtype=dtype)


# -------- Reduced / shape-only simulator --------

@dataclass
class GjrGarchReduced:
    """
    Shape-only (normalized) GJR–GARCH(1,1) simulator.

    Model (normalized):
        r'_t = ε'_t,    ε'_t = σ'_t z_t,   z_t ~ S_t_{ν,λ}(0,1)
        (σ'_{t+1})^2 = ω' + (α + γ * I[ε'_t < 0]) * (ε'_t)^2 + β * (σ'_t)^2

    with:
        κ = α + β + γ * P0,  P0 = E[z^2 | z<0]
        ω' = 1 - κ
        (σ'_0)^2 = initial_variance  (typically 1.0)
    """
    alpha: float
    gamma: float
    beta: float
    initial_variance: float = 1.0
    eta: float = 100.0
    lam: float = 0.0
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32

    # runtime state (private)
    _eps: float = field(default=1e-8, init=False, repr=False)
    _ti: int = field(default=0, init=False, repr=False)
    _var: torch.Tensor | None = field(default=None, init=False, repr=False)          # (σ'_t)^2
    _cum_returns: torch.Tensor | None = field(default=None, init=False, repr=False)  # Σ r'_s
    dist: HansenSkewedT = field(init=False, repr=False)
    P0: float = field(init=False)
    kappa: float = field(init=False)
    omega: float = field(init=False)  # ω' = 1 - κ

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.gamma = float(self.gamma)
        self.initial_variance = float(self.initial_variance)

        # Distribution and moments
        self.dist = HansenSkewedT(eta=float(self.eta), lam=float(self.lam), device=self.device)
        self.P0 = float(self.dist.second_moment_left())  # E[z^2 | z<0]

        # Stationarity / parameters
        self.kappa = self.alpha + self.beta + self.gamma * self.P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa:.6g} >= 1.")
        self.omega = 1.0 - self.kappa

    # ---- internal stepping ----

    def _reset(self, num_paths: int) -> None:
        self._ti = 0
        iv = max(self.initial_variance, self._eps)
        self._var = torch.full((num_paths,), iv, device=self.device, dtype=self.dtype)
        self._cum_returns = torch.zeros((num_paths,), device=self.device, dtype=self.dtype)

    def _require_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._var is None or self._cum_returns is None:
            raise RuntimeError("Simulator state is uninitialized. Call `_reset(num_paths)` first.")
        return self._var, self._cum_returns

    def _step(self) -> tuple[torch.Tensor, int]:
        self._ti += 1
        var, cum = self._require_state()

        z = self.dist.rvs(var.shape[0])  # standardized shocks
        shock = z * torch.sqrt(var)      # ε'_t

        # cum returns
        cum += shock

        # variance update to t+1
        is_neg = (shock < 0).to(var.dtype)
        var = self.omega + (self.alpha + self.gamma * is_neg) * shock.square() + self.beta * var
        self._var = torch.clamp(var, min=self._eps)

        return cum, self._ti

    # ---- public API ----

    def paths(self, time_points: Iterable[int], num_paths: int) -> torch.Tensor:
        """
        Returns cumulative normalized returns at requested times.
        Shape: (len(time_points), num_paths)
        """
        t = _validate_time_points(time_points)
        if t.size == 0:
            return torch.empty((0, num_paths), device=self.device, dtype=self.dtype)

        self._reset(num_paths)
        out = torch.empty((len(t), num_paths), device=self.device, dtype=self.dtype)

        for i, target_t in enumerate(t):
            while self._ti < int(target_t):
                self._step()
            out[i] = self._require_state()[1]  # cumulative returns

        return out

    def quantiles(
        self,
        time_points: Iterable[int],
        num_paths: int,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Row i contains quantiles of Σ r'_s at time_points[i].
        Shape: (len(time_points), size)
        
        Memory-efficient implementation: computes quantiles step-by-step
        without materializing all paths.
        """
        t = _validate_time_points(time_points)
        if t.size == 0:
            return torch.empty((0, size), device=self.device, dtype=self.dtype)

        q_levels = _linspace_tensor(lo, hi, size, device=self.device, dtype=self.dtype)
        
        # Initialize result tensor
        result = torch.empty((len(t), size), device=self.device, dtype=self.dtype)
        
        # Reset simulator
        self._reset(num_paths)
        
        # Track which time points we need to collect and current result index
        target_times = set(t)
        result_idx = 0
        
        # Step through simulation
        while self._ti < max(t):
            self._step()
            
            # Check if current time is one we need
            if self._ti in target_times:
                cum_returns = self._require_state()[1]  # Get cumulative returns
                
                if center:
                    cum_returns = cum_returns - cum_returns.mean()
                
                # Compute quantiles for this time point
                result[result_idx] = torch.quantile(cum_returns, q_levels)
                result_idx += 1
        
        return result


# -------- Full / raw simulator (thin wrapper) --------

@dataclass
class GjrGarch:
    """
    Full GJR–GARCH(1,1) simulator with mean and raw variance.

    Mapping to reduced model:
        α' = α, β' = β, γ' = γ, ω' = 1 - κ
        v = ω / (1 - κ),  r_t = μ + √v * r'_t,  σ_t^2 = v * (σ'_t)^2
    """
    mu: float
    omega: float
    alpha: float
    gamma: float
    beta: float
    initial_variance: float
    eta: float = 100.0
    lam: float = 0.0
    device: torch.device | str | None = None
    dtype: torch.dtype = torch.float32
    reduced_engine_cls: type[GjrGarchReduced] = GjrGarchReduced

    # derived
    reduced: GjrGarchReduced = field(init=False, repr=False)
    kappa: float = field(init=False)
    v: float = field(init=False)
    sqrt_v: float = field(init=False)

    def __post_init__(self) -> None:
        self.device = torch.device(self.device) if self.device is not None else torch.device("cpu")
        self.mu = float(self.mu)
        self.omega = float(self.omega)
        self.alpha = float(self.alpha)
        self.beta = float(self.beta)
        self.gamma = float(self.gamma)
        self.initial_variance = float(self.initial_variance)

        # Build reduced engine; it owns the distribution and P0
        self.reduced = self.reduced_engine_cls(
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            initial_variance=1.0,  # placeholder; we set the correct normalized σ0^2 below
            eta=self.eta,
            lam=self.lam,
            device=self.device,
            dtype=self.dtype,
        )

        # Stationarity via reduced.P0
        self.kappa = self.alpha + self.beta + self.gamma * self.reduced.P0
        if not (self.kappa < 1.0):
            raise ValueError(f"Stationarity violated: kappa={self.kappa:.6g} >= 1.")

        # Long-run variance v and normalized initial variance
        denom = (1.0 - self.kappa)
        self.v = self.omega / denom
        if not (self.v > 0.0):
            raise ValueError(f"Long-run variance must be positive; got v={self.v:.6g}.")
        self.sqrt_v = float(self.v ** 0.5)

        # Configure reduced engine parameters that depend on κ and v
        self.reduced.omega = 1.0 - self.kappa
        self.reduced.initial_variance = max(self.initial_variance / self.v, self.reduced._eps)

    # ---- public API: thin mapping to reduced ----

    def paths(self, time_points: Iterable[int], num_paths: int) -> torch.Tensor:
        """
        Returns cumulative raw returns at requested times.
        Shape: (len(time_points), num_paths)
        """
        t = _validate_time_points(time_points)
        if t.size == 0:
            return torch.empty((0, num_paths), device=self.device, dtype=self.dtype)

        reduced_paths = self.reduced.paths(t, num_paths)  # (T, N)
        # r_t = μ * t + √v * r'_t  (broadcast t over paths)
        t_tensor = torch.as_tensor(t, device=self.device, dtype=self.dtype).unsqueeze(1)
        return self.mu * t_tensor + self.sqrt_v * reduced_paths

    def quantiles(
        self,
        time_points: Iterable[int],
        num_paths: int,
        lo: float = 0.001,
        hi: float = 0.999,
        size: int = 512,
        center: bool = True,
    ) -> torch.Tensor:
        """
        Row i contains quantiles of Σ r_s at time_points[i].
        Shape: (len(time_points), size)
        """
        t = _validate_time_points(time_points)
        if t.size == 0:
            return torch.empty((0, size), device=self.device, dtype=self.dtype)

        # We can map quantiles directly without simulating raw paths:
        # - If center=False: Q_raw = μ * t + √v * Q_reduced
        # - If center=True:  Q_raw_centered = √v * Q_reduced_centered
        reduced_q = self.reduced.quantiles(t, num_paths, lo=lo, hi=hi, size=size, center=center)

        if center:
            return self.sqrt_v * reduced_q

        t_tensor = torch.as_tensor(t, device=self.device, dtype=self.dtype).unsqueeze(1)
        return self.mu * t_tensor + self.sqrt_v * reduced_q
