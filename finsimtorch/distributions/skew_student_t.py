import numpy as np
import torch
from scipy.integrate import quad
from scipy.stats import t as _t_dist

from ..utils.torch_utils import get_best_device


class HansenSkewedT:
    r"""
    Hansen's skewed Student-t distribution (Hansen, 1994).

    Parameters
    ----------
    eta : float
        Degrees of freedom \(\nu>2\).
    lam : float
        Skewness parameter \(\lambda \in (-1,1)\).
    device : str | torch.device | None
        Torch device for returned tensors and internal tables.

    Notes
    -----
    Define
    \[
      c = \frac{\Gamma\!\left(\frac{\nu+1}{2}\right)}
               {\sqrt{\pi(\nu-2)}\,\Gamma\!\left(\frac{\nu}{2}\right)},\quad
      a = \frac{4\lambda c(\nu-2)}{\nu-1},\quad
      b = \sqrt{1 + 3\lambda^2 - a^2},\quad
      s = \sqrt{1 - \frac{2}{\nu}}.
    \]
    The pdf is
    \[
      f(x) = b\,c \left(1 + \frac{1}{\nu-2}
      \left[\frac{b x + a}{1+\lambda\,\mathrm{sign}(x + a/b)}\right]^2\right)^{-(\nu+1)/2}.
    \]
    """

    # ------------------------ init & utilities ------------------------

    def __init__(
        self,
        eta: float = 10.0,
        lam: float = 0.0,
        device: str | torch.device | None = None,
    ):
        if eta <= 2:
            raise ValueError("eta (nu) must be > 2.")
        if not (-1.0 < lam < 1.0):
            raise ValueError("lam must be in (-1, 1).")

        self.eta = float(eta)
        self.lam = float(lam)
        self.device = torch.device(device) if device else get_best_device()

        # constants (floats)
        self.c: float = float(
            torch.exp(
                torch.lgamma(torch.tensor((eta + 1.0) / 2.0))
                - torch.lgamma(torch.tensor(eta / 2.0))
                - 0.5 * torch.log(torch.tensor(np.pi * (eta - 2.0)))
            ).item()
        )
        self.a: float = float(4.0 * lam * self.c * (eta - 2.0) / (eta - 1.0))
        self.b: float = float(np.sqrt(1.0 + 3.0 * lam * lam - self.a * self.a))
        self.scale: float = float(np.sqrt(1.0 - 2.0 / eta))
        self._log_bc: float = float(np.log(self.b * self.c))

        # inverse CDF table (for rvs)
        self.u_grid: torch.Tensor | None = None
        self.sample_grid: torch.Tensor | None = None

    @staticmethod
    def _to_tensor(
        x: torch.Tensor | np.ndarray | float,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=device, dtype=dtype)
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device=device, dtype=dtype)
        return torch.tensor(x, device=device, dtype=dtype)

    # ------------------------ sampling ------------------------

    def _precompute_icdf_table(self, num_grid: int = 100_000) -> None:
        """
        Precompute a linear-interp table for inverse-transform sampling.

        Mapping:
        \\[
          p =
          \begin{cases}
            \frac{u}{1-\\lambda}, & u<\frac{1-\\lambda}{2} \\
            \frac12 + \frac{u - (1-\\lambda)/2}{1+\\lambda}, & \text{otherwise}
          \\end{cases}\\!,
          \\quad t = T_\nu^{-1}(p),\\quad
          x = \frac{(1 + \\lambda\\,\\mathrm{sign}(u - \frac{1-\\lambda}{2}))\\, s\\, t - a}{b}.
        \\]
        """
        u_grid = np.linspace(1e-6, 1.0 - 1e-6, num_grid, dtype=np.float32)
        Fm = (1.0 - self.lam) / 2.0
        Fl = 1.0 - self.lam
        Fr = 1.0 + self.lam

        signed = np.sign(u_grid - Fm).astype(np.float32)
        u_trans = np.where(signed < 0.0, u_grid / Fl, 0.5 + (u_grid - Fm) / Fr)

        # SciPy t-PPF (CPU), then back to torch
        t_icdf_np = _t_dist.ppf(u_trans, df=self.eta).astype(np.float32)
        t_icdf = torch.from_numpy(t_icdf_np).to(self.device)
        signed_t = torch.from_numpy(signed).to(self.device)

        scaled = t_icdf * (1.0 + signed_t * self.lam) * self.scale
        samples = (scaled - self.a) / self.b

        self.u_grid = torch.from_numpy(u_grid).to(self.device)
        self.sample_grid = samples

    def rvs(self, size: int) -> torch.Tensor:
        r"""
        Random variates via inverse transform and precomputed table.

        \[
          x = \frac{(1 + \lambda\,\mathrm{sign}(u - \frac{1-\lambda}{2}))\, s\, T_\nu^{-1}(p(u)) - a}{b}.
        \]
        """
        if self.u_grid is None:
            self._precompute_icdf_table()

        umin = float(self.u_grid[0])
        umax = float(self.u_grid[-1])
        N = int(self.u_grid.numel())
        delta = (umax - umin) / (N - 1)

        u = torch.rand(size, device=self.device).clamp_(min=umin, max=umax)
        scaled_index = (u - umin) / delta
        idx = scaled_index.long().clamp_max(N - 2)
        frac = scaled_index - idx

        x0 = self.sample_grid[idx]
        x1 = self.sample_grid[idx + 1]
        return x0 + frac * (x1 - x0)

    # ------------------------ pdf / cdf ------------------------

    def logpdf(self, x: torch.Tensor | np.ndarray | float) -> torch.Tensor:
        r"""
        \[
          \log f(x) = \log(bc)
          - \frac{\nu+1}{2}\,\log\!\left(1+\frac{1}{\nu-2}
          \left[\frac{b x + a}{1+\lambda\,\mathrm{sign}(x+a/b)}\right]^2\right).
        \]
        """
        x = self._to_tensor(x, self.device)
        sgn = torch.sign(x + self.a / self.b)
        scale = 1.0 + self.lam * sgn
        arg = (self.b * x + self.a) / scale

        return torch.tensor(self._log_bc, device=self.device) - (
            (self.eta + 1.0) / 2.0
        ) * torch.log1p((arg * arg) / (self.eta - 2.0))

    def pdf(self, x: torch.Tensor | np.ndarray | float) -> torch.Tensor:
        r"""\(f(x)=\exp(\log f(x))\)."""
        return torch.exp(self.logpdf(x))

    def cdf(self, x: torch.Tensor | np.ndarray | float) -> torch.Tensor:
        r"""
        Piecewise mapping to t-CDF. Let \(z = b x + a\), \(s=\sqrt{1-2/\nu}\):
        \[
          F(x) =
          \begin{cases}
            (1-\lambda)\,T_\nu\!\left(\dfrac{z}{(1-\lambda)s}\right), & x < -a/b,\\[6pt]
            \dfrac{1-\lambda}{2} + (1+\lambda)\!\left[T_\nu\!\left(\dfrac{z}{(1+\lambda)s}\right)-\tfrac{1}{2}\right], & x \ge -a/b.
          \end{cases}
        \]
        """
        x = self._to_tensor(x, self.device)
        z = self.b * x + self.a
        out = torch.empty_like(x)

        mask = x < (-self.a / self.b)

        # Left piece
        if mask.any():
            t1 = (z[mask] / ((1.0 - self.lam) * self.scale)).detach().cpu().numpy()
            cdf_left = _t_dist.cdf(t1, df=self.eta).astype(np.float32)
            out[mask] = (1.0 - self.lam) * torch.from_numpy(cdf_left).to(self.device)

        # Right piece
        if (~mask).any():
            t2 = (z[~mask] / ((1.0 + self.lam) * self.scale)).detach().cpu().numpy()
            cdf_right = _t_dist.cdf(t2, df=self.eta).astype(np.float32)
            cdf_right_t = torch.from_numpy(cdf_right).to(self.device)
            out[~mask] = (1.0 - self.lam) / 2.0 + (1.0 + self.lam) * (cdf_right_t - 0.5)

        return out

    def F0(self) -> torch.Tensor:
        r"""Convenience: \(F(0)\)."""
        return self.cdf(torch.tensor(0.0, device=self.device))

    # ------------------------ internal: t helpers (SciPy) ------------------------

    def _t_ppf_torch(
        self, p: torch.Tensor | np.ndarray | float, df: float
    ) -> torch.Tensor:
        r"""
        Accurate \(T_\nu^{-1}(p)\) via SciPy.
        """
        # Always compute via SciPy on CPU, then move back
        if isinstance(p, torch.Tensor):
            p_np = p.detach().cpu().numpy().astype(np.float32)
            res_np = _t_dist.ppf(p_np, df=df).astype(np.float32)
            return torch.from_numpy(res_np).to(self.device)
        else:
            p_np = np.asarray(p, dtype=np.float32)
            res_np = _t_dist.ppf(p_np, df=df).astype(np.float32)
            return torch.from_numpy(res_np).to(self.device)

    # (Kept for completeness; not used when SciPy is available)
    def _t_pdf_torch(self, t: torch.Tensor, df: float) -> torch.Tensor:
        r"""
        Student-t pdf:
        \[
          f(t)=\frac{\Gamma\!\big(\frac{\nu+1}{2}\big)}{\sqrt{\nu\pi}\,\Gamma\!\big(\frac{\nu}{2}\big)}
          \left(1+\frac{t^2}{\nu}\right)^{-(\nu+1)/2}.
        \]
        """
        log_norm = (
            torch.lgamma(torch.tensor((df + 1.0) / 2.0, device=t.device))
            - torch.lgamma(torch.tensor(df / 2.0, device=t.device))
            - 0.5 * torch.log(torch.tensor(df * np.pi, device=t.device))
        )
        log_density = log_norm - ((df + 1.0) / 2.0) * torch.log1p((t * t) / df)
        return torch.exp(log_density)

    def second_moment_left(
        self, epsabs: float = 1e-10, epsrel: float = 1e-8, limit: int = 200
    ) -> torch.Tensor:
        r"""
        Compute the one-sided second moment
        \[
          \mathbb{E}\big[\epsilon^2\,\mathbf{1}\{\epsilon<0\}\big]
          \;=\;\int_{-\infty}^{0} x^2 f(x)\,dx
        \]
        by accurate numerical integration.

        Notes
        -----
        The integrand has a kink at \(x^\star=-a/b\) (where \(\mathrm{sign}(x + a/b)\) flips).
        We split the integral at \(x^\star\) if \(x^\star\in(-\infty,0)\) for better stability.
        """
        a, b, c, nu, lam = self.a, self.b, self.c, self.eta, self.lam

        def pdf_scalar(x: float) -> float:
            # scalar, numpy math
            sgn: float = float(np.sign(x + a / b))
            scale: float = 1.0 + lam * sgn
            arg: float = (b * x + a) / scale
            result: float = (b * c) * (1.0 + (arg * arg) / (nu - 2.0)) ** (
                -(nu + 1.0) / 2.0
            )
            return result

        def integrand(x: float) -> float:
            px = pdf_scalar(x)
            return (x * x) * px

        x_star = -a / b  # kink
        total: float = 0.0

        # Case 1: kink lies left of 0 -> split
        if x_star < 0.0:
            val1, _ = quad(
                integrand, -np.inf, x_star, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            val2, _ = quad(
                integrand, x_star, 0.0, epsabs=epsabs, epsrel=epsrel, limit=limit
            )
            total = val1 + val2
        else:
            # No kink in (-inf, 0)
            total, _ = quad(
                integrand, -np.inf, 0.0, epsabs=epsabs, epsrel=epsrel, limit=limit
            )

        return torch.tensor(total, dtype=torch.float32, device=self.device)

    def R_f(
        self, epsabs: float = 1e-10, epsrel: float = 1e-8, limit: int = 200
    ) -> torch.Tensor:
        """
        Compute R(f) = ∫ f(x)^2 dx for Hansen skew-t (mean 0, std 1).
        Splits at the kink x* = -a/b for stability.
        """
        a, b, c, nu, lam = self.a, self.b, self.c, self.eta, self.lam

        def pdf_scalar(x: float) -> float:
            # scalar, numpy math; matches self.pdf()
            a_val, b_val, c_val, nu_val, lam_val = (
                float(a),
                float(b),
                float(c),
                float(nu),
                float(lam),
            )
            sgn: float = float(np.sign(x + a_val / b_val))
            scale: float = 1.0 + lam_val * sgn
            arg: float = (b_val * x + a_val) / scale
            return float(
                (b_val * c_val)
                * (1.0 + (arg * arg) / (nu_val - 2.0)) ** (-(nu_val + 1.0) / 2.0)
            )

        def integrand(x: float) -> float:
            fx = pdf_scalar(x)
            return fx * fx

        x_star = -a / b  # kink

        # integrate on (-inf, x_star) and (x_star, inf)
        left, _ = quad(
            integrand, -np.inf, x_star, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        right, _ = quad(
            integrand, x_star, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        total = left + right

        return torch.tensor(total, dtype=torch.float32, device=self.device)

    def A_cdf_gaussian_kernel(self) -> torch.Tensor:
        """
        Bias constant A for CDF-smoothing with a Gaussian kernel (μ2=1):
        A = (1/4) * R(f).
        """
        return 0.25 * self.R_f()

    def R_fprime(
        self, epsabs: float = 1e-10, epsrel: float = 1e-8, limit: int = 400
    ) -> torch.Tensor:
        # numerical derivative of pdf, then square & integrate (split at kink)
        a, b = self.a, self.b
        x_star = -a / b

        def fprime(x: float, h: float = 1e-4) -> float:
            return float(self.pdf(x + h) - self.pdf(x - h)) / (2 * h)

        def integrand(x: float) -> float:
            v = fprime(x)
            return v * v

        left, _ = quad(
            integrand, -np.inf, x_star, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        right, _ = quad(
            integrand, x_star, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return torch.tensor(left + right, dtype=torch.float32, device=self.device)

    def C_integrated_variance(
        self, epsabs: float = 1e-10, epsrel: float = 1e-8, limit: int = 400
    ) -> torch.Tensor:
        a, b = self.a, self.b
        x_star = -a / b

        def integrand(x: float) -> float:
            Fx = float(self.cdf(x))
            return Fx * (1.0 - Fx)

        left, _ = quad(
            integrand, -np.inf, x_star, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        right, _ = quad(
            integrand, x_star, np.inf, epsabs=epsabs, epsrel=epsrel, limit=limit
        )
        return torch.tensor(left + right, dtype=torch.float32, device=self.device)

    def cdf_optimal_h_gaussian(self, n: int) -> torch.Tensor:
        Rfprime = self.R_fprime()
        C = self.C_integrated_variance()
        # D = (1/4) * R(f') for Gaussian kernel
        return (C / (Rfprime * n)) ** 0.25
