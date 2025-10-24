"""
finsimtorch: GPU-accelerated Monte Carlo simulation for quantitative finance.

A PyTorch-based library for efficient Monte Carlo simulation of stochastic price paths
in quantitative finance, focusing on well-known models like GJR-GARCH, Rough Heston,
and Rough Bergomi.
"""

__version__ = "0.1.0"
__author__ = "simu.ai"
__email__ = "thijs@simu.ai"

# Import main classes for easy access
from .models.base import MonteCarloSimulator
from .models.garch import GjrGarch, GjrGarchReduced
from .models.fractional_bm import FractionalBrownianMotion, generate_fbm_increments
from .distributions.skew_student_t import HansenSkewedT

# Re-export main functionality
__all__ = [
    "MonteCarloSimulator",
    "GjrGarch",
    "GjrGarchReduced", 
    "FractionalBrownianMotion",
    "generate_fbm_increments",
    "HansenSkewedT",
]
