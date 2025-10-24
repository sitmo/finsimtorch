"""Tests for Fractional Brownian Motion model."""

import pytest
import torch
import numpy as np
from finsimtorch import FractionalBrownianMotion


class TestFractionalBrownianMotion:
    """Test Fractional Brownian Motion model functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a test Fractional Brownian Motion model."""
        return FractionalBrownianMotion(hurst=0.3, device='cpu')
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.hurst == 0.3
        assert model.device == torch.device('cpu')
        assert model.dtype == torch.float32
    
    def test_invalid_hurst(self):
        """Test invalid Hurst parameter."""
        with pytest.raises(ValueError, match="Hurst parameter must be in"):
            FractionalBrownianMotion(hurst=1.5, device='cpu')
        
        with pytest.raises(ValueError, match="Hurst parameter must be in"):
            FractionalBrownianMotion(hurst=0.0, device='cpu')
    
    def test_simulation_cholesky(self, model):
        """Test basic simulation with Cholesky method."""
        time_points = [0.1, 0.5, 1.0]
        paths = model.paths(time_points, 100, method="cholesky")
        assert paths.shape == (3, 100)
        assert paths.dtype == model.dtype
        assert paths.device == model.device
    
    def test_simulation_approximate(self, model):
        """Test basic simulation with approximate method."""
        time_points = [0.1, 0.5, 1.0]
        paths = model.paths(time_points, 100, method="approximate")
        assert paths.shape == (3, 100)
        assert paths.dtype == model.dtype
        assert paths.device == model.device
    
    def test_quantiles(self, model):
        """Test quantiles computation."""
        time_points = [0.1, 0.5, 1.0]
        quantiles = model.quantiles(time_points, 100, size=20)
        assert quantiles.shape == (3, 20)
        assert quantiles.dtype == model.dtype
        assert quantiles.device == model.device
    
    def test_empty_time_points(self, model):
        """Test with empty time points."""
        paths = model.paths([], 100)
        assert paths.shape == (0, 100)
        
        quantiles = model.quantiles([], 100)
        assert quantiles.shape == (0, 512)  # Default size
    
    def test_invalid_time_points(self, model):
        """Test with invalid time points."""
        with pytest.raises(ValueError, match="Time points must be sorted"):
            model.paths([1.0, 0.5, 0.1], 100)
        
        with pytest.raises(ValueError, match="All time points must be positive"):
            model.paths([-0.1, 0.5, 1.0], 100)
    
    def test_invalid_quantile_bounds(self, model):
        """Test with invalid quantile bounds."""
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles([0.1, 0.5, 1.0], 100, lo=0.5, hi=0.1)
    
    def test_invalid_method(self, model):
        """Test with invalid method."""
        with pytest.raises(ValueError, match="Unknown method"):
            model.paths([0.1, 0.5, 1.0], 100, method="invalid")
    
    def test_different_hurst_values(self):
        """Test different Hurst parameter values."""
        for hurst in [0.1, 0.3, 0.5, 0.7, 0.9]:
            model = FractionalBrownianMotion(hurst=hurst, device='cpu')
            paths = model.paths([0.1, 0.5, 1.0], 50)
            assert paths.shape == (3, 50)
    
    def test_device_consistency(self):
        """Test device consistency."""
        if torch.cuda.is_available():
            model = FractionalBrownianMotion(hurst=0.3, device='cuda')
            paths = model.paths([0.1, 0.5, 1.0], 50)
            assert paths.device == torch.device('cuda')
    
    def test_dtype_consistency(self):
        """Test dtype consistency."""
        model = FractionalBrownianMotion(hurst=0.3, device='cpu', dtype=torch.float64)
        paths = model.paths([0.1, 0.5, 1.0], 50)
        assert paths.dtype == torch.float64