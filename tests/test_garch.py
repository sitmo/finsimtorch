"""Tests for GJR-GARCH models."""

import pytest
import torch
import numpy as np
from finsimtorch import GjrGarch, GjrGarchReduced


class TestGjrGarch:
    """Test GJR-GARCH model functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a test GJR-GARCH model."""
        return GjrGarch(
            mu=0.0,
            omega=0.1,
            alpha=0.1,
            gamma=0.05,
            beta=0.8,
            initial_variance=1.0,
            device='cpu'
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.mu == 0.0
        assert model.omega == 0.1
        assert model.alpha == 0.1
        assert model.gamma == 0.05
        assert model.beta == 0.8
        assert model.initial_variance == 1.0
        assert model.device == torch.device('cpu')
        assert model.dtype == torch.float32
    
    def test_simulation(self, model):
        """Test basic simulation functionality."""
        paths = model.paths([1, 5, 10], 100)
        assert paths.shape == (3, 100)
        assert paths.dtype == model.dtype
        assert paths.device == model.device
    
    def test_quantiles(self, model):
        """Test quantiles computation."""
        quantiles = model.quantiles([1, 5, 10], 100, size=20)
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
            model.paths([10, 5, 1], 100)
        
        with pytest.raises(ValueError, match="All time points must be positive"):
            model.paths([-1, 5, 10], 100)
    
    def test_invalid_quantile_bounds(self, model):
        """Test with invalid quantile bounds."""
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles([1, 5, 10], 100, lo=0.5, hi=0.1)


class TestGjrGarchReduced:
    """Test GJR-GARCH Reduced model functionality."""
    
    @pytest.fixture
    def model(self):
        """Create a test GJR-GARCH Reduced model."""
        return GjrGarchReduced(
            alpha=0.1,
            gamma=0.05,
            beta=0.8,
            initial_variance=1.0,
            device='cpu'
        )
    
    def test_model_initialization(self, model):
        """Test model initialization."""
        assert model.alpha == 0.1
        assert model.gamma == 0.05
        assert model.beta == 0.8
        assert model.initial_variance == 1.0
        assert model.device == torch.device('cpu')
        assert model.dtype == torch.float32
    
    def test_simulation(self, model):
        """Test basic simulation functionality."""
        paths = model.paths([1, 5, 10], 100)
        assert paths.shape == (3, 100)
        assert paths.dtype == model.dtype
        assert paths.device == model.device
    
    def test_quantiles(self, model):
        """Test quantiles computation."""
        quantiles = model.quantiles([1, 5, 10], 100, size=20)
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
            model.paths([10, 5, 1], 100)
        
        with pytest.raises(ValueError, match="All time points must be positive"):
            model.paths([-1, 5, 10], 100)
    
    def test_invalid_quantile_bounds(self, model):
        """Test with invalid quantile bounds."""
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles([1, 5, 10], 100, lo=0.5, hi=0.1)