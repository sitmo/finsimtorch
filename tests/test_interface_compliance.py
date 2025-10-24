"""Tests for interface compliance across all Monte Carlo models."""

import pytest
import torch
import numpy as np
from finsimtorch import GjrGarch, GjrGarchReduced, FractionalBrownianMotion, MonteCarloSimulator


# Test fixtures for each model type
@pytest.fixture
def garch_model():
    """Create a GjrGarch model for testing."""
    return GjrGarch(mu=0.0, omega=0.1, alpha=0.1, gamma=0.05, 
                    beta=0.8, initial_variance=1.0, device='cpu')

@pytest.fixture
def garch_reduced_model():
    """Create a GjrGarchReduced model for testing."""
    return GjrGarchReduced(alpha=0.1, gamma=0.05, beta=0.8, 
                           initial_variance=1.0, device='cpu')

@pytest.fixture
def fbm_model():
    """Create a FractionalBrownianMotion model for testing."""
    return FractionalBrownianMotion(hurst=0.3, device='cpu')


# Parametrized tests across all models
@pytest.mark.parametrize('model_fixture', ['garch_model', 'garch_reduced_model', 'fbm_model'])
class TestInterfaceCompliance:
    """Test that all models comply with MonteCarloSimulator protocol."""
    
    def test_has_required_attributes(self, model_fixture, request):
        """Test that models have required attributes."""
        model = request.getfixturevalue(model_fixture)
        assert hasattr(model, 'device')
        assert hasattr(model, 'dtype')
        assert isinstance(model.device, torch.device)
        assert isinstance(model.dtype, torch.dtype)
    
    def test_has_paths_method(self, model_fixture, request):
        """Test that models have paths method."""
        model = request.getfixturevalue(model_fixture)
        assert hasattr(model, 'paths')
        assert callable(model.paths)
    
    def test_has_quantiles_method(self, model_fixture, request):
        """Test that models have quantiles method."""
        model = request.getfixturevalue(model_fixture)
        assert hasattr(model, 'quantiles')
        assert callable(model.quantiles)
    
    def test_paths_returns_correct_shape(self, model_fixture, request):
        """Test that paths() returns correct tensor shape."""
        model = request.getfixturevalue(model_fixture)
        
        time_points = [1, 5, 10]
        num_paths = 100
        result = model.paths(time_points, num_paths)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape[0] == 3  # 3 time points
        assert result.shape[1] == num_paths
        assert result.dtype == model.dtype
        assert result.device == model.device
    
    def test_quantiles_returns_correct_shape(self, model_fixture, request):
        """Test that quantiles() returns correct tensor shape."""
        model = request.getfixturevalue(model_fixture)
        
        time_points = [1, 5, 10]
        num_paths = 100
        size = 50
        result = model.quantiles(time_points, num_paths, lo=0.1, hi=0.9, size=size)
        
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3, size)
        assert result.dtype == model.dtype
        assert result.device == model.device
    
    def test_protocol_compliance(self, model_fixture, request):
        """Test that isinstance works with Protocol (runtime_checkable)."""
        model = request.getfixturevalue(model_fixture)
        assert isinstance(model, MonteCarloSimulator)
    
    def test_paths_empty_time_points(self, model_fixture, request):
        """Test paths method with empty time points."""
        model = request.getfixturevalue(model_fixture)
        result = model.paths([], 10)
        assert result.shape == (0, 10)
    
    def test_quantiles_empty_time_points(self, model_fixture, request):
        """Test quantiles method with empty time points."""
        model = request.getfixturevalue(model_fixture)
        result = model.quantiles([], 10)
        assert result.shape == (0, 512)  # Default size
    
    def test_paths_invalid_time_points(self, model_fixture, request):
        """Test paths method with unsorted or non-positive time points."""
        model = request.getfixturevalue(model_fixture)
        with pytest.raises(ValueError, match="Time points must be sorted"):
            model.paths([10, 5, 1], 10)
        with pytest.raises(ValueError, match="All time points must be positive"):
            model.paths([-1, 5, 10], 10)
    
    def test_quantiles_invalid_time_points(self, model_fixture, request):
        """Test quantiles method with unsorted or non-positive time points."""
        model = request.getfixturevalue(model_fixture)
        with pytest.raises(ValueError, match="Time points must be sorted"):
            model.quantiles([10, 5, 1], 10)
        with pytest.raises(ValueError, match="All time points must be positive"):
            model.quantiles([-1, 5, 10], 10)
    
    def test_quantiles_invalid_bounds(self, model_fixture, request):
        """Test quantiles method with invalid quantile bounds."""
        model = request.getfixturevalue(model_fixture)
        time_points = [1, 2]
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles(time_points, 10, lo=0.5, hi=0.1)
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles(time_points, 10, lo=-0.1, hi=0.9)
        with pytest.raises(ValueError, match="Quantile bounds must satisfy"):
            model.quantiles(time_points, 10, lo=0.1, hi=1.1)