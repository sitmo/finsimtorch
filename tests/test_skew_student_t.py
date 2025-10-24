"""Tests for Hansen's Skewed Student-t distribution."""

import pytest
import torch
import numpy as np
from finsimtorch import HansenSkewedT


class TestHansenSkewedT:
    """Test Hansen's Skewed Student-t distribution."""
    
    @pytest.fixture
    def dist(self):
        """Create a test distribution."""
        return HansenSkewedT(eta=10.0, lam=0.1, device='cpu')
    
    def test_initialization(self, dist):
        """Test distribution initialization."""
        assert dist.eta == 10.0
        assert dist.lam == 0.1
        assert dist.device.type == 'cpu'
    
    def test_invalid_parameters(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="eta \\(nu\\) must be > 2"):
            HansenSkewedT(eta=1.0, lam=0.1)
        
        with pytest.raises(ValueError, match="lam must be in"):
            HansenSkewedT(eta=10.0, lam=1.5)
        
        with pytest.raises(ValueError, match="lam must be in"):
            HansenSkewedT(eta=10.0, lam=-1.5)
    
    def test_rvs(self, dist):
        """Test random variate generation."""
        samples = dist.rvs(1000)
        assert samples.shape == (1000,)
        assert torch.isfinite(samples).all()
    
    def test_pdf(self, dist):
        """Test probability density function."""
        x = torch.linspace(-5, 5, 100)
        pdf_vals = dist.pdf(x)
        
        assert pdf_vals.shape == (100,)
        assert torch.all(pdf_vals >= 0)
        assert torch.isfinite(pdf_vals).all()
    
    def test_logpdf(self, dist):
        """Test log probability density function."""
        x = torch.linspace(-5, 5, 100)
        logpdf_vals = dist.logpdf(x)
        
        assert logpdf_vals.shape == (100,)
        assert torch.isfinite(logpdf_vals).all()
    
    def test_cdf(self, dist):
        """Test cumulative distribution function."""
        x = torch.linspace(-5, 5, 100)
        cdf_vals = dist.cdf(x)
        
        assert cdf_vals.shape == (100,)
        assert torch.all((cdf_vals >= 0) & (cdf_vals <= 1))
        assert torch.isfinite(cdf_vals).all()
    
    def test_second_moment_left(self, dist):
        """Test second moment computation."""
        moment = dist.second_moment_left()
        assert moment > 0
        assert torch.isfinite(moment)
    
    def test_different_parameters(self):
        """Test distribution with different parameter values."""
        for eta in [5.0, 10.0, 20.0]:
            for lam in [-0.5, 0.0, 0.5]:
                dist = HansenSkewedT(eta=eta, lam=lam, device='cpu')
                samples = dist.rvs(100)
                assert samples.shape == (100,)
                assert torch.isfinite(samples).all()
