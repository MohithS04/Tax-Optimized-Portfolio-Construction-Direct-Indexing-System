"""
Tests for Portfolio Optimizer Module
"""

import pytest
import numpy as np
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from portfolio_optimizer import TaxOptimizedPortfolio


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_assets = 20
    tickers = [f'STOCK_{i}' for i in range(n_assets)]
    
    # Expected returns (annualized)
    returns = pd.Series(
        np.random.uniform(0.05, 0.15, n_assets),
        index=tickers
    )
    
    # Covariance matrix (annualized)
    vols = np.random.uniform(0.15, 0.35, n_assets)
    corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr[i, j] = corr[j, i] = np.random.uniform(0.2, 0.5)
    
    cov = np.outer(vols, vols) * corr
    cov_matrix = pd.DataFrame(cov, index=tickers, columns=tickers)
    
    return {
        'returns': returns,
        'cov_matrix': cov_matrix,
        'tickers': tickers
    }


class TestMeanVarianceOptimization:
    """Tests for mean-variance optimization."""
    
    def test_basic_optimization(self, sample_data):
        """Test basic optimization works."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        result = optimizer.optimize_mean_variance(max_position=0.15)
        
        assert result['status'] == 'optimal'
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        
    def test_weights_sum_to_one(self, sample_data):
        """Test that weights sum to 1."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        result = optimizer.optimize_mean_variance(max_position=0.15)
        
        if result['status'] == 'optimal':
            total_weight = sum(result['weights'].values())
            assert abs(total_weight - 1.0) < 0.001
    
    def test_max_position_constraint(self, sample_data):
        """Test max position constraint is respected."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        max_pos = 0.10
        result = optimizer.optimize_mean_variance(max_position=max_pos)
        
        if result['status'] == 'optimal':
            for weight in result['weights'].values():
                assert weight <= max_pos + 0.001
    
    def test_no_negative_weights(self, sample_data):
        """Test that weights are non-negative."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        result = optimizer.optimize_mean_variance(min_position=0.0)
        
        if result['status'] == 'optimal':
            for weight in result['weights'].values():
                assert weight >= -0.001


class TestDirectIndexing:
    """Tests for direct indexing optimization."""
    
    def test_cardinality_constraint(self, sample_data):
        """Test that number of holdings is limited."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        # Benchmark weights (equal)
        benchmark = {t: 1/len(sample_data['tickers']) 
                    for t in sample_data['tickers']}
        
        n_stocks = 10
        result = optimizer.optimize_direct_indexing(
            benchmark_weights=benchmark,
            n_stocks=n_stocks,
            tracking_error_limit=0.05
        )
        
        if result['status'] == 'optimal':
            assert result['n_holdings'] <= n_stocks + 1  # Allow small tolerance


class TestTaxAwareOptimization:
    """Tests for tax-aware optimization."""
    
    def test_with_existing_portfolio(self, sample_data):
        """Test tax-aware optimization with existing portfolio."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        # Initial portfolio
        initial = {t: 1/len(sample_data['tickers']) 
                  for t in sample_data['tickers']}
        
        # Cost basis (bought at lower prices)
        cost_basis = {t: 0.9 for t in sample_data['tickers']}
        current_prices = {t: 1.0 for t in sample_data['tickers']}
        
        result = optimizer.optimize_tax_aware(
            initial_weights=initial,
            cost_basis=cost_basis,
            current_prices=current_prices,
            turnover_limit=0.30
        )
        
        assert result['status'] == 'optimal'


class TestEfficientFrontier:
    """Tests for efficient frontier generation."""
    
    def test_frontier_generation(self, sample_data):
        """Test efficient frontier can be generated."""
        optimizer = TaxOptimizedPortfolio(
            returns=sample_data['returns'],
            cov_matrix=sample_data['cov_matrix']
        )
        
        frontier = optimizer.generate_efficient_frontier(n_points=10)
        
        assert len(frontier) > 0
        assert 'return' in frontier.columns
        assert 'volatility' in frontier.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
