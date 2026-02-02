"""
Tests for Tax-Loss Harvesting Module
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from tax_loss_harvesting import TaxLossHarvester, TaxLot


@pytest.fixture
def harvester():
    """Create a harvester instance."""
    return TaxLossHarvester(
        tax_rate_short_term=0.37,
        tax_rate_long_term=0.238
    )


@pytest.fixture
def sample_portfolio(harvester):
    """Create sample portfolio with positions."""
    # Add positions bought at different times
    harvester.add_purchase('AAPL', datetime(2023, 1, 15), 100, 150.00)
    harvester.add_purchase('MSFT', datetime(2023, 6, 1), 50, 340.00)
    harvester.add_purchase('GOOGL', datetime(2024, 1, 10), 75, 140.00)  # Recent
    
    return harvester


class TestTaxLot:
    """Tests for TaxLot class."""
    
    def test_lot_creation(self):
        """Test tax lot creation."""
        lot = TaxLot(
            ticker='AAPL',
            purchase_date=datetime(2023, 1, 1),
            shares=100,
            cost_basis=150.00
        )
        
        assert lot.ticker == 'AAPL'
        assert lot.shares == 100
        assert lot.cost_basis == 150.00
        assert lot.remaining_shares == 100
    
    def test_holding_period(self):
        """Test holding period calculation."""
        lot = TaxLot(
            ticker='AAPL',
            purchase_date=datetime(2023, 1, 1),
            shares=100,
            cost_basis=150.00
        )
        
        as_of = datetime(2024, 1, 1)
        days = lot.holding_days(as_of)
        
        assert days == 365 or days == 366  # Account for leap year
    
    def test_long_term_classification(self):
        """Test long-term vs short-term classification."""
        lot = TaxLot(
            ticker='AAPL',
            purchase_date=datetime(2023, 1, 1),
            shares=100,
            cost_basis=150.00
        )
        
        # Before 1 year
        assert not lot.is_long_term(datetime(2023, 12, 1))
        
        # After 1 year
        assert lot.is_long_term(datetime(2024, 1, 2))
    
    def test_unrealized_gain(self):
        """Test unrealized gain calculation."""
        lot = TaxLot(
            ticker='AAPL',
            purchase_date=datetime(2023, 1, 1),
            shares=100,
            cost_basis=150.00
        )
        
        # Price went up
        gain = lot.unrealized_gain(160.00)
        assert gain == 1000.00  # 100 shares * $10 gain
        
        # Price went down
        loss = lot.unrealized_gain(140.00)
        assert loss == -1000.00  # 100 shares * $10 loss


class TestHarvestOpportunities:
    """Tests for harvest opportunity identification."""
    
    def test_identify_losses(self, sample_portfolio):
        """Test that losses are identified."""
        current_prices = {
            'AAPL': 140.00,   # Loss
            'MSFT': 350.00,   # Gain
            'GOOGL': 130.00   # Loss
        }
        
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices=current_prices,
            current_date=datetime(2024, 6, 1),
            min_loss_pct=0.01
        )
        
        # Should find AAPL and GOOGL (both have losses)
        tickers_with_losses = [o['ticker'] for o in opportunities]
        assert 'AAPL' in tickers_with_losses
        assert 'GOOGL' in tickers_with_losses
        assert 'MSFT' not in tickers_with_losses
    
    def test_tax_benefit_calculation(self, sample_portfolio):
        """Test tax benefit is calculated correctly."""
        current_prices = {
            'AAPL': 140.00,   # $10 loss per share, 100 shares = $1000 loss
            'MSFT': 350.00,
            'GOOGL': 130.00
        }
        
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices=current_prices,
            current_date=datetime(2024, 6, 1)
        )
        
        # Find AAPL opportunity
        aapl_opp = next((o for o in opportunities if o['ticker'] == 'AAPL'), None)
        
        if aapl_opp:
            # AAPL held > 1 year, so long-term rate (23.8%)
            expected_benefit = 1000 * 0.238
            assert abs(aapl_opp['tax_benefit'] - expected_benefit) < 1
    
    def test_sorted_by_benefit(self, sample_portfolio):
        """Test opportunities are sorted by tax benefit."""
        current_prices = {
            'AAPL': 140.00,
            'MSFT': 300.00,   # Big loss
            'GOOGL': 130.00
        }
        
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices=current_prices,
            current_date=datetime(2024, 6, 1)
        )
        
        if len(opportunities) >= 2:
            # Should be sorted descending by tax benefit
            for i in range(len(opportunities) - 1):
                assert opportunities[i]['tax_benefit'] >= opportunities[i+1]['tax_benefit']


class TestHarvestExecution:
    """Tests for harvest execution."""
    
    def test_execute_harvest(self, sample_portfolio):
        """Test executing a harvest."""
        # First identify opportunities
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices={'AAPL': 140.00, 'MSFT': 350.00, 'GOOGL': 130.00},
            current_date=datetime(2024, 6, 1)
        )
        
        if opportunities:
            opp = opportunities[0]
            
            result = sample_portfolio.execute_harvest(
                ticker=opp['ticker'],
                lot_id=opp['lot_id'],
                sell_date=datetime(2024, 6, 1),
                sell_price=opp['current_price']
            )
            
            assert 'realized_loss' in result
            assert 'tax_benefit' in result
            assert result['realized_loss'] < 0  # Should be a loss
    
    def test_wash_sale_blocking(self, sample_portfolio):
        """Test that wash sale blocking is applied."""
        # Execute a harvest
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices={'AAPL': 140.00, 'MSFT': 350.00, 'GOOGL': 130.00},
            current_date=datetime(2024, 6, 1)
        )
        
        if opportunities:
            opp = opportunities[0]
            ticker = opp['ticker']
            
            sample_portfolio.execute_harvest(
                ticker=ticker,
                lot_id=opp['lot_id'],
                sell_date=datetime(2024, 6, 1),
                sell_price=opp['current_price']
            )
            
            # Check wash sale status
            status = sample_portfolio.check_wash_sale_status(
                ticker=ticker,
                date=datetime(2024, 6, 15)  # Within 30 days
            )
            
            assert status['blocked']
            assert status['days_remaining'] > 0


class TestReplacementSecurities:
    """Tests for replacement security finding."""
    
    def test_find_replacements(self, harvester):
        """Test finding replacement securities."""
        # Create sample returns data
        np.random.seed(42)
        dates = pd.date_range('2023-01-01', '2024-01-01', freq='D')
        
        # AAPL and MSFT highly correlated
        aapl_returns = np.random.normal(0.001, 0.02, len(dates))
        msft_returns = aapl_returns * 0.9 + np.random.normal(0, 0.005, len(dates))
        googl_returns = np.random.normal(0.001, 0.02, len(dates))
        
        returns_df = pd.DataFrame({
            'AAPL': aapl_returns,
            'MSFT': msft_returns,
            'GOOGL': googl_returns
        }, index=dates)
        
        # Find replacements for AAPL
        replacements = harvester.find_replacement_securities(
            ticker='AAPL',
            returns_df=returns_df,
            correlation_threshold=0.8
        )
        
        # MSFT should be found as replacement due to high correlation
        replacement_tickers = [r['ticker'] for r in replacements]
        assert 'MSFT' in replacement_tickers


class TestTaxAlpha:
    """Tests for tax alpha calculation."""
    
    def test_tax_alpha_calculation(self, sample_portfolio):
        """Test tax alpha is calculated."""
        # Execute some harvests
        opportunities = sample_portfolio.identify_harvest_opportunities(
            current_prices={'AAPL': 140.00, 'MSFT': 300.00, 'GOOGL': 130.00},
            current_date=datetime(2024, 6, 1)
        )
        
        for opp in opportunities[:2]:
            sample_portfolio.execute_harvest(
                ticker=opp['ticker'],
                lot_id=opp['lot_id'],
                sell_date=datetime(2024, 6, 1),
                sell_price=opp['current_price']
            )
        
        # Calculate tax alpha
        alpha = sample_portfolio.calculate_tax_alpha(
            portfolio_value=100000,
            years=1.0
        )
        
        assert 'tax_alpha_bps' in alpha
        assert 'total_tax_benefit' in alpha
        assert alpha['n_harvests'] >= 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
