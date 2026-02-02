"""
Tax-Optimized Portfolio Construction & Direct Indexing System

A sophisticated portfolio optimization system that implements tax-aware investment
strategies, direct indexing, and tax-loss harvesting for customized equity portfolios.
"""

__version__ = "1.0.0"
__author__ = "Portfolio Analytics Team"

from .data_preprocessing import DataPreprocessor
from .portfolio_optimizer import TaxOptimizedPortfolio
from .tax_loss_harvesting import TaxLossHarvester
from .factor_models import FactorModel
from .backtester import Backtester

__all__ = [
    "DataPreprocessor",
    "TaxOptimizedPortfolio", 
    "TaxLossHarvester",
    "FactorModel",
    "Backtester"
]
