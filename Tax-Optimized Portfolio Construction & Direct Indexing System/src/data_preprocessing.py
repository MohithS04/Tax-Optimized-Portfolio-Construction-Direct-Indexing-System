"""
Data Preprocessing Module for Tax-Optimized Portfolio Construction

This module handles:
    - Loading and cleaning downloaded datasets
    - Calculating returns and risk metrics
    - Estimating factor exposures
    - Preparing optimization inputs
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocesses financial data for portfolio optimization.
    
    Handles data loading, cleaning, returns calculation, and
    factor exposure estimation.
    """
    
    def __init__(self, data_dir='data'):
        """
        Initialize the data preprocessor.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing downloaded data files
        """
        self.data_dir = data_dir
        
        # Data containers
        self.stock_data = None
        self.adj_prices = None
        self.returns = None
        self.factors = None
        self.company_info = None
        self.rf_rate = None
        self.benchmark_data = None
        
        # Processed outputs
        self.cov_matrix = None
        self.expected_returns = None
        self.factor_exposures = None
        self.valid_tickers = None
        
    def load_all_data(self):
        """Load all downloaded datasets."""
        print("Loading datasets...")
        
        # Load stock prices (raw)
        try:
            with open(f'{self.data_dir}/stock_prices_raw.pkl', 'rb') as f:
                self.stock_data = pickle.load(f)
            print("  ✓ Raw stock prices loaded")
        except FileNotFoundError:
            print("  ⚠ Raw stock prices not found")
        
        # Load adjusted prices
        try:
            self.adj_prices = pd.read_csv(
                f'{self.data_dir}/adjusted_prices.csv',
                index_col=0,
                parse_dates=True
            )
            print(f"  ✓ Adjusted prices loaded ({len(self.adj_prices.columns)} stocks)")
        except FileNotFoundError:
            print("  ⚠ Adjusted prices not found")
            
        # Load Fama-French factors
        try:
            self.factors = pd.read_csv(
                f'{self.data_dir}/fama_french_factors.csv',
                index_col=0,
                parse_dates=True
            )
            print(f"  ✓ Factor data loaded ({len(self.factors.columns)} factors)")
        except FileNotFoundError:
            print("  ⚠ Factor data not found")
            
        # Load company info
        try:
            self.company_info = pd.read_csv(
                f'{self.data_dir}/company_info.csv'
            )
            print(f"  ✓ Company info loaded ({len(self.company_info)} companies)")
        except FileNotFoundError:
            print("  ⚠ Company info not found")
            
        # Load risk-free rate
        try:
            self.rf_rate = pd.read_csv(
                f'{self.data_dir}/risk_free_rate.csv',
                index_col=0,
                parse_dates=True
            )
            print("  ✓ Risk-free rate loaded")
        except FileNotFoundError:
            print("  ⚠ Risk-free rate not found")
            
        # Load benchmark data
        try:
            self.benchmark_data = pd.read_csv(
                f'{self.data_dir}/benchmark_indices.csv',
                index_col=0,
                parse_dates=True,
                header=[0, 1]
            )
            print("  ✓ Benchmark data loaded")
        except FileNotFoundError:
            print("  ⚠ Benchmark data not found")
            
        print("\n✓ All available data loaded successfully")
        
    def calculate_returns(self, method='simple'):
        """
        Calculate returns from adjusted prices.
        
        Parameters:
        -----------
        method : str
            'simple' for arithmetic returns, 'log' for log returns
            
        Returns:
        --------
        pd.DataFrame : Daily returns
        """
        if self.adj_prices is None:
            raise ValueError("Adjusted prices not loaded. Call load_all_data() first.")
        
        print(f"Calculating {method} returns...")
        
        if method == 'simple':
            self.returns = self.adj_prices.pct_change().dropna()
        elif method == 'log':
            self.returns = np.log(self.adj_prices / self.adj_prices.shift(1)).dropna()
        else:
            raise ValueError("Method must be 'simple' or 'log'")
        
        # Remove stocks with too many missing values
        missing_pct = self.returns.isnull().sum() / len(self.returns)
        valid_cols = missing_pct[missing_pct < 0.1].index
        self.returns = self.returns[valid_cols].dropna()
        
        print(f"  ✓ Returns calculated for {len(self.returns.columns)} stocks")
        print(f"  ✓ Date range: {self.returns.index[0].date()} to {self.returns.index[-1].date()}")
        
        return self.returns
    
    def calculate_excess_returns(self):
        """
        Calculate excess returns (returns minus risk-free rate).
        
        Returns:
        --------
        pd.DataFrame : Excess daily returns
        """
        if self.returns is None:
            self.calculate_returns()
            
        if self.rf_rate is None:
            print("  ⚠ No risk-free rate available, using returns as excess returns")
            return self.returns
        
        # Align dates
        common_dates = self.returns.index.intersection(self.rf_rate.index)
        rf_aligned = self.rf_rate.loc[common_dates, 'RF_Daily'].fillna(0)
        
        excess_returns = self.returns.loc[common_dates].sub(rf_aligned, axis=0)
        
        return excess_returns
    
    def estimate_factor_exposures(self, lookback_days=252):
        """
        Estimate factor loadings (betas) for each stock using regression.
        
        Parameters:
        -----------
        lookback_days : int
            Number of trading days for estimation
            
        Returns:
        --------
        pd.DataFrame : Factor exposures for each stock
        """
        if self.returns is None:
            self.calculate_returns()
            
        if self.factors is None:
            print("  ⚠ Factor data not available, skipping factor exposure estimation")
            return None
        
        print(f"Estimating factor exposures (lookback: {lookback_days} days)...")
        
        # Align returns and factors
        common_dates = self.returns.index.intersection(self.factors.index)
        
        if len(common_dates) < lookback_days:
            lookback_days = len(common_dates)
            print(f"  ⚠ Adjusted lookback to {lookback_days} days due to data availability")
        
        # Use most recent data
        common_dates = common_dates[-lookback_days:]
        
        returns_aligned = self.returns.loc[common_dates]
        factors_aligned = self.factors.loc[common_dates]
        
        # Prepare factor matrix (exclude RF if present)
        factor_cols = [c for c in factors_aligned.columns if c != 'RF']
        X = factors_aligned[factor_cols].values
        
        exposures = {}
        
        for ticker in returns_aligned.columns:
            try:
                y = returns_aligned[ticker].values
                
                # Remove NaN values
                valid_mask = ~np.isnan(y)
                if valid_mask.sum() < 30:  # Need minimum observations
                    continue
                
                y_valid = y[valid_mask]
                X_valid = X[valid_mask]
                
                # Run regression
                model = LinearRegression()
                model.fit(X_valid, y_valid)
                
                # Store results
                betas = dict(zip(factor_cols, model.coef_))
                betas['alpha'] = model.intercept_ * 252  # Annualized alpha
                betas['r_squared'] = model.score(X_valid, y_valid)
                
                exposures[ticker] = betas
                
            except Exception as e:
                continue
        
        self.factor_exposures = pd.DataFrame(exposures).T
        
        print(f"  ✓ Factor exposures estimated for {len(self.factor_exposures)} stocks")
        
        return self.factor_exposures
    
    def calculate_covariance_matrix(self, method='sample', shrinkage=0.1, window=252):
        """
        Calculate return covariance matrix.
        
        Parameters:
        -----------
        method : str
            'sample' for sample covariance
            'ewma' for exponentially weighted
            'shrinkage' for Ledoit-Wolf shrinkage
            'factor' for factor-based covariance
        shrinkage : float
            Shrinkage intensity (for shrinkage method)
        window : int
            Lookback window for EWMA
            
        Returns:
        --------
        pd.DataFrame : Covariance matrix (annualized)
        """
        if self.returns is None:
            self.calculate_returns()
        
        print(f"Calculating covariance matrix (method: {method})...")
        
        if method == 'sample':
            self.cov_matrix = self.returns.cov() * 252  # Annualized
            
        elif method == 'ewma':
            # Exponentially weighted covariance
            ewma_cov = self.returns.ewm(span=window).cov()
            # Get last covariance matrix
            last_date = self.returns.index[-1]
            self.cov_matrix = ewma_cov.loc[last_date] * 252
            
        elif method == 'shrinkage':
            # Ledoit-Wolf shrinkage estimator
            sample_cov = self.returns.cov() * 252
            
            # Target: diagonal matrix with average variance
            avg_var = np.trace(sample_cov.values) / len(sample_cov)
            target = np.eye(len(sample_cov)) * avg_var
            
            # Shrink towards target
            shrunk_cov = (1 - shrinkage) * sample_cov.values + shrinkage * target
            self.cov_matrix = pd.DataFrame(
                shrunk_cov,
                index=sample_cov.index,
                columns=sample_cov.columns
            )
            
        elif method == 'factor':
            # Factor-based covariance (requires factor exposures)
            if self.factor_exposures is None:
                self.estimate_factor_exposures()
            
            # Get factor covariance
            factor_cols = [c for c in self.factors.columns if c != 'RF']
            factor_cov = self.factors[factor_cols].cov() * 252
            
            # Common stocks
            common_stocks = self.factor_exposures.index.intersection(self.returns.columns)
            B = self.factor_exposures.loc[common_stocks, factor_cols].values
            
            # Factor-based covariance: B * F * B'
            factor_cov_matrix = B @ factor_cov.values @ B.T
            
            # Add idiosyncratic variance (diagonal)
            idio_var = self.returns[common_stocks].var() * 252
            
            full_cov = factor_cov_matrix + np.diag(idio_var.values)
            
            self.cov_matrix = pd.DataFrame(
                full_cov,
                index=common_stocks,
                columns=common_stocks
            )
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  ✓ Covariance matrix calculated ({len(self.cov_matrix)}x{len(self.cov_matrix)})")
        
        return self.cov_matrix
    
    def calculate_expected_returns(self, method='historical', lookback_days=252):
        """
        Calculate expected returns.
        
        Parameters:
        -----------
        method : str
            'historical' - simple historical mean
            'ewma' - exponentially weighted mean
            'capm' - CAPM-implied returns
            'factor' - factor model implied returns
        lookback_days : int
            Days for calculation
            
        Returns:
        --------
        pd.Series : Expected annual returns
        """
        if self.returns is None:
            self.calculate_returns()
        
        print(f"Calculating expected returns (method: {method})...")
        
        recent_returns = self.returns.iloc[-lookback_days:]
        
        if method == 'historical':
            self.expected_returns = recent_returns.mean() * 252  # Annualized
            
        elif method == 'ewma':
            self.expected_returns = recent_returns.ewm(span=lookback_days//2).mean().iloc[-1] * 252
            
        elif method == 'capm':
            # Need market returns and betas
            if self.factor_exposures is None:
                self.estimate_factor_exposures()
            
            # Market risk premium (historical)
            if self.factors is not None and 'Mkt-RF' in self.factors.columns:
                market_premium = self.factors['Mkt-RF'].mean() * 252
            else:
                market_premium = 0.06  # Default 6%
            
            # Risk-free rate
            if self.rf_rate is not None:
                rf = self.rf_rate['RF_Annual_Pct'].iloc[-1] / 100
            else:
                rf = 0.02  # Default 2%
            
            # CAPM: E[R] = Rf + beta * (E[Rm] - Rf)
            betas = self.factor_exposures['Mkt-RF'] if 'Mkt-RF' in self.factor_exposures.columns else 1.0
            self.expected_returns = rf + betas * market_premium
            
        elif method == 'factor':
            # Multi-factor model
            if self.factor_exposures is None:
                self.estimate_factor_exposures()
            
            # Factor risk premiums
            factor_cols = [c for c in self.factors.columns if c != 'RF']
            factor_premiums = self.factors[factor_cols].mean() * 252
            
            # Expected return = sum(beta_i * premium_i)
            common_stocks = self.factor_exposures.index.intersection(self.returns.columns)
            B = self.factor_exposures.loc[common_stocks, factor_cols]
            
            self.expected_returns = B @ factor_premiums
        else:
            raise ValueError(f"Unknown method: {method}")
        
        print(f"  ✓ Expected returns calculated for {len(self.expected_returns)} stocks")
        
        return self.expected_returns
    
    def get_sector_mapping(self):
        """
        Get sector mapping for stocks.
        
        Returns:
        --------
        dict : {ticker: sector}
        """
        if self.company_info is None:
            return {}
        
        return dict(zip(
            self.company_info['ticker'],
            self.company_info['sector']
        ))
    
    def get_market_caps(self):
        """
        Get market capitalizations.
        
        Returns:
        --------
        pd.Series : Market caps by ticker
        """
        if self.company_info is None:
            return None
        
        return pd.Series(
            self.company_info['marketCap'].values,
            index=self.company_info['ticker']
        )
    
    def calculate_benchmark_weights(self, method='market_cap'):
        """
        Calculate benchmark (market-cap) weights.
        
        Parameters:
        -----------
        method : str
            'market_cap' - market cap weighted
            'equal' - equal weighted
            
        Returns:
        --------
        pd.Series : Benchmark weights
        """
        market_caps = self.get_market_caps()
        
        if market_caps is None or method == 'equal':
            if self.valid_tickers is not None:
                weights = pd.Series(
                    1.0 / len(self.valid_tickers),
                    index=self.valid_tickers
                )
            else:
                weights = pd.Series(
                    1.0 / len(self.returns.columns),
                    index=self.returns.columns
                )
        else:
            # Filter to valid tickers
            if self.valid_tickers is not None:
                market_caps = market_caps[market_caps.index.isin(self.valid_tickers)]
            
            # Normalize to sum to 1
            weights = market_caps / market_caps.sum()
        
        return weights
    
    def prepare_optimization_inputs(self, 
                                   return_method='historical',
                                   cov_method='shrinkage',
                                   min_observations=200):
        """
        Prepare all inputs needed for portfolio optimization.
        
        Parameters:
        -----------
        return_method : str
            Method for expected returns calculation
        cov_method : str  
            Method for covariance calculation
        min_observations : int
            Minimum observations required for a stock
            
        Returns:
        --------
        dict : All optimization inputs
        """
        print("\n" + "=" * 50)
        print("PREPARING OPTIMIZATION INPUTS")
        print("=" * 50)
        
        # Calculate returns
        if self.returns is None:
            self.calculate_returns()
        
        # Filter stocks with sufficient data
        obs_count = self.returns.count()
        valid_stocks = obs_count[obs_count >= min_observations].index.tolist()
        
        print(f"\n  Filtered to {len(valid_stocks)} stocks with >= {min_observations} observations")
        
        # Calculate expected returns
        self.calculate_expected_returns(method=return_method)
        
        # Calculate covariance
        self.calculate_covariance_matrix(method=cov_method)
        
        # Estimate factor exposures
        self.estimate_factor_exposures()
        
        # Find intersection of all data
        if self.factor_exposures is not None:
            self.valid_tickers = list(set(valid_stocks) & 
                                     set(self.expected_returns.index) &
                                     set(self.cov_matrix.index) &
                                     set(self.factor_exposures.index))
        else:
            self.valid_tickers = list(set(valid_stocks) & 
                                     set(self.expected_returns.index) &
                                     set(self.cov_matrix.index))
        
        self.valid_tickers.sort()
        
        print(f"\n  Final universe: {len(self.valid_tickers)} stocks")
        
        # Get benchmark weights
        benchmark_weights = self.calculate_benchmark_weights()
        benchmark_weights = benchmark_weights[benchmark_weights.index.isin(self.valid_tickers)]
        benchmark_weights = benchmark_weights / benchmark_weights.sum()
        
        # Get sector mapping
        sector_mapping = self.get_sector_mapping()
        sector_mapping = {k: v for k, v in sector_mapping.items() if k in self.valid_tickers}
        
        # Prepare output dictionary
        optimization_inputs = {
            'returns': self.expected_returns[self.valid_tickers],
            'cov_matrix': self.cov_matrix.loc[self.valid_tickers, self.valid_tickers],
            'factor_exposures': self.factor_exposures.loc[self.valid_tickers] if self.factor_exposures is not None else None,
            'benchmark_weights': benchmark_weights,
            'sector_mapping': sector_mapping,
            'tickers': self.valid_tickers,
            'daily_returns': self.returns[self.valid_tickers]
        }
        
        print("\n✓ Optimization inputs prepared successfully!")
        
        return optimization_inputs
    
    def save_processed_data(self, filename='processed_data.pkl'):
        """Save processed data to pickle file."""
        optimization_inputs = self.prepare_optimization_inputs()
        
        filepath = f'{self.data_dir}/{filename}'
        with open(filepath, 'wb') as f:
            pickle.dump(optimization_inputs, f)
        
        print(f"\n✓ Processed data saved to {filepath}")
        
        return optimization_inputs


def main():
    """Main entry point for data preprocessing."""
    preprocessor = DataPreprocessor()
    preprocessor.load_all_data()
    
    # Prepare and save optimization inputs
    optimization_inputs = preprocessor.save_processed_data()
    
    print("\nData preprocessing complete!")
    print(f"Universe contains {len(optimization_inputs['tickers'])} stocks")


if __name__ == '__main__':
    main()
