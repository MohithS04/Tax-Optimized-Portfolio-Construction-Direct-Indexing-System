"""
Factor Model Module

This module implements:
    - Fama-French factor models (3, 5, 6 factor)
    - Factor exposure estimation
    - Factor-based risk decomposition
    - Factor-implied returns
    - Risk attribution
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
import warnings


class FactorModel:
    """
    Multi-factor model implementation for risk and return analysis.
    
    Supports Fama-French factors and custom factor definitions.
    """
    
    def __init__(self, 
                 factor_returns: pd.DataFrame,
                 risk_free_rate: Optional[pd.Series] = None):
        """
        Initialize factor model.
        
        Parameters:
        -----------
        factor_returns : pd.DataFrame
            Factor returns (daily), columns are factors
        risk_free_rate : pd.Series, optional
            Risk-free rate series (daily decimals)
        """
        self.factor_returns = factor_returns
        self.risk_free_rate = risk_free_rate
        
        # Factor names (exclude RF if present)
        self.factor_names = [
            col for col in factor_returns.columns 
            if col not in ['RF', 'Rf']
        ]
        
        # Store estimated exposures
        self.exposures = {}
        self.alphas = {}
        self.residual_vols = {}
        self.r_squared = {}
    
    def estimate_exposures(self,
                          stock_returns: pd.DataFrame,
                          method: str = 'ols',
                          window: Optional[int] = None,
                          min_observations: int = 60) -> pd.DataFrame:
        """
        Estimate factor exposures (betas) for each stock.
        
        Parameters:
        -----------
        stock_returns : pd.DataFrame
            Stock returns (daily)
        method : str
            Estimation method: 'ols', 'ridge', 'rolling'
        window : int, optional
            Rolling window for 'rolling' method
        min_observations : int
            Minimum observations for estimation
            
        Returns:
        --------
        pd.DataFrame : Factor exposures (betas)
        """
        # Align dates
        common_dates = stock_returns.index.intersection(self.factor_returns.index)
        
        if len(common_dates) < min_observations:
            raise ValueError(f"Insufficient data: {len(common_dates)} < {min_observations}")
        
        stock_aligned = stock_returns.loc[common_dates]
        factors_aligned = self.factor_returns.loc[common_dates, self.factor_names]
        
        # Get risk-free rate if available
        if self.risk_free_rate is not None:
            rf = self.risk_free_rate.loc[common_dates].fillna(0)
        else:
            rf = 0
        
        # Use specified window or all data
        if window and method != 'rolling':
            stock_aligned = stock_aligned.iloc[-window:]
            factors_aligned = factors_aligned.iloc[-window:]
            if isinstance(rf, pd.Series):
                rf = rf.iloc[-window:]
        
        exposures = {}
        alphas = {}
        residual_vols = {}
        r_squared = {}
        
        X = factors_aligned.values
        
        for ticker in stock_aligned.columns:
            try:
                # Stock excess returns
                y = stock_aligned[ticker].values
                if isinstance(rf, pd.Series):
                    y_excess = y - rf.values
                else:
                    y_excess = y - rf
                
                # Remove NaN
                valid_mask = ~np.isnan(y_excess) & ~np.any(np.isnan(X), axis=1)
                
                if valid_mask.sum() < min_observations:
                    continue
                
                y_valid = y_excess[valid_mask]
                X_valid = X[valid_mask]
                
                # Estimate model
                if method == 'ols':
                    model = LinearRegression()
                    model.fit(X_valid, y_valid)
                    betas = model.coef_
                    alpha = model.intercept_
                    r2 = model.score(X_valid, y_valid)
                    
                elif method == 'ridge':
                    model = Ridge(alpha=1.0)
                    model.fit(X_valid, y_valid)
                    betas = model.coef_
                    alpha = model.intercept_
                    r2 = model.score(X_valid, y_valid)
                    
                elif method == 'rolling':
                    # Use most recent window
                    if window:
                        y_valid = y_valid[-window:]
                        X_valid = X_valid[-window:]
                    model = LinearRegression()
                    model.fit(X_valid, y_valid)
                    betas = model.coef_
                    alpha = model.intercept_
                    r2 = model.score(X_valid, y_valid)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                # Calculate residual volatility
                y_pred = X_valid @ betas + alpha
                residuals = y_valid - y_pred
                resid_vol = np.std(residuals) * np.sqrt(252)  # Annualized
                
                exposures[ticker] = dict(zip(self.factor_names, betas))
                alphas[ticker] = alpha * 252  # Annualized
                residual_vols[ticker] = resid_vol
                r_squared[ticker] = r2
                
            except Exception as e:
                warnings.warn(f"Error estimating {ticker}: {e}")
                continue
        
        self.exposures = pd.DataFrame(exposures).T
        self.alphas = pd.Series(alphas)
        self.residual_vols = pd.Series(residual_vols)
        self.r_squared = pd.Series(r_squared)
        
        return self.exposures
    
    def estimate_exposures_robust(self,
                                 stock_returns: pd.DataFrame,
                                 confidence: float = 0.95) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate factor exposures with statistical significance.
        
        Uses statsmodels for robust standard errors.
        
        Parameters:
        -----------
        stock_returns : pd.DataFrame
            Stock returns
        confidence : float
            Confidence level for significance
            
        Returns:
        --------
        tuple : (exposures DataFrame, t-statistics DataFrame)
        """
        common_dates = stock_returns.index.intersection(self.factor_returns.index)
        stock_aligned = stock_returns.loc[common_dates]
        factors_aligned = self.factor_returns.loc[common_dates, self.factor_names]
        
        # Add constant for statsmodels
        X = sm.add_constant(factors_aligned)
        
        exposures = {}
        t_stats = {}
        p_values = {}
        
        for ticker in stock_aligned.columns:
            try:
                y = stock_aligned[ticker].dropna()
                X_valid = X.loc[y.index]
                
                if len(y) < 60:
                    continue
                
                # OLS with HAC standard errors
                model = sm.OLS(y, X_valid)
                results = model.fit(cov_type='HAC', cov_kwds={'maxlags': 5})
                
                # Extract results (skip constant)
                betas = results.params[1:]
                tstats = results.tvalues[1:]
                pvals = results.pvalues[1:]
                
                exposures[ticker] = dict(zip(self.factor_names, betas))
                t_stats[ticker] = dict(zip(self.factor_names, tstats))
                p_values[ticker] = dict(zip(self.factor_names, pvals))
                
            except Exception as e:
                continue
        
        self.exposures = pd.DataFrame(exposures).T
        t_stats_df = pd.DataFrame(t_stats).T
        
        return self.exposures, t_stats_df
    
    def calculate_factor_covariance(self, annualize: bool = True) -> pd.DataFrame:
        """
        Calculate factor return covariance matrix.
        
        Parameters:
        -----------
        annualize : bool
            Whether to annualize (multiply by 252)
            
        Returns:
        --------
        pd.DataFrame : Factor covariance matrix
        """
        factor_cov = self.factor_returns[self.factor_names].cov()
        
        if annualize:
            factor_cov = factor_cov * 252
        
        return factor_cov
    
    def calculate_stock_covariance(self,
                                  method: str = 'factor',
                                  shrinkage: float = 0.1) -> pd.DataFrame:
        """
        Calculate stock covariance matrix using factor model.
        
        Parameters:
        -----------
        method : str
            'factor' - Factor model covariance
            'hybrid' - Blend of factor and sample covariance
        shrinkage : float
            Shrinkage intensity for hybrid method
            
        Returns:
        --------
        pd.DataFrame : Stock covariance matrix
        """
        if self.exposures is None or len(self.exposures) == 0:
            raise ValueError("Run estimate_exposures() first")
        
        # Factor covariance
        factor_cov = self.calculate_factor_covariance()
        
        # Get exposure matrix
        tickers = self.exposures.index
        B = self.exposures.values
        
        # Factor-based covariance: B * F * B'
        factor_cov_stocks = B @ factor_cov.values @ B.T
        
        # Add idiosyncratic variance
        if hasattr(self, 'residual_vols') and len(self.residual_vols) > 0:
            resid_var = self.residual_vols.loc[tickers].values ** 2
            idio_var = np.diag(resid_var)
        else:
            idio_var = np.eye(len(tickers)) * 0.04  # Default 20% vol
        
        stock_cov = factor_cov_stocks + idio_var
        
        return pd.DataFrame(stock_cov, index=tickers, columns=tickers)
    
    def calculate_expected_returns(self,
                                  factor_premiums: Optional[Dict[str, float]] = None,
                                  use_historical: bool = True) -> pd.Series:
        """
        Calculate expected returns from factor model.
        
        Parameters:
        -----------
        factor_premiums : dict, optional
            Expected factor risk premiums {factor: premium}
        use_historical : bool
            Use historical factor means if premiums not provided
            
        Returns:
        --------
        pd.Series : Expected returns by stock
        """
        if self.exposures is None:
            raise ValueError("Run estimate_exposures() first")
        
        # Get factor premiums
        if factor_premiums is None and use_historical:
            factor_premiums = (self.factor_returns[self.factor_names].mean() * 252).to_dict()
        elif factor_premiums is None:
            # Default premiums based on historical research
            factor_premiums = {
                'Mkt-RF': 0.06,
                'SMB': 0.02,
                'HML': 0.03,
                'RMW': 0.03,
                'CMA': 0.02,
                'Mom': 0.04
            }
        
        # Calculate expected returns: E[R] = alpha + sum(beta_i * premium_i)
        expected_returns = self.alphas.copy() if hasattr(self, 'alphas') else pd.Series(0, index=self.exposures.index)
        
        for factor in self.factor_names:
            if factor in factor_premiums:
                expected_returns += self.exposures[factor] * factor_premiums[factor]
        
        return expected_returns
    
    def decompose_risk(self,
                      weights: pd.Series,
                      annualize: bool = True) -> Dict:
        """
        Decompose portfolio risk into factor and specific components.
        
        Parameters:
        -----------
        weights : pd.Series
            Portfolio weights
        annualize : bool
            Whether to annualize
            
        Returns:
        --------
        dict : Risk decomposition
        """
        # Align weights with exposures
        common = weights.index.intersection(self.exposures.index)
        w = weights.loc[common].values
        B = self.exposures.loc[common].values
        
        # Portfolio factor exposures
        port_exposures = B.T @ w
        
        # Factor covariance
        factor_cov = self.calculate_factor_covariance(annualize=annualize)
        
        # Factor risk (variance from factors)
        factor_var = port_exposures @ factor_cov.values @ port_exposures
        factor_vol = np.sqrt(factor_var)
        
        # Specific risk (idiosyncratic)
        if hasattr(self, 'residual_vols') and len(self.residual_vols) > 0:
            resid_var = self.residual_vols.loc[common].values ** 2
            specific_var = np.sum(w ** 2 * resid_var)
        else:
            specific_var = 0
        specific_vol = np.sqrt(specific_var)
        
        # Total risk
        total_var = factor_var + specific_var
        total_vol = np.sqrt(total_var)
        
        # Risk contribution by factor
        factor_contributions = {}
        for i, factor in enumerate(self.factor_names):
            factor_exposure = port_exposures[i]
            factor_variance = factor_cov.iloc[i, i]
            contribution = factor_exposure ** 2 * factor_variance / total_var if total_var > 0 else 0
            factor_contributions[factor] = contribution
        
        return {
            'total_volatility': total_vol,
            'factor_volatility': factor_vol,
            'specific_volatility': specific_vol,
            'factor_var_pct': factor_var / total_var if total_var > 0 else 0,
            'specific_var_pct': specific_var / total_var if total_var > 0 else 0,
            'portfolio_exposures': dict(zip(self.factor_names, port_exposures)),
            'factor_risk_contributions': factor_contributions
        }
    
    def decompose_return(self,
                        portfolio_return: float,
                        weights: pd.Series,
                        period_factor_returns: pd.Series) -> Dict:
        """
        Decompose portfolio return into factor and specific components.
        
        Parameters:
        -----------
        portfolio_return : float
            Actual portfolio return
        weights : pd.Series
            Portfolio weights
        period_factor_returns : pd.Series
            Factor returns for the period
            
        Returns:
        --------
        dict : Return attribution
        """
        # Align weights
        common = weights.index.intersection(self.exposures.index)
        w = weights.loc[common].values
        B = self.exposures.loc[common].values
        
        # Portfolio factor exposures
        port_exposures = B.T @ w
        
        # Factor return contributions
        factor_returns = {}
        total_factor_return = 0
        
        for i, factor in enumerate(self.factor_names):
            if factor in period_factor_returns.index:
                factor_ret = port_exposures[i] * period_factor_returns[factor]
                factor_returns[factor] = factor_ret
                total_factor_return += factor_ret
        
        # Alpha contribution (stock selection)
        if hasattr(self, 'alphas'):
            alpha_contribution = (w * self.alphas.loc[common].values).sum()
        else:
            alpha_contribution = 0
        
        # Specific return (residual)
        specific_return = portfolio_return - total_factor_return - alpha_contribution
        
        return {
            'total_return': portfolio_return,
            'factor_return': total_factor_return,
            'alpha_return': alpha_contribution,
            'specific_return': specific_return,
            'factor_contributions': factor_returns,
            'portfolio_exposures': dict(zip(self.factor_names, port_exposures))
        }
    
    def get_factor_summary(self) -> str:
        """Generate factor model summary."""
        summary = []
        summary.append("=" * 50)
        summary.append("FACTOR MODEL SUMMARY")
        summary.append("=" * 50)
        
        summary.append(f"\nFactors: {', '.join(self.factor_names)}")
        summary.append(f"Stocks with exposures: {len(self.exposures)}")
        
        # Factor statistics
        summary.append("\nFactor Statistics (Annualized):")
        factor_stats = pd.DataFrame({
            'Mean': self.factor_returns[self.factor_names].mean() * 252,
            'Vol': self.factor_returns[self.factor_names].std() * np.sqrt(252),
            'Sharpe': (self.factor_returns[self.factor_names].mean() * 252) / 
                     (self.factor_returns[self.factor_names].std() * np.sqrt(252))
        })
        summary.append(factor_stats.to_string())
        
        # Average exposures
        if len(self.exposures) > 0:
            summary.append("\nAverage Factor Exposures:")
            avg_exp = self.exposures.mean()
            for factor, exp in avg_exp.items():
                summary.append(f"  {factor}: {exp:.3f}")
        
        # R-squared distribution
        if hasattr(self, 'r_squared') and len(self.r_squared) > 0:
            summary.append(f"\nModel Fit (R²):")
            summary.append(f"  Mean: {self.r_squared.mean():.1%}")
            summary.append(f"  Median: {self.r_squared.median():.1%}")
            summary.append(f"  Min: {self.r_squared.min():.1%}")
            summary.append(f"  Max: {self.r_squared.max():.1%}")
        
        return "\n".join(summary)


def main():
    """Example usage of factor models."""
    import pickle
    
    # Load data
    try:
        factors = pd.read_csv('data/fama_french_factors.csv', 
                             index_col=0, parse_dates=True)
        adj_prices = pd.read_csv('data/adjusted_prices.csv',
                                index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data files not found. Run data_download.py first.")
        return
    
    # Calculate returns
    returns = adj_prices.pct_change().dropna()
    
    # Initialize factor model
    fm = FactorModel(factor_returns=factors)
    
    # Estimate exposures
    print("Estimating factor exposures...")
    exposures = fm.estimate_exposures(returns, method='ols')
    
    print(f"\nEstimated exposures for {len(exposures)} stocks")
    
    # Print summary
    print("\n" + fm.get_factor_summary())
    
    # Show top stocks by market beta
    if 'Mkt-RF' in exposures.columns:
        print("\nTop 10 stocks by market beta:")
        top_beta = exposures['Mkt-RF'].nlargest(10)
        for ticker, beta in top_beta.items():
            print(f"  {ticker}: {beta:.2f}")
    
    # Calculate expected returns
    expected = fm.calculate_expected_returns()
    print(f"\nExpected returns calculated for {len(expected)} stocks")
    print(f"Mean expected return: {expected.mean():.2%}")


if __name__ == '__main__':
    main()
