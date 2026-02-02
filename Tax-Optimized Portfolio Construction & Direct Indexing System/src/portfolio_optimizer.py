"""
Tax-Optimized Portfolio Optimizer

This module implements:
    - Mean-variance optimization with tax awareness
    - Direct indexing (index replication with subset of stocks)
    - Factor-tilted portfolio construction
    - Tracking error constrained optimization
    
Supports both Gurobi and CVXPY solvers for flexibility.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
import warnings

# Try to import optimization libraries
try:
    import gurobipy as gp
    from gurobipy import GRB
    GUROBI_AVAILABLE = True
except ImportError:
    GUROBI_AVAILABLE = False
    warnings.warn("Gurobi not available. Using CVXPY instead.")

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False

from scipy.optimize import minimize


class TaxOptimizedPortfolio:
    """
    Tax-aware portfolio optimization engine.
    
    Implements mean-variance optimization with tax considerations,
    direct indexing, and factor tilts.
    """
    
    def __init__(self, 
                 returns: pd.Series,
                 cov_matrix: pd.DataFrame,
                 factor_exposures: Optional[pd.DataFrame] = None,
                 tax_rate_short: float = 0.37,
                 tax_rate_long: float = 0.238,
                 risk_aversion: float = 2.5):
        """
        Initialize the portfolio optimizer.
        
        Parameters:
        -----------
        returns : pd.Series
            Expected annual returns for each asset
        cov_matrix : pd.DataFrame
            Covariance matrix of returns (annualized)
        factor_exposures : pd.DataFrame, optional
            Factor loadings for each asset
        tax_rate_short : float
            Short-term capital gains tax rate (default 37%)
        tax_rate_long : float
            Long-term capital gains tax rate (default 23.8% = 20% + 3.8% NIIT)
        risk_aversion : float
            Risk aversion parameter (lambda)
        """
        self.returns = returns
        self.cov_matrix = cov_matrix
        self.factor_exposures = factor_exposures
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        self.risk_aversion = risk_aversion
        
        # Asset universe
        self.tickers = returns.index.tolist()
        self.n_assets = len(self.tickers)
        
        # Convert to numpy arrays for optimization
        self.mu = returns.values
        self.sigma = cov_matrix.values
        
        # Validate covariance matrix
        self._validate_covariance()
        
    def _validate_covariance(self):
        """Ensure covariance matrix is positive semi-definite."""
        eigenvalues = np.linalg.eigvalsh(self.sigma)
        
        if np.any(eigenvalues < -1e-8):
            print("Warning: Covariance matrix has negative eigenvalues. Applying correction...")
            # Make matrix positive semi-definite
            min_eigenvalue = eigenvalues.min()
            self.sigma = self.sigma + (-min_eigenvalue + 1e-6) * np.eye(self.n_assets)
    
    def optimize_mean_variance(self,
                               target_return: Optional[float] = None,
                               target_volatility: Optional[float] = None,
                               max_position: float = 0.10,
                               min_position: float = 0.0,
                               sector_constraints: Optional[Dict] = None) -> Dict:
        """
        Standard mean-variance optimization.
        
        Parameters:
        -----------
        target_return : float, optional
            Target portfolio return (if None, maximizes Sharpe)
        target_volatility : float, optional
            Target portfolio volatility
        max_position : float
            Maximum weight per position
        min_position : float
            Minimum weight per position (0 allows no holding)
        sector_constraints : dict, optional
            {sector: (min_weight, max_weight)}
            
        Returns:
        --------
        dict : Optimization results
        """
        if CVXPY_AVAILABLE:
            return self._optimize_cvxpy(
                target_return=target_return,
                target_volatility=target_volatility,
                max_position=max_position,
                min_position=min_position
            )
        else:
            return self._optimize_scipy(
                target_return=target_return,
                max_position=max_position,
                min_position=min_position
            )
    
    def _optimize_cvxpy(self, 
                       target_return: Optional[float] = None,
                       target_volatility: Optional[float] = None,
                       max_position: float = 0.10,
                       min_position: float = 0.0) -> Dict:
        """Optimization using CVXPY."""
        # Decision variable
        w = cp.Variable(self.n_assets)
        
        # Portfolio return and risk
        port_return = self.mu @ w
        port_variance = cp.quad_form(w, self.sigma)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_position,
            w <= max_position
        ]
        
        if target_return is not None:
            constraints.append(port_return >= target_return)
            # Minimize variance subject to target return
            objective = cp.Minimize(port_variance)
        elif target_volatility is not None:
            constraints.append(port_variance <= target_volatility ** 2)
            # Maximize return subject to target volatility
            objective = cp.Maximize(port_return)
        else:
            # Maximize Sharpe ratio approximation (utility function)
            objective = cp.Maximize(port_return - self.risk_aversion * port_variance)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP, warm_start=True)
            
            if problem.status == 'optimal':
                weights = w.value
                weights = np.maximum(weights, 0)  # Clean up numerical errors
                weights = weights / weights.sum()  # Renormalize
                
                return self._create_result(weights, 'optimal')
            else:
                print(f"Optimization status: {problem.status}")
                return {'status': problem.status}
                
        except Exception as e:
            print(f"CVXPY optimization error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _optimize_scipy(self,
                       target_return: Optional[float] = None,
                       max_position: float = 0.10,
                       min_position: float = 0.0) -> Dict:
        """Fallback optimization using SciPy."""
        
        def neg_sharpe(w):
            port_return = np.dot(w, self.mu)
            port_vol = np.sqrt(np.dot(w, np.dot(self.sigma, w)))
            return -port_return / port_vol if port_vol > 0 else 0
        
        def portfolio_variance(w):
            return np.dot(w, np.dot(self.sigma, w))
        
        # Initial guess (equal weight)
        w0 = np.ones(self.n_assets) / self.n_assets
        
        # Constraints
        constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                'type': 'ineq', 
                'fun': lambda w: np.dot(w, self.mu) - target_return
            })
            objective = portfolio_variance
        else:
            objective = neg_sharpe
        
        # Bounds
        bounds = [(min_position, max_position) for _ in range(self.n_assets)]
        
        # Optimize
        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000}
        )
        
        if result.success:
            weights = result.x
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            return self._create_result(weights, 'optimal')
        else:
            return {'status': 'failed', 'message': result.message}
    
    def optimize_tax_aware(self,
                          initial_weights: Optional[Dict[str, float]] = None,
                          cost_basis: Optional[Dict[str, float]] = None,
                          current_prices: Optional[Dict[str, float]] = None,
                          max_position: float = 0.10,
                          min_position: float = 0.0,
                          turnover_limit: float = 0.30,
                          holding_periods: Optional[Dict[str, int]] = None) -> Dict:
        """
        Tax-aware portfolio optimization.
        
        Considers capital gains taxes when rebalancing.
        
        Parameters:
        -----------
        initial_weights : dict
            Current portfolio weights {ticker: weight}
        cost_basis : dict
            Cost basis for each position {ticker: cost_per_share}
        current_prices : dict
            Current prices {ticker: price}
        max_position : float
            Maximum weight per position
        min_position : float
            Minimum weight per position
        turnover_limit : float
            Maximum portfolio turnover (one-way)
        holding_periods : dict
            Days held for each position {ticker: days}
            
        Returns:
        --------
        dict : Optimization results with tax-adjusted metrics
        """
        if not CVXPY_AVAILABLE:
            print("Tax-aware optimization requires CVXPY")
            return self.optimize_mean_variance(max_position=max_position)
        
        # Decision variables
        w = cp.Variable(self.n_assets)
        
        # Portfolio return and risk
        port_return = self.mu @ w
        port_variance = cp.quad_form(w, self.sigma)
        
        # Base constraints
        constraints = [
            cp.sum(w) == 1,
            w >= min_position,
            w <= max_position
        ]
        
        # Tax cost estimation
        tax_cost = 0
        
        if initial_weights is not None:
            init_w = np.array([initial_weights.get(t, 0) for t in self.tickers])
            
            # Turnover constraint (absolute deviation)
            turnover = cp.norm(w - init_w, 1)
            constraints.append(turnover <= 2 * turnover_limit)
            
            # Calculate tax impact from selling
            if cost_basis is not None and current_prices is not None:
                for i, ticker in enumerate(self.tickers):
                    if ticker in cost_basis and ticker in current_prices:
                        basis = cost_basis[ticker]
                        current = current_prices[ticker]
                        
                        if current > basis:
                            # Unrealized gain percentage
                            gain_pct = (current - basis) / basis
                            
                            # Determine tax rate based on holding period
                            if holding_periods and ticker in holding_periods:
                                tax_rate = (self.tax_rate_long 
                                          if holding_periods[ticker] >= 365 
                                          else self.tax_rate_short)
                            else:
                                tax_rate = self.tax_rate_long
                            
                            # Tax on realized gains (approximation)
                            # Only pay tax on the amount sold
                            sell_amount = cp.maximum(init_w[i] - w[i], 0)
                            tax_cost += sell_amount * gain_pct * tax_rate
        
        # Objective: Maximize after-tax risk-adjusted return
        if isinstance(tax_cost, (int, float)):
            objective = cp.Maximize(port_return - self.risk_aversion * port_variance)
        else:
            objective = cp.Maximize(port_return - tax_cost - self.risk_aversion * port_variance)
        
        # Solve
        problem = cp.Problem(objective, constraints)
        
        try:
            problem.solve(solver=cp.OSQP)
            
            if problem.status == 'optimal':
                weights = w.value
                weights = np.maximum(weights, 0)
                weights = weights / weights.sum()
                
                result = self._create_result(weights, 'optimal')
                
                # Calculate actual tax cost
                if initial_weights and cost_basis and current_prices:
                    result['estimated_tax_cost'] = self._calculate_tax_cost(
                        initial_weights, weights, cost_basis, 
                        current_prices, holding_periods
                    )
                    result['turnover'] = np.sum(np.abs(weights - init_w)) / 2
                
                return result
            else:
                return {'status': problem.status}
                
        except Exception as e:
            print(f"Tax-aware optimization error: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _calculate_tax_cost(self, 
                           initial_weights: Dict,
                           new_weights: np.ndarray,
                           cost_basis: Dict,
                           current_prices: Dict,
                           holding_periods: Optional[Dict]) -> float:
        """Calculate actual tax cost from rebalancing."""
        tax_cost = 0
        
        for i, ticker in enumerate(self.tickers):
            init_w = initial_weights.get(ticker, 0)
            new_w = new_weights[i]
            
            # Only selling triggers tax
            if new_w < init_w and ticker in cost_basis and ticker in current_prices:
                sell_amount = init_w - new_w
                basis = cost_basis[ticker]
                current = current_prices[ticker]
                
                if current > basis:
                    gain = (current - basis) / basis * sell_amount
                    
                    if holding_periods and ticker in holding_periods:
                        tax_rate = (self.tax_rate_long 
                                  if holding_periods[ticker] >= 365 
                                  else self.tax_rate_short)
                    else:
                        tax_rate = self.tax_rate_long
                    
                    tax_cost += gain * tax_rate
        
        return tax_cost
    
    def optimize_direct_indexing(self,
                                 benchmark_weights: Dict[str, float],
                                 n_stocks: int = 100,
                                 tracking_error_limit: float = 0.015,
                                 max_position: float = 0.05,
                                 sector_neutral: bool = True,
                                 sector_mapping: Optional[Dict[str, str]] = None) -> Dict:
        """
        Direct indexing: replicate index with subset of stocks.
        
        Parameters:
        -----------
        benchmark_weights : dict
            Target index weights {ticker: weight}
        n_stocks : int
            Maximum number of stocks to hold
        tracking_error_limit : float
            Maximum tracking error vs benchmark
        max_position : float
            Maximum weight per position
        sector_neutral : bool
            If True, match sector weights to benchmark
        sector_mapping : dict
            {ticker: sector} mapping
            
        Returns:
        --------
        dict : Optimization results
        """
        # Convert benchmark weights to array
        benchmark_w = np.array([benchmark_weights.get(t, 0) for t in self.tickers])
        
        if GUROBI_AVAILABLE:
            return self._direct_indexing_gurobi(
                benchmark_w, n_stocks, tracking_error_limit, 
                max_position, sector_neutral, sector_mapping
            )
        elif CVXPY_AVAILABLE:
            return self._direct_indexing_cvxpy(
                benchmark_w, n_stocks, tracking_error_limit,
                max_position, sector_neutral, sector_mapping
            )
        else:
            print("Direct indexing requires Gurobi or CVXPY with integer support")
            return {'status': 'failed'}
    
    def _direct_indexing_gurobi(self,
                               benchmark_w: np.ndarray,
                               n_stocks: int,
                               tracking_error_limit: float,
                               max_position: float,
                               sector_neutral: bool,
                               sector_mapping: Optional[Dict]) -> Dict:
        """Direct indexing using Gurobi (supports integer constraints)."""
        model = gp.Model("DirectIndexing")
        model.setParam('OutputFlag', 0)
        
        # Binary variables: include stock or not
        include = model.addVars(self.n_assets, vtype=GRB.BINARY, name="include")
        
        # Continuous weights
        weights = model.addVars(self.n_assets, lb=0, ub=max_position, name="weights")
        
        # Link weights to inclusion
        for i in range(self.n_assets):
            model.addConstr(weights[i] <= include[i] * max_position)
            model.addConstr(weights[i] >= include[i] * 0.001)  # Minimum if included
        
        # Cardinality constraint
        model.addConstr(
            gp.quicksum(include[i] for i in range(self.n_assets)) <= n_stocks
        )
        
        # Budget constraint
        model.addConstr(
            gp.quicksum(weights[i] for i in range(self.n_assets)) == 1
        )
        
        # Tracking error constraint
        # TE^2 = (w - w_b)' * Sigma * (w - w_b)
        tracking_variance = gp.QuadExpr()
        for i in range(self.n_assets):
            for j in range(self.n_assets):
                tracking_variance += (
                    (weights[i] - benchmark_w[i]) * 
                    (weights[j] - benchmark_w[j]) * 
                    self.sigma[i, j]
                )
        
        model.addConstr(tracking_variance <= tracking_error_limit ** 2)
        
        # Sector constraints (if requested)
        if sector_neutral and sector_mapping:
            sectors = {}
            for i, ticker in enumerate(self.tickers):
                sector = sector_mapping.get(ticker, 'Unknown')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(i)
            
            for sector, indices in sectors.items():
                sector_benchmark = sum(benchmark_w[i] for i in indices)
                sector_weight = gp.quicksum(weights[i] for i in indices)
                
                # Allow 2% deviation
                model.addConstr(sector_weight >= sector_benchmark - 0.02)
                model.addConstr(sector_weight <= sector_benchmark + 0.02)
        
        # Objective: Minimize weight deviation from benchmark
        objective = gp.quicksum(
            (weights[i] - benchmark_w[i]) * (weights[i] - benchmark_w[i])
            for i in range(self.n_assets)
        )
        
        model.setObjective(objective, GRB.MINIMIZE)
        
        # Set time limit
        model.setParam('TimeLimit', 300)
        
        model.optimize()
        
        if model.status in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            selected = [i for i in range(self.n_assets) if include[i].X > 0.5]
            weights_arr = np.array([weights[i].X for i in range(self.n_assets)])
            
            # Renormalize
            weights_arr = weights_arr / weights_arr.sum()
            
            result = self._create_result(weights_arr, 'optimal')
            result['n_holdings'] = len(selected)
            result['selected_tickers'] = [self.tickers[i] for i in selected]
            result['tracking_error'] = np.sqrt(
                (weights_arr - benchmark_w) @ self.sigma @ (weights_arr - benchmark_w)
            )
            
            return result
        else:
            return {'status': 'failed', 'gurobi_status': model.status}
    
    def _direct_indexing_cvxpy(self,
                              benchmark_w: np.ndarray,
                              n_stocks: int,
                              tracking_error_limit: float,
                              max_position: float,
                              sector_neutral: bool,
                              sector_mapping: Optional[Dict]) -> Dict:
        """
        Direct indexing using CVXPY (continuous relaxation).
        Uses L1 penalty to encourage sparsity.
        """
        w = cp.Variable(self.n_assets)
        
        # Portfolio deviation from benchmark
        deviation = w - benchmark_w
        
        # Tracking error (variance of deviation)
        tracking_var = cp.quad_form(deviation, self.sigma)
        
        # Weight deviation (sum of squared differences)
        weight_deviation = cp.sum_squares(deviation)
        
        # Sparsity penalty (L1 norm encourages zeros)
        sparsity_penalty = cp.norm(w, 1)
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_position,
            tracking_var <= tracking_error_limit ** 2
        ]
        
        # Sector constraints
        if sector_neutral and sector_mapping:
            sectors = {}
            for i, ticker in enumerate(self.tickers):
                sector = sector_mapping.get(ticker, 'Unknown')
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(i)
            
            for sector, indices in sectors.items():
                sector_benchmark = sum(benchmark_w[i] for i in indices)
                sector_weight = cp.sum(w[indices])
                constraints.append(sector_weight >= sector_benchmark - 0.02)
                constraints.append(sector_weight <= sector_benchmark + 0.02)
        
        # Objective: minimize weight deviation + sparsity penalty
        lambda_sparse = 0.01 * (self.n_assets / n_stocks)  # Scale penalty
        objective = cp.Minimize(weight_deviation + lambda_sparse * sparsity_penalty)
        
        problem = cp.Problem(objective, constraints)
        # Use CLARABEL or SCS for problems with L1 norm (conic solver needed)
        try:
            problem.solve(solver=cp.CLARABEL)
        except Exception:
            try:
                problem.solve(solver=cp.SCS)
            except Exception:
                problem.solve()  # Let CVXPY choose
        
        if problem.status == 'optimal':
            weights = w.value
            
            # Apply threshold to create sparsity
            threshold = np.sort(weights)[-n_stocks] if len(weights) > n_stocks else 0
            weights[weights < max(threshold, 0.005)] = 0
            
            # Renormalize
            if weights.sum() > 0:
                weights = weights / weights.sum()
            
            result = self._create_result(weights, 'optimal')
            result['n_holdings'] = np.sum(weights > 0)
            result['selected_tickers'] = [
                self.tickers[i] for i in range(self.n_assets) 
                if weights[i] > 0
            ]
            result['tracking_error'] = np.sqrt(
                (weights - benchmark_w) @ self.sigma @ (weights - benchmark_w)
            )
            
            return result
        else:
            return {'status': problem.status}
    
    def optimize_factor_tilted(self,
                              factor_tilts: Dict[str, float],
                              benchmark_weights: Optional[Dict[str, float]] = None,
                              tracking_error_limit: float = 0.03,
                              max_position: float = 0.05) -> Dict:
        """
        Construct portfolio with factor tilts.
        
        Parameters:
        -----------
        factor_tilts : dict
            Desired factor exposures {factor: target_exposure}
            e.g., {'Mkt-RF': 1.0, 'SMB': 0.2, 'HML': 0.3}
        benchmark_weights : dict, optional
            Reference weights for tracking error
        tracking_error_limit : float
            Maximum tracking error
        max_position : float
            Maximum position size
            
        Returns:
        --------
        dict : Optimization results with factor exposures
        """
        if self.factor_exposures is None:
            print("Factor exposures not available")
            return {'status': 'failed', 'error': 'No factor exposures'}
        
        if not CVXPY_AVAILABLE:
            print("Factor-tilted optimization requires CVXPY")
            return {'status': 'failed'}
        
        # Get factor exposure matrix
        factor_cols = list(factor_tilts.keys())
        available_factors = [f for f in factor_cols if f in self.factor_exposures.columns]
        
        if not available_factors:
            print(f"None of the requested factors available. Have: {self.factor_exposures.columns.tolist()}")
            return {'status': 'failed'}
        
        B = self.factor_exposures[available_factors].values
        target_exposures = np.array([factor_tilts[f] for f in available_factors])
        
        # Decision variable
        w = cp.Variable(self.n_assets)
        
        # Portfolio factor exposures
        port_exposures = B.T @ w
        
        # Constraints
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= max_position
        ]
        
        # Factor exposure constraints (with some tolerance)
        for i, factor in enumerate(available_factors):
            target = target_exposures[i]
            constraints.append(port_exposures[i] >= target - 0.1)
            constraints.append(port_exposures[i] <= target + 0.1)
        
        # Tracking error constraint
        if benchmark_weights is not None:
            benchmark_w = np.array([benchmark_weights.get(t, 0) for t in self.tickers])
            deviation = w - benchmark_w
            tracking_var = cp.quad_form(deviation, self.sigma)
            constraints.append(tracking_var <= tracking_error_limit ** 2)
        
        # Objective: Maximize return while matching factor exposures
        port_return = self.mu @ w
        port_variance = cp.quad_form(w, self.sigma)
        
        # Penalize deviation from target factor exposures
        factor_penalty = cp.sum_squares(port_exposures - target_exposures)
        
        objective = cp.Maximize(
            port_return - self.risk_aversion * port_variance - 10 * factor_penalty
        )
        
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == 'optimal':
            weights = w.value
            weights = np.maximum(weights, 0)
            weights = weights / weights.sum()
            
            result = self._create_result(weights, 'optimal')
            
            # Add factor exposures
            actual_exposures = B.T @ weights
            result['factor_exposures'] = dict(zip(available_factors, actual_exposures))
            result['target_exposures'] = dict(zip(available_factors, target_exposures))
            
            return result
        else:
            return {'status': problem.status}
    
    def generate_efficient_frontier(self, 
                                   n_points: int = 50,
                                   max_position: float = 0.10) -> pd.DataFrame:
        """
        Generate the efficient frontier.
        
        Parameters:
        -----------
        n_points : int
            Number of points on the frontier
        max_position : float
            Maximum position size
            
        Returns:
        --------
        pd.DataFrame : Frontier with returns, volatilities, Sharpe ratios
        """
        # Find return range
        min_vol_result = self.optimize_mean_variance(
            target_return=self.mu.min(),
            max_position=max_position
        )
        
        max_return_result = self.optimize_mean_variance(
            target_return=self.mu.max() * 0.8,  # Slightly below max
            max_position=max_position
        )
        
        if min_vol_result['status'] != 'optimal':
            min_return = self.mu.min()
        else:
            min_return = min_vol_result['expected_return']
        
        if max_return_result['status'] != 'optimal':
            max_return = self.mu.max() * 0.5
        else:
            max_return = max_return_result['expected_return']
        
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier = []
        
        for target in target_returns:
            result = self.optimize_mean_variance(
                target_return=target,
                max_position=max_position
            )
            
            if result['status'] == 'optimal':
                frontier.append({
                    'return': result['expected_return'],
                    'volatility': result['volatility'],
                    'sharpe': result['sharpe_ratio'],
                    'weights': result['weights']
                })
        
        return pd.DataFrame(frontier)
    
    def _create_result(self, weights: np.ndarray, status: str) -> Dict:
        """Create standardized result dictionary."""
        # Calculate portfolio metrics
        port_return = np.dot(weights, self.mu)
        port_vol = np.sqrt(np.dot(weights, np.dot(self.sigma, weights)))
        sharpe = port_return / port_vol if port_vol > 0 else 0
        
        # Create weights dictionary
        weights_dict = {
            self.tickers[i]: weights[i] 
            for i in range(self.n_assets)
            if weights[i] > 1e-6
        }
        
        return {
            'status': status,
            'weights': weights_dict,
            'weights_array': weights,
            'expected_return': port_return,
            'volatility': port_vol,
            'sharpe_ratio': sharpe,
            'n_holdings': sum(1 for w in weights if w > 1e-6)
        }
    
    def get_portfolio_summary(self, result: Dict, 
                             sector_mapping: Optional[Dict] = None) -> str:
        """Generate human-readable portfolio summary."""
        if result['status'] != 'optimal':
            return f"Optimization failed with status: {result['status']}"
        
        summary = []
        summary.append("=" * 50)
        summary.append("PORTFOLIO OPTIMIZATION RESULTS")
        summary.append("=" * 50)
        summary.append(f"\nExpected Annual Return: {result['expected_return']:.2%}")
        summary.append(f"Annual Volatility:      {result['volatility']:.2%}")
        summary.append(f"Sharpe Ratio:           {result['sharpe_ratio']:.2f}")
        summary.append(f"Number of Holdings:     {result['n_holdings']}")
        
        # Top holdings
        weights_sorted = sorted(
            result['weights'].items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        summary.append(f"\nTop 10 Holdings:")
        for ticker, weight in weights_sorted:
            sector = sector_mapping.get(ticker, '') if sector_mapping else ''
            summary.append(f"  {ticker:6} {weight:6.2%}  {sector}")
        
        # Sector breakdown
        if sector_mapping:
            sector_weights = {}
            for ticker, weight in result['weights'].items():
                sector = sector_mapping.get(ticker, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            summary.append(f"\nSector Allocation:")
            for sector, weight in sorted(sector_weights.items(), key=lambda x: -x[1]):
                summary.append(f"  {sector:30} {weight:6.2%}")
        
        return "\n".join(summary)


def main():
    """Example usage of the portfolio optimizer."""
    import pickle
    
    # Load processed data
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Run data_preprocessing.py first.")
        return
    
    # Initialize optimizer
    optimizer = TaxOptimizedPortfolio(
        returns=data['returns'],
        cov_matrix=data['cov_matrix'],
        factor_exposures=data.get('factor_exposures'),
        tax_rate_long=0.238
    )
    
    print("\n" + "=" * 50)
    print("1. MEAN-VARIANCE OPTIMIZATION")
    print("=" * 50)
    
    result = optimizer.optimize_mean_variance(max_position=0.05)
    print(optimizer.get_portfolio_summary(result, data.get('sector_mapping')))
    
    print("\n" + "=" * 50)
    print("2. DIRECT INDEXING (100 stocks)")
    print("=" * 50)
    
    di_result = optimizer.optimize_direct_indexing(
        benchmark_weights=data['benchmark_weights'].to_dict(),
        n_stocks=100,
        tracking_error_limit=0.015,
        sector_neutral=True,
        sector_mapping=data.get('sector_mapping')
    )
    
    if di_result['status'] == 'optimal':
        print(f"Holdings: {di_result['n_holdings']}")
        print(f"Tracking Error: {di_result['tracking_error']:.2%}")
        print(optimizer.get_portfolio_summary(di_result, data.get('sector_mapping')))


if __name__ == '__main__':
    main()
