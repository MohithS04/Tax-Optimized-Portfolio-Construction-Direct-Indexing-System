"""
Backtesting Engine Module

This module implements:
    - Historical portfolio backtesting
    - Performance metrics calculation
    - Rebalancing simulation
    - Tax-aware backtesting
    - Benchmark comparison
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import warnings


@dataclass
class BacktestResult:
    """Container for backtest results."""
    portfolio_values: pd.Series
    returns: pd.Series
    weights_history: pd.DataFrame
    trades_history: List[Dict]
    metrics: Dict
    benchmark_values: Optional[pd.Series] = None
    tax_summary: Optional[Dict] = None


class Backtester:
    """
    Portfolio backtesting engine with tax awareness.
    
    Simulates historical portfolio performance with realistic
    rebalancing, transaction costs, and tax impacts.
    """
    
    def __init__(self,
                 prices: pd.DataFrame,
                 initial_capital: float = 1_000_000,
                 transaction_cost: float = 0.001,
                 tax_rate_short: float = 0.37,
                 tax_rate_long: float = 0.238):
        """
        Initialize backtester.
        
        Parameters:
        -----------
        prices : pd.DataFrame
            Historical prices (adjusted), index=dates, columns=tickers
        initial_capital : float
            Starting portfolio value
        transaction_cost : float
            Transaction cost as fraction (0.001 = 10 bps)
        tax_rate_short : float
            Short-term capital gains rate
        tax_rate_long : float
            Long-term capital gains rate
        """
        self.prices = prices
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        
        # Calculate returns
        self.returns = prices.pct_change().dropna()
        
        # State tracking
        self.current_weights = None
        self.current_shares = None
        self.cost_basis = {}
        self.purchase_dates = {}
        
    def run_backtest(self,
                    initial_weights: Dict[str, float],
                    rebalance_freq: str = 'M',
                    rebalance_threshold: float = 0.05,
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    benchmark_ticker: Optional[str] = None,
                    weight_generator: Optional[Callable] = None,
                    tax_aware: bool = False) -> BacktestResult:
        """
        Run portfolio backtest.
        
        Parameters:
        -----------
        initial_weights : dict
            Starting portfolio weights {ticker: weight}
        rebalance_freq : str
            Rebalancing frequency: 'D', 'W', 'M', 'Q', 'Y', or 'none'
        rebalance_threshold : float
            Rebalance if drift exceeds this threshold
        start_date : datetime, optional
            Backtest start date
        end_date : datetime, optional
            Backtest end date
        benchmark_ticker : str, optional
            Benchmark for comparison (e.g., '^GSPC')
        weight_generator : callable, optional
            Function to generate new weights at rebalance
        tax_aware : bool
            If True, track and calculate tax impacts
            
        Returns:
        --------
        BacktestResult : Complete backtest results
        """
        # Set date range
        if start_date is None:
            start_date = self.returns.index[0]
        if end_date is None:
            end_date = self.returns.index[-1]
        
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        
        # Filter data - use returns index as base since it's derived from prices
        mask = (self.returns.index >= start_date) & (self.returns.index <= end_date)
        returns = self.returns.loc[mask]
        # Filter prices to match returns index
        prices = self.prices.loc[returns.index]
        
        if len(returns) == 0:
            raise ValueError("No data in specified date range")
        
        # Get rebalance dates
        rebalance_dates = self._get_rebalance_dates(returns.index, rebalance_freq)
        
        # Initialize portfolio
        portfolio_value = self.initial_capital
        
        # Align weights with available tickers
        tickers = [t for t in initial_weights.keys() if t in returns.columns]
        weights = pd.Series({t: initial_weights[t] for t in tickers})
        weights = weights / weights.sum()  # Normalize
        
        # Initialize tracking
        self.current_weights = weights.copy()
        self.current_shares = (portfolio_value * weights) / prices.iloc[0][tickers]
        
        # Initialize cost basis for tax tracking
        if tax_aware:
            for ticker in tickers:
                self.cost_basis[ticker] = prices.iloc[0][ticker]
                self.purchase_dates[ticker] = prices.index[0]
        
        # Storage for results
        portfolio_values = []
        portfolio_returns = []
        weights_history = []
        trades_history = []
        realized_gains = []
        realized_losses = []
        
        # Run simulation
        prev_value = portfolio_value
        
        for i, date in enumerate(returns.index):
            # Calculate current portfolio value
            current_prices = prices.loc[date][tickers]
            position_values = self.current_shares * current_prices
            portfolio_value = position_values.sum()
            
            # Update weights
            self.current_weights = position_values / portfolio_value
            
            # Calculate return
            daily_return = (portfolio_value - prev_value) / prev_value if prev_value > 0 else 0
            
            portfolio_values.append(portfolio_value)
            portfolio_returns.append(daily_return)
            weights_history.append(self.current_weights.to_dict())
            
            # Check for rebalancing
            should_rebalance = False
            
            if rebalance_freq != 'none' and date in rebalance_dates:
                should_rebalance = True
            elif rebalance_threshold > 0:
                drift = np.abs(self.current_weights - weights).max()
                if drift > rebalance_threshold:
                    should_rebalance = True
            
            if should_rebalance:
                # Get new target weights
                if weight_generator is not None:
                    try:
                        target_weights = weight_generator(date, returns.loc[:date])
                        target_weights = pd.Series(target_weights)
                        target_weights = target_weights[target_weights.index.isin(tickers)]
                        target_weights = target_weights / target_weights.sum()
                    except Exception as e:
                        warnings.warn(f"Weight generator failed: {e}")
                        target_weights = weights
                else:
                    target_weights = weights
                
                # Execute rebalancing
                trades, gains, losses = self._execute_rebalance(
                    current_weights=self.current_weights,
                    target_weights=target_weights,
                    portfolio_value=portfolio_value,
                    current_prices=current_prices,
                    date=date,
                    tax_aware=tax_aware
                )
                
                if trades:
                    trades_history.extend(trades)
                    realized_gains.append(gains)
                    realized_losses.append(losses)
            
            prev_value = portfolio_value
        
        # Create result series
        portfolio_values = pd.Series(portfolio_values, index=returns.index)
        portfolio_returns = pd.Series(portfolio_returns, index=returns.index)
        weights_df = pd.DataFrame(weights_history, index=returns.index)
        
        # Calculate metrics
        metrics = self._calculate_metrics(portfolio_returns, portfolio_values)
        
        # Get benchmark if specified
        benchmark_values = None
        if benchmark_ticker and benchmark_ticker in self.prices.columns:
            benchmark_prices = self.prices.loc[mask, benchmark_ticker]
            benchmark_values = self.initial_capital * (benchmark_prices / benchmark_prices.iloc[0])
            
            # Add benchmark comparison metrics
            benchmark_returns = benchmark_prices.pct_change().dropna()
            metrics['benchmark_return'] = (benchmark_values.iloc[-1] / benchmark_values.iloc[0] - 1)
            metrics['alpha'] = metrics['total_return'] - metrics['benchmark_return']
            metrics['tracking_error'] = (portfolio_returns - benchmark_returns.loc[portfolio_returns.index]).std() * np.sqrt(252)
            metrics['information_ratio'] = metrics['alpha'] / metrics['tracking_error'] if metrics['tracking_error'] > 0 else 0
        
        # Tax summary
        tax_summary = None
        if tax_aware:
            total_gains = sum(realized_gains) if realized_gains else 0
            total_losses = sum(realized_losses) if realized_losses else 0
            tax_summary = {
                'realized_gains': total_gains,
                'realized_losses': total_losses,
                'net_realized': total_gains + total_losses,
                'estimated_tax': max(0, total_gains + total_losses) * self.tax_rate_long
            }
            metrics['tax_drag'] = tax_summary['estimated_tax'] / self.initial_capital
        
        return BacktestResult(
            portfolio_values=portfolio_values,
            returns=portfolio_returns,
            weights_history=weights_df,
            trades_history=trades_history,
            metrics=metrics,
            benchmark_values=benchmark_values,
            tax_summary=tax_summary
        )
    
    def _get_rebalance_dates(self, 
                            dates: pd.DatetimeIndex,
                            freq: str) -> set:
        """Get set of rebalancing dates."""
        if freq == 'none':
            return set()
        
        if freq == 'D':
            return set(dates)
        elif freq == 'W':
            return set(dates.to_period('W').to_timestamp().unique())
        elif freq == 'M':
            return set(dates.to_period('M').to_timestamp().unique())
        elif freq == 'Q':
            return set(dates.to_period('Q').to_timestamp().unique())
        elif freq == 'Y':
            return set(dates.to_period('Y').to_timestamp().unique())
        else:
            return set()
    
    def _execute_rebalance(self,
                          current_weights: pd.Series,
                          target_weights: pd.Series,
                          portfolio_value: float,
                          current_prices: pd.Series,
                          date: datetime,
                          tax_aware: bool) -> Tuple[List[Dict], float, float]:
        """Execute portfolio rebalancing."""
        trades = []
        realized_gains = 0
        realized_losses = 0
        
        # Calculate trade amounts
        current_values = current_weights * portfolio_value
        target_values = target_weights * portfolio_value
        
        # Ensure same tickers
        all_tickers = current_values.index.union(target_values.index)
        current_values = current_values.reindex(all_tickers, fill_value=0)
        target_values = target_values.reindex(all_tickers, fill_value=0)
        
        trade_values = target_values - current_values
        
        for ticker in all_tickers:
            trade_amount = trade_values[ticker]
            
            if abs(trade_amount) < 100:  # Skip tiny trades
                continue
            
            if ticker not in current_prices.index:
                continue
            
            price = current_prices[ticker]
            shares_traded = trade_amount / price
            
            # Transaction cost
            cost = abs(trade_amount) * self.transaction_cost
            
            trade_record = {
                'date': date,
                'ticker': ticker,
                'shares': shares_traded,
                'price': price,
                'value': trade_amount,
                'cost': cost,
                'type': 'buy' if trade_amount > 0 else 'sell'
            }
            
            # Tax impact for sells
            if tax_aware and trade_amount < 0 and ticker in self.cost_basis:
                cost_basis = self.cost_basis[ticker]
                gain_loss = (price - cost_basis) * abs(shares_traded)
                trade_record['realized_gain_loss'] = gain_loss
                
                if gain_loss > 0:
                    realized_gains += gain_loss
                else:
                    realized_losses += gain_loss
            
            trades.append(trade_record)
            
            # Update shares
            if ticker in self.current_shares.index:
                self.current_shares[ticker] += shares_traded
            
            # Update cost basis for buys
            if tax_aware and trade_amount > 0:
                if ticker in self.cost_basis:
                    # Average cost basis
                    old_value = self.cost_basis[ticker] * (current_values[ticker] / price)
                    new_value = price * shares_traded
                    total_shares = (current_values[ticker] / price) + shares_traded
                    self.cost_basis[ticker] = (old_value * self.cost_basis[ticker] + new_value * price) / (old_value + new_value) if (old_value + new_value) > 0 else price
                else:
                    self.cost_basis[ticker] = price
                    self.purchase_dates[ticker] = date
        
        # Update weights
        self.current_weights = target_weights.copy()
        
        return trades, realized_gains, realized_losses
    
    def _calculate_metrics(self, 
                          returns: pd.Series,
                          values: pd.Series) -> Dict:
        """Calculate comprehensive performance metrics."""
        # Basic returns
        total_return = values.iloc[-1] / values.iloc[0] - 1
        days = (values.index[-1] - values.index[0]).days
        years = days / 365.25
        cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
        
        # Risk metrics
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Risk-adjusted returns
        rf_rate = 0.02 / 252  # Assume 2% annual
        excess_returns = returns - rf_rate
        sharpe = np.sqrt(252) * excess_returns.mean() / returns.std() if returns.std() > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_vol = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.0001
        sortino = (cagr - 0.02) / downside_vol if downside_vol > 0 else 0
        
        # Drawdown analysis
        rolling_max = values.cummax()
        drawdown = (values - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar ratio
        calmar = cagr / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        win_days = (returns > 0).sum()
        total_days = len(returns)
        win_rate = win_days / total_days if total_days > 0 else 0
        
        # Best/worst periods
        best_day = returns.max()
        worst_day = returns.min()
        best_month = returns.resample('M').sum().max() if len(returns) > 20 else returns.max()
        worst_month = returns.resample('M').sum().min() if len(returns) > 20 else returns.min()
        
        # Skewness and kurtosis
        skewness = returns.skew()
        kurtosis = returns.kurtosis()
        
        return {
            'total_return': total_return,
            'cagr': cagr,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'sortino_ratio': sortino,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'best_day': best_day,
            'worst_day': worst_day,
            'best_month': best_month,
            'worst_month': worst_month,
            'skewness': skewness,
            'kurtosis': kurtosis,
            'total_days': total_days,
            'years': years
        }
    
    def run_monte_carlo(self,
                       initial_weights: Dict[str, float],
                       n_simulations: int = 1000,
                       n_days: int = 252,
                       rebalance_freq: str = 'M') -> Dict:
        """
        Run Monte Carlo simulation for portfolio.
        
        Parameters:
        -----------
        initial_weights : dict
            Portfolio weights
        n_simulations : int
            Number of simulation paths
        n_days : int
            Days to simulate
        rebalance_freq : str
            Rebalancing frequency
            
        Returns:
        --------
        dict : Simulation results with percentiles
        """
        # Get tickers and returns
        tickers = [t for t in initial_weights.keys() if t in self.returns.columns]
        returns = self.returns[tickers]
        weights = pd.Series({t: initial_weights[t] for t in tickers})
        weights = weights / weights.sum()
        
        # Historical statistics
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Generate simulations
        final_values = []
        max_drawdowns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mean_returns.values,
                cov_matrix.values,
                n_days
            )
            
            # Portfolio returns
            port_returns = random_returns @ weights.values
            
            # Calculate path
            values = self.initial_capital * np.cumprod(1 + port_returns)
            
            final_values.append(values[-1])
            
            # Max drawdown
            rolling_max = np.maximum.accumulate(values)
            drawdown = (values - rolling_max) / rolling_max
            max_drawdowns.append(drawdown.min())
        
        final_values = np.array(final_values)
        max_drawdowns = np.array(max_drawdowns)
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'percentile_5': np.percentile(final_values, 5),
            'percentile_25': np.percentile(final_values, 25),
            'percentile_75': np.percentile(final_values, 75),
            'percentile_95': np.percentile(final_values, 95),
            'prob_loss': np.mean(final_values < self.initial_capital),
            'mean_max_drawdown': np.mean(max_drawdowns),
            'worst_max_drawdown': np.min(max_drawdowns),
            'var_95': np.percentile(final_values, 5),
            'cvar_95': np.mean(final_values[final_values <= np.percentile(final_values, 5)])
        }
    
    def generate_report(self, result: BacktestResult) -> str:
        """Generate human-readable backtest report."""
        report = []
        report.append("=" * 60)
        report.append("PORTFOLIO BACKTEST REPORT")
        report.append("=" * 60)
        
        m = result.metrics
        
        report.append(f"\nBacktest Period: {result.portfolio_values.index[0].date()} to {result.portfolio_values.index[-1].date()}")
        report.append(f"Trading Days: {m['total_days']}")
        report.append(f"Years: {m['years']:.2f}")
        
        report.append("\n" + "-" * 40)
        report.append("RETURNS")
        report.append("-" * 40)
        report.append(f"Total Return:        {m['total_return']:>10.2%}")
        report.append(f"CAGR:                {m['cagr']:>10.2%}")
        report.append(f"Best Day:            {m['best_day']:>10.2%}")
        report.append(f"Worst Day:           {m['worst_day']:>10.2%}")
        report.append(f"Best Month:          {m['best_month']:>10.2%}")
        report.append(f"Worst Month:         {m['worst_month']:>10.2%}")
        
        report.append("\n" + "-" * 40)
        report.append("RISK")
        report.append("-" * 40)
        report.append(f"Annual Volatility:   {m['annual_volatility']:>10.2%}")
        report.append(f"Max Drawdown:        {m['max_drawdown']:>10.2%}")
        report.append(f"Skewness:            {m['skewness']:>10.2f}")
        report.append(f"Kurtosis:            {m['kurtosis']:>10.2f}")
        
        report.append("\n" + "-" * 40)
        report.append("RISK-ADJUSTED")
        report.append("-" * 40)
        report.append(f"Sharpe Ratio:        {m['sharpe_ratio']:>10.2f}")
        report.append(f"Sortino Ratio:       {m['sortino_ratio']:>10.2f}")
        report.append(f"Calmar Ratio:        {m['calmar_ratio']:>10.2f}")
        report.append(f"Win Rate:            {m['win_rate']:>10.1%}")
        
        # Benchmark comparison
        if 'benchmark_return' in m:
            report.append("\n" + "-" * 40)
            report.append("VS BENCHMARK")
            report.append("-" * 40)
            report.append(f"Benchmark Return:    {m['benchmark_return']:>10.2%}")
            report.append(f"Alpha:               {m['alpha']:>10.2%}")
            report.append(f"Tracking Error:      {m['tracking_error']:>10.2%}")
            report.append(f"Information Ratio:   {m['information_ratio']:>10.2f}")
        
        # Tax summary
        if result.tax_summary:
            report.append("\n" + "-" * 40)
            report.append("TAX IMPACT")
            report.append("-" * 40)
            ts = result.tax_summary
            report.append(f"Realized Gains:      ${ts['realized_gains']:>12,.2f}")
            report.append(f"Realized Losses:     ${ts['realized_losses']:>12,.2f}")
            report.append(f"Net Realized:        ${ts['net_realized']:>12,.2f}")
            report.append(f"Estimated Tax:       ${ts['estimated_tax']:>12,.2f}")
            report.append(f"Tax Drag:            {m.get('tax_drag', 0):>10.2%}")
        
        # Trading activity
        if result.trades_history:
            report.append("\n" + "-" * 40)
            report.append("TRADING ACTIVITY")
            report.append("-" * 40)
            n_trades = len(result.trades_history)
            total_cost = sum(t.get('cost', 0) for t in result.trades_history)
            report.append(f"Total Trades:        {n_trades:>10}")
            report.append(f"Transaction Costs:   ${total_cost:>12,.2f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)


def main():
    """Example backtest."""
    import pickle
    
    # Load data
    try:
        adj_prices = pd.read_csv('data/adjusted_prices.csv',
                                index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Data not found. Run data_download.py first.")
        return
    
    # Simple equal-weight portfolio of top stocks
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
               'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ']
    
    # Filter to available tickers
    available = [t for t in tickers if t in adj_prices.columns]
    weights = {t: 1/len(available) for t in available}
    
    print(f"Backtesting portfolio with {len(available)} stocks")
    
    # Initialize backtester
    backtester = Backtester(
        prices=adj_prices,
        initial_capital=1_000_000,
        transaction_cost=0.001
    )
    
    # Run backtest
    result = backtester.run_backtest(
        initial_weights=weights,
        rebalance_freq='M',
        rebalance_threshold=0.05,
        start_date='2020-01-01',
        tax_aware=True
    )
    
    # Print report
    print(backtester.generate_report(result))
    
    # Monte Carlo simulation
    print("\n" + "=" * 60)
    print("MONTE CARLO SIMULATION (1 Year Forward)")
    print("=" * 60)
    
    mc_results = backtester.run_monte_carlo(
        initial_weights=weights,
        n_simulations=1000,
        n_days=252
    )
    
    print(f"\nExpected Final Value: ${mc_results['mean_final_value']:,.0f}")
    print(f"Median Final Value: ${mc_results['median_final_value']:,.0f}")
    print(f"5th Percentile: ${mc_results['percentile_5']:,.0f}")
    print(f"95th Percentile: ${mc_results['percentile_95']:,.0f}")
    print(f"Probability of Loss: {mc_results['prob_loss']:.1%}")
    print(f"Expected Max Drawdown: {mc_results['mean_max_drawdown']:.1%}")


if __name__ == '__main__':
    main()
