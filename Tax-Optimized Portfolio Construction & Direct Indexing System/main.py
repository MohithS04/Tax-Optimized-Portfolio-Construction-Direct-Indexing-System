#!/usr/bin/env python3
"""
Tax-Optimized Portfolio Construction & Direct Indexing System

Main entry point for running the complete portfolio optimization workflow.

Usage:
    python main.py download    # Download all data
    python main.py preprocess  # Preprocess data
    python main.py optimize    # Run portfolio optimization
    python main.py harvest     # Identify tax-loss harvesting opportunities
    python main.py backtest    # Run backtest simulation
    python main.py full        # Run complete pipeline
    python main.py demo        # Run demonstration with sample data
"""

import argparse
import sys
import os
import pickle
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))


def download_data():
    """Download all required datasets."""
    print("\n" + "=" * 60)
    print("STEP 1: DOWNLOADING DATA")
    print("=" * 60)
    
    from data_download import DataDownloader
    
    downloader = DataDownloader(
        data_dir='data',
        start_date='2014-01-01'
    )
    
    results = downloader.download_all()
    
    return results


def preprocess_data():
    """Preprocess downloaded data for optimization."""
    print("\n" + "=" * 60)
    print("STEP 2: PREPROCESSING DATA")
    print("=" * 60)
    
    from data_preprocessing import DataPreprocessor
    
    preprocessor = DataPreprocessor(data_dir='data')
    preprocessor.load_all_data()
    
    # Prepare and save optimization inputs
    optimization_inputs = preprocessor.save_processed_data()
    
    return optimization_inputs


def run_optimization():
    """Run portfolio optimization."""
    print("\n" + "=" * 60)
    print("STEP 3: PORTFOLIO OPTIMIZATION")
    print("=" * 60)
    
    from portfolio_optimizer import TaxOptimizedPortfolio
    
    # Load processed data
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print("Processed data not found. Running preprocessing first...")
        data = preprocess_data()
    
    # Initialize optimizer
    optimizer = TaxOptimizedPortfolio(
        returns=data['returns'],
        cov_matrix=data['cov_matrix'],
        factor_exposures=data.get('factor_exposures'),
        tax_rate_long=0.238,
        risk_aversion=2.5
    )
    
    # 1. Mean-Variance Optimization
    print("\n--- Mean-Variance Optimization ---")
    mv_result = optimizer.optimize_mean_variance(
        max_position=0.05,
        min_position=0.0
    )
    
    if mv_result['status'] == 'optimal':
        print(optimizer.get_portfolio_summary(mv_result, data.get('sector_mapping')))
    
    # 2. Direct Indexing
    print("\n--- Direct Indexing (100 stocks) ---")
    di_result = optimizer.optimize_direct_indexing(
        benchmark_weights=data['benchmark_weights'].to_dict(),
        n_stocks=100,
        tracking_error_limit=0.015,
        sector_neutral=True,
        sector_mapping=data.get('sector_mapping')
    )
    
    if di_result['status'] == 'optimal':
        print(f"Number of Holdings: {di_result['n_holdings']}")
        print(f"Expected Tracking Error: {di_result.get('tracking_error', 0):.2%}")
    
    # Save results
    results = {
        'mean_variance': mv_result,
        'direct_indexing': di_result,
        'timestamp': datetime.now()
    }
    
    with open('data/optimization_results.pkl', 'wb') as f:
        pickle.dump(results, f)
    
    print("\n✓ Optimization results saved to data/optimization_results.pkl")
    
    return results


def run_tax_harvesting():
    """Identify tax-loss harvesting opportunities."""
    print("\n" + "=" * 60)
    print("STEP 4: TAX-LOSS HARVESTING ANALYSIS")
    print("=" * 60)
    
    from tax_loss_harvesting import TaxLossHarvester
    import pandas as pd
    
    # Load price data
    try:
        adj_prices = pd.read_csv('data/adjusted_prices.csv',
                                index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Price data not found. Please run data download first.")
        return None
    
    # Initialize harvester
    harvester = TaxLossHarvester(
        tax_rate_short_term=0.37,
        tax_rate_long_term=0.238
    )
    
    # Simulate a portfolio purchase from 1 year ago
    purchase_date = datetime.now().replace(year=datetime.now().year - 1)
    
    # Get random sample of stocks for demo
    tickers = adj_prices.columns[:50].tolist()
    
    # Get prices from purchase date
    purchase_idx = adj_prices.index.get_indexer([purchase_date], method='nearest')[0]
    purchase_prices = adj_prices.iloc[purchase_idx]
    
    # Add positions
    for ticker in tickers:
        if ticker in purchase_prices.index and pd.notna(purchase_prices[ticker]):
            # Random position size
            shares = 100
            price = purchase_prices[ticker]
            harvester.add_purchase(ticker, purchase_date, shares, price)
    
    # Get current prices
    current_prices = adj_prices.iloc[-1].to_dict()
    current_date = adj_prices.index[-1]
    
    # Find opportunities
    opportunities = harvester.identify_harvest_opportunities(
        current_prices=current_prices,
        current_date=current_date,
        min_loss_pct=0.05
    )
    
    print(f"\nFound {len(opportunities)} harvesting opportunities")
    
    if opportunities:
        print("\nTop 10 Opportunities:")
        for opp in opportunities[:10]:
            print(f"\n  {opp['ticker']}:")
            print(f"    Loss: ${opp['unrealized_loss']:,.2f} ({opp['unrealized_loss_pct']:.1%})")
            print(f"    Tax Benefit: ${opp['tax_benefit']:,.2f}")
            print(f"    Holding: {opp['holding_days']} days ({'LT' if opp['is_long_term'] else 'ST'})")
    
    # Save opportunities
    with open('data/harvest_opportunities.pkl', 'wb') as f:
        pickle.dump(opportunities, f)
    
    print("\n✓ Harvest opportunities saved to data/harvest_opportunities.pkl")
    
    return opportunities


def run_backtest():
    """Run portfolio backtest."""
    print("\n" + "=" * 60)
    print("STEP 5: PORTFOLIO BACKTESTING")
    print("=" * 60)
    
    from backtester import Backtester
    import pandas as pd
    
    # Load price data
    try:
        adj_prices = pd.read_csv('data/adjusted_prices.csv',
                                index_col=0, parse_dates=True)
    except FileNotFoundError:
        print("Price data not found. Please run data download first.")
        return None
    
    # Load optimization results if available
    try:
        with open('data/optimization_results.pkl', 'rb') as f:
            opt_results = pickle.load(f)
        weights = opt_results['mean_variance']['weights']
    except FileNotFoundError:
        # Default equal-weight portfolio
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 
                  'META', 'TSLA', 'BRK-B', 'UNH', 'JNJ',
                  'JPM', 'V', 'PG', 'HD', 'MA']
        available = [t for t in tickers if t in adj_prices.columns]
        weights = {t: 1/len(available) for t in available}
    
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
    print("\n" + "-" * 40)
    print("MONTE CARLO SIMULATION (1 Year Forward)")
    print("-" * 40)
    
    mc_results = backtester.run_monte_carlo(
        initial_weights=weights,
        n_simulations=1000,
        n_days=252
    )
    
    print(f"\nExpected Final Value: ${mc_results['mean_final_value']:,.0f}")
    print(f"Median Final Value: ${mc_results['median_final_value']:,.0f}")
    print(f"5th Percentile (VaR): ${mc_results['percentile_5']:,.0f}")
    print(f"95th Percentile: ${mc_results['percentile_95']:,.0f}")
    print(f"Probability of Loss: {mc_results['prob_loss']:.1%}")
    
    # Save results
    with open('data/backtest_results.pkl', 'wb') as f:
        pickle.dump({
            'backtest': result,
            'monte_carlo': mc_results
        }, f)
    
    print("\n✓ Backtest results saved to data/backtest_results.pkl")
    
    return result


def run_demo():
    """Run demonstration with sample data."""
    print("\n" + "=" * 60)
    print("TAX-OPTIMIZED PORTFOLIO SYSTEM - DEMO")
    print("=" * 60)
    
    import pandas as pd
    import numpy as np
    
    # Create sample data
    print("\nGenerating sample data...")
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2024-01-01', freq='B')
    n_stocks = 50
    tickers = [f'STOCK_{i:02d}' for i in range(n_stocks)]
    
    # Generate correlated returns
    mean_returns = np.random.uniform(0.0003, 0.0008, n_stocks)
    vols = np.random.uniform(0.015, 0.035, n_stocks)
    
    # Correlation matrix
    corr_matrix = np.eye(n_stocks)
    for i in range(n_stocks):
        for j in range(i+1, n_stocks):
            corr_matrix[i, j] = corr_matrix[j, i] = np.random.uniform(0.2, 0.6)
    
    # Covariance matrix
    cov_matrix = np.outer(vols, vols) * corr_matrix
    
    # Generate returns
    returns = np.random.multivariate_normal(mean_returns, cov_matrix, len(dates))
    returns_df = pd.DataFrame(returns, index=dates, columns=tickers)
    
    # Generate prices
    prices = (1 + returns_df).cumprod() * 100
    
    print(f"Created {n_stocks} stocks with {len(dates)} trading days")
    
    # Portfolio optimization
    print("\n--- Portfolio Optimization ---")
    
    from portfolio_optimizer import TaxOptimizedPortfolio
    
    expected_returns = pd.Series(returns_df.mean() * 252, index=tickers)
    cov_df = pd.DataFrame(returns_df.cov() * 252, index=tickers, columns=tickers)
    
    optimizer = TaxOptimizedPortfolio(
        returns=expected_returns,
        cov_matrix=cov_df,
        risk_aversion=2.5
    )
    
    result = optimizer.optimize_mean_variance(max_position=0.10)
    
    if result['status'] == 'optimal':
        print(f"\nOptimal Portfolio:")
        print(f"  Expected Return: {result['expected_return']:.2%}")
        print(f"  Volatility: {result['volatility']:.2%}")
        print(f"  Sharpe Ratio: {result['sharpe_ratio']:.2f}")
        print(f"  Holdings: {result['n_holdings']}")
    
    # Tax-loss harvesting demo
    print("\n--- Tax-Loss Harvesting Demo ---")
    
    from tax_loss_harvesting import TaxLossHarvester
    
    harvester = TaxLossHarvester()
    
    # Add positions (simulate buying 1 year ago)
    purchase_date = dates[0]
    purchase_prices = prices.iloc[0]
    
    for ticker in tickers[:20]:
        harvester.add_purchase(ticker, purchase_date, 100, purchase_prices[ticker])
    
    # Find opportunities
    current_prices = prices.iloc[-1].to_dict()
    opportunities = harvester.identify_harvest_opportunities(
        current_prices, dates[-1], min_loss_pct=0.03
    )
    
    print(f"\nFound {len(opportunities)} harvest opportunities")
    if opportunities:
        total_benefit = sum(o['tax_benefit'] for o in opportunities)
        print(f"Total Potential Tax Benefit: ${total_benefit:,.2f}")
    
    # Backtest demo
    print("\n--- Backtest Demo ---")
    
    from backtester import Backtester
    
    backtester = Backtester(prices=prices, initial_capital=1_000_000)
    
    # Equal weight the top holdings
    top_holdings = sorted(
        [t for t in result['weights'].keys() if result['weights'][t] > 0.01],
        key=lambda t: result['weights'][t],
        reverse=True
    )[:20]
    weights = {t: 1/len(top_holdings) for t in top_holdings}
    
    bt_result = backtester.run_backtest(
        initial_weights=weights,
        rebalance_freq='Q',
        tax_aware=True
    )
    
    print(f"\nBacktest Results (2020-2024):")
    print(f"  Total Return: {bt_result.metrics['total_return']:.2%}")
    print(f"  CAGR: {bt_result.metrics['cagr']:.2%}")
    print(f"  Volatility: {bt_result.metrics['annual_volatility']:.2%}")
    print(f"  Sharpe Ratio: {bt_result.metrics['sharpe_ratio']:.2f}")
    print(f"  Max Drawdown: {bt_result.metrics['max_drawdown']:.2%}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETE")
    print("=" * 60)
    print("\nTo run with real data:")
    print("  1. python main.py download")
    print("  2. python main.py full")


def run_full_pipeline():
    """Run the complete pipeline."""
    print("\n" + "=" * 60)
    print("RUNNING COMPLETE PIPELINE")
    print("=" * 60)
    
    # Step 1: Download data
    download_data()
    
    # Step 2: Preprocess
    preprocess_data()
    
    # Step 3: Optimize
    run_optimization()
    
    # Step 4: Tax harvesting
    run_tax_harvesting()
    
    # Step 5: Backtest
    run_backtest()
    
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE!")
    print("=" * 60)
    print("\nResults saved in 'data/' directory:")
    print("  - processed_data.pkl")
    print("  - optimization_results.pkl")
    print("  - harvest_opportunities.pkl")
    print("  - backtest_results.pkl")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Tax-Optimized Portfolio Construction System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download     Download all market data
  python main.py preprocess   Preprocess data for optimization
  python main.py optimize     Run portfolio optimization
  python main.py harvest      Analyze tax-loss harvesting
  python main.py backtest     Run backtesting simulation
  python main.py full         Run complete pipeline
  python main.py demo         Run demo with sample data
        """
    )
    
    parser.add_argument(
        'command',
        choices=['download', 'preprocess', 'optimize', 'harvest', 'backtest', 'full', 'demo'],
        help='Command to run'
    )
    
    args = parser.parse_args()
    
    commands = {
        'download': download_data,
        'preprocess': preprocess_data,
        'optimize': run_optimization,
        'harvest': run_tax_harvesting,
        'backtest': run_backtest,
        'full': run_full_pipeline,
        'demo': run_demo
    }
    
    try:
        commands[args.command]()
    except KeyboardInterrupt:
        print("\n\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
