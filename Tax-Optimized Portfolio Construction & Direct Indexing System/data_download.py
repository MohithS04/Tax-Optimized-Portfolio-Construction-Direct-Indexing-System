"""
Complete Data Download Script for Tax-Optimized Portfolio Project
Downloads all necessary datasets from free sources

Usage:
    python data_download.py

This script downloads:
    1. S&P 500 constituent list from Wikipedia
    2. Historical stock prices (10 years) from Yahoo Finance
    3. Fama-French factor data from Kenneth French's website
    4. Risk-free rate from FRED
    5. Company fundamentals from Yahoo Finance
    6. Benchmark indices (S&P 500, Dow Jones, NASDAQ)
"""

import yfinance as yf
import pandas as pd
import pandas_datareader as pdr
from datetime import datetime, timedelta
import os
import time
import warnings
from tqdm import tqdm

warnings.filterwarnings('ignore')


class DataDownloader:
    """Handles downloading all required datasets for the portfolio system."""
    
    def __init__(self, data_dir='data', start_date='2014-01-01', end_date=None):
        """
        Initialize the data downloader.
        
        Parameters:
        -----------
        data_dir : str
            Directory to save downloaded data
        start_date : str
            Start date for historical data (YYYY-MM-DD)
        end_date : str or None
            End date for historical data (defaults to today)
        """
        self.data_dir = data_dir
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        
        # Create data directory
        os.makedirs(data_dir, exist_ok=True)
        
        # Store tickers
        self.tickers = []
        self.sp500_table = None
        
    def download_sp500_list(self):
        """Download S&P 500 constituent list from Wikipedia."""
        print("\n[1/7] Downloading S&P 500 constituent list...")
        
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        
        try:
            tables = pd.read_html(url)
            self.sp500_table = tables[0]
            
            # Clean ticker symbols (replace . with - for Yahoo Finance)
            self.tickers = self.sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
            
            # Save constituent list
            self.sp500_table.to_csv(f'{self.data_dir}/sp500_constituents.csv', index=False)
            
            print(f"✓ Found {len(self.tickers)} S&P 500 stocks")
            return self.tickers
            
        except Exception as e:
            print(f"✗ Error downloading S&P 500 list: {e}")
            # Fallback to a predefined list of major stocks
            self.tickers = self._get_fallback_tickers()
            return self.tickers
    
    def _get_fallback_tickers(self):
        """Return fallback list of major stocks if Wikipedia fails."""
        return [
            'AAPL', 'MSFT', 'AMZN', 'NVDA', 'GOOGL', 'META', 'TSLA', 'BRK-B',
            'UNH', 'XOM', 'JNJ', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
            'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'WMT', 'MCD', 'CSCO',
            'TMO', 'ACN', 'ABT', 'DHR', 'NEE', 'VZ', 'ADBE', 'CRM', 'NKE',
            'CMCSA', 'TXN', 'PM', 'UPS', 'RTX', 'BMY', 'ORCL', 'HON', 'QCOM',
            'T', 'MS', 'AMGN', 'IBM', 'GS', 'CAT', 'LOW', 'BA', 'SPGI', 'BLK'
        ]
    
    def download_stock_prices(self):
        """Download historical stock prices from Yahoo Finance."""
        print(f"\n[2/7] Downloading historical stock prices ({self.start_date} to {self.end_date})...")
        print("This may take 5-10 minutes...")
        
        if not self.tickers:
            self.download_sp500_list()
        
        try:
            # Download all stock data
            stock_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                group_by='ticker',
                auto_adjust=False,
                threads=True,
                progress=True
            )
            
            # Save raw data
            stock_data.to_pickle(f'{self.data_dir}/stock_prices_raw.pkl')
            print("✓ Stock price data saved (raw)")
            
            return stock_data
            
        except Exception as e:
            print(f"✗ Error downloading stock prices: {e}")
            return None
    
    def download_adjusted_prices(self):
        """Download adjusted close prices for returns calculation."""
        print("\n[3/7] Downloading adjusted prices...")
        
        if not self.tickers:
            self.download_sp500_list()
        
        try:
            adj_data = yf.download(
                self.tickers,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )
            
            # Extract Close prices (which are now adjusted)
            if 'Close' in adj_data.columns.get_level_values(0):
                adj_close = adj_data['Close']
            else:
                adj_close = adj_data
            
            adj_close.to_csv(f'{self.data_dir}/adjusted_prices.csv')
            print("✓ Adjusted prices saved")
            
            return adj_close
            
        except Exception as e:
            print(f"✗ Error downloading adjusted prices: {e}")
            return None
    
    def download_fama_french_factors(self):
        """Download Fama-French factor data."""
        print("\n[4/7] Downloading Fama-French factor data...")
        
        try:
            # Download 5 Factors (daily)
            ff5_daily = pdr.DataReader(
                'F-F_Research_Data_5_Factors_2x3_daily',
                'famafrench',
                start=self.start_date
            )[0]
            
            # Download Momentum Factor
            mom_daily = pdr.DataReader(
                'F-F_Momentum_Factor_daily',
                'famafrench',
                start=self.start_date
            )[0]
            
            # Combine factors
            factors = pd.concat([ff5_daily, mom_daily], axis=1)
            factors.index = pd.to_datetime(factors.index, format='%Y%m%d')
            factors = factors / 100  # Convert to decimal returns
            
            # Save factors
            factors.to_csv(f'{self.data_dir}/fama_french_factors.csv')
            print("✓ Fama-French factors saved")
            
            # Download industry portfolios
            try:
                industries_10 = pdr.DataReader(
                    '10_Industry_Portfolios_daily',
                    'famafrench',
                    start=self.start_date
                )[0]
                industries_10.index = pd.to_datetime(industries_10.index, format='%Y%m%d')
                industries_10 = industries_10 / 100
                industries_10.to_csv(f'{self.data_dir}/industry_returns.csv')
                print("✓ Industry portfolio returns saved")
            except Exception as e:
                print(f"⚠ Could not download industry returns: {e}")
            
            return factors
            
        except Exception as e:
            print(f"✗ Error downloading factor data: {e}")
            print("  You may need to install pandas-datareader: pip install pandas-datareader")
            return None
    
    def download_risk_free_rate(self):
        """Download risk-free rate from FRED."""
        print("\n[5/7] Downloading risk-free rate (3-Month T-Bill)...")
        
        try:
            rf_rate = pdr.DataReader('DGS3MO', 'fred', start=self.start_date)
            rf_rate.columns = ['RF_Annual_Pct']
            
            # Convert annual percentage to daily decimal
            rf_rate['RF_Daily'] = rf_rate['RF_Annual_Pct'] / 100 / 252
            
            rf_rate.to_csv(f'{self.data_dir}/risk_free_rate.csv')
            print("✓ Risk-free rate saved")
            
            return rf_rate
            
        except Exception as e:
            print(f"✗ Error downloading risk-free rate: {e}")
            return None
    
    def download_company_info(self):
        """Download company fundamentals from Yahoo Finance."""
        print("\n[6/7] Downloading company fundamentals...")
        print("This may take 10-15 minutes...")
        
        if not self.tickers:
            self.download_sp500_list()
        
        company_data = []
        failed_tickers = []
        
        for ticker in tqdm(self.tickers, desc="Downloading company info"):
            try:
                stock = yf.Ticker(ticker)
                info = stock.info
                
                company_data.append({
                    'ticker': ticker,
                    'name': info.get('longName', info.get('shortName', '')),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'marketCap': info.get('marketCap', 0),
                    'beta': info.get('beta', 1.0),
                    'dividendYield': info.get('dividendYield', 0),
                    'trailingPE': info.get('trailingPE', None),
                    'forwardPE': info.get('forwardPE', None),
                    'priceToBook': info.get('priceToBook', None),
                    'profitMargins': info.get('profitMargins', None),
                    'returnOnEquity': info.get('returnOnEquity', None)
                })
                
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                failed_tickers.append(ticker)
                continue
        
        company_info = pd.DataFrame(company_data)
        company_info.to_csv(f'{self.data_dir}/company_info.csv', index=False)
        
        print(f"✓ Company info saved ({len(company_info)} stocks)")
        if failed_tickers:
            print(f"  ⚠ Failed to download info for {len(failed_tickers)} stocks")
        
        return company_info
    
    def download_benchmark_indices(self):
        """Download benchmark index data."""
        print("\n[7/7] Downloading benchmark indices...")
        
        benchmarks = ['^GSPC', '^DJI', '^IXIC', '^RUT']  # S&P 500, Dow, NASDAQ, Russell 2000
        
        try:
            benchmark_data = yf.download(
                benchmarks,
                start=self.start_date,
                end=self.end_date,
                auto_adjust=True,
                progress=False
            )
            
            benchmark_data.to_csv(f'{self.data_dir}/benchmark_indices.csv')
            print("✓ Benchmark data saved")
            
            return benchmark_data
            
        except Exception as e:
            print(f"✗ Error downloading benchmark indices: {e}")
            return None
    
    def download_all(self):
        """Download all datasets."""
        print("=" * 60)
        print("TAX-OPTIMIZED PORTFOLIO DATA DOWNLOAD")
        print("=" * 60)
        
        results = {}
        
        # Download all data
        results['tickers'] = self.download_sp500_list()
        results['stock_prices'] = self.download_stock_prices()
        results['adjusted_prices'] = self.download_adjusted_prices()
        results['factors'] = self.download_fama_french_factors()
        results['rf_rate'] = self.download_risk_free_rate()
        results['company_info'] = self.download_company_info()
        results['benchmarks'] = self.download_benchmark_indices()
        
        # Print summary
        self._print_summary()
        
        return results
    
    def _print_summary(self):
        """Print download summary."""
        print("\n" + "=" * 60)
        print("DOWNLOAD COMPLETE!")
        print("=" * 60)
        print(f"\nFiles created in '{self.data_dir}/' directory:")
        
        files = [
            ('sp500_constituents.csv', 'Current S&P 500 list'),
            ('stock_prices_raw.pkl', 'Raw OHLCV data (all stocks)'),
            ('adjusted_prices.csv', 'Adjusted close prices'),
            ('fama_french_factors.csv', '6 factor returns (daily)'),
            ('industry_returns.csv', '10 industry portfolio returns'),
            ('risk_free_rate.csv', '3-month T-Bill rates'),
            ('company_info.csv', 'Fundamentals & sector data'),
            ('benchmark_indices.csv', 'S&P 500, Dow, NASDAQ, Russell')
        ]
        
        for filename, description in files:
            filepath = f'{self.data_dir}/{filename}'
            status = "✓" if os.path.exists(filepath) else "✗"
            print(f"  {status} {filename:30} - {description}")
        
        print("\n" + "=" * 60)
        print("You can now proceed with data preprocessing and modeling!")
        print("=" * 60)


def main():
    """Main entry point for data download."""
    downloader = DataDownloader(
        data_dir='data',
        start_date='2014-01-01'
    )
    
    downloader.download_all()


if __name__ == '__main__':
    main()
