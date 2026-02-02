# 📊 Tax-Optimized Portfolio Construction & Direct Indexing System

<div align="center">

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=for-the-badge&logo=python)
![Dash](https://img.shields.io/badge/Dash-Plotly-green?style=for-the-badge&logo=plotly)
![Cloudflare](https://img.shields.io/badge/Cloudflare-Tunnel-orange?style=for-the-badge&logo=cloudflare)
![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)

**A sophisticated portfolio optimization system with tax-aware strategies, direct indexing, and real-time analytics dashboard**

[Features](#-features) • [Quick Start](#-quick-start) • [Dashboard](#-interactive-dashboard) • [Public Access](#-public-access) • [Documentation](#-documentation)

</div>

---

## 🎯 Overview

This system implements institutional-grade portfolio optimization with a focus on **tax efficiency**. It combines modern portfolio theory with tax-aware strategies to help maximize after-tax returns through:

- **Direct Indexing** - Replicate indices with individual stocks for tax optimization
- **Tax-Loss Harvesting** - Automatically identify and execute loss harvesting opportunities
- **Factor-Based Analysis** - Fama-French multi-factor risk decomposition
- **Real-Time Dashboard** - Professional banking-themed analytics interface with custom background

---

## ✨ Features

### 📈 Portfolio Optimization
| Strategy | Description |
|----------|-------------|
| Mean-Variance | Classic Markowitz optimization with constraints |
| Minimum Variance | Lowest volatility portfolio |
| Maximum Sharpe | Optimal risk-adjusted returns |
| Risk Parity | Equal risk contribution from all assets |
| Direct Indexing | Track index with fewer stocks for tax efficiency |
| Tax-Aware Rebalancing | Minimize tax impact during rebalancing |

### 💰 Tax Optimization
- **Tax-Loss Harvesting** - Identify unrealized losses for harvesting
- **Wash Sale Compliance** - Automatic 30-day wash sale rule checking
- **Tax Lot Accounting** - FIFO, LIFO, and specific lot identification
- **Tax Alpha Calculation** - Measure tax efficiency benefits

### 📊 Analytics & Visualization
- **Interactive Dashboard** - Real-time portfolio analytics
- **Performance Charts** - Portfolio value, returns, drawdowns
- **Risk Metrics** - VaR, Sharpe, Sortino, Max Drawdown
- **Monte Carlo Simulation** - Forward-looking projections
- **Factor Exposure** - Fama-French factor decomposition

### 🎨 Professional Dashboard Design
- Custom banking/finance background image
- Dark theme with blue accent colors
- Animated gradient overlays
- Glass-morphism transparent cards
- Glowing ambient orb effects
- Real-time clock and live indicators
- Scrolling ticker tape with top holdings
- Mobile-responsive design

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9+ (recommended: 3.10 or 3.11)
- pip package manager
- Internet connection

### Installation

```bash
# Navigate to project directory
cd "Tax-Optimized Portfolio Construction & Direct Indexing System"

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run Demo

```bash
# Run complete demo (downloads data, optimizes, backtests)
python main.py demo

# Launch interactive dashboard
python dashboard.py
```

Open http://localhost:8050 in your browser.

---

## 🌐 Public Access

Share your dashboard with anyone using Cloudflare Tunnel:

### Quick Method (Recommended)

```bash
# One-click public access
./start_public.sh
```

### Manual Method

```bash
# Terminal 1: Start dashboard
python dashboard.py --public

# Terminal 2: Create tunnel
cloudflared tunnel --url http://localhost:8050
```

You'll receive a public URL like:
```
https://xxxx-xxxx-xxxx-xxxx.trycloudflare.com
```

**Features:**
- ✅ No password required - direct access
- ✅ Works from any device (laptop, mobile, tablet)
- ✅ HTTPS secured
- ✅ Shareable worldwide

### Access Options

| Type | URL/Command |
|------|-------------|
| **Local** | http://localhost:8050 |
| **Network** | http://YOUR_IP:8050 (with --public flag) |
| **Internet** | `./start_public.sh` or `cloudflared tunnel` |

---

## 📱 Interactive Dashboard

<div align="center">

### 🏦 Professional Banking Theme

</div>

The dashboard features a sophisticated design inspired by institutional trading platforms:

| Component | Description |
|-----------|-------------|
| **Background** | Custom finance/banking image with globe and charts |
| **KPI Cards** | Portfolio value, returns, Sharpe, drawdown, tax savings |
| **Performance Chart** | Interactive portfolio value with 50-day MA |
| **Drawdown Chart** | Underwater analysis showing peak-to-trough |
| **Allocation** | Donut chart + sector bars + holdings table |
| **Tax Harvest** | Opportunities with potential savings |
| **Monte Carlo** | Forward projections with confidence intervals |
| **Live Indicators** | Real-time clock, system status, ticker tape |

### Visual Features
- 🌌 Custom background image with dark overlay
- 💎 Glass-morphism transparent cards
- ✨ Animated gradient effects
- 🔵 Glowing ambient orbs
- 📊 Color-coded accent bars on each card
- 🎯 Smooth hover transitions

---

## 📋 Command Reference

```bash
# Data Operations
python main.py download     # Download market data from Yahoo Finance
python main.py preprocess   # Preprocess data for optimization

# Analysis
python main.py optimize     # Run portfolio optimization
python main.py backtest     # Run historical backtesting
python main.py harvest      # Find tax-loss harvest opportunities

# All-in-One
python main.py demo         # Run complete demo pipeline

# Dashboard
python dashboard.py              # Launch local dashboard
python dashboard.py --public     # Launch with network access
python dashboard.py --port 8051  # Use different port
./start_public.sh                # Launch with public internet access
```

---

## 📁 Project Structure

```
Tax-Optimized Portfolio Construction & Direct Indexing System/
│
├── 📄 main.py                    # CLI entry point
├── 📄 dashboard.py               # Interactive web dashboard
├── 📄 data_download.py           # Data acquisition module
├── 📄 requirements.txt           # Python dependencies
├── 📄 execution.txt              # Detailed execution guide
├── 📄 start_public.sh            # Public access launcher
│
├── 📁 assets/                    # Static assets
│   └── dashboard-bg.png          # Custom background image
│
├── 📁 src/                       # Source modules
│   ├── data_preprocessing.py     # Data cleaning & preparation
│   ├── portfolio_optimizer.py    # Optimization algorithms
│   ├── tax_loss_harvesting.py    # Tax-loss harvesting logic
│   ├── factor_models.py          # Fama-French factor analysis
│   ├── backtester.py             # Historical backtesting
│   └── visualization.py          # Chart generation
│
├── 📁 data/                      # Generated data files
│   ├── sp500_prices.pkl          # Stock price data
│   ├── ff_factors.pkl            # Fama-French factors
│   ├── processed_data.pkl        # Preprocessed data
│   ├── optimization_results.pkl  # Optimization results
│   ├── backtest_results.pkl      # Backtest results
│   └── harvest_opportunities.pkl # Tax harvesting data
│
└── 📁 notebooks/                 # Jupyter notebooks (optional)
```

---

## 📊 Data Sources

All data sources are **free** and publicly available:

| Data | Source | Description |
|------|--------|-------------|
| Stock Prices | Yahoo Finance | Daily OHLCV data via `yfinance` |
| Factor Returns | Kenneth French Library | Fama-French 3, 5, 6-factor + Momentum |
| Risk-Free Rate | FRED | 3-Month Treasury Bill rates |
| S&P 500 Constituents | Wikipedia | Current index components |
| Sector Classification | Yahoo Finance | GICS sector mapping |

---

## 🔧 Usage Examples

### Portfolio Optimization

```python
from src.portfolio_optimizer import TaxOptimizedPortfolio

optimizer = TaxOptimizedPortfolio(
    returns=expected_returns,
    cov_matrix=covariance_matrix,
    risk_free_rate=0.05
)

# Mean-variance optimization
result = optimizer.optimize_mean_variance(max_position=0.05)
print(f"Expected Return: {result['expected_return']:.2%}")
print(f"Sharpe Ratio: {result['sharpe_ratio']:.2f}")

# Direct indexing (track S&P 500 with 50 stocks)
result = optimizer.optimize_direct_indexing(
    benchmark_weights=sp500_weights,
    n_stocks=50,
    tracking_error_limit=0.02
)
```

### Tax-Loss Harvesting

```python
from src.tax_loss_harvesting import TaxLossHarvester

harvester = TaxLossHarvester(
    tax_rate_short_term=0.37,
    tax_rate_long_term=0.238
)

# Add positions
harvester.add_purchase('AAPL', '2023-01-15', 100, 150.00)

# Find opportunities
opportunities = harvester.identify_harvest_opportunities(
    current_prices={'AAPL': 140.00},
    current_date=datetime.now()
)

for opp in opportunities:
    print(f"{opp['ticker']}: ${opp['tax_benefit']:.2f} potential savings")
```

### Backtesting

```python
from src.backtester import Backtester

backtester = Backtester(
    prices=historical_prices,
    initial_capital=1_000_000,
    transaction_cost=0.001
)

result = backtester.run_backtest(
    initial_weights=portfolio_weights,
    rebalance_freq='M'
)

print(f"CAGR: {result.metrics['cagr']:.2%}")
print(f"Sharpe: {result.metrics['sharpe_ratio']:.2f}")
print(f"Max Drawdown: {result.metrics['max_drawdown']:.2%}")

# Monte Carlo simulation
mc = backtester.run_monte_carlo(n_simulations=1000)
print(f"95% VaR: ${mc['percentile_5']:,.0f}")
```

---

## 📈 Performance Metrics

| Category | Metrics |
|----------|---------|
| **Returns** | Total Return, CAGR, Best/Worst Day/Month |
| **Risk** | Volatility, Max Drawdown, VaR, CVaR |
| **Risk-Adjusted** | Sharpe Ratio, Sortino Ratio, Calmar Ratio |
| **Tax** | Tax Alpha, Realized Gains/Losses, Harvest Opportunities |

---

## 💵 Tax Rates (US Default)

| Type | Rate | Holding Period |
|------|------|----------------|
| Short-Term | 37% | < 1 year |
| Long-Term | 23.8% | ≥ 1 year (20% + 3.8% NIIT) |

*Rates are configurable for different tax situations.*

---

## 🛠 Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| numpy/cvxpy conflict | `pip install "numpy>=1.24,<2.0" "cvxpy>=1.4,<1.8" --force-reinstall` |
| No data files | `python main.py download && python main.py preprocess` |
| Dashboard won't start | `pip install dash dash-bootstrap-components plotly` |
| Port 8050 in use | `python dashboard.py --port 8051` |
| Permission denied | `chmod +x start_public.sh` |
| cloudflared not found | `brew install cloudflared` (macOS) |
| Background not showing | Ensure `assets/dashboard-bg.png` exists |

---

## 📚 Documentation

- **[execution.txt](execution.txt)** - Detailed step-by-step execution guide
- **Module Docstrings** - `help(TaxOptimizedPortfolio)` in Python
- **Notebooks** - Example Jupyter notebooks in `notebooks/` folder

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This software is for **educational and research purposes only**. It is not financial advice. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

---

## 🙏 Acknowledgments

- [Kenneth French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) - Factor data
- [Yahoo Finance](https://finance.yahoo.com/) - Market data via `yfinance`
- [Plotly](https://plotly.com/) - Interactive visualizations
- [Cloudflare](https://www.cloudflare.com/) - Public tunnel access

---

<div align="center">

**Built with ❤️ for tax-efficient investing**

[⬆ Back to Top](#-tax-optimized-portfolio-construction--direct-indexing-system)

</div>
