"""
Visualization Module

This module provides:
    - Portfolio performance charts
    - Risk analytics visualizations
    - Factor exposure plots
    - Tax harvesting dashboards
    - Interactive reports
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import warnings

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.ticker import PercentFormatter
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    warnings.warn("Matplotlib not available")

try:
    import seaborn as sns
    SEABORN_AVAILABLE = True
except ImportError:
    SEABORN_AVAILABLE = False

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = None  # Placeholder for type hints
    warnings.warn("Plotly not available - interactive charts disabled")


class PortfolioVisualizer:
    """
    Visualization suite for portfolio analytics.
    
    Supports both static (matplotlib) and interactive (plotly) charts.
    """
    
    def __init__(self, style: str = 'seaborn'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        style : str
            Matplotlib style ('seaborn', 'ggplot', 'default')
        """
        if MATPLOTLIB_AVAILABLE:
            plt.style.use(style if style != 'seaborn' else 'seaborn-v0_8-whitegrid')
        
        # Color palettes
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'positive': '#2ca02c',
            'negative': '#d62728',
            'neutral': '#7f7f7f',
            'palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', 
                       '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        }
    
    def plot_portfolio_performance(self,
                                   portfolio_values: pd.Series,
                                   benchmark_values: Optional[pd.Series] = None,
                                   title: str = "Portfolio Performance",
                                   interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot portfolio value over time.
        
        Parameters:
        -----------
        portfolio_values : pd.Series
            Portfolio values over time
        benchmark_values : pd.Series, optional
            Benchmark values for comparison
        title : str
            Chart title
        interactive : bool
            Use Plotly for interactive chart
            
        Returns:
        --------
        go.Figure or None : Plotly figure if interactive
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            # Portfolio line
            fig.add_trace(go.Scatter(
                x=portfolio_values.index,
                y=portfolio_values.values,
                mode='lines',
                name='Portfolio',
                line=dict(color=self.colors['primary'], width=2)
            ))
            
            # Benchmark line
            if benchmark_values is not None:
                fig.add_trace(go.Scatter(
                    x=benchmark_values.index,
                    y=benchmark_values.values,
                    mode='lines',
                    name='Benchmark',
                    line=dict(color=self.colors['secondary'], width=2, dash='dash')
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Value ($)',
                hovermode='x unified',
                template='plotly_white',
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
            )
            
            fig.update_yaxes(tickformat='$,.0f')
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            ax.plot(portfolio_values.index, portfolio_values.values,
                   label='Portfolio', color=self.colors['primary'], linewidth=2)
            
            if benchmark_values is not None:
                ax.plot(benchmark_values.index, benchmark_values.values,
                       label='Benchmark', color=self.colors['secondary'], 
                       linewidth=2, linestyle='--')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Value ($)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format y-axis as currency
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
            
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_returns_distribution(self,
                                  returns: pd.Series,
                                  title: str = "Returns Distribution",
                                  interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot histogram of returns with statistics.
        """
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns.values,
                nbinsx=50,
                name='Daily Returns',
                marker_color=self.colors['primary'],
                opacity=0.75
            ))
            
            # Add vertical lines for statistics
            mean_ret = returns.mean()
            median_ret = returns.median()
            
            fig.add_vline(x=mean_ret, line_dash="dash", 
                         line_color=self.colors['secondary'],
                         annotation_text=f"Mean: {mean_ret:.2%}")
            fig.add_vline(x=0, line_dash="solid", line_color="black", line_width=1)
            
            fig.update_layout(
                title=title,
                xaxis_title='Daily Return',
                yaxis_title='Frequency',
                template='plotly_white',
                showlegend=False
            )
            
            fig.update_xaxes(tickformat='.1%')
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            ax.hist(returns.values, bins=50, color=self.colors['primary'], 
                   alpha=0.75, edgecolor='white')
            
            # Statistics
            mean_ret = returns.mean()
            std_ret = returns.std()
            
            ax.axvline(mean_ret, color=self.colors['secondary'], 
                      linestyle='--', label=f'Mean: {mean_ret:.2%}')
            ax.axvline(0, color='black', linestyle='-', linewidth=1)
            
            # Add text box with stats
            stats_text = f'Mean: {mean_ret:.2%}\nStd: {std_ret:.2%}\nSkew: {returns.skew():.2f}\nKurt: {returns.kurtosis():.2f}'
            ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Daily Return')
            ax.set_ylabel('Frequency')
            ax.xaxis.set_major_formatter(PercentFormatter(1))
            ax.legend()
            
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_drawdown(self,
                     portfolio_values: pd.Series,
                     title: str = "Portfolio Drawdown",
                     interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot underwater chart (drawdown over time).
        """
        # Calculate drawdown
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=drawdown.index,
                y=drawdown.values,
                fill='tozeroy',
                fillcolor='rgba(214, 39, 40, 0.3)',
                line=dict(color=self.colors['negative'], width=1),
                name='Drawdown'
            ))
            
            # Highlight max drawdown
            max_dd_idx = drawdown.idxmin()
            max_dd_val = drawdown.min()
            
            fig.add_annotation(
                x=max_dd_idx,
                y=max_dd_val,
                text=f"Max DD: {max_dd_val:.1%}",
                showarrow=True,
                arrowhead=2
            )
            
            fig.update_layout(
                title=title,
                xaxis_title='Date',
                yaxis_title='Drawdown',
                template='plotly_white'
            )
            
            fig.update_yaxes(tickformat='.0%')
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 5))
            
            ax.fill_between(drawdown.index, drawdown.values, 0,
                           color=self.colors['negative'], alpha=0.3)
            ax.plot(drawdown.index, drawdown.values, 
                   color=self.colors['negative'], linewidth=1)
            
            # Mark max drawdown
            max_dd_idx = drawdown.idxmin()
            max_dd_val = drawdown.min()
            ax.annotate(f'Max DD: {max_dd_val:.1%}',
                       xy=(max_dd_idx, max_dd_val),
                       xytext=(10, 30), textcoords='offset points',
                       arrowprops=dict(arrowstyle='->', color='black'),
                       fontsize=10)
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Date')
            ax.set_ylabel('Drawdown')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_ylim(drawdown.min() * 1.1, 0.05)
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_weights_allocation(self,
                               weights: Dict[str, float],
                               title: str = "Portfolio Allocation",
                               top_n: int = 15,
                               interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot portfolio weights as pie/bar chart.
        """
        # Sort and limit to top N
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        
        if len(sorted_weights) > top_n:
            top_weights = dict(sorted_weights[:top_n])
            other_weight = sum(w for _, w in sorted_weights[top_n:])
            top_weights['Other'] = other_weight
        else:
            top_weights = dict(sorted_weights)
        
        labels = list(top_weights.keys())
        values = list(top_weights.values())
        
        if interactive and PLOTLY_AVAILABLE:
            fig = make_subplots(rows=1, cols=2, 
                               specs=[[{'type': 'pie'}, {'type': 'bar'}]],
                               subplot_titles=('Allocation', 'Weights'))
            
            # Pie chart
            fig.add_trace(
                go.Pie(labels=labels, values=values, hole=0.4,
                      textinfo='label+percent',
                      textposition='outside'),
                row=1, col=1
            )
            
            # Bar chart
            fig.add_trace(
                go.Bar(x=labels, y=values, marker_color=self.colors['primary']),
                row=1, col=2
            )
            
            fig.update_layout(
                title_text=title,
                template='plotly_white',
                showlegend=False
            )
            
            fig.update_yaxes(tickformat='.1%', row=1, col=2)
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            
            # Pie chart
            ax1.pie(values, labels=labels, autopct='%1.1f%%',
                   colors=self.colors['palette'][:len(values)])
            ax1.set_title('Allocation', fontsize=12)
            
            # Bar chart
            bars = ax2.barh(labels, values, color=self.colors['primary'])
            ax2.set_xlabel('Weight')
            ax2.set_title('Weights', fontsize=12)
            ax2.xaxis.set_major_formatter(PercentFormatter(1))
            
            # Add value labels
            for bar, val in zip(bars, values):
                ax2.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                        f'{val:.1%}', va='center', fontsize=9)
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_sector_allocation(self,
                              weights: Dict[str, float],
                              sector_mapping: Dict[str, str],
                              title: str = "Sector Allocation",
                              interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot allocation by sector.
        """
        # Aggregate by sector
        sector_weights = {}
        for ticker, weight in weights.items():
            sector = sector_mapping.get(ticker, 'Unknown')
            sector_weights[sector] = sector_weights.get(sector, 0) + weight
        
        # Sort by weight
        sector_weights = dict(sorted(sector_weights.items(), 
                                    key=lambda x: x[1], reverse=True))
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=list(sector_weights.keys()),
                    y=list(sector_weights.values()),
                    marker_color=self.colors['palette'][:len(sector_weights)],
                    text=[f'{v:.1%}' for v in sector_weights.values()],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title='Sector',
                yaxis_title='Weight',
                template='plotly_white'
            )
            
            fig.update_yaxes(tickformat='.0%')
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            sectors = list(sector_weights.keys())
            weights_vals = list(sector_weights.values())
            
            bars = ax.bar(sectors, weights_vals, 
                         color=self.colors['palette'][:len(sectors)])
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Sector')
            ax.set_ylabel('Weight')
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            
            # Rotate labels
            plt.xticks(rotation=45, ha='right')
            
            # Add value labels
            for bar, val in zip(bars, weights_vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                       f'{val:.1%}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_factor_exposures(self,
                             exposures: pd.Series,
                             title: str = "Portfolio Factor Exposures",
                             interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot factor exposures as bar chart.
        """
        factors = exposures.index.tolist()
        values = exposures.values
        
        # Color based on positive/negative
        colors = [self.colors['positive'] if v >= 0 else self.colors['negative'] 
                 for v in values]
        
        if interactive and PLOTLY_AVAILABLE:
            fig = go.Figure(data=[
                go.Bar(
                    x=factors,
                    y=values,
                    marker_color=colors,
                    text=[f'{v:.2f}' for v in values],
                    textposition='outside'
                )
            ])
            
            fig.add_hline(y=0, line_dash="solid", line_color="black", line_width=1)
            fig.add_hline(y=1, line_dash="dash", line_color="gray", line_width=1,
                         annotation_text="Market (1.0)")
            
            fig.update_layout(
                title=title,
                xaxis_title='Factor',
                yaxis_title='Exposure (Beta)',
                template='plotly_white'
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, ax = plt.subplots(figsize=(10, 6))
            
            bars = ax.bar(factors, values, color=colors)
            
            ax.axhline(y=0, color='black', linewidth=1)
            ax.axhline(y=1, color='gray', linewidth=1, linestyle='--', alpha=0.5)
            ax.text(len(factors)-0.5, 1.05, 'Market (1.0)', fontsize=9, color='gray')
            
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.set_xlabel('Factor')
            ax.set_ylabel('Exposure (Beta)')
            
            # Add value labels
            for bar, val in zip(bars, values):
                y_pos = val + 0.05 if val >= 0 else val - 0.1
                ax.text(bar.get_x() + bar.get_width()/2, y_pos,
                       f'{val:.2f}', ha='center', fontsize=9)
            
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_tax_harvest_opportunities(self,
                                       opportunities: List[Dict],
                                       title: str = "Tax-Loss Harvest Opportunities",
                                       interactive: bool = False) -> Optional[go.Figure]:
        """
        Visualize tax-loss harvesting opportunities.
        """
        if not opportunities:
            print("No harvest opportunities to display")
            return None
        
        df = pd.DataFrame(opportunities)
        
        if interactive and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Tax Benefit by Stock', 'Loss % Distribution',
                               'Holding Period', 'Cumulative Tax Benefit'),
                specs=[[{'type': 'bar'}, {'type': 'histogram'}],
                      [{'type': 'scatter'}, {'type': 'scatter'}]]
            )
            
            # Tax benefit by stock
            fig.add_trace(
                go.Bar(x=df['ticker'], y=df['tax_benefit'],
                      marker_color=self.colors['primary'],
                      name='Tax Benefit'),
                row=1, col=1
            )
            
            # Loss distribution
            fig.add_trace(
                go.Histogram(x=df['unrealized_loss_pct'], nbinsx=20,
                           marker_color=self.colors['negative'],
                           name='Loss %'),
                row=1, col=2
            )
            
            # Holding period scatter
            fig.add_trace(
                go.Scatter(x=df['holding_days'], y=df['tax_benefit'],
                          mode='markers',
                          marker=dict(
                              size=np.abs(df['unrealized_loss']) / 100,
                              color=df['is_long_term'].map({True: self.colors['positive'],
                                                            False: self.colors['secondary']}),
                              sizemin=5
                          ),
                          text=df['ticker'],
                          name='Opportunities'),
                row=2, col=1
            )
            
            # Cumulative benefit
            df_sorted = df.sort_values('tax_benefit', ascending=False)
            df_sorted['cumulative_benefit'] = df_sorted['tax_benefit'].cumsum()
            
            fig.add_trace(
                go.Scatter(x=list(range(1, len(df_sorted)+1)),
                          y=df_sorted['cumulative_benefit'],
                          mode='lines+markers',
                          marker_color=self.colors['positive'],
                          name='Cumulative'),
                row=2, col=2
            )
            
            fig.update_layout(
                title_text=title,
                template='plotly_white',
                showlegend=False,
                height=600
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Tax benefit by stock
            top_10 = df.nlargest(10, 'tax_benefit')
            axes[0, 0].barh(top_10['ticker'], top_10['tax_benefit'],
                          color=self.colors['primary'])
            axes[0, 0].set_xlabel('Tax Benefit ($)')
            axes[0, 0].set_title('Top 10 Harvest Opportunities')
            
            # Loss distribution
            axes[0, 1].hist(df['unrealized_loss_pct'] * 100, bins=20,
                          color=self.colors['negative'], alpha=0.75)
            axes[0, 1].set_xlabel('Loss (%)')
            axes[0, 1].set_ylabel('Count')
            axes[0, 1].set_title('Loss Distribution')
            
            # Holding period scatter
            colors = [self.colors['positive'] if lt else self.colors['secondary'] 
                     for lt in df['is_long_term']]
            sizes = np.abs(df['unrealized_loss']) / 50
            axes[1, 0].scatter(df['holding_days'], df['tax_benefit'],
                             c=colors, s=sizes, alpha=0.6)
            axes[1, 0].set_xlabel('Holding Days')
            axes[1, 0].set_ylabel('Tax Benefit ($)')
            axes[1, 0].set_title('Benefit vs Holding Period')
            axes[1, 0].axvline(x=365, color='gray', linestyle='--', alpha=0.5)
            
            # Cumulative benefit
            df_sorted = df.sort_values('tax_benefit', ascending=False)
            cumulative = df_sorted['tax_benefit'].cumsum()
            axes[1, 1].plot(range(1, len(cumulative)+1), cumulative,
                          color=self.colors['positive'], marker='o', markersize=3)
            axes[1, 1].set_xlabel('Number of Harvests')
            axes[1, 1].set_ylabel('Cumulative Tax Benefit ($)')
            axes[1, 1].set_title('Cumulative Benefit')
            
            plt.suptitle(title, fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            return None
    
    def plot_rolling_metrics(self,
                            returns: pd.Series,
                            window: int = 252,
                            metrics: List[str] = ['volatility', 'sharpe'],
                            interactive: bool = False) -> Optional[go.Figure]:
        """
        Plot rolling performance metrics.
        """
        rolling_data = {}
        
        if 'volatility' in metrics:
            rolling_data['Volatility'] = returns.rolling(window).std() * np.sqrt(252)
        
        if 'sharpe' in metrics:
            rolling_mean = returns.rolling(window).mean() * 252
            rolling_std = returns.rolling(window).std() * np.sqrt(252)
            rolling_data['Sharpe'] = rolling_mean / rolling_std
        
        if 'return' in metrics:
            rolling_data['Return'] = returns.rolling(window).mean() * 252
        
        df = pd.DataFrame(rolling_data)
        
        if interactive and PLOTLY_AVAILABLE:
            fig = make_subplots(
                rows=len(metrics), cols=1,
                shared_xaxes=True,
                subplot_titles=[m.title() for m in metrics]
            )
            
            for i, (name, data) in enumerate(df.items(), 1):
                fig.add_trace(
                    go.Scatter(x=data.index, y=data.values,
                              mode='lines', name=name,
                              line=dict(color=self.colors['palette'][i-1])),
                    row=i, col=1
                )
            
            fig.update_layout(
                title=f"Rolling {window}-Day Metrics",
                template='plotly_white',
                height=200 * len(metrics) + 100
            )
            
            return fig
            
        elif MATPLOTLIB_AVAILABLE:
            fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 4*len(metrics)),
                                    sharex=True)
            
            if len(metrics) == 1:
                axes = [axes]
            
            for ax, (name, data) in zip(axes, df.items()):
                ax.plot(data.index, data.values, 
                       color=self.colors['primary'], linewidth=1)
                ax.set_ylabel(name)
                ax.set_title(f'Rolling {window}-Day {name}')
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Date')
            
            plt.suptitle(f'Rolling {window}-Day Metrics', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.show()
            
            return None
    
    def create_dashboard(self,
                        portfolio_values: pd.Series,
                        returns: pd.Series,
                        weights: Dict[str, float],
                        metrics: Dict,
                        benchmark_values: Optional[pd.Series] = None,
                        sector_mapping: Optional[Dict] = None) -> Optional[go.Figure]:
        """
        Create comprehensive interactive dashboard.
        """
        if not PLOTLY_AVAILABLE:
            print("Dashboard requires Plotly. Install with: pip install plotly")
            return None
        
        fig = make_subplots(
            rows=3, cols=2,
            specs=[[{'colspan': 2}, None],
                   [{'type': 'domain'}, {'type': 'bar'}],
                   [{'colspan': 2}, None]],
            subplot_titles=('Portfolio Performance', 'Allocation', 'Top Holdings', 'Drawdown'),
            row_heights=[0.4, 0.3, 0.3],
            vertical_spacing=0.1
        )
        
        # Performance chart
        fig.add_trace(
            go.Scatter(x=portfolio_values.index, y=portfolio_values.values,
                      mode='lines', name='Portfolio',
                      line=dict(color=self.colors['primary'], width=2)),
            row=1, col=1
        )
        
        if benchmark_values is not None:
            fig.add_trace(
                go.Scatter(x=benchmark_values.index, y=benchmark_values.values,
                          mode='lines', name='Benchmark',
                          line=dict(color=self.colors['secondary'], width=2, dash='dash')),
                row=1, col=1
            )
        
        # Pie chart for allocation
        sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
        other = sum(w for t, w in weights.items() if t not in dict(sorted_weights))
        labels = [t for t, w in sorted_weights] + (['Other'] if other > 0 else [])
        values = [w for t, w in sorted_weights] + ([other] if other > 0 else [])
        
        fig.add_trace(
            go.Pie(labels=labels, values=values, hole=0.4,
                  textinfo='label+percent', textposition='outside',
                  marker_colors=self.colors['palette']),
            row=2, col=1
        )
        
        # Top holdings bar
        top_10 = sorted_weights[:10]
        fig.add_trace(
            go.Bar(x=[t for t, w in top_10], y=[w for t, w in top_10],
                  marker_color=self.colors['primary'],
                  text=[f'{w:.1%}' for t, w in top_10],
                  textposition='outside'),
            row=2, col=2
        )
        
        # Drawdown
        rolling_max = portfolio_values.cummax()
        drawdown = (portfolio_values - rolling_max) / rolling_max
        
        fig.add_trace(
            go.Scatter(x=drawdown.index, y=drawdown.values,
                      fill='tozeroy', mode='lines',
                      fillcolor='rgba(214, 39, 40, 0.3)',
                      line=dict(color=self.colors['negative'], width=1),
                      name='Drawdown'),
            row=3, col=1
        )
        
        # Add metrics annotation
        metrics_text = (
            f"Total Return: {metrics.get('total_return', 0):.1%}<br>"
            f"CAGR: {metrics.get('cagr', 0):.1%}<br>"
            f"Volatility: {metrics.get('annual_volatility', 0):.1%}<br>"
            f"Sharpe: {metrics.get('sharpe_ratio', 0):.2f}<br>"
            f"Max DD: {metrics.get('max_drawdown', 0):.1%}"
        )
        
        fig.add_annotation(
            xref="paper", yref="paper",
            x=0.98, y=0.98,
            text=metrics_text,
            showarrow=False,
            font=dict(size=11),
            align="right",
            bgcolor="white",
            bordercolor="gray",
            borderwidth=1
        )
        
        fig.update_layout(
            title=dict(text="Portfolio Dashboard", font=dict(size=20)),
            template='plotly_white',
            height=900,
            showlegend=True,
            legend=dict(yanchor="top", y=0.99, xanchor="right", x=0.99)
        )
        
        return fig
    
    def save_report(self,
                   figures: List[go.Figure],
                   filename: str = "portfolio_report.html"):
        """
        Save multiple figures as HTML report.
        """
        if not PLOTLY_AVAILABLE:
            print("HTML export requires Plotly")
            return
        
        from plotly.io import write_html
        
        html_content = """
        <html>
        <head>
            <title>Portfolio Analytics Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #333; }
                .chart { margin: 20px 0; border: 1px solid #ddd; padding: 10px; }
            </style>
        </head>
        <body>
            <h1>Portfolio Analytics Report</h1>
            <p>Generated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M') + """</p>
        """
        
        for i, fig in enumerate(figures):
            html_content += f'<div class="chart">{fig.to_html(full_html=False)}</div>'
        
        html_content += "</body></html>"
        
        with open(filename, 'w') as f:
            f.write(html_content)
        
        print(f"Report saved to {filename}")


def main():
    """Example visualization."""
    import pickle
    
    # Try to load backtest results or create sample data
    try:
        adj_prices = pd.read_csv('data/adjusted_prices.csv',
                                index_col=0, parse_dates=True)
        
        # Create sample portfolio
        tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA']
        available = [t for t in tickers if t in adj_prices.columns]
        
        # Simple equal weight returns
        weights = {t: 1/len(available) for t in available}
        returns = adj_prices[available].pct_change().dropna()
        portfolio_returns = (returns * pd.Series(weights)).sum(axis=1)
        portfolio_values = 1_000_000 * (1 + portfolio_returns).cumprod()
        
    except FileNotFoundError:
        print("Data not found. Creating sample data...")
        
        # Generate sample data
        dates = pd.date_range('2020-01-01', '2024-01-01', freq='D')
        np.random.seed(42)
        returns = pd.Series(np.random.normal(0.0004, 0.015, len(dates)), index=dates)
        portfolio_values = 1_000_000 * (1 + returns).cumprod()
        portfolio_returns = returns
        weights = {'AAPL': 0.2, 'MSFT': 0.2, 'GOOGL': 0.15, 
                  'AMZN': 0.15, 'NVDA': 0.1, 'Other': 0.2}
    
    # Initialize visualizer
    viz = PortfolioVisualizer()
    
    print("Creating visualizations...")
    
    # Performance chart
    viz.plot_portfolio_performance(
        portfolio_values,
        title="Portfolio Performance (2020-2024)"
    )
    
    # Returns distribution
    viz.plot_returns_distribution(
        portfolio_returns,
        title="Daily Returns Distribution"
    )
    
    # Drawdown
    viz.plot_drawdown(portfolio_values)
    
    # Allocation
    viz.plot_weights_allocation(weights)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
