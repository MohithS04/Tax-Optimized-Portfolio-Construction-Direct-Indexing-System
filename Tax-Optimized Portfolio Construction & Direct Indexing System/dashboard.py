#!/usr/bin/env python3
"""
Tax-Optimized Portfolio - Interactive Dashboard

A professional, real-time themed dashboard for portfolio analytics.

Usage:
    python dashboard.py              # Start dashboard server
    python dashboard.py --port 8050  # Custom port
"""

import pickle
import pandas as pd
import numpy as np
import argparse
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Dash imports
from dash import Dash, html, dcc, callback, Output, Input
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ============================================================================
# THEME CONFIGURATION
# ============================================================================

# Professional banking/finance dark theme colors
THEME = {
    'background': '#0a0e17',
    'background_gradient': 'linear-gradient(135deg, #0a0e17 0%, #1a1f2e 50%, #0d1220 100%)',
    'card_bg': 'rgba(15, 20, 35, 0.75)',
    'card_bg_solid': 'rgba(22, 27, 45, 0.9)',
    'card_border': 'rgba(59, 130, 246, 0.25)',
    'card_glow': '0 0 40px rgba(59, 130, 246, 0.15)',
    'text_primary': '#f0f6fc',
    'text_secondary': '#a0aec0',
    'accent_blue': '#3b82f6',
    'accent_cyan': '#06b6d4',
    'accent_green': '#10b981',
    'accent_red': '#ef4444',
    'accent_yellow': '#f59e0b',
    'accent_purple': '#8b5cf6',
    'accent_gold': '#fbbf24',
    'chart_grid': 'rgba(30, 41, 59, 0.5)',
    'gradient_start': '#3b82f6',
    'gradient_end': '#8b5cf6',
}

# CSS for animated banking background with custom image
BANKING_BACKGROUND_CSS = """
/* Custom Background Image */
.background-image {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -3;
    background-image: url('/assets/dashboard-bg.png');
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}

/* Dark overlay for readability */
.background-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -2;
    background: rgba(10, 14, 23, 0.75);
}

/* Animated Banking Background */
@keyframes gradientShift {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

@keyframes pulse {
    0%, 100% { opacity: 0.02; }
    50% { opacity: 0.06; }
}

@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
}

@keyframes scan {
    0% { top: -100%; }
    100% { top: 100%; }
}

.banking-background {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background: linear-gradient(135deg, rgba(10, 14, 23, 0.3) 0%, rgba(15, 23, 42, 0.2) 25%, rgba(30, 27, 75, 0.3) 50%, rgba(15, 23, 42, 0.2) 75%, rgba(10, 14, 23, 0.3) 100%);
    background-size: 400% 400%;
    animation: gradientShift 15s ease infinite;
}

.grid-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100%;
    height: 100%;
    z-index: -1;
    background-image: 
        linear-gradient(rgba(59, 130, 246, 0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(59, 130, 246, 0.02) 1px, transparent 1px);
    background-size: 50px 50px;
    animation: pulse 4s ease-in-out infinite;
}

.scan-line {
    position: fixed;
    top: -100%;
    left: 0;
    width: 100%;
    height: 200px;
    background: linear-gradient(180deg, 
        transparent 0%, 
        rgba(59, 130, 246, 0.03) 50%, 
        transparent 100%);
    z-index: -1;
    animation: scan 8s linear infinite;
}

.glow-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    z-index: -1;
    animation: pulse 6s ease-in-out infinite;
}

.glow-orb-1 {
    top: 10%;
    left: 10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.15) 0%, transparent 70%);
}

.glow-orb-2 {
    top: 60%;
    right: 10%;
    width: 300px;
    height: 300px;
    background: radial-gradient(circle, rgba(139, 92, 246, 0.1) 0%, transparent 70%);
    animation-delay: 2s;
}

.glow-orb-3 {
    bottom: 10%;
    left: 30%;
    width: 350px;
    height: 350px;
    background: radial-gradient(circle, rgba(6, 182, 212, 0.08) 0%, transparent 70%);
    animation-delay: 4s;
}

/* Card hover effects */
.dashboard-card {
    backdrop-filter: blur(10px);
    transition: all 0.3s ease;
    border: 1px solid rgba(59, 130, 246, 0.1);
}

.dashboard-card:hover {
    border-color: rgba(59, 130, 246, 0.3);
    box-shadow: 0 0 30px rgba(59, 130, 246, 0.15);
    transform: translateY(-2px);
}

/* KPI card glow effect */
.kpi-card {
    position: relative;
    overflow: hidden;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 100%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(59, 130, 246, 0.1), transparent);
    transition: left 0.5s ease;
}

.kpi-card:hover::before {
    left: 100%;
}

/* Live indicator pulse */
@keyframes livePulse {
    0%, 100% { 
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    50% { 
        box-shadow: 0 0 0 10px rgba(16, 185, 129, 0);
    }
}

.live-indicator {
    width: 8px;
    height: 8px;
    background: #10b981;
    border-radius: 50%;
    display: inline-block;
    animation: livePulse 2s infinite;
}

/* Ticker tape animation */
@keyframes tickerScroll {
    0% { transform: translateX(100%); }
    100% { transform: translateX(-100%); }
}

.ticker-tape {
    overflow: hidden;
    white-space: nowrap;
}

.ticker-content {
    display: inline-block;
    animation: tickerScroll 30s linear infinite;
}

/* Number counter animation */
@keyframes countUp {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.counter-value {
    animation: countUp 0.5s ease-out;
}
"""

# Plotly template for dark theme with transparent backgrounds
PLOTLY_TEMPLATE = {
    'layout': {
        'paper_bgcolor': 'rgba(15, 20, 35, 0.6)',
        'plot_bgcolor': 'rgba(15, 20, 35, 0.4)',
        'font': {'color': THEME['text_primary'], 'family': 'Inter, sans-serif'},
        'xaxis': {
            'gridcolor': 'rgba(59, 130, 246, 0.1)',
            'zerolinecolor': 'rgba(59, 130, 246, 0.15)',
            'tickfont': {'color': THEME['text_secondary']}
        },
        'yaxis': {
            'gridcolor': 'rgba(59, 130, 246, 0.1)',
            'zerolinecolor': 'rgba(59, 130, 246, 0.15)',
            'tickfont': {'color': THEME['text_secondary']}
        },
        'colorway': [THEME['accent_blue'], THEME['accent_green'], THEME['accent_purple'],
                    THEME['accent_yellow'], THEME['accent_red']],
        'margin': {'l': 40, 'r': 20, 't': 40, 'b': 40}
    }
}

# ============================================================================
# DATA LOADING
# ============================================================================

def load_all_data():
    """Load all saved results."""
    data = {}
    
    try:
        with open('data/backtest_results.pkl', 'rb') as f:
            data['backtest'] = pickle.load(f)
    except FileNotFoundError:
        data['backtest'] = None
    
    try:
        with open('data/optimization_results.pkl', 'rb') as f:
            data['optimization'] = pickle.load(f)
    except FileNotFoundError:
        data['optimization'] = None
    
    try:
        with open('data/harvest_opportunities.pkl', 'rb') as f:
            data['harvest'] = pickle.load(f)
    except FileNotFoundError:
        data['harvest'] = None
    
    try:
        with open('data/processed_data.pkl', 'rb') as f:
            data['processed'] = pickle.load(f)
    except FileNotFoundError:
        data['processed'] = None
    
    return data

# Load data globally
DATA = load_all_data()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def format_currency(value):
    """Format number as currency."""
    if value >= 1e9:
        return f"${value/1e9:.2f}B"
    elif value >= 1e6:
        return f"${value/1e6:.2f}M"
    elif value >= 1e3:
        return f"${value/1e3:.1f}K"
    else:
        return f"${value:,.2f}"

def format_percent(value):
    """Format number as percentage."""
    return f"{value*100:.2f}%"

def get_trend_icon(value):
    """Return trend icon based on value."""
    if value > 0:
        return "▲"
    elif value < 0:
        return "▼"
    return "●"

def get_trend_color(value):
    """Return color based on value."""
    if value > 0:
        return THEME['accent_green']
    elif value < 0:
        return THEME['accent_red']
    return THEME['text_secondary']

# ============================================================================
# COMPONENT BUILDERS
# ============================================================================

def create_kpi_card(title, value, subtitle=None, trend=None, icon=None):
    """Create a KPI metric card with banking theme."""
    trend_color = get_trend_color(trend) if trend else THEME['text_secondary']
    trend_icon = get_trend_icon(trend) if trend else ""
    
    # Determine accent color based on metric type
    accent_color = THEME['accent_blue']
    if 'Return' in title or 'CAGR' in title:
        accent_color = THEME['accent_green']
    elif 'Drawdown' in title:
        accent_color = THEME['accent_red']
    elif 'Tax' in title:
        accent_color = THEME['accent_gold']
    elif 'Sharpe' in title:
        accent_color = THEME['accent_cyan']
    
    return dbc.Card([
        # Top accent line
        html.Div(style={
            'height': '3px',
            'background': f'linear-gradient(90deg, {accent_color}, transparent)',
            'borderRadius': '12px 12px 0 0'
        }),
        dbc.CardBody([
            html.Div([
                html.Span(icon, style={
                    'fontSize': '20px',
                    'marginRight': '8px',
                    'opacity': '0.9'
                }) if icon else None,
                html.Span(title, style={
                    'color': THEME['text_secondary'],
                    'fontSize': '11px',
                    'textTransform': 'uppercase',
                    'letterSpacing': '1.5px',
                    'fontWeight': '500'
                })
            ], style={'marginBottom': '12px', 'display': 'flex', 'alignItems': 'center'}),
            html.Div([
                html.Span(value, className='counter-value', style={
                    'fontSize': '26px',
                    'fontWeight': '700',
                    'color': THEME['text_primary'],
                    'fontFamily': 'JetBrains Mono, monospace'
                }),
                html.Span(f" {trend_icon} {format_percent(trend) if trend else ''}", style={
                    'fontSize': '13px',
                    'color': trend_color,
                    'marginLeft': '8px',
                    'fontWeight': '600'
                }) if trend else None
            ]),
            html.Div(subtitle, style={
                'color': THEME['text_secondary'],
                'fontSize': '11px',
                'marginTop': '8px',
                'opacity': '0.8'
            }) if subtitle else None
        ], style={'padding': '16px 20px'})
    ], className='kpi-card dashboard-card', style={
        'backgroundColor': THEME['card_bg'],
        'border': f"1px solid {THEME['card_border']}",
        'borderRadius': '12px',
        'height': '100%',
        'backdropFilter': 'blur(10px)',
        'boxShadow': THEME['card_glow']
    })

def create_header():
    """Create dashboard header with banking theme."""
    # Get some metrics for the ticker
    ticker_items = []
    if DATA.get('optimization') and 'mean_variance' in DATA['optimization']:
        opt = DATA['optimization']['mean_variance']
        if opt.get('status') == 'optimal':
            weights = opt.get('weights', {})
            for ticker, weight in sorted(weights.items(), key=lambda x: -x[1])[:8]:
                ticker_items.append(f"{ticker} {weight*100:.1f}%")
    
    ticker_text = "  •  ".join(ticker_items) if ticker_items else "Portfolio Analytics Dashboard"
    
    return html.Div([
        # Top bar with gradient
        html.Div(style={
            'height': '2px',
            'background': f'linear-gradient(90deg, {THEME["accent_blue"]}, {THEME["accent_purple"]}, {THEME["accent_cyan"]})',
            'backgroundSize': '200% 200%',
            'animation': 'gradientShift 3s ease infinite'
        }),
        
        # Main header
        dbc.Container([
            dbc.Row([
                dbc.Col([
                    html.Div([
                        html.Div([
                            html.Span("◆", style={
                                'color': THEME['accent_blue'],
                                'fontSize': '24px',
                                'marginRight': '12px',
                                'filter': 'drop-shadow(0 0 10px rgba(59, 130, 246, 0.5))'
                            }),
                            html.Span("TAX-OPTIMIZED", style={
                                'color': THEME['accent_blue'],
                                'fontWeight': '300',
                                'fontSize': '24px',
                                'letterSpacing': '2px'
                            }),
                            html.Span(" PORTFOLIO", style={
                                'color': THEME['text_primary'],
                                'fontWeight': '700',
                                'fontSize': '24px',
                                'letterSpacing': '2px'
                            })
                        ], style={'display': 'flex', 'alignItems': 'center'}),
                        html.P("Direct Indexing & Tax-Loss Harvesting System", style={
                            'color': THEME['text_secondary'],
                            'marginBottom': '0',
                            'fontSize': '12px',
                            'marginTop': '4px',
                            'marginLeft': '36px',
                            'letterSpacing': '1px'
                        })
                    ])
                ], width=6),
                dbc.Col([
                    html.Div([
                        # Live status
                        html.Div([
                            html.Span(className='live-indicator', style={'marginRight': '8px'}),
                            html.Span("LIVE", style={
                                'color': THEME['accent_green'],
                                'fontSize': '10px',
                                'fontWeight': '700',
                                'letterSpacing': '2px',
                                'marginRight': '15px'
                            }),
                            html.Span("|", style={'color': THEME['card_border'], 'marginRight': '15px'}),
                            html.Span(id='current-time', style={
                                'color': THEME['text_primary'],
                                'fontSize': '13px',
                                'fontFamily': 'JetBrains Mono, monospace'
                            })
                        ], style={'textAlign': 'right', 'marginBottom': '8px', 'display': 'flex', 'alignItems': 'center', 'justifyContent': 'flex-end'}),
                        html.Div([
                            html.Span("Session Active", style={
                                'color': THEME['text_secondary'],
                                'fontSize': '10px',
                                'textTransform': 'uppercase',
                                'letterSpacing': '1px'
                            })
                        ], style={'textAlign': 'right'})
                    ])
                ], width=6)
            ], align='center')
        ], fluid=True, style={'padding': '20px 30px'}),
        
        # Ticker tape
        html.Div([
            html.Div([
                html.Span("TOP HOLDINGS:  ", style={
                    'color': THEME['accent_blue'],
                    'fontWeight': '600',
                    'marginRight': '10px'
                }),
                html.Span(ticker_text + "  •  " + ticker_text, style={
                    'color': THEME['text_secondary']
                })
            ], className='ticker-content', style={
                'fontSize': '11px',
                'fontFamily': 'JetBrains Mono, monospace',
                'letterSpacing': '0.5px'
            })
        ], className='ticker-tape', style={
            'backgroundColor': 'rgba(59, 130, 246, 0.05)',
            'padding': '8px 30px',
            'borderTop': f"1px solid {THEME['card_border']}",
            'borderBottom': f"1px solid {THEME['card_border']}"
        })
    ])

def create_kpi_row():
    """Create KPI metrics row."""
    metrics = {}
    
    if DATA.get('backtest') and 'backtest' in DATA['backtest']:
        m = DATA['backtest']['backtest'].metrics
        pv = DATA['backtest']['backtest'].portfolio_values
        metrics = {
            'total_return': m.get('total_return', 0),
            'cagr': m.get('cagr', 0),
            'volatility': m.get('annual_volatility', 0),
            'sharpe': m.get('sharpe_ratio', 0),
            'max_dd': m.get('max_drawdown', 0),
            'portfolio_value': pv.iloc[-1] if len(pv) > 0 else 1000000
        }
    
    if DATA.get('optimization') and 'mean_variance' in DATA['optimization']:
        opt = DATA['optimization']['mean_variance']
        metrics['holdings'] = opt.get('n_holdings', 0)
        metrics['expected_return'] = opt.get('expected_return', 0)
    
    tax_benefit = 0
    if DATA.get('harvest'):
        tax_benefit = sum(h['tax_benefit'] for h in DATA['harvest'])
    
    return dbc.Container([
        dbc.Row([
            dbc.Col(create_kpi_card(
                "Portfolio Value",
                format_currency(metrics.get('portfolio_value', 1000000)),
                "Current NAV",
                metrics.get('total_return', 0),
                "💰"
            ), width=2),
            dbc.Col(create_kpi_card(
                "Total Return",
                format_percent(metrics.get('total_return', 0)),
                "Since Inception",
                icon="📈"
            ), width=2),
            dbc.Col(create_kpi_card(
                "CAGR",
                format_percent(metrics.get('cagr', 0)),
                "Annualized",
                icon="📊"
            ), width=2),
            dbc.Col(create_kpi_card(
                "Sharpe Ratio",
                f"{metrics.get('sharpe', 0):.2f}",
                "Risk-Adjusted",
                icon="⚖️"
            ), width=2),
            dbc.Col(create_kpi_card(
                "Max Drawdown",
                format_percent(metrics.get('max_dd', 0)),
                "Peak to Trough",
                icon="📉"
            ), width=2),
            dbc.Col(create_kpi_card(
                "Tax Savings",
                format_currency(tax_benefit),
                f"{len(DATA.get('harvest', []))} Opportunities",
                icon="💵"
            ), width=2),
        ], className='g-3')
    ], fluid=True, style={'padding': '20px 30px'})

def create_performance_chart():
    """Create main performance chart."""
    fig = go.Figure()
    
    if DATA.get('backtest') and 'backtest' in DATA['backtest']:
        pv = DATA['backtest']['backtest'].portfolio_values
        
        # Portfolio value line
        fig.add_trace(go.Scatter(
            x=pv.index,
            y=pv.values,
            mode='lines',
            name='Portfolio Value',
            line=dict(color=THEME['accent_blue'], width=2),
            fill='tozeroy',
            fillcolor=f"rgba(88, 166, 255, 0.1)",
            hovertemplate='%{x}<br>Value: $%{y:,.0f}<extra></extra>'
        ))
        
        # Add moving average
        ma_50 = pv.rolling(50).mean()
        fig.add_trace(go.Scatter(
            x=ma_50.index,
            y=ma_50.values,
            mode='lines',
            name='50-Day MA',
            line=dict(color=THEME['accent_yellow'], width=1, dash='dot'),
            hovertemplate='%{x}<br>MA50: $%{y:,.0f}<extra></extra>'
        ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Portfolio Performance', font=dict(size=16)),
        xaxis_title='',
        yaxis_title='Value ($)',
        yaxis_tickformat='$,.0f',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
            bgcolor='rgba(0,0,0,0)'
        ),
        hovermode='x unified',
        height=350
    )
    
    return fig

def create_drawdown_chart():
    """Create drawdown chart."""
    fig = go.Figure()
    
    if DATA.get('backtest') and 'backtest' in DATA['backtest']:
        pv = DATA['backtest']['backtest'].portfolio_values
        rolling_max = pv.cummax()
        drawdown = (pv - rolling_max) / rolling_max
        
        fig.add_trace(go.Scatter(
            x=drawdown.index,
            y=drawdown.values,
            mode='lines',
            fill='tozeroy',
            fillcolor=f"rgba(248, 81, 73, 0.3)",
            line=dict(color=THEME['accent_red'], width=1),
            hovertemplate='%{x}<br>Drawdown: %{y:.1%}<extra></extra>'
        ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Underwater Chart (Drawdown)', font=dict(size=16)),
        xaxis_title='',
        yaxis_title='Drawdown',
        yaxis_tickformat='.0%',
        showlegend=False,
        height=200
    )
    
    return fig

def create_allocation_chart():
    """Create allocation donut chart."""
    fig = go.Figure()
    
    if DATA.get('optimization') and 'mean_variance' in DATA['optimization']:
        opt = DATA['optimization']['mean_variance']
        if opt.get('status') == 'optimal':
            weights = opt['weights']
            sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:8]
            other = sum(w for t, w in weights.items() if t not in dict(sorted_weights))
            
            labels = [t for t, w in sorted_weights] + (['Other'] if other > 0.02 else [])
            values = [w for t, w in sorted_weights] + ([other] if other > 0.02 else [])
            
            colors = [THEME['accent_blue'], THEME['accent_green'], THEME['accent_purple'],
                     THEME['accent_yellow'], '#f97583', '#56d4dd', '#ff9f43', THEME['text_secondary'], '#666']
            
            fig.add_trace(go.Pie(
                labels=labels,
                values=values,
                hole=0.6,
                marker=dict(colors=colors[:len(labels)]),
                textinfo='label+percent',
                textposition='outside',
                textfont=dict(size=10, color=THEME['text_primary']),
                hovertemplate='%{label}<br>Weight: %{percent}<br>Value: %{value:.2%}<extra></extra>'
            ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Portfolio Allocation', font=dict(size=16)),
        showlegend=False,
        height=300,
        annotations=[dict(
            text='<b>Holdings</b>',
            x=0.5, y=0.5,
            font=dict(size=14, color=THEME['text_primary']),
            showarrow=False
        )]
    )
    
    return fig

def create_sector_chart():
    """Create sector allocation bar chart."""
    fig = go.Figure()
    
    if DATA.get('optimization') and DATA.get('processed'):
        opt = DATA['optimization'].get('mean_variance', {})
        if opt.get('status') == 'optimal':
            weights = opt['weights']
            sector_mapping = DATA['processed'].get('sector_mapping', {})
            
            sector_weights = {}
            for ticker, weight in weights.items():
                sector = sector_mapping.get(ticker, 'Unknown')
                sector_weights[sector] = sector_weights.get(sector, 0) + weight
            
            sector_weights = dict(sorted(sector_weights.items(), key=lambda x: -x[1]))
            
            colors = [THEME['accent_blue'] if i == 0 else THEME['accent_green'] if i == 1 
                     else THEME['accent_purple'] if i == 2 else THEME['text_secondary']
                     for i in range(len(sector_weights))]
            
            fig.add_trace(go.Bar(
                y=list(sector_weights.keys()),
                x=list(sector_weights.values()),
                orientation='h',
                marker=dict(
                    color=colors,
                    line=dict(width=0)
                ),
                text=[f'{w:.1%}' for w in sector_weights.values()],
                textposition='outside',
                textfont=dict(color=THEME['text_primary'], size=10),
                hovertemplate='%{y}<br>Weight: %{x:.2%}<extra></extra>'
            ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Sector Allocation', font=dict(size=16)),
        xaxis_title='',
        xaxis_tickformat='.0%',
        yaxis_title='',
        showlegend=False,
        height=300
    )
    
    return fig

def create_harvest_chart():
    """Create tax harvest opportunities chart."""
    fig = go.Figure()
    
    if DATA.get('harvest') and len(DATA['harvest']) > 0:
        harvest = DATA['harvest'][:10]
        
        fig.add_trace(go.Bar(
            x=[h['ticker'] for h in harvest],
            y=[h['tax_benefit'] for h in harvest],
            marker=dict(
                color=[THEME['accent_green'] if h['tax_benefit'] > 1000 else THEME['accent_yellow']
                      for h in harvest],
                line=dict(width=0)
            ),
            text=[f"${h['tax_benefit']:,.0f}" for h in harvest],
            textposition='outside',
            textfont=dict(color=THEME['text_primary'], size=10),
            hovertemplate='<b>%{x}</b><br>Tax Benefit: $%{y:,.2f}<extra></extra>'
        ))
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Tax-Loss Harvest Opportunities', font=dict(size=16)),
        xaxis_title='',
        yaxis_title='Tax Benefit ($)',
        yaxis_tickformat='$,.0f',
        showlegend=False,
        height=300
    )
    
    return fig

def create_returns_distribution():
    """Create returns distribution chart."""
    fig = go.Figure()
    
    if DATA.get('backtest') and 'backtest' in DATA['backtest']:
        returns = DATA['backtest']['backtest'].returns
        
        fig.add_trace(go.Histogram(
            x=returns.values,
            nbinsx=50,
            marker=dict(
                color=THEME['accent_blue'],
                line=dict(color=THEME['card_border'], width=1)
            ),
            opacity=0.8,
            hovertemplate='Return: %{x:.2%}<br>Count: %{y}<extra></extra>'
        ))
        
        # Add mean line
        mean_ret = returns.mean()
        fig.add_vline(
            x=mean_ret,
            line_dash="dash",
            line_color=THEME['accent_yellow'],
            annotation_text=f"Mean: {mean_ret:.2%}",
            annotation_font_color=THEME['text_primary']
        )
        
        # Add zero line
        fig.add_vline(x=0, line_color=THEME['text_secondary'], line_width=1)
    
    fig.update_layout(
        **PLOTLY_TEMPLATE['layout'],
        title=dict(text='Returns Distribution', font=dict(size=16)),
        xaxis_title='Daily Return',
        xaxis_tickformat='.1%',
        yaxis_title='Frequency',
        showlegend=False,
        height=250
    )
    
    return fig

def create_holdings_table():
    """Create top holdings table."""
    if not DATA.get('optimization') or 'mean_variance' not in DATA['optimization']:
        return html.Div("No data available")
    
    opt = DATA['optimization']['mean_variance']
    if opt.get('status') != 'optimal':
        return html.Div("Optimization not available")
    
    weights = opt['weights']
    sector_mapping = DATA['processed'].get('sector_mapping', {}) if DATA.get('processed') else {}
    
    sorted_weights = sorted(weights.items(), key=lambda x: x[1], reverse=True)[:10]
    
    rows = []
    for i, (ticker, weight) in enumerate(sorted_weights):
        sector = sector_mapping.get(ticker, 'Unknown')
        rows.append(
            html.Tr([
                html.Td(str(i+1), style={'color': THEME['text_secondary'], 'width': '30px'}),
                html.Td([
                    html.Span(ticker, style={'fontWeight': '600', 'color': THEME['text_primary']}),
                ]),
                html.Td(sector, style={'color': THEME['text_secondary'], 'fontSize': '12px'}),
                html.Td(f"{weight:.2%}", style={
                    'color': THEME['accent_green'] if weight > 0.04 else THEME['text_primary'],
                    'fontWeight': '500',
                    'textAlign': 'right'
                })
            ], style={'borderBottom': f"1px solid {THEME['chart_grid']}"})
        )
    
    return html.Table([
        html.Thead(html.Tr([
            html.Th('#', style={'color': THEME['text_secondary'], 'fontWeight': '500', 'fontSize': '11px'}),
            html.Th('Ticker', style={'color': THEME['text_secondary'], 'fontWeight': '500', 'fontSize': '11px'}),
            html.Th('Sector', style={'color': THEME['text_secondary'], 'fontWeight': '500', 'fontSize': '11px'}),
            html.Th('Weight', style={'color': THEME['text_secondary'], 'fontWeight': '500', 'fontSize': '11px', 'textAlign': 'right'})
        ])),
        html.Tbody(rows)
    ], style={'width': '100%', 'borderCollapse': 'collapse'})

def create_monte_carlo_info():
    """Create Monte Carlo simulation info panel."""
    if not DATA.get('backtest') or 'monte_carlo' not in DATA['backtest']:
        return html.Div("Monte Carlo data not available")
    
    mc = DATA['backtest']['monte_carlo']
    
    return html.Div([
        html.Div([
            html.Div("Expected (1Y)", style={'color': THEME['text_secondary'], 'fontSize': '11px'}),
            html.Div(format_currency(mc.get('mean_final_value', 0)), style={
                'color': THEME['text_primary'], 'fontSize': '20px', 'fontWeight': '600'
            })
        ], style={'marginBottom': '15px'}),
        
        html.Div([
            html.Div([
                html.Span("5th Percentile (VaR)", style={'color': THEME['text_secondary'], 'fontSize': '11px'}),
                html.Div(format_currency(mc.get('percentile_5', 0)), style={
                    'color': THEME['accent_red'], 'fontWeight': '500'
                })
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("95th Percentile", style={'color': THEME['text_secondary'], 'fontSize': '11px'}),
                html.Div(format_currency(mc.get('percentile_95', 0)), style={
                    'color': THEME['accent_green'], 'fontWeight': '500'
                })
            ], style={'marginBottom': '10px'}),
            html.Div([
                html.Span("Probability of Loss", style={'color': THEME['text_secondary'], 'fontSize': '11px'}),
                html.Div(format_percent(mc.get('prob_loss', 0)), style={
                    'color': THEME['accent_yellow'], 'fontWeight': '500'
                })
            ])
        ])
    ])

# ============================================================================
# APP LAYOUT
# ============================================================================

app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Tax-Optimized Portfolio Dashboard"

# Inject custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
        <style>
''' + BANKING_BACKGROUND_CSS + '''
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

app.layout = html.Div([
    # Background Image Layer
    html.Div(className='background-image'),
    html.Div(className='background-overlay'),
    
    # Animated Background Elements
    html.Div(className='banking-background'),
    html.Div(className='grid-overlay'),
    html.Div(className='scan-line'),
    html.Div(className='glow-orb glow-orb-1'),
    html.Div(className='glow-orb glow-orb-2'),
    html.Div(className='glow-orb glow-orb-3'),
    
    # Interval for time updates
    dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
    
    # Header
    create_header(),
    
    # KPI Row
    create_kpi_row(),
    
    # Main Content
    dbc.Container([
        # Row 1: Performance Chart
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_blue"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='performance-chart',
                            figure=create_performance_chart(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=12)
        ], className='mb-3'),
        
        # Row 2: Drawdown
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_red"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='drawdown-chart',
                            figure=create_drawdown_chart(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=12)
        ], className='mb-3'),
        
        # Row 3: Allocation, Sectors, Holdings
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_purple"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='allocation-chart',
                            figure=create_allocation_chart(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_cyan"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='sector-chart',
                            figure=create_sector_chart(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_green"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        html.Div([
                            html.Span("📋", style={'marginRight': '8px'}),
                            html.Span("Top Holdings", style={
                                'color': THEME['text_primary'],
                                'fontSize': '14px',
                                'fontWeight': '600',
                                'letterSpacing': '0.5px'
                            })
                        ], style={'marginBottom': '15px'}),
                        create_holdings_table()
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=4)
        ], className='mb-3'),
        
        # Row 4: Tax Harvest, Returns Distribution, Monte Carlo
        dbc.Row([
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_gold"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='harvest-chart',
                            figure=create_harvest_chart(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=5),
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_blue"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        dcc.Graph(
                            id='returns-dist',
                            figure=create_returns_distribution(),
                            config={'displayModeBar': False}
                        )
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=4),
            dbc.Col([
                dbc.Card([
                    html.Div(style={
                        'height': '2px',
                        'background': f'linear-gradient(90deg, {THEME["accent_purple"]}, transparent)',
                        'borderRadius': '12px 12px 0 0'
                    }),
                    dbc.CardBody([
                        html.Div([
                            html.Span("🎯", style={'marginRight': '8px'}),
                            html.Span("Monte Carlo Forecast", style={
                                'color': THEME['text_primary'],
                                'fontSize': '14px',
                                'fontWeight': '600',
                                'letterSpacing': '0.5px'
                            })
                        ], style={'marginBottom': '15px'}),
                        create_monte_carlo_info()
                    ])
                ], className='dashboard-card', style={
                    'backgroundColor': THEME['card_bg'],
                    'border': f"1px solid {THEME['card_border']}",
                    'borderRadius': '12px',
                    'height': '100%',
                    'backdropFilter': 'blur(10px)',
                    'boxShadow': THEME['card_glow']
                })
            ], width=3)
        ], className='mb-4'),
        
        # Footer
        dbc.Row([
            dbc.Col([
                html.Div([
                    html.Div([
                        html.Span("◆", style={'color': THEME['accent_blue'], 'marginRight': '8px'}),
                        html.Span("TAX-OPTIMIZED PORTFOLIO SYSTEM", style={
                            'color': THEME['text_secondary'],
                            'fontSize': '11px',
                            'letterSpacing': '2px',
                            'fontWeight': '500'
                        }),
                    ], style={'marginBottom': '8px'}),
                    html.Div([
                        html.Span("Powered by ", style={'color': THEME['text_secondary'], 'fontSize': '10px'}),
                        html.Span("Direct Indexing Engine", style={
                            'color': THEME['accent_blue'],
                            'fontSize': '10px',
                            'fontWeight': '500'
                        }),
                        html.Span(" • ", style={'color': THEME['card_border'], 'margin': '0 8px'}),
                        html.Span("Real-Time Analytics", style={
                            'color': THEME['accent_cyan'],
                            'fontSize': '10px',
                            'fontWeight': '500'
                        }),
                        html.Span(" • ", style={'color': THEME['card_border'], 'margin': '0 8px'}),
                        html.Span("Tax-Loss Harvesting", style={
                            'color': THEME['accent_green'],
                            'fontSize': '10px',
                            'fontWeight': '500'
                        })
                    ])
                ], style={
                    'textAlign': 'center',
                    'padding': '25px 0',
                    'borderTop': f"1px solid {THEME['card_border']}"
                })
            ])
        ])
        
    ], fluid=True, style={'padding': '0 30px'})
    
], style={
    'backgroundColor': THEME['background'],
    'minHeight': '100vh',
    'fontFamily': 'Inter, -apple-system, BlinkMacSystemFont, sans-serif'
})

# ============================================================================
# CALLBACKS
# ============================================================================

@callback(
    Output('current-time', 'children'),
    Input('interval-component', 'n_intervals')
)
def update_time(n):
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Portfolio Dashboard')
    parser.add_argument('--port', type=int, default=8050, help='Port number')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='Host address (use 0.0.0.0 for public access)')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--public', action='store_true', help='Enable public access (sets host to 0.0.0.0)')
    args = parser.parse_args()
    
    # If public flag is set, use 0.0.0.0
    host = '0.0.0.0' if args.public else args.host
    
    print("\n" + "=" * 60)
    print("🚀 TAX-OPTIMIZED PORTFOLIO DASHBOARD")
    print("=" * 60)
    
    if host == '0.0.0.0':
        # Get local IP for display
        import socket
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
        except:
            local_ip = "YOUR_IP"
        
        print(f"\n📊 Dashboard starting in PUBLIC mode")
        print(f"\n   Local access:    http://localhost:{args.port}")
        print(f"   Network access:  http://{local_ip}:{args.port}")
        print(f"\n   💡 For internet sharing, use ngrok:")
        print(f"      ngrok http {args.port}")
    else:
        print(f"\n📊 Starting server at: http://localhost:{args.port}")
        print(f"\n   💡 For public access, restart with:")
        print(f"      python dashboard.py --public")
    
    print("\n   Press Ctrl+C to stop\n")
    
    app.run(debug=args.debug, port=args.port, host=host)


if __name__ == '__main__':
    main()
