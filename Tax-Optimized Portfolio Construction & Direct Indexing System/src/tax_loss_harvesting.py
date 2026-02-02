"""
Tax-Loss Harvesting Module

This module implements:
    - Tax lot tracking and management
    - Loss harvesting opportunity identification
    - Wash sale rule compliance
    - Replacement security selection
    - Tax alpha calculation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import warnings


@dataclass
class TaxLot:
    """Represents a single tax lot (purchase)."""
    ticker: str
    purchase_date: datetime
    shares: float
    cost_basis: float  # Price per share at purchase
    remaining_shares: float = None
    sold_date: Optional[datetime] = None
    lot_id: str = ""
    
    def __post_init__(self):
        if self.remaining_shares is None:
            self.remaining_shares = self.shares
        if not self.lot_id:
            self.lot_id = f"{self.ticker}_{self.purchase_date.strftime('%Y%m%d')}_{id(self)}"
    
    @property
    def total_cost(self) -> float:
        """Total cost basis for the lot."""
        return self.shares * self.cost_basis
    
    @property
    def remaining_cost(self) -> float:
        """Cost basis for remaining shares."""
        return self.remaining_shares * self.cost_basis
    
    def holding_days(self, as_of_date: datetime) -> int:
        """Days held as of given date."""
        return (as_of_date - self.purchase_date).days
    
    def is_long_term(self, as_of_date: datetime) -> bool:
        """Whether holding qualifies for long-term treatment."""
        return self.holding_days(as_of_date) >= 365
    
    def unrealized_gain(self, current_price: float) -> float:
        """Calculate unrealized gain/loss."""
        return (current_price - self.cost_basis) * self.remaining_shares
    
    def unrealized_gain_pct(self, current_price: float) -> float:
        """Calculate unrealized gain/loss percentage."""
        if self.cost_basis == 0:
            return 0
        return (current_price - self.cost_basis) / self.cost_basis


class TaxLossHarvester:
    """
    Tax-loss harvesting engine with full lot tracking.
    
    Implements IRS wash sale rules and optimizes tax harvesting
    while maintaining portfolio exposure.
    """
    
    def __init__(self,
                 tax_rate_short_term: float = 0.37,
                 tax_rate_long_term: float = 0.238,
                 wash_sale_days: int = 30,
                 min_harvest_amount: float = 100):
        """
        Initialize the tax-loss harvester.
        
        Parameters:
        -----------
        tax_rate_short_term : float
            Short-term capital gains tax rate (ordinary income)
        tax_rate_long_term : float
            Long-term capital gains tax rate
        wash_sale_days : int
            Days before/after sale that trigger wash sale (IRS = 30)
        min_harvest_amount : float
            Minimum tax benefit to execute harvest
        """
        self.tax_rate_st = tax_rate_short_term
        self.tax_rate_lt = tax_rate_long_term
        self.wash_sale_days = wash_sale_days
        self.min_harvest_amount = min_harvest_amount
        
        # Tax lot storage: {ticker: [TaxLot, ...]}
        self.tax_lots: Dict[str, List[TaxLot]] = {}
        
        # Wash sale tracking: {ticker: blocked_until_date}
        self.wash_sale_blocked: Dict[str, datetime] = {}
        
        # Harvest history
        self.harvest_history: List[Dict] = []
        
        # Replacement tracking
        self.replacements: Dict[str, str] = {}  # {sold_ticker: replacement_ticker}
    
    def add_purchase(self, 
                    ticker: str,
                    date: datetime,
                    shares: float,
                    price: float) -> TaxLot:
        """
        Record a purchase (create tax lot).
        
        Parameters:
        -----------
        ticker : str
            Security ticker
        date : datetime
            Purchase date
        shares : float
            Number of shares purchased
        price : float
            Price per share
            
        Returns:
        --------
        TaxLot : The created tax lot
        """
        date = pd.to_datetime(date)
        
        lot = TaxLot(
            ticker=ticker,
            purchase_date=date,
            shares=shares,
            cost_basis=price
        )
        
        if ticker not in self.tax_lots:
            self.tax_lots[ticker] = []
        
        self.tax_lots[ticker].append(lot)
        
        # Clear wash sale if buying back
        if ticker in self.wash_sale_blocked:
            if date <= self.wash_sale_blocked[ticker]:
                warnings.warn(f"Wash sale triggered for {ticker}")
        
        return lot
    
    def add_portfolio(self,
                     holdings: Dict[str, Dict],
                     purchase_date: datetime):
        """
        Add multiple holdings at once.
        
        Parameters:
        -----------
        holdings : dict
            {ticker: {'shares': n, 'price': p}}
        purchase_date : datetime
            Date of purchase
        """
        for ticker, info in holdings.items():
            self.add_purchase(
                ticker=ticker,
                date=purchase_date,
                shares=info['shares'],
                price=info['price']
            )
    
    def get_position_summary(self, ticker: str, current_price: float) -> Dict:
        """Get summary of position across all lots."""
        if ticker not in self.tax_lots:
            return None
        
        lots = [lot for lot in self.tax_lots[ticker] if lot.remaining_shares > 0]
        
        if not lots:
            return None
        
        total_shares = sum(lot.remaining_shares for lot in lots)
        total_cost = sum(lot.remaining_cost for lot in lots)
        avg_cost = total_cost / total_shares if total_shares > 0 else 0
        current_value = total_shares * current_price
        
        return {
            'ticker': ticker,
            'total_shares': total_shares,
            'average_cost': avg_cost,
            'current_price': current_price,
            'current_value': current_value,
            'total_cost_basis': total_cost,
            'unrealized_gain': current_value - total_cost,
            'unrealized_gain_pct': (current_value - total_cost) / total_cost if total_cost > 0 else 0,
            'n_lots': len(lots)
        }
    
    def identify_harvest_opportunities(self,
                                      current_prices: Dict[str, float],
                                      current_date: datetime,
                                      min_loss_pct: float = 0.05) -> List[Dict]:
        """
        Identify positions with harvestable losses.
        
        Parameters:
        -----------
        current_prices : dict
            {ticker: current_price}
        current_date : datetime
            Current date for evaluation
        min_loss_pct : float
            Minimum loss percentage to consider
            
        Returns:
        --------
        list : Harvest opportunities sorted by tax benefit
        """
        current_date = pd.to_datetime(current_date)
        opportunities = []
        
        for ticker, lots in self.tax_lots.items():
            if ticker not in current_prices:
                continue
            
            current_price = current_prices[ticker]
            
            # Check wash sale restriction
            if ticker in self.wash_sale_blocked:
                if current_date < self.wash_sale_blocked[ticker]:
                    continue
            
            for lot in lots:
                if lot.remaining_shares <= 0:
                    continue
                
                # Calculate unrealized loss
                unrealized_pnl = lot.unrealized_gain(current_price)
                unrealized_pnl_pct = lot.unrealized_gain_pct(current_price)
                
                # Only consider losses above threshold
                if unrealized_pnl >= 0 or abs(unrealized_pnl_pct) < min_loss_pct:
                    continue
                
                # Determine holding period and tax rate
                holding_days = lot.holding_days(current_date)
                is_long_term = holding_days >= 365
                tax_rate = self.tax_rate_lt if is_long_term else self.tax_rate_st
                
                # Calculate tax benefit
                tax_benefit = abs(unrealized_pnl) * tax_rate
                
                if tax_benefit >= self.min_harvest_amount:
                    opportunities.append({
                        'ticker': ticker,
                        'lot_id': lot.lot_id,
                        'purchase_date': lot.purchase_date,
                        'shares': lot.remaining_shares,
                        'cost_basis': lot.cost_basis,
                        'current_price': current_price,
                        'unrealized_loss': unrealized_pnl,
                        'unrealized_loss_pct': unrealized_pnl_pct,
                        'tax_benefit': tax_benefit,
                        'tax_rate': tax_rate,
                        'is_long_term': is_long_term,
                        'holding_days': holding_days
                    })
        
        # Sort by tax benefit (highest first)
        opportunities.sort(key=lambda x: x['tax_benefit'], reverse=True)
        
        return opportunities
    
    def execute_harvest(self,
                       ticker: str,
                       lot_id: str,
                       sell_date: datetime,
                       sell_price: float,
                       shares: Optional[float] = None) -> Dict:
        """
        Execute a tax-loss harvest.
        
        Parameters:
        -----------
        ticker : str
            Security to sell
        lot_id : str
            Specific lot ID to sell
        sell_date : datetime
            Date of sale
        sell_price : float
            Price per share
        shares : float, optional
            Number of shares to sell (None = all)
            
        Returns:
        --------
        dict : Harvest execution details
        """
        sell_date = pd.to_datetime(sell_date)
        
        # Find the lot
        if ticker not in self.tax_lots:
            raise ValueError(f"No lots found for {ticker}")
        
        lot = None
        for l in self.tax_lots[ticker]:
            if l.lot_id == lot_id:
                lot = l
                break
        
        if lot is None:
            raise ValueError(f"Lot {lot_id} not found for {ticker}")
        
        if lot.remaining_shares <= 0:
            raise ValueError(f"Lot {lot_id} has no remaining shares")
        
        # Determine shares to sell
        shares_to_sell = shares if shares else lot.remaining_shares
        shares_to_sell = min(shares_to_sell, lot.remaining_shares)
        
        # Calculate realized loss
        proceeds = shares_to_sell * sell_price
        cost_basis_sold = shares_to_sell * lot.cost_basis
        realized_loss = proceeds - cost_basis_sold
        
        # Determine tax rate and benefit
        is_long_term = lot.is_long_term(sell_date)
        tax_rate = self.tax_rate_lt if is_long_term else self.tax_rate_st
        tax_benefit = abs(realized_loss) * tax_rate if realized_loss < 0 else 0
        
        # Update lot
        lot.remaining_shares -= shares_to_sell
        if lot.remaining_shares <= 0:
            lot.sold_date = sell_date
        
        # Set wash sale block
        wash_sale_end = sell_date + timedelta(days=self.wash_sale_days)
        self.wash_sale_blocked[ticker] = wash_sale_end
        
        # Record harvest
        harvest_record = {
            'ticker': ticker,
            'lot_id': lot_id,
            'sell_date': sell_date,
            'shares_sold': shares_to_sell,
            'sell_price': sell_price,
            'cost_basis': lot.cost_basis,
            'proceeds': proceeds,
            'realized_loss': realized_loss,
            'tax_benefit': tax_benefit,
            'tax_rate': tax_rate,
            'is_long_term': is_long_term,
            'wash_sale_blocked_until': wash_sale_end
        }
        
        self.harvest_history.append(harvest_record)
        
        return harvest_record
    
    def find_replacement_securities(self,
                                   ticker: str,
                                   returns_df: pd.DataFrame,
                                   correlation_threshold: float = 0.90,
                                   n_candidates: int = 5) -> List[Dict]:
        """
        Find highly correlated replacement securities.
        
        Parameters:
        -----------
        ticker : str
            Security to replace
        returns_df : pd.DataFrame
            Returns for all securities
        correlation_threshold : float
            Minimum correlation to qualify
        n_candidates : int
            Maximum candidates to return
            
        Returns:
        --------
        list : Candidate replacements with correlations
        """
        if ticker not in returns_df.columns:
            return []
        
        # Calculate correlations
        correlations = returns_df.corr()[ticker].drop(ticker, errors='ignore')
        
        # Filter by threshold
        candidates = correlations[correlations >= correlation_threshold]
        candidates = candidates.sort_values(ascending=False)
        
        # Check wash sale restrictions
        valid_candidates = []
        current_date = datetime.now()
        
        for candidate_ticker, corr in candidates.items():
            # Skip if wash sale blocked
            if candidate_ticker in self.wash_sale_blocked:
                if current_date < self.wash_sale_blocked[candidate_ticker]:
                    continue
            
            # Skip if already a replacement in use
            if candidate_ticker in self.replacements.values():
                continue
            
            valid_candidates.append({
                'ticker': candidate_ticker,
                'correlation': corr,
                'is_same_sector': False  # Would need sector data
            })
            
            if len(valid_candidates) >= n_candidates:
                break
        
        return valid_candidates
    
    def execute_replacement(self,
                           sold_ticker: str,
                           replacement_ticker: str,
                           buy_date: datetime,
                           shares: float,
                           price: float) -> Dict:
        """
        Execute replacement purchase after harvest.
        
        Parameters:
        -----------
        sold_ticker : str
            Original security that was sold
        replacement_ticker : str
            Replacement security to buy
        buy_date : datetime
            Purchase date
        shares : float
            Number of shares
        price : float
            Price per share
            
        Returns:
        --------
        dict : Replacement execution details
        """
        # Record replacement
        self.replacements[sold_ticker] = replacement_ticker
        
        # Create new lot
        lot = self.add_purchase(
            ticker=replacement_ticker,
            date=buy_date,
            shares=shares,
            price=price
        )
        
        return {
            'original_ticker': sold_ticker,
            'replacement_ticker': replacement_ticker,
            'buy_date': buy_date,
            'shares': shares,
            'price': price,
            'total_cost': shares * price,
            'lot_id': lot.lot_id
        }
    
    def calculate_ytd_harvested_losses(self, year: int = None) -> Dict:
        """
        Calculate year-to-date harvested losses.
        
        Parameters:
        -----------
        year : int
            Year to calculate (default: current year)
            
        Returns:
        --------
        dict : Summary of YTD harvesting activity
        """
        if year is None:
            year = datetime.now().year
        
        ytd_harvests = [
            h for h in self.harvest_history 
            if h['sell_date'].year == year
        ]
        
        if not ytd_harvests:
            return {
                'year': year,
                'total_harvests': 0,
                'total_losses': 0,
                'total_tax_benefit': 0,
                'short_term_losses': 0,
                'long_term_losses': 0
            }
        
        total_losses = sum(h['realized_loss'] for h in ytd_harvests)
        total_tax_benefit = sum(h['tax_benefit'] for h in ytd_harvests)
        
        st_losses = sum(
            h['realized_loss'] for h in ytd_harvests 
            if not h['is_long_term']
        )
        lt_losses = sum(
            h['realized_loss'] for h in ytd_harvests 
            if h['is_long_term']
        )
        
        return {
            'year': year,
            'total_harvests': len(ytd_harvests),
            'total_losses': total_losses,
            'total_tax_benefit': total_tax_benefit,
            'short_term_losses': st_losses,
            'long_term_losses': lt_losses,
            'average_tax_rate': total_tax_benefit / abs(total_losses) if total_losses != 0 else 0
        }
    
    def calculate_tax_alpha(self,
                           portfolio_value: float,
                           years: float = 1.0) -> Dict:
        """
        Calculate annualized tax alpha from harvesting.
        
        Parameters:
        -----------
        portfolio_value : float
            Current portfolio value
        years : float
            Investment period in years
            
        Returns:
        --------
        dict : Tax alpha metrics
        """
        total_tax_benefit = sum(h['tax_benefit'] for h in self.harvest_history)
        
        # Annualized tax alpha
        annual_tax_benefit = total_tax_benefit / years if years > 0 else total_tax_benefit
        tax_alpha = annual_tax_benefit / portfolio_value if portfolio_value > 0 else 0
        
        return {
            'total_tax_benefit': total_tax_benefit,
            'annual_tax_benefit': annual_tax_benefit,
            'tax_alpha_bps': tax_alpha * 10000,  # Basis points
            'tax_alpha_pct': tax_alpha * 100,
            'n_harvests': len(self.harvest_history),
            'years': years
        }
    
    def check_wash_sale_status(self, ticker: str, date: datetime) -> Dict:
        """
        Check wash sale status for a ticker.
        
        Parameters:
        -----------
        ticker : str
            Security to check
        date : datetime
            Date to check
            
        Returns:
        --------
        dict : Wash sale status
        """
        date = pd.to_datetime(date)
        
        if ticker not in self.wash_sale_blocked:
            return {
                'ticker': ticker,
                'blocked': False,
                'blocked_until': None,
                'days_remaining': 0
            }
        
        blocked_until = self.wash_sale_blocked[ticker]
        is_blocked = date < blocked_until
        days_remaining = (blocked_until - date).days if is_blocked else 0
        
        return {
            'ticker': ticker,
            'blocked': is_blocked,
            'blocked_until': blocked_until,
            'days_remaining': max(0, days_remaining)
        }
    
    def get_harvest_summary(self) -> str:
        """Generate human-readable harvest summary."""
        summary = []
        summary.append("=" * 50)
        summary.append("TAX-LOSS HARVESTING SUMMARY")
        summary.append("=" * 50)
        
        if not self.harvest_history:
            summary.append("\nNo harvests executed yet.")
            return "\n".join(summary)
        
        # Overall stats
        total_losses = sum(h['realized_loss'] for h in self.harvest_history)
        total_benefit = sum(h['tax_benefit'] for h in self.harvest_history)
        
        summary.append(f"\nTotal Harvests: {len(self.harvest_history)}")
        summary.append(f"Total Losses Realized: ${total_losses:,.2f}")
        summary.append(f"Total Tax Benefit: ${total_benefit:,.2f}")
        
        # By year
        years = set(h['sell_date'].year for h in self.harvest_history)
        for year in sorted(years):
            ytd = self.calculate_ytd_harvested_losses(year)
            summary.append(f"\n{year}:")
            summary.append(f"  Harvests: {ytd['total_harvests']}")
            summary.append(f"  Losses: ${ytd['total_losses']:,.2f}")
            summary.append(f"  Tax Benefit: ${ytd['total_tax_benefit']:,.2f}")
        
        # Currently blocked tickers
        current_date = datetime.now()
        blocked = [
            (t, d) for t, d in self.wash_sale_blocked.items()
            if d > current_date
        ]
        
        if blocked:
            summary.append(f"\nWash Sale Blocked Securities:")
            for ticker, until in sorted(blocked, key=lambda x: x[1]):
                days = (until - current_date).days
                summary.append(f"  {ticker}: {days} days remaining")
        
        return "\n".join(summary)
    
    def optimize_harvest_schedule(self,
                                 current_prices: Dict[str, float],
                                 current_date: datetime,
                                 target_losses: float,
                                 returns_df: pd.DataFrame) -> List[Dict]:
        """
        Create optimized harvest schedule to achieve target losses.
        
        Parameters:
        -----------
        current_prices : dict
            Current prices
        current_date : datetime
            Current date
        target_losses : float
            Target loss amount to harvest
        returns_df : pd.DataFrame
            Returns for finding replacements
            
        Returns:
        --------
        list : Recommended harvest actions
        """
        opportunities = self.identify_harvest_opportunities(
            current_prices, current_date
        )
        
        if not opportunities:
            return []
        
        recommended = []
        cumulative_loss = 0
        
        for opp in opportunities:
            if cumulative_loss >= target_losses:
                break
            
            # Find replacement
            replacements = self.find_replacement_securities(
                opp['ticker'],
                returns_df
            )
            
            if replacements:
                replacement = replacements[0]
            else:
                replacement = None
            
            recommended.append({
                'action': 'harvest',
                'ticker': opp['ticker'],
                'lot_id': opp['lot_id'],
                'shares': opp['shares'],
                'loss_amount': opp['unrealized_loss'],
                'tax_benefit': opp['tax_benefit'],
                'replacement': replacement
            })
            
            cumulative_loss += abs(opp['unrealized_loss'])
        
        return recommended


def main():
    """Example usage of the tax-loss harvester."""
    harvester = TaxLossHarvester()
    
    # Simulate portfolio purchases
    print("Creating sample portfolio...")
    
    purchases = [
        ('AAPL', '2023-01-15', 100, 150.00),
        ('MSFT', '2023-03-01', 50, 340.00),
        ('GOOGL', '2023-06-10', 75, 135.00),
        ('AMZN', '2023-09-01', 30, 145.00),
        ('NVDA', '2024-01-15', 40, 550.00),
    ]
    
    for ticker, date, shares, price in purchases:
        harvester.add_purchase(ticker, date, shares, price)
    
    # Simulate current prices (some with losses)
    current_prices = {
        'AAPL': 145.00,   # Loss
        'MSFT': 380.00,   # Gain
        'GOOGL': 125.00,  # Loss
        'AMZN': 155.00,   # Gain
        'NVDA': 480.00,   # Loss
    }
    
    current_date = datetime(2024, 12, 1)
    
    # Find harvest opportunities
    print("\nIdentifying harvest opportunities...")
    opportunities = harvester.identify_harvest_opportunities(
        current_prices, current_date
    )
    
    print(f"\nFound {len(opportunities)} harvest opportunities:")
    for opp in opportunities:
        print(f"\n  {opp['ticker']}:")
        print(f"    Loss: ${opp['unrealized_loss']:,.2f} ({opp['unrealized_loss_pct']:.1%})")
        print(f"    Tax Benefit: ${opp['tax_benefit']:,.2f}")
        print(f"    Holding: {opp['holding_days']} days ({'LT' if opp['is_long_term'] else 'ST'})")
    
    # Execute a harvest
    if opportunities:
        print("\n" + "=" * 50)
        print("Executing top harvest...")
        opp = opportunities[0]
        
        result = harvester.execute_harvest(
            ticker=opp['ticker'],
            lot_id=opp['lot_id'],
            sell_date=current_date,
            sell_price=opp['current_price']
        )
        
        print(f"  Sold: {result['shares_sold']} shares of {result['ticker']}")
        print(f"  Realized Loss: ${result['realized_loss']:,.2f}")
        print(f"  Tax Benefit: ${result['tax_benefit']:,.2f}")
        print(f"  Wash Sale Block Until: {result['wash_sale_blocked_until'].date()}")
    
    # Print summary
    print("\n" + harvester.get_harvest_summary())


if __name__ == '__main__':
    main()
