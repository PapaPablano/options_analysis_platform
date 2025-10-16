"""
Portfolio Analysis for Options Trading
Multi-position management, Greeks aggregation, and risk analysis
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

from monte_carlo import MonteCarloSimulator


class OptionsPortfolio:
    """Options portfolio management and analysis"""
    
    def __init__(self, name: str = "Options Portfolio"):
        """
        Initialize options portfolio
        
        Args:
            name (str): Portfolio name
        """
        self.name = name
        self.positions = []
        self.underlying_price = 0
        self.risk_free_rate = 0.05
        self.volatility = 0.2
        
    def add_position(self, symbol: str, option_type: str, strike: float, 
                    expiration: str, quantity: int, premium_paid: float = 0):
        """
        Add a position to the portfolio
        
        Args:
            symbol (str): Underlying symbol
            option_type (str): 'call' or 'put'
            strike (float): Strike price
            expiration (str): Expiration date
            quantity (int): Number of contracts (positive for long, negative for short)
            premium_paid (float): Premium paid/received for the position
        """
        position = {
            'symbol': symbol,
            'option_type': option_type,
            'strike': strike,
            'expiration': expiration,
            'quantity': quantity,
            'premium_paid': premium_paid,
            'added_at': datetime.now().isoformat()
        }
        
        self.positions.append(position)
        print(f"Added position: {quantity} {option_type} contracts at ${strike} strike")
    
    def remove_position(self, position_index: int):
        """Remove a position from the portfolio"""
        if 0 <= position_index < len(self.positions):
            removed = self.positions.pop(position_index)
            print(f"Removed position: {removed['option_type']} at ${removed['strike']}")
            return True
        return False
    
    def get_portfolio_summary(self) -> Dict:
        """Get portfolio summary statistics"""
        if not self.positions:
            return {
                'total_positions': 0,
                'total_contracts': 0,
                'net_calls': 0,
                'net_puts': 0,
                'total_premium_paid': 0
            }
        
        total_contracts = sum(abs(pos['quantity']) for pos in self.positions)
        net_calls = sum(pos['quantity'] for pos in self.positions if pos['option_type'] == 'call')
        net_puts = sum(pos['quantity'] for pos in self.positions if pos['option_type'] == 'put')
        total_premium = sum(pos['premium_paid'] * pos['quantity'] for pos in self.positions)
        
        return {
            'total_positions': len(self.positions),
            'total_contracts': total_contracts,
            'net_calls': net_calls,
            'net_puts': net_puts,
            'total_premium_paid': total_premium
        }
    
    def calculate_portfolio_greeks(self, current_price: float, time_to_exp: float) -> Dict:
        """
        Calculate aggregate portfolio Greeks
        
        Args:
            current_price (float): Current underlying price
            time_to_exp (float): Time to expiration in years
            
        Returns:
            Dict: Portfolio Greeks
        """
        if not self.positions:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        portfolio_delta = 0
        portfolio_gamma = 0
        portfolio_theta = 0
        portfolio_vega = 0
        portfolio_rho = 0
        
        for position in self.positions:
            # Calculate individual option Greeks (simplified Black-Scholes)
            greeks = self._calculate_option_greeks(
                current_price, position['strike'], self.risk_free_rate,
                self.volatility, time_to_exp, position['option_type']
            )
            
            # Aggregate Greeks (multiply by quantity and 100 for contract multiplier)
            multiplier = position['quantity'] * 100
            
            portfolio_delta += greeks['delta'] * multiplier
            portfolio_gamma += greeks['gamma'] * multiplier
            portfolio_theta += greeks['theta'] * multiplier
            portfolio_vega += greeks['vega'] * multiplier
            portfolio_rho += greeks['rho'] * multiplier
        
        return {
            'delta': portfolio_delta,
            'gamma': portfolio_gamma,
            'theta': portfolio_theta,
            'vega': portfolio_vega,
            'rho': portfolio_rho
        }
    
    def _calculate_option_greeks(self, S: float, K: float, r: float, sigma: float, 
                                T: float, option_type: str) -> Dict:
        """Calculate Greeks for a single option using Black-Scholes"""
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        # Black-Scholes parameters
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
        d2 = d1 - sigma*np.sqrt(T)
        
        # Standard normal CDF and PDF approximations
        def norm_cdf(x):
            return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
        
        def norm_pdf(x):
            return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
        
        # Calculate Greeks
        if option_type.lower() == 'call':
            delta = np.exp(-r*T) * norm_cdf(d1)
            theta = (-S * norm_pdf(d1) * sigma / (2*np.sqrt(T)) 
                    - r*K*np.exp(-r*T)*norm_cdf(d2)) / 365
            rho = K * T * np.exp(-r*T) * norm_cdf(d2) / 100
        else:  # put
            delta = -np.exp(-r*T) * norm_cdf(-d1)
            theta = (-S * norm_pdf(d1) * sigma / (2*np.sqrt(T)) 
                    + r*K*np.exp(-r*T)*norm_cdf(-d2)) / 365
            rho = -K * T * np.exp(-r*T) * norm_cdf(-d2) / 100
        
        gamma = np.exp(-r*T) * norm_pdf(d1) / (S * sigma * np.sqrt(T))
        vega = S * np.sqrt(T) * norm_pdf(d1) / 100
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def calculate_portfolio_payoff(self, underlying_prices: np.ndarray, time_to_exp: float) -> np.ndarray:
        """
        Calculate portfolio payoff for different underlying prices
        
        Args:
            underlying_prices (np.ndarray): Array of underlying prices
            time_to_exp (float): Time to expiration in years
            
        Returns:
            np.ndarray: Portfolio payoffs
        """
        if not self.positions:
            return np.zeros_like(underlying_prices)
        
        portfolio_payoffs = np.zeros_like(underlying_prices)
        
        for position in self.positions:
            strike = position['strike']
            quantity = position['quantity']
            option_type = position['option_type']
            premium = position['premium_paid']
            
            # Calculate option payoff
            if option_type.lower() == 'call':
                option_payoff = np.maximum(underlying_prices - strike, 0)
            else:  # put
                option_payoff = np.maximum(strike - underlying_prices, 0)
            
            # Add to portfolio (multiply by quantity and 100 for contract size)
            position_payoff = quantity * (option_payoff - premium) * 100
            portfolio_payoffs += position_payoff
        
        return portfolio_payoffs
    
    def monte_carlo_analysis(self, n_simulations: int = 10000) -> Dict:
        """
        Run Monte Carlo analysis on the portfolio
        
        Args:
            n_simulations (int): Number of simulations
            
        Returns:
            Dict: Monte Carlo results
        """
        if not self.positions:
            return {'error': 'No positions in portfolio'}
        
        # Prepare positions for Monte Carlo
        mc_positions = []
        for pos in self.positions:
            mc_positions.append({
                'strike': pos['strike'],
                'quantity': pos['quantity'],
                'option_type': pos['option_type']
            })
        
        # Run Monte Carlo simulation
        simulator = MonteCarloSimulator(n_simulations=n_simulations)
        
        # Use average time to expiration
        avg_time_to_exp = 0.25  # Default to 3 months
        
        result = simulator.portfolio_mc_analysis(
            mc_positions, self.underlying_price, self.risk_free_rate,
            self.volatility, avg_time_to_exp
        )
        
        return {
            'mean_payoff': result['mean_value'],
            'std_payoff': result['std_value'],
            'var_95': result['var_95'],
            'var_99': result['var_99'],
            'max_payoff': result['max_value'],
            'min_payoff': result['min_value']
        }
    
    def plot_portfolio_payoff(self, price_range: Tuple[float, float] = None, 
                             n_points: int = 100):
        """
        Plot portfolio payoff diagram
        
        Args:
            price_range (Tuple[float, float]): Price range for analysis
            n_points (int): Number of points to plot
        """
        if not self.positions:
            print("No positions to plot")
            return
        
        if price_range is None:
            # Default price range based on strikes
            strikes = [pos['strike'] for pos in self.positions]
            min_strike = min(strikes)
            max_strike = max(strikes)
            price_range = (min_strike * 0.7, max_strike * 1.3)
        
        # Generate price array
        prices = np.linspace(price_range[0], price_range[1], n_points)
        
        # Calculate payoffs
        payoffs = self.calculate_portfolio_payoff(prices, 0)
        
        # Create plot
        plt.figure(figsize=(12, 8))
        plt.plot(prices, payoffs, 'b-', linewidth=2, label='Portfolio Payoff')
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        plt.axvline(x=self.underlying_price, color='red', linestyle='--', 
                   alpha=0.7, label=f'Current Price: ${self.underlying_price:.2f}')
        
        plt.title(f'Portfolio Payoff Diagram - {self.name}')
        plt.xlabel('Underlying Price')
        plt.ylabel('Portfolio Payoff')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
        
        return payoffs


class PortfolioAnalyzer:
    """Advanced portfolio analysis tools"""
    
    def __init__(self, portfolio: OptionsPortfolio):
        """
        Initialize portfolio analyzer
        
        Args:
            portfolio (OptionsPortfolio): Portfolio to analyze
        """
        self.portfolio = portfolio
        
    def analyze_risk_metrics(self, current_price: float, time_to_exp: float) -> Dict:
        """
        Analyze portfolio risk metrics
        
        Args:
            current_price (float): Current underlying price
            time_to_exp (float): Time to expiration
            
        Returns:
            Dict: Risk analysis results
        """
        greeks = self.portfolio.calculate_portfolio_greeks(current_price, time_to_exp)
        
        # Calculate risk metrics
        delta_exposure = greeks['delta'] * current_price
        gamma_exposure = greeks['gamma'] * (current_price ** 2)
        theta_daily = greeks['theta']
        vega_exposure = greeks['vega'] * 0.01  # 1% vol change
        
        return {
            'greeks': greeks,
            'delta_exposure': delta_exposure,
            'gamma_exposure': gamma_exposure,
            'theta_daily': theta_daily,
            'vega_exposure': vega_exposure,
            'risk_summary': self._generate_risk_summary(greeks)
        }
    
    def _generate_risk_summary(self, greeks: Dict) -> str:
        """Generate human-readable risk summary"""
        delta = greeks['delta']
        gamma = greeks['gamma']
        theta = greeks['theta']
        vega = greeks['vega']
        
        summary = []
        
        if abs(delta) > 100:
            direction = "bullish" if delta > 0 else "bearish"
            summary.append(f"High {direction} exposure (Delta: {delta:.0f})")
        
        if abs(gamma) > 50:
            summary.append(f"High gamma risk (Gamma: {gamma:.0f})")
        
        if theta < -50:
            summary.append(f"High time decay (Theta: {theta:.0f})")
        
        if abs(vega) > 100:
            vol_sensitivity = "high" if vega > 0 else "low"
            summary.append(f"{vol_sensitivity.capitalize()} volatility sensitivity (Vega: {vega:.0f})")
        
        return "; ".join(summary) if summary else "Moderate risk profile"
    
    def optimize_portfolio(self, target_delta: float = 0, max_positions: int = 10) -> Dict:
        """
        Suggest portfolio optimizations
        
        Args:
            target_delta (float): Target portfolio delta
            max_positions (int): Maximum number of positions
            
        Returns:
            Dict: Optimization suggestions
        """
        current_greeks = self.portfolio.calculate_portfolio_greeks(
            self.portfolio.underlying_price, 0.25
        )
        
        current_delta = current_greeks['delta']
        delta_drift = target_delta - current_delta
        
        suggestions = []
        
        if abs(delta_drift) > 50:
            if delta_drift > 0:
                suggestions.append("Consider adding long call positions or reducing put positions")
            else:
                suggestions.append("Consider adding long put positions or reducing call positions")
        
        if current_greeks['gamma'] > 100:
            suggestions.append("High gamma risk - consider reducing position sizes")
        
        if current_greeks['theta'] < -100:
            suggestions.append("High time decay - consider shorter-term strategies")
        
        return {
            'current_delta': current_delta,
            'target_delta': target_delta,
            'delta_drift': delta_drift,
            'suggestions': suggestions,
            'current_greeks': current_greeks
        }


def main():
    """Example usage of portfolio analyzer"""
    print("Options Portfolio Analyzer")
    print("=" * 40)
    
    # Create portfolio
    portfolio = OptionsPortfolio("Test Portfolio")
    portfolio.underlying_price = 100
    portfolio.risk_free_rate = 0.05
    portfolio.volatility = 0.2
    
    # Add some positions
    portfolio.add_position("AAPL", "call", 105, "2024-03-15", 1, 2.50)
    portfolio.add_position("AAPL", "put", 95, "2024-03-15", -1, 1.80)
    portfolio.add_position("AAPL", "call", 110, "2024-03-15", 2, 1.20)
    
    # Portfolio summary
    summary = portfolio.get_portfolio_summary()
    print(f"Portfolio Summary:")
    print(f"  Total Positions: {summary['total_positions']}")
    print(f"  Total Contracts: {summary['total_contracts']}")
    print(f"  Net Calls: {summary['net_calls']}")
    print(f"  Net Puts: {summary['net_puts']}")
    
    # Calculate Greeks
    greeks = portfolio.calculate_portfolio_greeks(100, 0.25)
    print(f"\nPortfolio Greeks:")
    print(f"  Delta: {greeks['delta']:.2f}")
    print(f"  Gamma: {greeks['gamma']:.2f}")
    print(f"  Theta: {greeks['theta']:.2f}")
    print(f"  Vega: {greeks['vega']:.2f}")
    print(f"  Rho: {greeks['rho']:.2f}")
    
    # Risk analysis
    analyzer = PortfolioAnalyzer(portfolio)
    risk_metrics = analyzer.analyze_risk_metrics(100, 0.25)
    print(f"\nRisk Summary: {risk_metrics['risk_summary']}")
    
    # Monte Carlo analysis
    print(f"\nRunning Monte Carlo Analysis...")
    mc_results = portfolio.monte_carlo_analysis()
    print(f"  Mean Payoff: ${mc_results['mean_payoff']:.2f}")
    print(f"  Std Payoff: ${mc_results['std_payoff']:.2f}")
    print(f"  95% VaR: ${mc_results['var_95']['var']:.2f}")
    print(f"  99% VaR: ${mc_results['var_99']['var']:.2f}")


if __name__ == "__main__":
    main()
