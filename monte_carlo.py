"""
Monte Carlo Simulation for Options Analysis
Risk analysis, VaR, CVaR, and option pricing
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class MonteCarloSimulator:
    """Monte Carlo simulation for options and portfolio analysis"""
    
    def __init__(self, n_simulations: int = 10000, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator
        
        Args:
            n_simulations (int): Number of simulation paths
            random_seed (int): Random seed for reproducibility
        """
        self.n_simulations = n_simulations
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
    def price_option_mc(self, S0: float, K: float, r: float, sigma: float, 
                       T: float, option_type: str, n_steps: int = 252) -> Dict:
        """
        Price option using Monte Carlo simulation
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility
            T (float): Time to expiration (years)
            option_type (str): 'call' or 'put'
            n_steps (int): Number of time steps
            
        Returns:
            Dict: Simulation results
        """
        dt = T / n_steps
        
        # Generate random price paths
        price_paths = np.zeros((self.n_simulations, n_steps + 1))
        price_paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            # Generate random shocks
            Z = np.random.standard_normal(self.n_simulations)
            
            # Update prices using geometric Brownian motion
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
        
        # Calculate payoffs
        final_prices = price_paths[:, -1]
        
        if option_type.lower() == 'call':
            payoffs = np.maximum(final_prices - K, 0)
        else:  # put
            payoffs = np.maximum(K - final_prices, 0)
        
        # Discount payoffs to present value
        option_price = np.exp(-r * T) * np.mean(payoffs)
        
        # Calculate statistics
        std_error = np.std(payoffs) / np.sqrt(self.n_simulations)
        confidence_interval = 1.96 * std_error
        
        return {
            'option_price': option_price,
            'payoffs': payoffs,
            'price_paths': price_paths,
            'std_error': std_error,
            'confidence_interval': confidence_interval,
            'max_payoff': np.max(payoffs),
            'min_payoff': np.min(payoffs),
            'mean_payoff': np.mean(payoffs)
        }
    
    def calculate_var_cvar(self, returns: np.ndarray, confidence_level: float = 0.05) -> Dict:
        """
        Calculate Value at Risk (VaR) and Conditional Value at Risk (CVaR)
        
        Args:
            returns (np.ndarray): Array of returns or payoffs
            confidence_level (float): Confidence level (e.g., 0.05 for 95% VaR)
            
        Returns:
            Dict: VaR and CVaR results
        """
        # Sort returns in ascending order
        sorted_returns = np.sort(returns)
        
        # Calculate VaR
        var_index = int(confidence_level * len(sorted_returns))
        var = sorted_returns[var_index]
        
        # Calculate CVaR (expected shortfall)
        cvar = np.mean(sorted_returns[:var_index])
        
        return {
            'var': var,
            'cvar': cvar,
            'confidence_level': confidence_level,
            'percentile': confidence_level * 100
        }
    
    def greeks_mc(self, S0: float, K: float, r: float, sigma: float, 
                  T: float, option_type: str, bump: float = 0.01) -> Dict:
        """
        Calculate Greeks using Monte Carlo simulation
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility
            T (float): Time to expiration
            option_type (str): 'call' or 'put'
            bump (float): Bump size for numerical derivatives
            
        Returns:
            Dict: Greeks values
        """
        # Base case
        base_result = self.price_option_mc(S0, K, r, sigma, T, option_type)
        base_price = base_result['option_price']
        
        # Delta: sensitivity to underlying price
        delta_result = self.price_option_mc(S0 + bump, K, r, sigma, T, option_type)
        delta = (delta_result['option_price'] - base_price) / bump
        
        # Gamma: second derivative with respect to price
        gamma_result = self.price_option_mc(S0 - bump, K, r, sigma, T, option_type)
        gamma = (delta_result['option_price'] - 2 * base_price + gamma_result['option_price']) / (bump**2)
        
        # Theta: sensitivity to time
        theta_result = self.price_option_mc(S0, K, r, sigma, T - bump/365, option_type)
        theta = (theta_result['option_price'] - base_price) / (bump/365)
        
        # Vega: sensitivity to volatility
        vega_result = self.price_option_mc(S0, K, r, sigma + bump, T, option_type)
        vega = (vega_result['option_price'] - base_price) / bump
        
        # Rho: sensitivity to interest rate
        rho_result = self.price_option_mc(S0, K, r + bump, sigma, T, option_type)
        rho = (rho_result['option_price'] - base_price) / bump
        
        return {
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }
    
    def portfolio_mc_analysis(self, positions: List[Dict], S0: float, r: float, 
                             sigma: float, T: float, n_steps: int = 252) -> Dict:
        """
        Monte Carlo analysis for a portfolio of options
        
        Args:
            positions (List[Dict]): List of position dictionaries
            S0 (float): Current underlying price
            r (float): Risk-free rate
            sigma (float): Volatility
            T (float): Time to expiration
            n_steps (int): Number of time steps
            
        Returns:
            Dict: Portfolio analysis results
        """
        dt = T / n_steps
        
        # Generate price paths
        price_paths = np.zeros((self.n_simulations, n_steps + 1))
        price_paths[:, 0] = S0
        
        for t in range(1, n_steps + 1):
            Z = np.random.standard_normal(self.n_simulations)
            price_paths[:, t] = price_paths[:, t-1] * np.exp(
                (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z
            )
        
        # Calculate portfolio payoffs for each path
        portfolio_payoffs = np.zeros(self.n_simulations)
        
        for position in positions:
            strike = position['strike']
            quantity = position['quantity']
            option_type = position['option_type']
            
            # Calculate position payoff
            if option_type.lower() == 'call':
                position_payoff = np.maximum(price_paths[:, -1] - strike, 0)
            else:  # put
                position_payoff = np.maximum(strike - price_paths[:, -1], 0)
            
            portfolio_payoffs += quantity * position_payoff
        
        # Discount to present value
        portfolio_value = np.exp(-r * T) * portfolio_payoffs
        
        # Calculate risk metrics
        var_95 = self.calculate_var_cvar(portfolio_value, 0.05)
        var_99 = self.calculate_var_cvar(portfolio_value, 0.01)
        
        return {
            'portfolio_value': portfolio_value,
            'mean_value': np.mean(portfolio_value),
            'std_value': np.std(portfolio_value),
            'var_95': var_95,
            'var_99': var_99,
            'price_paths': price_paths,
            'max_value': np.max(portfolio_value),
            'min_value': np.min(portfolio_value)
        }
    
    def stress_test(self, S0: float, K: float, r: float, sigma: float, 
                   T: float, option_type: str, stress_scenarios: Dict) -> Dict:
        """
        Stress test option under various market scenarios
        
        Args:
            S0 (float): Current stock price
            K (float): Strike price
            r (float): Risk-free rate
            sigma (float): Volatility
            T (float): Time to expiration
            option_type (str): 'call' or 'put'
            stress_scenarios (Dict): Dictionary of stress scenarios
            
        Returns:
            Dict: Stress test results
        """
        results = {}
        
        for scenario_name, scenario_params in stress_scenarios.items():
            stressed_S0 = S0 * scenario_params.get('price_shock', 1.0)
            stressed_sigma = sigma * scenario_params.get('vol_shock', 1.0)
            stressed_r = r + scenario_params.get('rate_shock', 0.0)
            
            result = self.price_option_mc(
                stressed_S0, K, stressed_r, stressed_sigma, T, option_type
            )
            
            results[scenario_name] = {
                'option_price': result['option_price'],
                'mean_payoff': result['mean_payoff'],
                'max_payoff': result['max_payoff'],
                'scenario_params': scenario_params
            }
        
        return results


def main():
    """Example usage of Monte Carlo simulator"""
    print("Monte Carlo Options Simulator")
    print("=" * 40)
    
    # Initialize simulator
    simulator = MonteCarloSimulator(n_simulations=10000)
    
    # Example parameters
    S0 = 100  # Current stock price
    K = 100   # Strike price
    r = 0.05  # Risk-free rate
    sigma = 0.2  # Volatility
    T = 0.25  # Time to expiration (3 months)
    
    # Price a call option
    print("Pricing Call Option...")
    call_result = simulator.price_option_mc(S0, K, r, sigma, T, 'call')
    print(f"Call Option Price: ${call_result['option_price']:.4f}")
    print(f"Standard Error: ${call_result['std_error']:.4f}")
    print(f"95% Confidence Interval: ±${call_result['confidence_interval']:.4f}")
    
    # Price a put option
    print("\nPricing Put Option...")
    put_result = simulator.price_option_mc(S0, K, r, sigma, T, 'put')
    print(f"Put Option Price: ${put_result['option_price']:.4f}")
    
    # Calculate Greeks
    print("\nCalculating Greeks...")
    greeks = simulator.greeks_mc(S0, K, r, sigma, T, 'call')
    print(f"Delta: {greeks['delta']:.4f}")
    print(f"Gamma: {greeks['gamma']:.4f}")
    print(f"Theta: {greeks['theta']:.4f}")
    print(f"Vega: {greeks['vega']:.4f}")
    print(f"Rho: {greeks['rho']:.4f}")
    
    # Risk analysis
    print("\nRisk Analysis...")
    var_95 = simulator.calculate_var_cvar(call_result['payoffs'], 0.05)
    var_99 = simulator.calculate_var_cvar(call_result['payoffs'], 0.01)
    
    print(f"95% VaR: ${var_95['var']:.2f}")
    print(f"99% VaR: ${var_99['var']:.2f}")
    print(f"95% CVaR: ${var_95['cvar']:.2f}")
    
    # Stress testing
    print("\nStress Testing...")
    stress_scenarios = {
        'market_crash': {'price_shock': 0.7, 'vol_shock': 2.0},
        'market_rally': {'price_shock': 1.3, 'vol_shock': 0.8},
        'vol_spike': {'price_shock': 1.0, 'vol_shock': 2.5},
        'rate_increase': {'price_shock': 1.0, 'vol_shock': 1.0, 'rate_shock': 0.02}
    }
    
    stress_results = simulator.stress_test(S0, K, r, sigma, T, 'call', stress_scenarios)
    
    for scenario, result in stress_results.items():
        print(f"{scenario}: ${result['option_price']:.4f}")


if __name__ == "__main__":
    main()
