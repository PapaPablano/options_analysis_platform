"""
Options Data Analyzer
Core module for fetching and analyzing options data
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class OptionsAnalyzer:
    """Main options analysis class"""
    
    def __init__(self, ticker):
        """
        Initialize options analyzer
        
        Args:
            ticker (str): Stock ticker symbol
        """
        self.ticker = ticker.upper()
        self.stock = yf.Ticker(self.ticker)
        self.current_price = None
        self.options_data = {}
        
    def fetch_stock_info(self):
        """Fetch basic stock information"""
        try:
            info = self.stock.info
            self.current_price = info.get('currentPrice', info.get('regularMarketPrice', 0))
            
            if self.current_price == 0:
                # Fallback to recent price
                hist = self.stock.history(period='1d')
                if not hist.empty:
                    self.current_price = hist['Close'].iloc[-1]
            
            return self.current_price > 0
        except Exception as e:
            print(f"Error fetching stock info: {e}")
            return False
    
    def get_expiration_dates(self):
        """Get available expiration dates"""
        try:
            expirations = self.stock.options
            return sorted(expirations) if expirations else []
        except Exception as e:
            print(f"Error fetching expiration dates: {e}")
            return []
    
    def fetch_options_chain(self, expiration_date):
        """Fetch options chain for specific expiration"""
        try:
            opt_chain = self.stock.option_chain(expiration_date)
            
            # Combine calls and puts
            calls = opt_chain.calls.copy()
            puts = opt_chain.puts.copy()
            
            calls['optionType'] = 'call'
            puts['optionType'] = 'put'
            
            # Combine and clean data
            options_df = pd.concat([calls, puts], ignore_index=True)
            
            # Clean column names
            options_df.columns = options_df.columns.str.replace(' ', '')
            
            # Store in options_data
            self.options_data[expiration_date] = options_df
            
            return True
        except Exception as e:
            print(f"Error fetching options chain: {e}")
            return False
    
    def calculate_greeks(self, expiration_date):
        """Calculate Greeks for options"""
        if expiration_date not in self.options_data:
            return False
        
        try:
            df = self.options_data[expiration_date].copy()
            
            # Calculate time to expiration
            exp_date = datetime.strptime(expiration_date, '%Y-%m-%d')
            today = datetime.now()
            T = (exp_date - today).days / 365.0
            
            # Risk-free rate (simplified)
            risk_free_rate = 0.05
            
            # Calculate Greeks using Black-Scholes approximation
            for idx, row in df.iterrows():
                S = self.current_price
                K = row['strike']
                r = risk_free_rate
                T = max(T, 0.001)  # Avoid division by zero
                sigma = row['impliedVolatility'] if pd.notna(row['impliedVolatility']) else 0.2
                
                # Black-Scholes Greeks calculation
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                
                # Delta
                if row['optionType'] == 'call':
                    delta = np.exp(-r*T) * self._norm_cdf(d1)
                else:  # put
                    delta = -np.exp(-r*T) * self._norm_cdf(-d1)
                
                # Gamma
                gamma = np.exp(-r*T) * self._norm_pdf(d1) / (S * sigma * np.sqrt(T))
                
                # Theta
                if row['optionType'] == 'call':
                    theta = (-S * self._norm_pdf(d1) * sigma / (2*np.sqrt(T)) 
                            - r*K*np.exp(-r*T)*self._norm_cdf(d2)) / 365
                else:  # put
                    theta = (-S * self._norm_pdf(d1) * sigma / (2*np.sqrt(T)) 
                            + r*K*np.exp(-r*T)*self._norm_cdf(-d2)) / 365
                
                # Vega
                vega = S * np.sqrt(T) * self._norm_pdf(d1) / 100
                
                # Rho
                if row['optionType'] == 'call':
                    rho = K * T * np.exp(-r*T) * self._norm_cdf(d2) / 100
                else:  # put
                    rho = -K * T * np.exp(-r*T) * self._norm_cdf(-d2) / 100
                
                # Store Greeks
                df.at[idx, 'delta'] = delta
                df.at[idx, 'gamma'] = gamma
                df.at[idx, 'theta'] = theta
                df.at[idx, 'vega'] = vega
                df.at[idx, 'rho'] = rho
            
            self.options_data[expiration_date] = df
            return True
            
        except Exception as e:
            print(f"Error calculating Greeks: {e}")
            return False
    
    def _norm_cdf(self, x):
        """Cumulative distribution function of standard normal"""
        return 0.5 * (1 + np.sign(x) * np.sqrt(1 - np.exp(-2 * x**2 / np.pi)))
    
    def _norm_pdf(self, x):
        """Probability density function of standard normal"""
        return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)
    
    def analyze_volatility(self, expiration_date):
        """Analyze implied volatility patterns"""
        if expiration_date not in self.options_data:
            return None
        
        df = self.options_data[expiration_date]
        
        # Filter out options with missing IV
        df_iv = df.dropna(subset=['impliedVolatility'])
        
        if df_iv.empty:
            return None
        
        # Overall statistics
        overall_stats = {
            'mean_iv': df_iv['impliedVolatility'].mean() * 100,
            'median_iv': df_iv['impliedVolatility'].median() * 100,
            'std_iv': df_iv['impliedVolatility'].std() * 100,
            'min_iv': df_iv['impliedVolatility'].min() * 100,
            'max_iv': df_iv['impliedVolatility'].max() * 100
        }
        
        # Calls vs Puts
        calls = df_iv[df_iv['optionType'] == 'call']
        puts = df_iv[df_iv['optionType'] == 'put']
        
        calls_stats = {
            'count': len(calls),
            'mean_iv': calls['impliedVolatility'].mean() * 100 if len(calls) > 0 else 0
        }
        
        puts_stats = {
            'count': len(puts),
            'mean_iv': puts['impliedVolatility'].mean() * 100 if len(puts) > 0 else 0
        }
        
        return {
            'overall_stats': overall_stats,
            'calls_stats': calls_stats,
            'puts_stats': puts_stats
        }
    
    def analyze_strategies(self, expiration_date):
        """Analyze common options strategies"""
        if expiration_date not in self.options_data:
            return None
        
        df = self.options_data[expiration_date]
        strategies = {}
        
        # Covered Call Analysis
        calls = df[df['optionType'] == 'call']
        if len(calls) > 0:
            # Find ATM call
            atm_calls = calls[np.abs(calls['strike'] - self.current_price) == 
                            np.abs(calls['strike'] - self.current_price).min()]
            
            if len(atm_calls) > 0:
                atm_call = atm_calls.iloc[0]
                premium = atm_call['lastPrice']
                strike = atm_call['strike']
                
                # Calculate metrics
                max_profit = (strike - self.current_price) + premium
                days_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days
                annualized_return = (premium / self.current_price) * (365 / max(days_to_exp, 1)) * 100
                
                strategies['covered_call'] = {
                    'strike': strike,
                    'premium': premium,
                    'max_profit': max_profit,
                    'annualized_return': annualized_return
                }
        
        # Cash Secured Put Analysis
        puts = df[df['optionType'] == 'put']
        if len(puts) > 0:
            # Find ATM put
            atm_puts = puts[np.abs(puts['strike'] - self.current_price) == 
                          np.abs(puts['strike'] - self.current_price).min()]
            
            if len(atm_puts) > 0:
                atm_put = atm_puts.iloc[0]
                premium = atm_put['lastPrice']
                strike = atm_put['strike']
                
                # Calculate metrics
                max_profit = premium
                days_to_exp = (datetime.strptime(expiration_date, '%Y-%m-%d') - datetime.now()).days
                annualized_return = (premium / strike) * (365 / max(days_to_exp, 1)) * 100
                
                strategies['cash_secured_put'] = {
                    'strike': strike,
                    'premium': premium,
                    'max_profit': max_profit,
                    'annualized_return': annualized_return
                }
        
        return strategies


def main():
    """Example usage"""
    print("Options Analyzer - Example Usage")
    print("=" * 40)
    
    # Initialize analyzer
    analyzer = OptionsAnalyzer("AAPL")
    
    # Fetch stock info
    if analyzer.fetch_stock_info():
        print(f"Current Price: ${analyzer.current_price:.2f}")
        
        # Get expiration dates
        expirations = analyzer.get_expiration_dates()
        print(f"Available Expirations: {len(expirations)}")
        
        if expirations:
            # Analyze first expiration
            expiration = expirations[0]
            print(f"Analyzing: {expiration}")
            
            if analyzer.fetch_options_chain(expiration):
                print("Options chain loaded successfully")
                
                if analyzer.calculate_greeks(expiration):
                    print("Greeks calculated successfully")
                
                # Analyze volatility
                vol_analysis = analyzer.analyze_volatility(expiration)
                if vol_analysis:
                    print(f"Mean IV: {vol_analysis['overall_stats']['mean_iv']:.2f}%")
                
                # Analyze strategies
                strategies = analyzer.analyze_strategies(expiration)
                if strategies:
                    print(f"Found {len(strategies)} strategies")
            else:
                print("Failed to load options chain")
    else:
        print("Failed to fetch stock information")


if __name__ == "__main__":
    main()
