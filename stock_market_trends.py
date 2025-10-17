"""
Stock Market Trends Analysis
Comprehensive ML models for price forecasting and volatility modeling
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

# Time series libraries
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model

# Deep learning
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# XGBoost
import xgboost as xgb

# Data fetching
import yfinance as yf

# Plotting
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px


class StockMarketTrendsAnalyzer:
    """
    Comprehensive stock market trends analysis with multiple ML models
    """
    
    def __init__(self, ticker: str, lookback_days: int = 252*2):
        """
        Initialize the analyzer
        
        Args:
            ticker (str): Stock ticker symbol
            lookback_days (int): Number of days to look back for training data
        """
        self.ticker = ticker.upper()
        self.lookback_days = lookback_days
        self.data = None
        self.features = None
        self.target = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
    def fetch_data(self, start_date: str = None, end_date: str = None) -> bool:
        """
        Fetch historical stock data
        
        Args:
            start_date (str): Start date in YYYY-MM-DD format
            end_date (str): End date in YYYY-MM-DD format
            
        Returns:
            bool: Success status
        """
        try:
            if start_date is None:
                start_date = (datetime.now() - timedelta(days=self.lookback_days)).strftime('%Y-%m-%d')
            if end_date is None:
                end_date = datetime.now().strftime('%Y-%m-%d')
            
            # Create ticker with custom headers to avoid browser impersonation issues
            import requests
            session = requests.Session()
            session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            stock = yf.Ticker(self.ticker, session=session)
            self.data = stock.history(start=start_date, end=end_date)
            
            if self.data.empty:
                print(f"No data found for {self.ticker}")
                return False
            
            # Clean data
            self.data = self.data.dropna()
            print(f"Fetched {len(self.data)} days of data for {self.ticker}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            # Try alternative approach
            try:
                stock = yf.Ticker(self.ticker)
                self.data = stock.history(period="2y")  # Use period instead of dates
                if not self.data.empty:
                    self.data = self.data.dropna()
                    print(f"Fetched {len(self.data)} days of data for {self.ticker} (fallback method)")
                    return True
            except Exception as e2:
                print(f"Fallback also failed: {e2}")
            return False
    
    def create_features(self, lags: int = 20, technical_indicators: bool = True) -> pd.DataFrame:
        """
        Create comprehensive feature set
        
        Args:
            lags (int): Number of lag features to create
            technical_indicators (bool): Whether to include technical indicators
            
        Returns:
            pd.DataFrame: Feature matrix
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call fetch_data() first.")
        
        df = self.data.copy()
        
        # Price-based features
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Lag features
        for i in range(1, lags + 1):
            df[f'close_lag_{i}'] = df['Close'].shift(i)
            df[f'volume_lag_{i}'] = df['Volume'].shift(i)
            df[f'returns_lag_{i}'] = df['returns'].shift(i)
        
        # Technical indicators
        if technical_indicators:
            # Moving averages
            df['sma_5'] = df['Close'].rolling(window=5).mean()
            df['sma_20'] = df['Close'].rolling(window=20).mean()
            df['sma_50'] = df['Close'].rolling(window=50).mean()
            df['ema_12'] = df['Close'].ewm(span=12).mean()
            df['ema_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['macd'] = df['ema_12'] - df['ema_26']
            df['macd_signal'] = df['macd'].ewm(span=9).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
            
            # RSI
            delta = df['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # Bollinger Bands
            df['bb_middle'] = df['Close'].rolling(window=20).mean()
            bb_std = df['Close'].rolling(window=20).std()
            df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
            df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_position'] = (df['Close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volume indicators
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
            
            # Price patterns
            df['high_low_ratio'] = df['High'] / df['Low']
            df['close_open_ratio'] = df['Close'] / df['Open']
            
            # Volatility indicators
            df['atr'] = self._calculate_atr(df)
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(window=50).mean()
        
        # Market regime features
        df['trend_5'] = (df['Close'] > df['sma_5']).astype(int)
        df['trend_20'] = (df['Close'] > df['sma_20']).astype(int)
        df['trend_50'] = (df['Close'] > df['sma_50']).astype(int)
        
        # Time-based features
        df['day_of_week'] = df.index.dayofweek
        df['month'] = df.index.month
        df['quarter'] = df.index.quarter
        
        # Drop rows with NaN values
        df = df.dropna()
        
        self.features = df
        return df
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        return true_range.rolling(window=period).mean()
    
    def prepare_target(self, target_type: str = 'returns', horizon: int = 1) -> pd.Series:
        """
        Prepare target variable
        
        Args:
            target_type (str): Type of target ('returns', 'price', 'volatility')
            horizon (int): Prediction horizon in days
            
        Returns:
            pd.Series: Target variable
        """
        if self.features is None:
            raise ValueError("Features not created. Call create_features() first.")
        
        if target_type == 'returns':
            self.target = self.features['returns'].shift(-horizon)
        elif target_type == 'price':
            self.target = self.features['Close'].shift(-horizon)
        elif target_type == 'volatility':
            self.target = self.features['volatility'].shift(-horizon)
        else:
            raise ValueError("target_type must be 'returns', 'price', or 'volatility'")
        
        return self.target
    
    def arima_garch_model(self, order: Tuple[int, int, int] = (1, 1, 1), 
                         garch_order: Tuple[int, int] = (1, 1)) -> Dict:
        """
        ARIMA-GARCH model for price forecasting and volatility modeling
        
        Args:
            order (Tuple): ARIMA order (p, d, q)
            garch_order (Tuple): GARCH order (p, q)
            
        Returns:
            Dict: Model results and predictions
        """
        if self.features is None:
            raise ValueError("Features not created. Call create_features() first.")
        
        try:
            # Prepare data
            returns = self.features['returns'].dropna()
            
            # Test for stationarity
            adf_result = adfuller(returns)
            is_stationary = adf_result[1] < 0.05
            
            if not is_stationary:
                print("Data is not stationary, applying differencing")
                returns = returns.diff().dropna()
            
            # Split data
            split_idx = int(len(returns) * 0.8)
            train_data = returns[:split_idx]
            test_data = returns[split_idx:]
            
            # Fit ARIMA model
            arima_model = ARIMA(train_data, order=order)
            arima_fit = arima_model.fit()
            
            # Get ARIMA residuals
            arima_residuals = arima_fit.resid
            
            # Fit GARCH model on residuals
            garch_model = arch_model(arima_residuals, vol='Garch', p=garch_order[0], q=garch_order[1])
            garch_fit = garch_model.fit(disp='off')
            
            # Make predictions
            arima_forecast = arima_fit.forecast(steps=len(test_data))
            garch_forecast = garch_fit.forecast(horizon=len(test_data))
            
            # Calculate metrics
            mse = mean_squared_error(test_data, arima_forecast)
            mae = mean_absolute_error(test_data, arima_forecast)
            rmse = np.sqrt(mse)
            
            self.models['arima_garch'] = {
                'arima_model': arima_fit,
                'garch_model': garch_fit,
                'predictions': arima_forecast,
                'volatility_forecast': garch_forecast,
                'metrics': {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
            }
            
            return self.models['arima_garch']
            
        except Exception as e:
            print(f"Error in ARIMA-GARCH model: {e}")
            return {}
    
    def ridge_lasso_regression(self, alpha_ridge: float = 1.0, alpha_lasso: float = 0.1) -> Dict:
        """
        Ridge and Lasso regression with feature selection
        
        Args:
            alpha_ridge (float): Ridge regularization parameter
            alpha_lasso (float): Lasso regularization parameter
            
        Returns:
            Dict: Model results
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target not prepared. Call create_features() and prepare_target() first.")
        
        # Prepare feature matrix
        feature_cols = [col for col in self.features.columns 
                       if col not in ['returns', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X = self.features[feature_cols].dropna()
        y = self.target.dropna()
        
        # Align X and y
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Remove infinite values
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Ridge Regression
        ridge_model = Ridge(alpha=alpha_ridge)
        ridge_scores = cross_val_score(ridge_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
        ridge_model.fit(X_scaled, y)
        
        # Lasso Regression
        lasso_model = Lasso(alpha=alpha_lasso, max_iter=10000)
        lasso_scores = cross_val_score(lasso_model, X_scaled, y, cv=tscv, scoring='neg_mean_squared_error')
        lasso_model.fit(X_scaled, y)
        
        # Feature selection with Lasso
        lasso_coef = lasso_model.coef_
        selected_features = X.columns[lasso_coef != 0].tolist()
        
        self.models['ridge_lasso'] = {
            'ridge_model': ridge_model,
            'lasso_model': lasso_model,
            'ridge_scores': ridge_scores,
            'lasso_scores': lasso_scores,
            'selected_features': selected_features,
            'feature_importance': dict(zip(X.columns, np.abs(lasso_coef)))
        }
        
        return self.models['ridge_lasso']
    
    def random_forest_model(self, n_estimators: int = 100, max_depth: int = 10) -> Dict:
        """
        Random Forest model for maximum predictive accuracy
        
        Args:
            n_estimators (int): Number of trees
            max_depth (int): Maximum depth of trees
            
        Returns:
            Dict: Model results
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target not prepared.")
        
        # Prepare data (same as ridge/lasso)
        feature_cols = [col for col in self.features.columns 
                       if col not in ['returns', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X = self.features[feature_cols].dropna()
        y = self.target.dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Random Forest
        rf_model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        
        rf_scores = cross_val_score(rf_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        rf_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, rf_model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.models['random_forest'] = {
            'model': rf_model,
            'scores': rf_scores,
            'feature_importance': feature_importance,
            'top_features': sorted_features[:10]
        }
        
        return self.models['random_forest']
    
    def xgboost_model(self, n_estimators: int = 100, max_depth: int = 6, learning_rate: float = 0.1) -> Dict:
        """
        XGBoost model for maximum predictive accuracy
        
        Args:
            n_estimators (int): Number of boosting rounds
            max_depth (int): Maximum depth of trees
            learning_rate (float): Learning rate
            
        Returns:
            Dict: Model results
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target not prepared.")
        
        # Prepare data
        feature_cols = [col for col in self.features.columns 
                       if col not in ['returns', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X = self.features[feature_cols].dropna()
        y = self.target.dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        # Time series split
        tscv = TimeSeriesSplit(n_splits=5)
        
        # XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42,
            n_jobs=-1
        )
        
        xgb_scores = cross_val_score(xgb_model, X, y, cv=tscv, scoring='neg_mean_squared_error')
        xgb_model.fit(X, y)
        
        # Feature importance
        feature_importance = dict(zip(X.columns, xgb_model.feature_importances_))
        sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        self.models['xgboost'] = {
            'model': xgb_model,
            'scores': xgb_scores,
            'feature_importance': feature_importance,
            'top_features': sorted_features[:10]
        }
        
        return self.models['xgboost']
    
    def lstm_model(self, sequence_length: int = 60, lstm_units: int = 50, 
                   epochs: int = 100, batch_size: int = 32) -> Dict:
        """
        LSTM model for capturing complex temporal patterns
        
        Args:
            sequence_length (int): Length of input sequences
            lstm_units (int): Number of LSTM units
            epochs (int): Number of training epochs
            batch_size (int): Batch size
            
        Returns:
            Dict: Model results
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target not prepared.")
        
        # Prepare data
        feature_cols = [col for col in self.features.columns 
                       if col not in ['returns', 'Close', 'Open', 'High', 'Low', 'Volume']]
        X = self.features[feature_cols].dropna()
        y = self.target.dropna()
        
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
        y = y.loc[X.index]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        y_scaled = StandardScaler().fit_transform(y.values.reshape(-1, 1)).flatten()
        
        # Create sequences
        def create_sequences(X, y, seq_length):
            X_seq, y_seq = [], []
            for i in range(seq_length, len(X)):
                X_seq.append(X[i-seq_length:i])
                y_seq.append(y[i])
            return np.array(X_seq), np.array(y_seq)
        
        X_seq, y_seq = create_sequences(X_scaled, y_scaled, sequence_length)
        
        # Split data
        split_idx = int(len(X_seq) * 0.8)
        X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
        y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]
        
        # Build LSTM model
        model = Sequential([
            LSTM(lstm_units, return_sequences=True, input_shape=(sequence_length, X_scaled.shape[1])),
            Dropout(0.2),
            LSTM(lstm_units // 2, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        
        # Callbacks
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
        
        # Train model
        history = model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=0
        )
        
        # Make predictions
        y_pred = model.predict(X_test, verbose=0).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        
        self.models['lstm'] = {
            'model': model,
            'history': history.history,
            'predictions': y_pred,
            'actual': y_test,
            'metrics': {
                'mse': mse,
                'mae': mae,
                'rmse': rmse
            }
        }
        
        return self.models['lstm']
    
    def hybrid_arima_garch_lstm(self, arima_order: Tuple = (1, 1, 1), 
                               garch_order: Tuple = (1, 1),
                               lstm_units: int = 50) -> Dict:
        """
        Hybrid ARIMA-GARCH-LSTM model combining all approaches
        
        Args:
            arima_order (Tuple): ARIMA order
            garch_order (Tuple): GARCH order
            lstm_units (int): LSTM units
            
        Returns:
            Dict: Hybrid model results
        """
        if self.features is None or self.target is None:
            raise ValueError("Features and target not prepared.")
        
        try:
            # Get ARIMA-GARCH results
            arima_garch_results = self.arima_garch_model(arima_order, garch_order)
            
            # Get LSTM results
            lstm_results = self.lstm_model(lstm_units=lstm_units)
            
            # Combine predictions (simple ensemble)
            if 'arima_garch' in self.models and 'lstm' in self.models:
                # This is a simplified combination - in practice, you'd want more sophisticated ensemble methods
                arima_pred = self.models['arima_garch']['predictions']
                lstm_pred = self.models['lstm']['predictions']
                
                # Align predictions (this is simplified)
                min_len = min(len(arima_pred), len(lstm_pred))
                combined_pred = (arima_pred[:min_len] + lstm_pred[:min_len]) / 2
                
                self.models['hybrid'] = {
                    'arima_garch': arima_garch_results,
                    'lstm': lstm_results,
                    'combined_predictions': combined_pred,
                    'ensemble_method': 'simple_average'
                }
                
                return self.models['hybrid']
            
        except Exception as e:
            print(f"Error in hybrid model: {e}")
            return {}
    
    def time_series_cross_validation(self, model_name: str, n_splits: int = 5) -> Dict:
        """
        Time series cross-validation to avoid look-ahead bias
        
        Args:
            model_name (str): Name of the model to validate
            n_splits (int): Number of CV splits
            
        Returns:
            Dict: Cross-validation results
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found. Train the model first.")
        
        # This is a simplified implementation
        # In practice, you'd implement proper time series CV for each model type
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        # Get model scores if available
        if 'scores' in self.models[model_name]:
            scores = self.models[model_name]['scores']
            cv_results = {
                'mean_score': np.mean(scores),
                'std_score': np.std(scores),
                'scores': scores.tolist()
            }
        else:
            cv_results = {
                'message': f'Cross-validation not implemented for {model_name}',
                'scores': []
            }
        
        return cv_results
    
    def plot_results(self, model_name: str = None):
        """
        Plot model results and predictions
        
        Args:
            model_name (str): Specific model to plot, or None for all
        """
        if model_name and model_name in self.models:
            self._plot_single_model(model_name)
        else:
            self._plot_all_models()
    
    def _plot_single_model(self, model_name: str):
        """Plot results for a single model"""
        model_data = self.models[model_name]
        
        if model_name == 'lstm':
            # Plot LSTM training history
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Training history
            ax1.plot(model_data['history']['loss'], label='Training Loss')
            ax1.plot(model_data['history']['val_loss'], label='Validation Loss')
            ax1.set_title('LSTM Training History')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('Loss')
            ax1.legend()
            ax1.grid(True)
            
            # Predictions vs Actual
            ax2.scatter(model_data['actual'], model_data['predictions'], alpha=0.6)
            ax2.plot([model_data['actual'].min(), model_data['actual'].max()], 
                    [model_data['actual'].min(), model_data['actual'].max()], 'r--', lw=2)
            ax2.set_xlabel('Actual')
            ax2.set_ylabel('Predicted')
            ax2.set_title('LSTM: Predicted vs Actual')
            ax2.grid(True)
            
        elif model_name in ['random_forest', 'xgboost']:
            # Plot feature importance
            top_features = model_data['top_features'][:10]
            features, importance = zip(*top_features)
            
            plt.figure(figsize=(10, 6))
            plt.barh(features, importance)
            plt.xlabel('Feature Importance')
            plt.title(f'{model_name.title()} - Top 10 Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
        
        plt.show()
    
    def _plot_all_models(self):
        """Plot results for all trained models"""
        n_models = len(self.models)
        if n_models == 0:
            print("No models trained yet.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (name, model_data) in enumerate(self.models.items()):
            if i >= 4:  # Limit to 4 plots
                break
                
            ax = axes[i]
            
            if 'scores' in model_data:
                scores = model_data['scores']
                ax.hist(scores, bins=10, alpha=0.7)
                ax.set_title(f'{name.title()} - CV Scores Distribution')
                ax.set_xlabel('Score')
                ax.set_ylabel('Frequency')
            elif 'metrics' in model_data:
                metrics = model_data['metrics']
                metric_names = list(metrics.keys())
                metric_values = list(metrics.values())
                ax.bar(metric_names, metric_values)
                ax.set_title(f'{name.title()} - Metrics')
                ax.set_ylabel('Value')
        
        plt.tight_layout()
        plt.show()
    
    def get_model_summary(self) -> Dict:
        """
        Get summary of all trained models
        
        Returns:
            Dict: Model summary
        """
        summary = {}
        
        for name, model_data in self.models.items():
            if 'scores' in model_data:
                summary[name] = {
                    'mean_cv_score': np.mean(model_data['scores']),
                    'std_cv_score': np.std(model_data['scores']),
                    'best_score': np.max(model_data['scores'])
                }
            elif 'metrics' in model_data:
                summary[name] = model_data['metrics']
            else:
                summary[name] = {'status': 'trained', 'details': 'available'}
        
        return summary


def main():
    """Example usage of StockMarketTrendsAnalyzer"""
    print("Stock Market Trends Analyzer")
    print("=" * 50)
    
    # Initialize analyzer
    analyzer = StockMarketTrendsAnalyzer("AAPL", lookback_days=252*3)
    
    # Fetch data
    if analyzer.fetch_data():
        print("✓ Data fetched successfully")
        
        # Create features
        analyzer.create_features(lags=20, technical_indicators=True)
        print("✓ Features created")
        
        # Prepare target
        analyzer.prepare_target(target_type='returns', horizon=1)
        print("✓ Target prepared")
        
        # Train models
        print("\nTraining models...")
        
        # ARIMA-GARCH
        print("Training ARIMA-GARCH...")
        arima_garch_results = analyzer.arima_garch_model()
        
        # Ridge/Lasso
        print("Training Ridge/Lasso...")
        ridge_lasso_results = analyzer.ridge_lasso_regression()
        
        # Random Forest
        print("Training Random Forest...")
        rf_results = analyzer.random_forest_model()
        
        # XGBoost
        print("Training XGBoost...")
        xgb_results = analyzer.xgboost_model()
        
        # LSTM
        print("Training LSTM...")
        lstm_results = analyzer.lstm_model()
        
        # Hybrid model
        print("Training Hybrid ARIMA-GARCH-LSTM...")
        hybrid_results = analyzer.hybrid_arima_garch_lstm()
        
        # Get summary
        print("\n" + "="*50)
        print("MODEL SUMMARY")
        print("="*50)
        
        summary = analyzer.get_model_summary()
        for model_name, results in summary.items():
            print(f"\n{model_name.upper()}:")
            for metric, value in results.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.4f}")
                else:
                    print(f"  {metric}: {value}")
        
        # Plot results
        print("\nGenerating plots...")
        analyzer.plot_results()
        
        print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
