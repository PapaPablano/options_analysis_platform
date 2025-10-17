#!/usr/bin/env python3
"""
Diagnostic script for the Stock Market Trends Platform
"""

import sys
import time
import traceback
from datetime import datetime

def test_basic_imports():
    """Test basic Python imports"""
    print("🔍 Testing basic imports...")
    try:
        import pandas as pd
        import numpy as np
        import streamlit as st
        import plotly.graph_objects as go
        import yfinance as yf
        import sklearn
        print("✅ All basic imports successful")
        return True
    except Exception as e:
        print(f"❌ Import error: {e}")
        return False

def test_data_fetching():
    """Test data fetching capabilities"""
    print("\n🔍 Testing data fetching...")
    try:
        import yfinance as yf
        
        # Test stock data
        stock = yf.Ticker('AAPL')
        data = stock.history(period='5d')
        print(f"✅ Stock data: {len(data)} days fetched")
        
        # Test options data
        expirations = stock.options
        print(f"✅ Options expirations: {len(expirations) if expirations else 0}")
        
        if expirations:
            opt_chain = stock.option_chain(expirations[0])
            print(f"✅ Options chain: {len(opt_chain.calls)} calls, {len(opt_chain.puts)} puts")
        
        return True
    except Exception as e:
        print(f"❌ Data fetching error: {e}")
        return False

def test_options_analyzer():
    """Test options analyzer functionality"""
    print("\n🔍 Testing Options Analyzer...")
    try:
        from options_analyzer import OptionsAnalyzer
        
        analyzer = OptionsAnalyzer('AAPL')
        success = analyzer.fetch_stock_info()
        print(f"✅ Stock info fetch: {success}")
        
        if success:
            expirations = analyzer.get_expiration_dates()
            print(f"✅ Expirations: {len(expirations)} available")
            
            if expirations:
                opt_success = analyzer.fetch_options_chain(expirations[0])
                print(f"✅ Options chain: {opt_success}")
                
                if opt_success:
                    greeks_success = analyzer.calculate_greeks(expirations[0])
                    print(f"✅ Greeks calculation: {greeks_success}")
        
        return True
    except Exception as e:
        print(f"❌ Options Analyzer error: {e}")
        traceback.print_exc()
        return False

def test_ml_imports():
    """Test ML library imports"""
    print("\n🔍 Testing ML imports...")
    try:
        import sklearn
        from sklearn.ensemble import RandomForestRegressor
        from sklearn.linear_model import Ridge, Lasso
        print("✅ Scikit-learn imports successful")
        
        # Test XGBoost (optional)
        try:
            import xgboost as xgb
            print("✅ XGBoost import successful")
        except Exception as e:
            print(f"⚠️ XGBoost import failed: {e}")
        
        # Test TensorFlow (optional)
        try:
            import tensorflow as tf
            print("✅ TensorFlow import successful")
        except Exception as e:
            print(f"⚠️ TensorFlow import failed: {e}")
        
        # Test ARCH (optional)
        try:
            import arch
            print("✅ ARCH import successful")
        except Exception as e:
            print(f"⚠️ ARCH import failed: {e}")
        
        return True
    except Exception as e:
        print(f"❌ ML imports error: {e}")
        return False

def test_stock_trends_analyzer():
    """Test stock trends analyzer (lightweight test)"""
    print("\n🔍 Testing Stock Trends Analyzer (lightweight)...")
    try:
        from stock_market_trends import StockMarketTrendsAnalyzer
        
        analyzer = StockMarketTrendsAnalyzer('AAPL')
        success = analyzer.fetch_data()
        print(f"✅ Data fetch: {success}")
        
        if success:
            analyzer.create_features()
            print(f"✅ Features created: {len(analyzer.features.columns)} columns")
            
            analyzer.prepare_target()
            print(f"✅ Target prepared: {len(analyzer.target)} values")
        
        return True
    except Exception as e:
        print(f"❌ Stock Trends Analyzer error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all diagnostic tests"""
    print("🚀 Stock Market Trends Platform - Diagnostic Test")
    print("=" * 60)
    print(f"Timestamp: {datetime.now()}")
    print(f"Python version: {sys.version}")
    print("=" * 60)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Fetching", test_data_fetching),
        ("Options Analyzer", test_options_analyzer),
        ("ML Imports", test_ml_imports),
        ("Stock Trends Analyzer", test_stock_trends_analyzer)
    ]
    
    results = []
    for test_name, test_func in tests:
        start_time = time.time()
        try:
            result = test_func()
            duration = time.time() - start_time
            results.append((test_name, result, duration))
            print(f"⏱️ {test_name}: {duration:.2f}s")
        except Exception as e:
            duration = time.time() - start_time
            results.append((test_name, False, duration))
            print(f"❌ {test_name} failed: {e}")
            print(f"⏱️ {test_name}: {duration:.2f}s")
    
    print("\n" + "=" * 60)
    print("📊 DIAGNOSTIC SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result, duration in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name:<25} ({duration:.2f}s)")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Platform is ready for deployment.")
    else:
        print("⚠️ Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
