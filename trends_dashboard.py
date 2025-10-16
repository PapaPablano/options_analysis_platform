"""
Stock Market Trends Dashboard
Interactive Streamlit dashboard for ML-based stock market analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from stock_market_trends import StockMarketTrendsAnalyzer


def main():
    """Main dashboard application"""
    st.set_page_config(
        page_title="Stock Market Trends Analysis",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Stock Market Trends Analysis")
    st.markdown("Advanced ML models for price forecasting and volatility modeling")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Ticker selection
    ticker = st.sidebar.text_input(
        "Stock Ticker", 
        value="AAPL", 
        help="Enter a stock ticker symbol (e.g., AAPL, SPY, TSLA)"
    ).upper()
    
    # Date range selection
    col1, col2 = st.sidebar.columns(2)
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=datetime.now() - timedelta(days=252*3),
            help="Start date for historical data"
        )
    with col2:
        end_date = st.date_input(
            "End Date",
            value=datetime.now(),
            help="End date for historical data"
        )
    
    # Model parameters
    st.sidebar.subheader("Model Parameters")
    
    # ARIMA-GARCH parameters
    st.sidebar.markdown("**ARIMA-GARCH**")
    arima_p = st.sidebar.slider("ARIMA p", 0, 3, 1)
    arima_d = st.sidebar.slider("ARIMA d", 0, 2, 1)
    arima_q = st.sidebar.slider("ARIMA q", 0, 3, 1)
    garch_p = st.sidebar.slider("GARCH p", 1, 2, 1)
    garch_q = st.sidebar.slider("GARCH q", 1, 2, 1)
    
    # Regression parameters
    st.sidebar.markdown("**Ridge/Lasso**")
    ridge_alpha = st.sidebar.slider("Ridge Alpha", 0.1, 10.0, 1.0, 0.1)
    lasso_alpha = st.sidebar.slider("Lasso Alpha", 0.01, 1.0, 0.1, 0.01)
    
    # Tree-based parameters
    st.sidebar.markdown("**Random Forest**")
    rf_estimators = st.sidebar.slider("RF Estimators", 50, 200, 100)
    rf_depth = st.sidebar.slider("RF Max Depth", 5, 20, 10)
    
    st.sidebar.markdown("**XGBoost**")
    xgb_estimators = st.sidebar.slider("XGB Estimators", 50, 200, 100)
    xgb_depth = st.sidebar.slider("XGB Max Depth", 3, 10, 6)
    xgb_lr = st.sidebar.slider("XGB Learning Rate", 0.01, 0.3, 0.1, 0.01)
    
    # LSTM parameters
    st.sidebar.markdown("**LSTM**")
    sequence_length = st.sidebar.slider("Sequence Length", 30, 120, 60)
    lstm_units = st.sidebar.slider("LSTM Units", 25, 100, 50)
    epochs = st.sidebar.slider("Epochs", 50, 200, 100)
    
    # Analysis options
    st.sidebar.subheader("Analysis Options")
    target_type = st.sidebar.selectbox(
        "Target Variable",
        ["returns", "price", "volatility"],
        help="What to predict"
    )
    prediction_horizon = st.sidebar.slider(
        "Prediction Horizon (days)", 
        1, 30, 1,
        help="How many days ahead to predict"
    )
    
    # Initialize analyzer
    if 'analyzer' not in st.session_state or st.session_state.get('current_ticker') != ticker:
        with st.spinner(f"Initializing analyzer for {ticker}..."):
            st.session_state.analyzer = StockMarketTrendsAnalyzer(ticker)
            st.session_state.current_ticker = ticker
    
    analyzer = st.session_state.analyzer
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Data Overview", 
        "🤖 Model Training", 
        "📈 Predictions", 
        "🔍 Feature Analysis", 
        "📋 Model Comparison", 
        "⚙️ Advanced Settings"
    ])
    
    with tab1:
        st.header("📊 Data Overview")
        
        if st.button("🔄 Load Data", type="primary"):
            with st.spinner("Fetching data..."):
                success = analyzer.fetch_data(
                    start_date.strftime('%Y-%m-%d'),
                    end_date.strftime('%Y-%m-%d')
                )
                
                if success:
                    st.success(f"✅ Loaded {len(analyzer.data)} days of data for {ticker}")
                    
                    # Display basic statistics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Current Price", f"${analyzer.data['Close'].iloc[-1]:.2f}")
                    with col2:
                        returns = analyzer.data['Close'].pct_change().dropna()
                        st.metric("Daily Volatility", f"{returns.std()*100:.2f}%")
                    with col3:
                        st.metric("Total Return", f"{((analyzer.data['Close'].iloc[-1] / analyzer.data['Close'].iloc[0]) - 1)*100:.2f}%")
                    with col4:
                        st.metric("Data Points", len(analyzer.data))
                    
                    # Price chart
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=analyzer.data.index,
                        y=analyzer.data['Close'],
                        mode='lines',
                        name='Close Price',
                        line=dict(color='blue', width=2)
                    ))
                    
                    fig.update_layout(
                        title=f"{ticker} Stock Price",
                        xaxis_title="Date",
                        yaxis_title="Price ($)",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Volume chart
                    fig_vol = go.Figure()
                    fig_vol.add_trace(go.Bar(
                        x=analyzer.data.index,
                        y=analyzer.data['Volume'],
                        name='Volume',
                        marker_color='lightblue'
                    ))
                    
                    fig_vol.update_layout(
                        title=f"{ticker} Trading Volume",
                        xaxis_title="Date",
                        yaxis_title="Volume",
                        height=300
                    )
                    
                    st.plotly_chart(fig_vol, use_container_width=True)
                    
                    # Create features
                    with st.spinner("Creating features..."):
                        analyzer.create_features(lags=20, technical_indicators=True)
                        analyzer.prepare_target(target_type=target_type, horizon=prediction_horizon)
                        st.success("✅ Features created successfully")
                        
                        # Display feature summary
                        st.subheader("Feature Summary")
                        feature_cols = [col for col in analyzer.features.columns 
                                       if col not in ['returns', 'Close', 'Open', 'High', 'Low', 'Volume']]
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write(f"**Total Features:** {len(feature_cols)}")
                            st.write(f"**Data Points:** {len(analyzer.features)}")
                        with col2:
                            st.write(f"**Target Variable:** {target_type}")
                            st.write(f"**Prediction Horizon:** {prediction_horizon} days")
                        
                        # Show feature categories
                        lag_features = [col for col in feature_cols if 'lag' in col]
                        technical_features = [col for col in feature_cols if col not in lag_features and col not in ['day_of_week', 'month', 'quarter']]
                        time_features = [col for col in feature_cols if col in ['day_of_week', 'month', 'quarter']]
                        
                        st.write(f"**Lag Features:** {len(lag_features)}")
                        st.write(f"**Technical Indicators:** {len(technical_features)}")
                        st.write(f"**Time Features:** {len(time_features)}")
                        
                else:
                    st.error("❌ Failed to load data. Please check the ticker symbol.")
    
    with tab2:
        st.header("🤖 Model Training")
        
        if analyzer.data is None:
            st.warning("⚠️ Please load data first in the Data Overview tab.")
        else:
            st.subheader("Train Models")
            
            col1, col2 = st.columns(2)
            
            with col1:
                train_arima_garch = st.button("🚀 Train ARIMA-GARCH", type="primary")
                train_ridge_lasso = st.button("📊 Train Ridge/Lasso")
                train_rf = st.button("🌲 Train Random Forest")
            
            with col2:
                train_xgb = st.button("⚡ Train XGBoost")
                train_lstm = st.button("🧠 Train LSTM")
                train_hybrid = st.button("🔄 Train Hybrid Model")
            
            # Training progress
            if train_arima_garch:
                with st.spinner("Training ARIMA-GARCH model..."):
                    try:
                        results = analyzer.arima_garch_model(
                            order=(arima_p, arima_d, arima_q),
                            garch_order=(garch_p, garch_q)
                        )
                        if results:
                            st.success("✅ ARIMA-GARCH model trained successfully")
                            st.session_state.arima_garch_trained = True
                        else:
                            st.error("❌ Failed to train ARIMA-GARCH model")
                    except Exception as e:
                        st.error(f"❌ Error training ARIMA-GARCH: {str(e)}")
            
            if train_ridge_lasso:
                with st.spinner("Training Ridge/Lasso models..."):
                    try:
                        results = analyzer.ridge_lasso_regression(
                            alpha_ridge=ridge_alpha,
                            alpha_lasso=lasso_alpha
                        )
                        if results:
                            st.success("✅ Ridge/Lasso models trained successfully")
                            st.session_state.ridge_lasso_trained = True
                        else:
                            st.error("❌ Failed to train Ridge/Lasso models")
                    except Exception as e:
                        st.error(f"❌ Error training Ridge/Lasso: {str(e)}")
            
            if train_rf:
                with st.spinner("Training Random Forest model..."):
                    try:
                        results = analyzer.random_forest_model(
                            n_estimators=rf_estimators,
                            max_depth=rf_depth
                        )
                        if results:
                            st.success("✅ Random Forest model trained successfully")
                            st.session_state.rf_trained = True
                        else:
                            st.error("❌ Failed to train Random Forest model")
                    except Exception as e:
                        st.error(f"❌ Error training Random Forest: {str(e)}")
            
            if train_xgb:
                with st.spinner("Training XGBoost model..."):
                    try:
                        results = analyzer.xgboost_model(
                            n_estimators=xgb_estimators,
                            max_depth=xgb_depth,
                            learning_rate=xgb_lr
                        )
                        if results:
                            st.success("✅ XGBoost model trained successfully")
                            st.session_state.xgb_trained = True
                        else:
                            st.error("❌ Failed to train XGBoost model")
                    except Exception as e:
                        st.error(f"❌ Error training XGBoost: {str(e)}")
            
            if train_lstm:
                with st.spinner("Training LSTM model (this may take a while)..."):
                    try:
                        results = analyzer.lstm_model(
                            sequence_length=sequence_length,
                            lstm_units=lstm_units,
                            epochs=epochs
                        )
                        if results:
                            st.success("✅ LSTM model trained successfully")
                            st.session_state.lstm_trained = True
                        else:
                            st.error("❌ Failed to train LSTM model")
                    except Exception as e:
                        st.error(f"❌ Error training LSTM: {str(e)}")
            
            if train_hybrid:
                with st.spinner("Training Hybrid model..."):
                    try:
                        results = analyzer.hybrid_arima_garch_lstm(
                            arima_order=(arima_p, arima_d, arima_q),
                            garch_order=(garch_p, garch_q),
                            lstm_units=lstm_units
                        )
                        if results:
                            st.success("✅ Hybrid model trained successfully")
                            st.session_state.hybrid_trained = True
                        else:
                            st.error("❌ Failed to train Hybrid model")
                    except Exception as e:
                        st.error(f"❌ Error training Hybrid: {str(e)}")
            
            # Show training status
            st.subheader("Training Status")
            models_status = {
                "ARIMA-GARCH": st.session_state.get('arima_garch_trained', False),
                "Ridge/Lasso": st.session_state.get('ridge_lasso_trained', False),
                "Random Forest": st.session_state.get('rf_trained', False),
                "XGBoost": st.session_state.get('xgb_trained', False),
                "LSTM": st.session_state.get('lstm_trained', False),
                "Hybrid": st.session_state.get('hybrid_trained', False)
            }
            
            for model, status in models_status.items():
                if status:
                    st.success(f"✅ {model}")
                else:
                    st.info(f"⏳ {model} - Not trained")
    
    with tab3:
        st.header("📈 Predictions")
        
        if not any(st.session_state.get(f'{model}_trained', False) for model in ['arima_garch', 'ridge_lasso', 'rf', 'xgb', 'lstm', 'hybrid']):
            st.warning("⚠️ Please train at least one model first.")
        else:
            # Show predictions for trained models
            if 'arima_garch' in analyzer.models:
                st.subheader("ARIMA-GARCH Predictions")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{analyzer.models['arima_garch']['metrics']['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{analyzer.models['arima_garch']['metrics']['mae']:.4f}")
                
                # Plot predictions
                if 'predictions' in analyzer.models['arima_garch']:
                    pred_data = analyzer.models['arima_garch']['predictions']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=pred_data,
                        mode='lines',
                        name='ARIMA-GARCH Predictions',
                        line=dict(color='blue')
                    ))
                    fig.update_layout(
                        title="ARIMA-GARCH Predictions",
                        xaxis_title="Time",
                        yaxis_title="Predicted Returns",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
            
            if 'lstm' in analyzer.models:
                st.subheader("LSTM Predictions")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("RMSE", f"{analyzer.models['lstm']['metrics']['rmse']:.4f}")
                with col2:
                    st.metric("MAE", f"{analyzer.models['lstm']['metrics']['mae']:.4f}")
                
                # Plot LSTM training history
                if 'history' in analyzer.models['lstm']:
                    history = analyzer.models['lstm']['history']
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=history['loss'],
                        mode='lines',
                        name='Training Loss',
                        line=dict(color='blue')
                    ))
                    fig.add_trace(go.Scatter(
                        y=history['val_loss'],
                        mode='lines',
                        name='Validation Loss',
                        line=dict(color='red')
                    ))
                    fig.update_layout(
                        title="LSTM Training History",
                        xaxis_title="Epoch",
                        yaxis_title="Loss",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Plot predictions vs actual
                if 'predictions' in analyzer.models['lstm'] and 'actual' in analyzer.models['lstm']:
                    pred = analyzer.models['lstm']['predictions']
                    actual = analyzer.models['lstm']['actual']
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=actual,
                        y=pred,
                        mode='markers',
                        name='Predictions vs Actual',
                        marker=dict(color='blue', opacity=0.6)
                    ))
                    fig.add_trace(go.Scatter(
                        x=[actual.min(), actual.max()],
                        y=[actual.min(), actual.max()],
                        mode='lines',
                        name='Perfect Prediction',
                        line=dict(color='red', dash='dash')
                    ))
                    fig.update_layout(
                        title="LSTM: Predictions vs Actual",
                        xaxis_title="Actual",
                        yaxis_title="Predicted",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.header("🔍 Feature Analysis")
        
        if analyzer.features is None:
            st.warning("⚠️ Please load data and create features first.")
        else:
            # Feature importance for tree-based models
            if 'random_forest' in analyzer.models:
                st.subheader("Random Forest Feature Importance")
                
                top_features = analyzer.models['random_forest']['top_features'][:15]
                features, importance = zip(*top_features)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color='lightblue'
                ))
                fig.update_layout(
                    title="Top 15 Features by Importance",
                    xaxis_title="Importance",
                    yaxis_title="Features",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            if 'xgboost' in analyzer.models:
                st.subheader("XGBoost Feature Importance")
                
                top_features = analyzer.models['xgboost']['top_features'][:15]
                features, importance = zip(*top_features)
                
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    x=importance,
                    y=features,
                    orientation='h',
                    marker_color='lightgreen'
                ))
                fig.update_layout(
                    title="Top 15 Features by Importance",
                    xaxis_title="Importance",
                    yaxis_title="Features",
                    height=500
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Lasso feature selection
            if 'ridge_lasso' in analyzer.models:
                st.subheader("Lasso Feature Selection")
                
                selected_features = analyzer.models['ridge_lasso']['selected_features']
                st.write(f"**Selected Features ({len(selected_features)}):**")
                
                if selected_features:
                    # Show selected features
                    feature_importance = analyzer.models['ridge_lasso']['feature_importance']
                    selected_importance = {k: v for k, v in feature_importance.items() if k in selected_features}
                    
                    if selected_importance:
                        fig = go.Figure()
                        fig.add_trace(go.Bar(
                            x=list(selected_importance.values()),
                            y=list(selected_importance.keys()),
                            orientation='h',
                            marker_color='orange'
                        ))
                        fig.update_layout(
                            title="Lasso Selected Features",
                            xaxis_title="Coefficient Magnitude",
                            yaxis_title="Features",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No features selected by Lasso (all coefficients are zero)")
    
    with tab5:
        st.header("📋 Model Comparison")
        
        if not analyzer.models:
            st.warning("⚠️ No models trained yet.")
        else:
            # Model summary
            summary = analyzer.get_model_summary()
            
            # Create comparison table
            comparison_data = []
            for model_name, results in summary.items():
                if 'mean_cv_score' in results:
                    comparison_data.append({
                        'Model': model_name.title(),
                        'Mean CV Score': f"{results['mean_cv_score']:.4f}",
                        'Std CV Score': f"{results['std_cv_score']:.4f}",
                        'Best Score': f"{results['best_score']:.4f}"
                    })
                elif 'mse' in results:
                    comparison_data.append({
                        'Model': model_name.title(),
                        'MSE': f"{results['mse']:.4f}",
                        'MAE': f"{results['mae']:.4f}",
                        'RMSE': f"{results['rmse']:.4f}"
                    })
            
            if comparison_data:
                comparison_df = pd.DataFrame(comparison_data)
                st.subheader("Model Performance Comparison")
                st.dataframe(comparison_df, use_container_width=True)
                
                # Best model
                if 'Mean CV Score' in comparison_df.columns:
                    best_model = comparison_df.loc[comparison_df['Mean CV Score'].astype(float).idxmax()]
                    st.success(f"🏆 Best Model: {best_model['Model']} (CV Score: {best_model['Mean CV Score']})")
                elif 'RMSE' in comparison_df.columns:
                    best_model = comparison_df.loc[comparison_df['RMSE'].astype(float).idxmin()]
                    st.success(f"🏆 Best Model: {best_model['Model']} (RMSE: {best_model['RMSE']})")
            
            # Cross-validation results
            st.subheader("Cross-Validation Results")
            
            for model_name in analyzer.models.keys():
                if st.button(f"Show CV for {model_name.title()}"):
                    cv_results = analyzer.time_series_cross_validation(model_name)
                    
                    if 'scores' in cv_results and cv_results['scores']:
                        st.write(f"**{model_name.title()} Cross-Validation:**")
                        st.write(f"Mean Score: {cv_results['mean_score']:.4f}")
                        st.write(f"Std Score: {cv_results['std_score']:.4f}")
                        
                        # Plot CV scores
                        fig = go.Figure()
                        fig.add_trace(go.Box(
                            y=cv_results['scores'],
                            name=model_name.title()
                        ))
                        fig.update_layout(
                            title=f"{model_name.title()} Cross-Validation Scores",
                            yaxis_title="Score",
                            height=400
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info(f"Cross-validation not available for {model_name}")
    
    with tab6:
        st.header("⚙️ Advanced Settings")
        
        st.subheader("Model Configuration")
        
        # Advanced parameters
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Data Processing**")
            lag_features = st.slider("Number of Lag Features", 5, 50, 20)
            include_technical = st.checkbox("Include Technical Indicators", value=True)
            include_time_features = st.checkbox("Include Time Features", value=True)
            
        with col2:
            st.markdown("**Cross-Validation**")
            cv_splits = st.slider("CV Splits", 3, 10, 5)
            test_size = st.slider("Test Size (%)", 10, 30, 20)
        
        # Model ensemble settings
        st.subheader("Ensemble Settings")
        ensemble_method = st.selectbox(
            "Ensemble Method",
            ["Simple Average", "Weighted Average", "Stacking"],
            help="Method to combine multiple model predictions"
        )
        
        if ensemble_method == "Weighted Average":
            st.write("Configure model weights:")
            weights = {}
            for model in analyzer.models.keys():
                weights[model] = st.slider(f"Weight for {model}", 0.0, 1.0, 0.5)
        
        # Export settings
        st.subheader("Export Options")
        
        col1, col2 = st.columns(2)
        
        with col1:
            export_predictions = st.checkbox("Export Predictions")
            export_features = st.checkbox("Export Features")
            
        with col2:
            export_models = st.checkbox("Export Trained Models")
            export_plots = st.checkbox("Export Plots")
        
        if st.button("📥 Export Results", type="primary"):
            st.info("Export functionality would be implemented here")
        
        # Model persistence
        st.subheader("Model Persistence")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("💾 Save Models"):
                st.info("Model saving functionality would be implemented here")
        
        with col2:
            if st.button("📂 Load Models"):
                st.info("Model loading functionality would be implemented here")
        
        # Performance monitoring
        st.subheader("Performance Monitoring")
        
        if st.button("📊 Generate Performance Report"):
            st.info("Performance report generation would be implemented here")
        
        # System information
        st.subheader("System Information")
        
        import sys
        import platform
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write(f"**Python Version:** {sys.version}")
            st.write(f"**Platform:** {platform.system()}")
            
        with col2:
            st.write(f"**Streamlit Version:** {st.__version__}")
            st.write(f"**Models Trained:** {len(analyzer.models)}")


if __name__ == "__main__":
    main()
