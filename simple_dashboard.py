"""
Simple Options Analysis Dashboard
A streamlined version for immediate use
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import time

from options_analyzer import OptionsAnalyzer


def main():
    """Simple Streamlit dashboard"""
    st.set_page_config(
        page_title="Options Analysis Platform",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    st.title("📈 Options Analysis Platform")
    st.markdown("Real-time options analysis and strategy evaluation")
    
    # Sidebar
    st.sidebar.header("Configuration")
    
    # Ticker input
    ticker = st.sidebar.text_input(
        "Enter Ticker Symbol", 
        value="AAPL",
        help="Enter a stock ticker symbol (e.g., AAPL, SPY, TSLA)"
    ).upper()
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = None
    if 'options_data' not in st.session_state:
        st.session_state.options_data = {}
    if 'current_price' not in st.session_state:
        st.session_state.current_price = None
    
    # Load data button
    if st.sidebar.button("Load Options Data", type="primary"):
        with st.spinner(f"Loading options data for {ticker}..."):
            try:
                # Initialize analyzer
                analyzer = OptionsAnalyzer(ticker)
                
                # Fetch stock info
                if not analyzer.fetch_stock_info():
                    st.error(f"Failed to fetch stock information for {ticker}")
                    st.stop()
                
                # Get expiration dates
                expirations = analyzer.get_expiration_dates()
                if not expirations:
                    st.error(f"No options data found for {ticker}")
                    st.stop()
                
                # Store in session state
                st.session_state.analyzer = analyzer
                st.session_state.current_price = analyzer.current_price
                st.session_state.expirations = expirations
                
                st.success(f"Successfully loaded data for {ticker}")
                st.success(f"Current Price: ${analyzer.current_price:.2f}")
                st.success(f"Found {len(expirations)} expiration dates")
                
            except Exception as e:
                st.error(f"Error loading data: {str(e)}")
                st.stop()
    
    # Check if data is loaded
    if st.session_state.analyzer is None:
        st.info("👈 Please enter a ticker symbol and click 'Load Options Data' to begin")
        st.stop()
    
    analyzer = st.session_state.analyzer
    current_price = st.session_state.current_price
    expirations = st.session_state.expirations
    
    # Expiration date selector
    st.sidebar.subheader("Analysis Options")
    selected_expiration = st.sidebar.selectbox(
        "Select Expiration Date",
        options=expirations,
        index=0
    )
    
    # Load options chain for selected expiration
    if selected_expiration not in st.session_state.options_data:
        with st.spinner(f"Loading options chain for {selected_expiration}..."):
            if analyzer.fetch_options_chain(selected_expiration):
                analyzer.calculate_greeks(selected_expiration)
                st.session_state.options_data[selected_expiration] = analyzer.options_data[selected_expiration]
            else:
                st.error(f"Failed to load options chain for {selected_expiration}")
                st.stop()
    
    options_df = st.session_state.options_data[selected_expiration]
    
    # Main content
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Current Price", f"${current_price:.2f}")
    
    with col2:
        total_contracts = len(options_df)
        st.metric("Total Contracts", f"{total_contracts:,}")
    
    with col3:
        total_volume = options_df['volume'].sum()
        st.metric("Total Volume", f"{total_volume:,}")
    
    with col4:
        total_oi = options_df['openInterest'].sum()
        st.metric("Total Open Interest", f"{total_oi:,}")
    
    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Options Chain", 
        "📊 Volatility Analysis", 
        "🎯 Strategy Analysis",
        "📋 Data Table"
    ])
    
    with tab1:
        st.header("Options Chain Analysis")
        
        # Options chain display
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Interactive options chain
            fig = create_interactive_chain_plotly(options_df, ticker, current_price, selected_expiration)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Options chain table
            st.subheader("Options Chain Data")
            
            # Filter options
            option_type = st.selectbox("Option Type", ["All", "Calls", "Puts"])
            min_volume = st.number_input("Minimum Volume", min_value=0, value=0)
            
            # Apply filters
            filtered_df = options_df.copy()
            if option_type != "All":
                filtered_df = filtered_df[filtered_df['optionType'] == option_type.lower()]
            if min_volume > 0:
                filtered_df = filtered_df[filtered_df['volume'] >= min_volume]
            
            # Display table
            display_columns = ['contractSymbol', 'optionType', 'strike', 'lastPrice', 
                             'volume', 'openInterest', 'impliedVolatility']
            
            if 'delta' in filtered_df.columns:
                display_columns.extend(['delta', 'gamma', 'theta', 'vega'])
            
            st.dataframe(
                filtered_df[display_columns].round(4),
                use_container_width=True,
                height=400
            )
    
    with tab2:
        st.header("Volatility Analysis")
        
        # Volatility analysis
        vol_analysis = analyzer.analyze_volatility(selected_expiration)
        
        if vol_analysis:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Volatility Statistics")
                stats = vol_analysis['overall_stats']
                st.metric("Mean IV", f"{stats['mean_iv']:.2f}%")
                st.metric("Median IV", f"{stats['median_iv']:.2f}%")
                st.metric("IV Range", f"{stats['min_iv']:.2f}% - {stats['max_iv']:.2f}%")
                st.metric("IV Std Dev", f"{stats['std_iv']:.2f}%")
            
            with col2:
                st.subheader("Calls vs Puts")
                calls_stats = vol_analysis['calls_stats']
                puts_stats = vol_analysis['puts_stats']
                
                st.metric("Calls Count", calls_stats['count'])
                st.metric("Calls Mean IV", f"{calls_stats['mean_iv']:.2f}%")
                st.metric("Puts Count", puts_stats['count'])
                st.metric("Puts Mean IV", f"{puts_stats['mean_iv']:.2f}%")
            
            # Volatility smile plot
            st.subheader("Volatility Smile")
            fig = create_volatility_smile_plotly(options_df, ticker, current_price, selected_expiration)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.header("Strategy Analysis")
        
        # Strategy analysis
        strategies = analyzer.analyze_strategies(selected_expiration)
        
        if strategies:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Covered Call")
                if strategies.get('covered_call'):
                    cc = strategies['covered_call']
                    st.metric("Strike", f"${cc['strike']:.2f}")
                    st.metric("Premium", f"${cc['premium']:.2f}")
                    st.metric("Annualized Return", f"{cc['annualized_return']:.2f}%")
                    st.metric("Max Profit", f"${cc['max_profit']:.2f}")
                else:
                    st.info("No covered call opportunities found")
            
            with col2:
                st.subheader("Cash Secured Put")
                if strategies.get('cash_secured_put'):
                    csp = strategies['cash_secured_put']
                    st.metric("Strike", f"${csp['strike']:.2f}")
                    st.metric("Premium", f"${csp['premium']:.2f}")
                    st.metric("Annualized Return", f"{csp['annualized_return']:.2f}%")
                    st.metric("Max Profit", f"${csp['max_profit']:.2f}")
                else:
                    st.info("No cash secured put opportunities found")
        else:
            st.info("No strategies found for this expiration")
    
    with tab4:
        st.header("Raw Options Data")
        
        # Display full options data
        st.subheader("All Options Data")
        
        # Add filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            option_type_filter = st.selectbox("Filter by Type", ["All", "Calls", "Puts"], key="filter_type")
        
        with col2:
            min_strike = st.number_input("Min Strike", min_value=0.0, value=0.0)
        
        with col3:
            max_strike = st.number_input("Max Strike", min_value=0.0, value=float(current_price * 2))
        
        # Apply filters
        filtered_data = options_df.copy()
        
        if option_type_filter != "All":
            filtered_data = filtered_data[filtered_data['optionType'] == option_type_filter.lower()]
        
        filtered_data = filtered_data[
            (filtered_data['strike'] >= min_strike) & 
            (filtered_data['strike'] <= max_strike)
        ]
        
        # Display filtered data
        st.dataframe(
            filtered_data.round(4),
            use_container_width=True,
            height=600
        )
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="Download Filtered Data as CSV",
            data=csv,
            file_name=f"{ticker}_options_{selected_expiration}.csv",
            mime="text/csv"
        )


def create_interactive_chain_plotly(options_df, ticker, current_price, expiration_date):
    """Create interactive options chain plot"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Volume', 'Open Interest', 'Implied Volatility', 'Last Price'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Separate calls and puts
    calls = options_df[options_df['optionType'] == 'call']
    puts = options_df[options_df['optionType'] == 'put']
    
    # Plot 1: Volume
    if len(calls) > 0:
        fig.add_trace(
            go.Scatter(x=calls['strike'], y=calls['volume'], 
                      mode='markers', name='Calls Volume', 
                      marker=dict(color='green', size=8)),
            row=1, col=1
        )
    if len(puts) > 0:
        fig.add_trace(
            go.Scatter(x=puts['strike'], y=puts['volume'], 
                      mode='markers', name='Puts Volume', 
                      marker=dict(color='red', size=8, symbol='triangle-up')),
            row=1, col=1
        )
    
    # Plot 2: Open Interest
    if len(calls) > 0:
        fig.add_trace(
            go.Scatter(x=calls['strike'], y=calls['openInterest'], 
                      mode='markers', name='Calls OI', 
                      marker=dict(color='green', size=8), showlegend=False),
            row=1, col=2
        )
    if len(puts) > 0:
        fig.add_trace(
            go.Scatter(x=puts['strike'], y=puts['openInterest'], 
                      mode='markers', name='Puts OI', 
                      marker=dict(color='red', size=8, symbol='triangle-up'), showlegend=False),
            row=1, col=2
        )
    
    # Plot 3: Implied Volatility
    if len(calls) > 0:
        fig.add_trace(
            go.Scatter(x=calls['strike'], y=calls['impliedVolatility'], 
                      mode='markers', name='Calls IV', 
                      marker=dict(color='green', size=8), showlegend=False),
            row=2, col=1
        )
    if len(puts) > 0:
        fig.add_trace(
            go.Scatter(x=puts['strike'], y=puts['impliedVolatility'], 
                      mode='markers', name='Puts IV', 
                      marker=dict(color='red', size=8, symbol='triangle-up'), showlegend=False),
            row=2, col=1
        )
    
    # Plot 4: Last Price
    if len(calls) > 0:
        fig.add_trace(
            go.Scatter(x=calls['strike'], y=calls['lastPrice'], 
                      mode='markers', name='Calls Price', 
                      marker=dict(color='green', size=8), showlegend=False),
            row=2, col=2
        )
    if len(puts) > 0:
        fig.add_trace(
            go.Scatter(x=puts['strike'], y=puts['lastPrice'], 
                      mode='markers', name='Puts Price', 
                      marker=dict(color='red', size=8, symbol='triangle-up'), showlegend=False),
            row=2, col=2
        )
    
    # Add current price lines
    for row in [1, 2]:
        for col in [1, 2]:
            fig.add_vline(x=current_price, line_dash="dash", 
                         line_color="blue", opacity=0.7, 
                         annotation_text=f"Current: ${current_price:.2f}",
                         row=row, col=col)
    
    # Update layout
    fig.update_layout(
        title=f'{ticker} Options Chain - {expiration_date}',
        height=600,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Strike Price", row=2, col=1)
    fig.update_xaxes(title_text="Strike Price", row=2, col=2)
    fig.update_yaxes(title_text="Volume", row=1, col=1)
    fig.update_yaxes(title_text="Open Interest", row=1, col=2)
    fig.update_yaxes(title_text="Implied Volatility (%)", row=2, col=1)
    fig.update_yaxes(title_text="Last Price", row=2, col=2)
    
    return fig


def create_volatility_smile_plotly(options_df, ticker, current_price, expiration_date):
    """Create volatility smile plot"""
    fig = go.Figure()
    
    # Separate calls and puts
    calls = options_df[options_df['optionType'] == 'call']
    puts = options_df[options_df['optionType'] == 'put']
    
    # Filter out options with missing IV
    calls_iv = calls.dropna(subset=['impliedVolatility'])
    puts_iv = puts.dropna(subset=['impliedVolatility'])
    
    # Plot calls
    if len(calls_iv) > 0:
        fig.add_trace(go.Scatter(
            x=calls_iv['strike'], 
            y=calls_iv['impliedVolatility'],
            mode='markers',
            name='Calls',
            marker=dict(color='green', size=8)
        ))
    
    # Plot puts
    if len(puts_iv) > 0:
        fig.add_trace(go.Scatter(
            x=puts_iv['strike'], 
            y=puts_iv['impliedVolatility'],
            mode='markers',
            name='Puts',
            marker=dict(color='red', size=8, symbol='triangle-up')
        ))
    
    # Add current price line
    fig.add_vline(x=current_price, line_dash="dash", 
                 line_color="blue", opacity=0.7, 
                 annotation_text=f"Current Price: ${current_price:.2f}")
    
    fig.update_layout(
        title=f'{ticker} Volatility Smile - {expiration_date}',
        xaxis_title='Strike Price',
        yaxis_title='Implied Volatility (%)',
        height=500
    )
    
    return fig


if __name__ == "__main__":
    main()
