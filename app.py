import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.train import train_model, predict
from src.strategy import TradingStrategy
import torch

st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Price Predictor & Strategy Simulator")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)

if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching Data and Training Model..."):
        # 1. Load Data
        loader = DataLoader(ticker, start_date, end_date)
        try:
            loader.fetch_stock_data()
            X, y, scaler = loader.preprocess_data()
            
            # 2. Train Model
            # Split data
            train_size = int(len(X) * 0.8)
            X_train, y_train = X[:train_size], y[:train_size]
            X_test, y_test = X[train_size:], y[train_size:]
            
            model = train_model(X_train, y_train, num_epochs=epochs)
            
            # 3. Predict
            train_preds = predict(model, X_train, scaler)
            test_preds = predict(model, X_test, scaler)
            
            # Align Predictions for Plotting
            original_data = loader.get_raw_data()
            close_prices = original_data['Close'].values.reshape(-1, 1)
            
            # Indices
            train_idx = range(60, 60+len(train_preds))
            test_idx = range(60+len(train_preds), 60+len(train_preds)+len(test_preds))
            
            # 4. Visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=original_data.index, y=original_data['Close'], mode='lines', name='Actual Price'))
            
            # Plot Train Predictions (Need to map back to dates)
            train_dates = original_data.index[train_idx]
            fig.add_trace(go.Scatter(x=train_dates, y=train_preds.flatten(), mode='lines', name='Train Prediction', line=dict(color='orange')))
            
            # Plot Test Predictions
            test_dates = original_data.index[test_idx]
            fig.add_trace(go.Scatter(x=test_dates, y=test_preds.flatten(), mode='lines', name='Test Prediction', line=dict(color='green')))
            
            st.plotly_chart(fig, use_container_width=True)
            
            # 5. Sentiment Analysis
            sentiment_score = loader.fetch_news_sentiment()
            st.subheader("Sentiment Analysis")
            st.metric("Latest News Sentiment (Polarity)", f"{sentiment_score:.2f}")
            if sentiment_score > 0:
                st.success("Overall Positive Sentiment")
            elif sentiment_score < 0:
                st.error("Overall Negative Sentiment")
            else:
                st.info("Neutral Sentiment")

            # 6. Strategy Simulation
            st.subheader("Trading Strategy Simulation (Test Set)")
            strategy = TradingStrategy(initial_capital=10000)
            
            # Get actual prices for test set
            actual_test_prices = original_data['Close'].values[test_idx]
            
            # Run strategy
            # Note: strategy expects equal length arrays. test_preds is aligned with actual_test_prices
            signals, portfolio_value = strategy.simple_strategy(actual_test_prices, test_preds.flatten())
            
            # Plot Portfolio Value
            st.write(f"Initial Capital: $10,000 | Final Value: ${portfolio_value[-1]:,.2f}")
            
            strat_fig = go.Figure()
            strat_fig.add_trace(go.Scatter(x=test_dates[:-1], y=portfolio_value, mode='lines', name='Portfolio Value'))
            st.plotly_chart(strat_fig, use_container_width=True)
            
        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)


