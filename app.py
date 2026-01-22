import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.train import train_model, predict
from src.strategy import TradingStrategy


st.set_page_config(page_title="Stock Predictor", layout="wide")
st.title("📈 Stock Price Predictor & Strategy Simulator")

st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

st.sidebar.subheader("Model Config")
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
hidden_size = st.sidebar.slider("Hidden Units", 16, 128, 50)
num_layers = st.sidebar.slider("LSTM Layers", 1, 3, 2)
lookback = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)
include_sentiment = st.sidebar.checkbox("Include Sentiment", value=False, help="Add live news sentiment as a model feature")

st.sidebar.subheader("Strategy Config")
selected_strategy = st.sidebar.selectbox(
    "Trading Strategy", ["MA Crossover", "Simple Threshold", "Darvas Box", "Robust Strategy"]
)
stop_loss = 0.01
if selected_strategy != "Robust Strategy":
    stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 10.0, 2.0, step=0.5) / 100


@st.cache_data
def get_stock_data(ticker, start, end, sequence_length, include_sentiment=False):
    loader = DataLoader(ticker, start, end)
    loader.fetch_stock_data()
    X_train, y_train, X_test, y_test, scaler, input_size, offset = loader.preprocess_data(
        sequence_length=sequence_length,
        include_sentiment=include_sentiment
    )
    return loader, X_train, y_train, X_test, y_test, scaler, input_size, offset


@st.cache_data
def get_news_sentiment(ticker):
    loader = DataLoader(ticker, None, None)
    return loader.fetch_news_sentiment()


if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching Data and Training Model..."):
        try:
            loader, X_train, y_train, X_test, y_test, scaler, input_size, offset = get_stock_data(
                ticker, start_date, end_date, lookback, include_sentiment
            )

            model = train_model(
                X_train,
                y_train,
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                num_epochs=epochs,
                dropout=dropout,
                seed=42,
            )

            train_preds_scaled = predict(model, X_train, scaler)
            
            if len(X_test) == 0:
                st.error("Not enough data to generate test predictions. Reduce lookback or choose a longer date range.")
                st.stop()
                
            test_preds_scaled = predict(model, X_test, scaler)

            original_data = loader.data
            close_prices = original_data["Close"].values

            train_len = len(train_preds_scaled)
            test_len = len(test_preds_scaled)

            train_start_idx_pp = lookback
            train_start_idx_raw = train_start_idx_pp + offset

            train_prev_close = close_prices[train_start_idx_raw - 1 : train_start_idx_raw + train_len - 1]
            train_preds_ret = train_preds_scaled.flatten()
            train_preds_price = train_prev_close * np.exp(train_preds_ret)

            train_size = train_len + lookback
            
            test_start_idx_pp = train_size
            test_start_idx_raw = test_start_idx_pp + offset

            test_prev_close = close_prices[test_start_idx_raw - 1 : test_start_idx_raw + test_len - 1]
            test_preds_ret = test_preds_scaled.flatten()
            test_preds_price = test_prev_close * np.exp(test_preds_ret)

            train_idx = range(train_start_idx_raw, train_start_idx_raw + train_len)
            test_idx = range(test_start_idx_raw, test_start_idx_raw + test_len)

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(x=original_data.index, y=original_data["Close"], mode="lines", name="Actual Price")
            )

            if len(train_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=original_data.index[train_idx],
                        y=train_preds_price,
                        mode="lines",
                        name="Train Pred (Reconstructed)",
                        line=dict(color="orange"),
                    )
                )

            if len(test_idx) > 0:
                fig.add_trace(
                    go.Scatter(
                        x=original_data.index[test_idx],
                        y=test_preds_price,
                        mode="lines",
                        name="Test Pred (Reconstructed)",
                        line=dict(color="green"),
                    )
                )

            st.plotly_chart(fig, width="stretch")

            sentiment_score, sentiment_details = get_news_sentiment(ticker)

            st.subheader("Live Market Sentiment 📰")
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("VADER Sentiment Score", f"{sentiment_score:.2f}")
                if sentiment_score > 0.05:
                    st.success("Overall Positive")
                elif sentiment_score < -0.05:
                    st.error("Overall Negative")
                else:
                    st.info("Neutral")
                st.caption("Note: This reflects *current* news, not historical data.")
            with col2:
                with st.expander("View News Headlines & Scores"):
                    st.dataframe(pd.DataFrame(sentiment_details))

            st.subheader(f"Trading Strategy Simulation ({selected_strategy})")
            strategy = TradingStrategy(initial_capital=10000)

            actual_test_prices = original_data["Close"].values[test_idx]

            if selected_strategy == "Simple Threshold":
                signals, portfolio_value = strategy.simple_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )
            elif selected_strategy == "Darvas Box":
                signals, portfolio_value = strategy.darvas_box_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )
            elif selected_strategy == "Robust Strategy":
                signals, portfolio_value = strategy.robust_strategy(
                    original_data, test_idx, test_preds_price, risk_per_trade=0.02
                )
            else:
                signals, portfolio_value = strategy.ma_crossover_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )

            final_val = portfolio_value[-1]
            profit = final_val - 10000

            st.metric(label="Final Portfolio Value", value=f"${final_val:,.2f}", delta=f"{profit:,.2f}")

            strat_fig = go.Figure()
            strat_fig.add_trace(
                go.Scatter(x=original_data.index[test_idx], y=portfolio_value[1:], mode="lines", name="Portfolio Value")
            )
            st.plotly_chart(strat_fig, width="stretch")

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
