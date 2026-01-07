import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.data_loader import DataLoader
from src.train import train_model, predict
from src.strategy import TradingStrategy


st.set_page_config(page_title="Stock Predictor", layout="wide")

st.title("📈 Stock Price Predictor & Strategy Simulator")

# Sidebar
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Ticker Symbol", "AAPL")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("today"))

# Model Config
st.sidebar.subheader("Model Config")
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50)
hidden_size = st.sidebar.slider("Hidden Units", 16, 128, 50)
num_layers = st.sidebar.slider("LSTM Layers", 1, 3, 2)
lookback = st.sidebar.slider("Lookback (Days)", 30, 90, 60)
dropout = st.sidebar.slider("Dropout", 0.0, 0.5, 0.2, step=0.05)

# Strategy Config
st.sidebar.subheader("Strategy Config")
selected_strategy = st.sidebar.selectbox("Trading Strategy", ["Simple Threshold", "MA Crossover", "Darvas Box"])
stop_loss = st.sidebar.slider("Stop Loss (%)", 0.0, 10.0, 2.0, step=0.5) / 100


@st.cache_data
def get_stock_data(ticker, start, end, sequence_length):
    loader = DataLoader(ticker, start, end)
    loader.fetch_stock_data()
    # Now returns split data directly
    X_train, y_train, X_test, y_test, scaler, input_size, offset = loader.preprocess_data(
        sequence_length=sequence_length
    )
    return loader, X_train, y_train, X_test, y_test, scaler, input_size, offset


@st.cache_data
def get_news_sentiment(ticker):
    loader = DataLoader(ticker, None, None)
    return loader.fetch_news_sentiment()


if st.sidebar.button("Run Prediction"):
    with st.spinner("Fetching Data and Training Model..."):
        try:
            # 1. Load Data
            loader, X_train, y_train, X_test, y_test, scaler, input_size, offset = get_stock_data(
                ticker, start_date, end_date, lookback
            )

            # 2. Train Model
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

            # 3. Predict (Output is Scaled Log Returns)
            train_preds_scaled = predict(
                model, X_train, scaler
            )  # Actually this uses target_scaler implicitly? No, predict uses passed scaler.
            test_preds_scaled = predict(model, X_test, scaler)

            # 4. Reconstruct Prices
            # Pred_Price_t = Price_{t-1} * exp(Pred_Log_Ret_t)
            original_data = loader.get_raw_data()
            close_prices = original_data["Close"].values

            # Indices calculations
            # Data pipeline:
            # 1. Raw Data (Length N)
            # 2. Preprocess (Drops 'offset' rows, usually ~14 for indicators) -> Length N - offset
            # 3. Features/Target Creation. Target y[k] corresponds to Preprocessed[k].
            #    Preprocessed[k] is Raw[k + offset].

            # Reconstruction:
            # y[t] is Log Return for time t (relative to Preprocessed).
            # Corresponds to Raw[t + offset].
            # LogRet_t = ln(Price_t / Price_{t-1})
            # Price_t = Price_{t-1} * exp(LogRet_t)

            # We need Price_{t-1}.
            # If y[t] is for Raw[t + offset], then Price_{t-1} is Raw[t + offset - 1].

            # Train Data:
            # X comes from `create_sequences` starting at `lookback` (seq_length).
            # First Train Target y[0] corresponds to Preprocessed[lookback].
            # Which is Raw[lookback + offset].
            # We need Raw[lookback + offset - 1] as base.

            train_len = len(train_preds_scaled)
            test_len = len(test_preds_scaled)

            # Train Reconstruction
            # Start index in PREPROCESSED frame
            train_start_idx_pp = lookback
            # Start index in RAW frame
            train_start_idx_raw = train_start_idx_pp + offset

            # We need Previous Closes: Raw[start-1 : start+len-1]
            train_prev_close = close_prices[train_start_idx_raw - 1 : train_start_idx_raw + train_len - 1]

            train_preds_ret = train_preds_scaled.flatten()
            train_preds_price = train_prev_close * np.exp(train_preds_ret)

            # Test Reconstruction
            # Test starts after Train in PREPROCESSED frame
            # Split point: int(len(dataset) * 0.8)
            # But we need to know the split index used in data_loader.
            # Rerun logic briefly or assume standard 0.8
            # In data_loader: train_size = int(len(dataset) * 0.8)
            # dataset length = len(df) - NaNs? No, dataset = df.values.
            # So split is on (N - offset).

            # Let's trust the lengths.
            # Test follows Train directly? No, usually gap of 'lookback' in TimeSeriesSplit?
            # data_loader uses:
            # train = data[:train_size]
            # test = data[train_size:]
            # X_test, y_test = create_sequences(test)
            # creates from [lookback : end].
            # So first Test Target is at test[lookback].
            # test[0] is data[train_size].
            # So test target[0] is data[train_size + lookback].
            # Raw index = train_size + lookback + offset.

            # Calculate train_size based on knowns
            # train_len = train_size - lookback
            train_size = train_len + lookback

            test_start_idx_pp = train_size + lookback
            test_start_idx_raw = test_start_idx_pp + offset

            test_prev_close = close_prices[test_start_idx_raw - 1 : test_start_idx_raw + test_len - 1]
            test_preds_ret = test_preds_scaled.flatten()
            test_preds_price = test_prev_close * np.exp(test_preds_ret)

            # Re-verify Test Indices plotting
            # Plot x-axis needs Raw Indices
            train_idx = range(train_start_idx_raw, train_start_idx_raw + train_len)
            test_idx = range(test_start_idx_raw, test_start_idx_raw + test_len)

            # 5. Visualization
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

            st.plotly_chart(fig, use_container_width=True)

            # 6. Live Market Sentiment
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

            # 7. Strategy Simulation
            st.subheader(f"Trading Strategy Simulation ({selected_strategy})")
            strategy = TradingStrategy(initial_capital=10000)

            # Get actual prices for test set
            actual_test_prices = original_data["Close"].values[test_idx]

            # Run strategy
            # Note: Now passing reconstructed prices
            if selected_strategy == "Simple Threshold":
                signals, portfolio_value = strategy.simple_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )
            elif selected_strategy == "Darvas Box":
                signals, portfolio_value = strategy.darvas_box_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )
            else:
                signals, portfolio_value = strategy.ma_crossover_strategy(
                    actual_test_prices, test_preds_price, stop_loss_pct=stop_loss
                )

            # Plot Portfolio Value
            final_val = portfolio_value[-1]
            profit = final_val - 10000

            st.metric(label="Final Portfolio Value", value=f"${final_val:,.2f}", delta=f"${profit:,.2f}")

            strat_fig = go.Figure()
            strat_fig.add_trace(
                go.Scatter(x=original_data.index[test_idx], y=portfolio_value[1:], mode="lines", name="Portfolio Value")
            )
            st.plotly_chart(strat_fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.exception(e)
