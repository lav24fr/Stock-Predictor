# Stock Predictor Project Overview

## 📖 Project Description

The **Stock Predictor** is a Streamlit-based application designed to forecast stock prices using Deep Learning (LSTM) and simulate algorithmic trading strategies. It integrates:

- **Historical Data**: Fetched via `yfinance`.
- **Sentiment Analysis**: VADER sentiment scoring on news headlines.
- **Deep Learning**: An LSTM (Long Short-Term Memory) neural network for time-series forecasting.
- **Strategy Simulation**: Backtesting engine for various trading strategies (Simple, MA Crossover, Darvas Box).

---

## 🚀 How to Use

### 1. Setup & Installation

Ensure dependencies are installed:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

### 2. Configuration (Sidebar)

- **Ticker Symbol**: Enter the stock ticker (e.g., `AAPL`, `TSLA`, `NVDA`).
- **Date Range**: Select start and end dates for training/testing.
- **Model Config**:
  - *Epochs*: Number of training iterations. Higher = longer training, risk of overfitting.
  - *Hidden Units*: Size of LSTM hidden layer. Complexity of patterns it can learn.
  - *Layers*: Number of stacked LSTM layers.
  - *Lookback*: Number of past days used to predict the next day.
  - *Dropout*: Regularization to prevent overfitting.
- **Strategy Config**:
  - *Strategy*: Choose the logic for buy/sell signals.
  - *Stop Loss*: Percentage drop to trigger an automatic sell (e.g., 2.0%).

### 3. Execution

Click **"Run Prediction"** to:

1. Fetch data.
2. Train the LSTM model on-the-fly.
3. Visualize `Actual` vs `Predicted` prices.
4. View Live Market Sentiment.
5. Simulate the selected trading strategy and view Portfolio Value performance.

---

## 🧠 Trading Strategies & Analysis

### 1. Simple Threshold Strategy

**How it works:**

- Calculates the % difference between **Predicted Next Price** and **Current Price**.
- **Buy Signal**: If `(Predicted - Current) / Current > 0.5%`.
- **Sell/Short Signal**: If `(Predicted - Current) / Current < -0.5%`.

**⚠️ Problems/Limitations:**

- **Model Dependency**: It blindly trusts the specific point prediction of the LSTM. If the model is slightly off (which is common in financial time-series), the strategy fails.
- **Whipsaw Risk**: In volatile markets, predictions might oscillate around the threshold, causing excessive buying/selling (racking up transaction costs, though ignored here).
- **Lag**: LSTMs often learn to just predict "Price(t) ≈ Price(t-1)". This strategy might just be momentum chasing in disguise.

### 2. Moving Average (MA) Crossover

**How it works:**

- Calculates Short (5-day) and Long (20-day) Moving Averages **on the Predicted Prices**.
- **Buy Signal**: Short MA crosses *above* Long MA.
- **Sell Signal**: Short MA crosses *below* Long MA.

**⚠️ Problems/Limitations:**

- **Double Lag**: Moving Averages are lagging indicators. Applying them to *Predicted* prices (which are often smoothed versions of reality) compounds this lag.
- **Sideways Markets**: Like all MA strategies, it loses money in choppy/sideways markets due to false crossover signals.
- **Signal Delay**: Entries and exits are often too late to capture rapid moves.

### 3. Darvas Box Strategy

**How it works:**

- Identifies consolidation ranges ("Boxes") based on recent Highs and Lows.
- **Top**: Established after 3 days of not exceeding a new high.
- **Bottom**: Established after 3 days of not dropping below a new low.
- **Buy Signal**: Price breaks *above* the Box Top (Breakout).
- **Sell Signal**: Price breaks *below* the Box Bottom (Breakdown) or hits Stop Loss.
- *Note*: This implementation builds boxes using **Predicted Prices**.

**⚠️ Problems/Limitations:**

- **Prediction Variance**: Neural networks (Mean Squared Error loss) tend to output smooth lines with lower variance than reality. They might rarely produce the jagged "highs" needed to form valid Darvas boxes.
- **False Breakouts**: If the model overshoots a prediction, it might trigger a breakout signal that doesn't exist in the real market.
- **Regime Mismatch**: Darvas is a *trend-following* method for volatile growth stocks. It performs poorly on mean-reverting or stable assets.

---

## 🛠 Model & Technical Architecture

- **Input**: `Log Returns`, `Volume Change`, `RSI`, `MACD`, `Signal Line`.
- **Target**: Next day's `Log Return`.
- **Reconstruction**: Predictions are converted back to Prices ($ P_t = P_{t-1} \times e^{ret} $) for visualization and strategy execution.
- **Data Leakage Prevention**: Scalers are fit *only* on the training set.
