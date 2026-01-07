
# 📈 AI Stock Predictor & Strategy Simulator

A powerful, interactive Streamlit application that uses Deep Learning (LSTM) to forecast stock prices and simulate algorithmic trading strategies.

## 🚀 Key Features

* **Deep Learning Forecast**: Uses a PyTorch-based Long Short-Term Memory (LSTM) network to predict future price movements based on historical data.
* **Live Sentiment Analysis**: Integrates news sentiment from VADER analysis to adjust predictions in real-time.
* **Strategy Simulation**: Backtest three distinct trading strategies on the predicted data.
  * **MA Crossover**: Trend-following using forecasted Moving Averages.
  * **Darvas Box**: Breakout detection for strong trends.
  * **Simple Threshold**: High-risk, high-reward signal based on raw momentum.
  * **Robust Strategy**: ML-Enhanced strategy with ATR Volatility sizing and SMA Trend filtering.
* **Interactive Dashboard**: Fully customizable simulation parameters (Lookback, Stop Loss, Model Complexity) via Streamlit.

## 🛠️ Installation

1. **Clone the Repository**

    ```bash
    git clone https://github.com/lav24fr/Stock-Predictor.git
    cd Stock-Predictor
    ```

2. **Install Dependencies**

    ```bash
    pip install -r requirements.txt
    ```

3. **Run the App**

    ```bash
    streamlit run app.py
    ```

## 📘 Trading Strategies

For a deep dive into the optimal parameters and logic for each strategy, please read our **[Strategy Guide](STRATEGY_GUIDE.md)**.

**Quick Summary:**

* 🏆 **Best Overall**: **MA Crossover** (Short=5, Long=50, SL=1%). Reliable and trend-safe.
* 📦 **Safest**: **Darvas Box**. Preserves capital in choppy markets.
* ⚡ **High Risk**: **Simple Threshold**. Only use with a simple model (20 epochs) to avoid overfitting.

## 🧠 Model Configuration

The application allows you to tweak the AI model. Our research shows:

* **Simple Models (20 Epochs, 32 Hidden Units)** perform **better** for trading than complex ones.
* Complex models tends to "memorize" noise, leading to whipsaws in trading signals.

## 📂 Project Structure

* `app.py`: Main Streamlit dashboard and UI logic.
* `src/`: Core logic modules.
  * `model.py`: LSTM PyTorch architecture.
  * `train.py`: Training loop and prediction logic.
  * `strategy.py`: Implementation of trading algorithms.
  * `data_loader.py`: Data fetching (yfinance) and preprocessing.
* `tests/`: Unit tests for critical logic.
