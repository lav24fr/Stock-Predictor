# Stock Price Predictor & Strategy Simulator 📈

An advanced LSTM-based stock price prediction application that simulates various trading strategies. Built with **PyTorch**, **Streamlit**, and **yfinance**.

![App Screenshot](https://via.placeholder.com/800x400?text=Stock+Predictor+Dashboard)

## Features

- **Deep Learning Model**: Two-layer LSTM with Dropout and L2 Regularization, trained on Log Returns.
- **Sentiment Analysis**: Integration with VADER Sentiment to analyze live news headlines.
- **Trading Strategies**:
  - **MA Crossover**: Classic Golden/Death Cross strategy.
  - **Darvas Box**: Automatic box theory implementation.
  - **Robust Strategy**: Trend-following with ATR-based entry/exit and predicted price targets.
  - **Simple Threshold**: Momentum-based entry on strong prediction signals.
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR.
- **Interactive Dashboard**: Real-time backtesting and PnL visualization.

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/stock-predictor.git
   cd stock-predictor
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

## Project Structure

- `app.py`: Main Streamlit dashboard application.
- `src/model.py`: PyTorch LSTM model definition.
- `src/train.py`: Training loop with Early Stopping and LR Scheduling.
- `src/data_loader.py`: Data fetching (yfinance) and preprocessing pipeline.
- `src/strategy.py`: Implementation of trading logic.

## Strategy Details

- **Robust Strategy**: Confirms uptrend with 50-day SMA. Enters if predicted price move > 0.2% OR > 0.02 *ATR. Uses Trailing Stop Loss (2* ATR).
- **MA Crossover**: Buys when Short MA crosses above Long MA.
- **Simple Threshold**: Buys if next-day predicted return > 0.5%.

## License

MIT
