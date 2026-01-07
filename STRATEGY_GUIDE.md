# 🧠 Trading Strategy Guide & Optimization Report

This document details the three algorithmic trading strategies available in the Stock Predictor, along with their optimal configurations based on a calibration optimization run on AAPL (2020-2023).

### 🏆 Key Optimization Finding

Across all strategies, a **Simpler Model** (20 Epochs, 32 Hidden Units, 1 Layer) significantly outperformed complex models. Complex models tended to overfit noise, leading to losses in trading.

---

## 1. Moving Average (MA) Crossover 🏆 **(MOST EFFICIENT)**

### 📖 How it Works

This is a trend-following strategy that generates signals based on the crossing of two moving averages derived from **Predicted Future Prices**.

- **Buy Signal**: When the *Short Window Forecasted MA* crosses *above* the *Long Window Forecasted MA*. This indicates a potential uptake in trend.
- **Sell Signal**: When the Short MA crosses below the Long MA.

### ⚙️ Optimal Parameters

Based on recent optimization:

- **Short Window**: `5 Days`
- **Long Window**: `50 Days`
- **Stop Loss**: `1.0%`

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
- **Layers**: `1`
- **Dropout**: `0.0`
*Reason*: Complex models tend to overfit noise. A simpler model produces smoother trend lines, which works perfectly with Moving Averages.

### 📊 Performance Note

In our calibration test (Test Period: ~3 months), this strategy yielded a **+6.9% Return**, turning $10,000 into **$10,691**.

- **Why it works**: The 5/50 combination allows it to capture medium-term trends while filtering out daily noise. Using a tight 1% Stop Loss prevents large drawdowns during false breakouts.

---

## 2. Darvas Box Strategy 📦 **(NEUTRAL / SAFE)**

### 📖 How it Works

This strategy looks for consolidation periods ("Boxes") where price trades within a range.

- **Box Top**: Established after 3 days of failing to make a new high.
- **Box Bottom**: Established after 3 days of failing to make a new low.
- **Buy Signal**: Price breaks *above* the Box Top (Breakout).
- **Sell Signal**: Price breaks *below* the Box Bottom (Breakdown).

### ⚙️ Optimal Parameters

- **Stop Loss**: `1.0%`
- **Timeframe**: Works best on *Weekly* charts, but adapted here for *Daily* data.

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
*Reason*: While one might think a complex model captures the volatility needed for boxes, our tests showed the Simple model was safer, preserving capital ($10,000) while complex models often triggered false breakouts leading to losses.

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
*Reason*: While one might think a complex model captures the volatility needed for boxes, our tests showed the Simple model was safer, preserving capital ($10,000) while complex models often triggered false breakouts leading to losses.

### 📊 Performance Note

In calibration, this strategy returned **$10,000 (0% gain/loss)**.

- **Why**: Darvas Boxes require strong, volatile trends to form valid boxes and breakouts. In choppy or "efficient" market periods, or when using smoothed LSTM predictions, boxes may fail to form, resulting in no trades (capital preservation).

---

## 3. Simple Threshold Strategy ⚠️ **(HIGH RISK)**

### 📖 How it Works

This strategy makes a trade decision every single day based on the raw predicted return for tomorrow.

- **Buy**: If `(Predicted_Next - Current) > 0.5%`.
- **Short**: If `(Predicted_Next - Current) < -0.5%`.

### ⚙️ Optimal Parameters

- **Stop Loss**: `5.0%` (Needs a wide stop because it is very volatile).

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
- **Layers**: `1`
- **Dropout**: `0.0`
*Critical Insight*: This strategy is **highly sensitive** to model noise.
- **Complex Model (50 epochs, 128 hidden)**: **-$1,700 Loss**.
- **Simple Model (20 epochs, 32 hidden)**: **+$2,364 Profit**.
The simple model prevents overfitting to daily noise, allowing the threshold strategy to capture genuine momentum.

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
- **Layers**: `1`
- **Dropout**: `0.0`
*Critical Insight*: This strategy is **highly sensitive** to model noise.
- **Complex Model (50 epochs, 128 hidden)**: **-$1,700 Loss**.
- **Simple Model (20 epochs, 32 hidden)**: **+$2,364 Profit**.
The simple model prevents overfitting to daily noise, allowing the threshold strategy to capture genuine momentum.

### 📊 Performance Note

In calibration, this strategy performed poorly, returning **$7,823 (-21.7% Loss)**.

- **Why**: It is "too fast." It reacts to every minor fluctuation in the model's prediction. Since LSTM predictions often contain noise or lag, this strategy enters many false trades, getting whipsawed by the market.

---

## 4. Robust Strategy 🛡️ **(ADVANCED / BALANCED)**

### 📖 How it Works

This strategy combines Machine Learning predictions with classical technical analysis to filter out bad trades and manage risk dynamically.

- **Trend Filter**: Only considers buying if the price is above the **50-Day SMA** (Bullish Regime).
- **Volatility Sizing**: Uses **Average True Range (ATR)** to calculate position size and stop-loss distance, ensuring risk is normalized based on market volatility.
- **Signal**: Buys when the Model predicts a price increase greater than `0.5 * ATR` (meaningful move).

### ⚙️ Parameters (Auto-Configured)

- **Stop Loss**: Dynamic, set at `2 * ATR` below entry.
- **Take Profit**: Dynamic, set at `3 * ATR` above entry (1.5 Risk-Reward Ratio).
- **Risk Per Trade**: Fixed at `2%` of capital.

### 🤖 Best Model Configuration

- **Complexity**: **Low (Fast/Simple)**
- **Epochs**: `20`
- **Hidden Units**: `32`
*Reason*: Since this strategy relies on the model primarily for *directional confirmation* rather than precise price targets, a simple, stable model is sufficient and preferred.

### 📊 Performance Note

This strategy is designed to be **robust**. It trades less frequently than the Simple Threshold strategy but with higher conviction. It avoids trading against the major trend (thanks to SMA) and avoids getting stopped out by normal noise (changes stop distance based on volatility).

---

## 💡 Recommendation

For the best balance of risk and return using this AI model:

1. **Use MA Crossover**: Set Short=5, Long=50, Stop Loss=1.0%.
2. **For High Risk/Reward**: Use **Simple Threshold** but ONLY with a **Simple Model (20 Epochs)**. If you overtrain (100 epochs), you will likely lose money.
