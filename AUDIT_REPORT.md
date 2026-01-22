# Red Team Audit Report: Stock Predictor

**Date:** 2024-05-23
**Auditor:** Jules (AI Senior Quantitative Developer)
**Scope:** `src/data_loader.py`, `src/strategy.py`, `src/train.py`

---

## 🚨 Critical Vulnerabilities

### 1. Sentiment Integration "Silent Failure"
**Severity:** Critical
**Location:** `src/data_loader.py` -> `preprocess_data`

**Description:**
The model is trained with the `Sentiment` feature set to a constant `0.0` for all historical rows. However, during inference (and the very end of the test set), live sentiment values (non-zero) are injected.
- **Root Cause:** `StandardScaler` fitted on a constant column (all zeros) results in `mean=0` and `variance=0`. Sklearn defaults `scale_=1.0` in this case.
- **Mechanism:**
    1. During training, the LSTM learns to ignore this input (or rather, the weights associated with it are never updated by gradient descent because the input is always 0, so they remain at their random initialization values).
    2. During live inference, a non-zero sentiment score (e.g., `0.5`) is fed in.
    3. This value is multiplied by the *random, unoptimized weights*.
- **Impact:**
    This injects purely random noise into the prediction during live trading. The model's behavior in production will be erratic and uncorrelated with its backtest performance (which likely used 0.0 for the bulk of testing). This is a catastrophic "works in dev, fails in prod" bug.

### 2. Missing Dependency
**Severity:** High (Operational)
**Location:** `src/data_loader.py` imports `vaderSentiment`

**Description:**
The `vaderSentiment` library is imported but missing from `requirements.txt`.
- **Impact:** The application will crash immediately upon startup for any new user attempting to install dependencies via the provided requirements file.

---

## ⚠️ Edge Cases & Logic Fragility

### 1. Indicator Calculation Gaps
**Location:** `src/data_loader.py` -> `_add_indicators`

**Description:**
The method calculates technical indicators (RSI, MACD) using `.diff()` and rolling windows on a concatenated DataFrame (`warmup_df` + `df`).
- **Risk:** If `warmup_df` and `df` are not strictly contiguous in time (e.g., if there is a gap of days/weeks between the end of warmup and start of df), the `.diff()` operation will calculate the price change across that gap as if it were a single time step.
- **Consequence:** This creates a massive artificial "return" spike at the first index of `df`, distorting RSI and MACD values for the subsequent window (14-26 steps). While current usage in `walk_forward_splits` appears to maintain continuity, this function is fragile to any future changes in data splitting logic.

### 2. Strategy Data Alignment Assumption
**Location:** `src/strategy.py` -> `robust_strategy`

**Description:**
The strategy relies on `predicted_prices[i+1]` corresponding to the day *after* `test_indices[i]`.
- **Analysis:** This assumption holds true given the current `create_sequences` logic (where target $y$ is $t+1$). However, it implicitly assumes `predicted_prices` and `test_indices` are perfectly aligned and of the same length. Any misalignment (e.g., if one array is truncated) could lead to look-ahead bias or index errors. The current code handles length mismatch via truncation (`min_len`), which is safe but silently masks potential upstream data issues.

---

## ⚡ Optimization Opportunities

### 1. Vectorization of Strategy Loop
**Location:** `src/strategy.py` -> `robust_strategy`

**Description:**
The strategy iterates row-by-row using `data.iloc[idx]`. DataFrame row access is significantly slower than Numpy array access.
- **Recommendation:** Convert relevant columns (`Open`, `High`, `Low`, `Close`, `ATR`, `SMA_50`) to Numpy arrays before the loop. Accessing `opens[idx]` instead of `data.iloc[idx]['Open']` can yield a 50-100x speedup for the loop execution.

---

## 🔍 Specific Questions Answered

### Leakage Regression Test
**Question:** Look closely at `DataLoader._add_indicators` and how it handles the `warmup_df`. Is there ANY edge case where information from row $N$ could leak into row $N-1$?
**Answer:** **No backward leakage found.** The functions used (`diff`, `rolling`, `ewm`) are causal (only use past data). `warmup_df` is prepended, so flow is strictly $Past \to Future$.
**Caveat:** As noted in "Edge Cases", while leakage (future to past) is not present, **Gap Distortion** is a risk if inputs are not contiguous.

### Strategy Execution Logic
**Question:** In `TradingStrategy.robust_strategy`, verify the execution timing. If a signal is generated at index $i$, are we strictly entering at $i+1$? Are the stop-loss checks for the existing position using strictly current available data?
**Answer:** **Yes, the logic is sound.**
- **Signal Generation:** Uses `predicted_prices[i+1]` (prediction for tomorrow) available at Close of Day $i$. Sets `pending_entry = True`.
- **Entry Execution:** Checks `if pending_entry` on iteration $i+1$ and uses `current_open` (Open of Day $i+1$) to enter. This correctly simulates "Market on Open" orders submitted after the previous close.
- **Stop-Loss/Take-Profit:** Checks `current_low` and `current_high` of the *current* bar ($i+1$) to trigger exits. This correctly simulates intraday stop orders.

### Walk-Forward Correctness
**Question:** Check `DataLoader.walk_forward_splits`. Does the expanding window logic correctly preserve time order? Are there off-by-one errors?
**Answer:** **Logic is correct.**
- The split logic `test_start = train_end` ensures strict continuity (index $K$ is excluded from train, included in test).
- The expanding window approach ($N-5S \dots N$) correctly covers the dataset without overlap or gaps in the test segments.

### Sentiment Integration
**Question:** Verify how preprocessing handles sentiment. Does filling historical values with 0.0 introduce bias?
**Answer:** **YES. This is the critical failure.**
- Filling history with `0.0` while feeding live values during inference breaks the IID assumption.
- The model treats "Sentiment" as a constant zero feature during training (learning nothing).
- During inference, it receives non-zero inputs which are multiplied by random, untrained weights.
- **Recommendation:** Either obtain historical sentiment data to train properly, or remove the feature entirely. Do not mix "dummy" 0.0 history with "live" real data.
