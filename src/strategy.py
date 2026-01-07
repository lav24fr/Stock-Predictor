import pandas as pd


class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def simple_strategy(self, actual_prices, predicted_prices, stop_loss_pct=0.0):
        """
        Simple strategy: If predicted price > current price (with threshold), Buy/Go Long.
        If predicted price < current price, Sell/Go Short.
        Includes Stop Loss.
        """
        capital = self.initial_capital
        position = 0  # Number of shares (+ for Long, - for Short)
        entry_price = 0
        portfolio_value = [capital]

        signals = []  # 1: Buy, -1: Sell, 0: Hold

        # Determine signals
        for i in range(len(actual_prices) - 1):
            curr_price = actual_prices[i]
            # Use prediction for NEXT day (i+1) vs Current Price (i)
            # predicted_prices contains predictions aligned such that index j is prediction for time j
            # We want pred_next_price to be prediction for time i+1
            pred_next_price = predicted_prices[i+1]

            # Stop Loss Check
            triggered_sl = False
            if position != 0 and stop_loss_pct > 0:
                # Long Position SL: Price drops below entry
                if position > 0 and curr_price < entry_price * (1 - stop_loss_pct):
                    capital += position * curr_price
                    position = 0
                    triggered_sl = True
                    signals.append(-1)
                # Short Position SL: Price rises above entry
                elif position < 0 and curr_price > entry_price * (1 + stop_loss_pct):
                    capital += position * curr_price  # position is negative, so this subtracts cost to cover
                    position = 0
                    triggered_sl = True
                    signals.append(1)  # Buy to cover

            if triggered_sl:
                portfolio_value.append(capital)
                continue

            # Trading Logic
            # Percentage diff
            diff = (pred_next_price - curr_price) / curr_price

            # Bullish Signal
            if diff > 0.005:
                # If Short, Cover First
                if position < 0:
                    capital += position * curr_price
                    position = 0

                # Go Long if Neutral
                if position == 0:
                    position = capital / curr_price
                    entry_price = curr_price
                    capital = 0
                    signals.append(1)
                else:
                    # Already Long
                    signals.append(0)

            # Bearish Signal
            elif diff < -0.005:
                # If Long, Sell First
                if position > 0:
                    capital += position * curr_price
                    position = 0

                # Go Short if Neutral
                if position == 0:
                    # Shorting: We get cash? No, usually margin.
                    # Simplified: We treat 'capital' as collateral.
                    # We sell X shares. We assume we can leverage 1x.
                    # Position = - (Capital / Price)
                    max_shares = capital / curr_price
                    position = -max_shares
                    entry_price = curr_price
                    # Capital stays as collateral (conceptually)
                    # For simple arithmetic: Portfolio = Capital + (Position * Price)
                    # When we enter short: Value = C + (-C/P * P) = 0? No.
                    # Standard Sim: Cash increases by short sale proceeds.
                    capital += abs(position) * curr_price
                    signals.append(-1)
                else:
                    # Already Short
                    signals.append(0)
            else:
                signals.append(0)

            # Update portfolio value
            # Equity = Cash + Market Value of Positions
            current_val = capital + (position * curr_price)
            portfolio_value.append(current_val)

        return signals, portfolio_value

    def ma_crossover_strategy(self, actual_prices, predicted_prices, short_window=5, long_window=20, stop_loss_pct=0.0):
        """
        Moving Average Crossover Strategy using Predicted Prices.
        Buy when Short MA crosses above Long MA (Go Long).
        Sell when Short MA crosses below Long MA (Go Short).
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        # Convert to Series for easy MA calculation
        pred_series = pd.Series(predicted_prices.flatten())
        short_ma = pred_series.rolling(window=short_window).mean()
        long_ma = pred_series.rolling(window=long_window).mean()

        # We can only trade after long_window available
        # We can only trade after long_window is available in the PREDICTION series
        # We start checking at i such that i+1 >= long_window (since we look at i+1)
        # So i >= long_window - 1
        
        # Determine signals
        # We iterate up to len-1 because we look at i+1
        for i in range(len(predicted_prices) - 1):
            price = actual_prices[i] if i < len(actual_prices) else predicted_prices[i]

            # We need valid MA at i+1
            if i + 1 < long_window:
                signals.append(0)
                portfolio_value.append(capital + (position * price))
                continue

            # Stop Loss Check
            triggered_sl = False
            if position != 0 and stop_loss_pct > 0:
                if position > 0 and price < entry_price * (1 - stop_loss_pct):
                    capital += position * price
                    position = 0
                    triggered_sl = True
                    signals.append(-1)
                elif position < 0 and price > entry_price * (1 + stop_loss_pct):
                    capital += position * price
                    position = 0
                    triggered_sl = True
                    signals.append(1)

            if triggered_sl:
                portfolio_value.append(capital)
                continue

            # Crossover Check using FORECASTED MAs (i+1)
            # Available at time i (since Pred[i+1] is known at i)
            curr_short = short_ma[i+1]
            curr_long = long_ma[i+1]
            prev_short = short_ma[i]
            prev_long = long_ma[i]

            # Bullish Cross (Short > Long)
            if curr_short > curr_long and prev_short <= prev_long:
                # Cover Short
                if position < 0:
                    capital += position * price
                    position = 0
                # Go Long
                if position == 0:
                    position = capital / price
                    entry_price = price
                    capital = 0
                    signals.append(1)
                else:
                    signals.append(0)
            # Bearish Cross (Short < Long)
            elif short_ma[i] < long_ma[i] and short_ma[i - 1] >= long_ma[i - 1]:
                # Sell Long
                if position > 0:
                    capital += position * price
                    position = 0
                # Go Short
                if position == 0:
                    max_shares = capital / price
                    position = -max_shares
                    entry_price = price
                    capital += abs(position) * price
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)

            current_val = capital + (position * price)
            portfolio_value.append(current_val)

        return signals, portfolio_value

    def darvas_box_strategy(self, actual_prices, predicted_prices, stop_loss_pct=0.0):
        """
        Darvas Box Strategy adapted for Predicted Close Prices.
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        # Darvas State Variables
        box_top = -1.0
        box_bottom = -1.0
        state = "NO_BOX"

        # Tracking Highs/Lows for Box formation
        current_period_high = -1.0
        current_period_low = 1e9
        days_since_high = 0
        days_since_low = 0

        # Persistent Stop Loss (The "Floor")
        trailing_stop_price = -1.0

        prices = predicted_prices.flatten()
        actuals = actual_prices

        for i in range(len(prices)):
            price = prices[i]
            # Execution price: use actuals if available (avoids lookahead bias in execution)
            exec_price = actuals[i] if i < len(actuals) else price

            signal = 0  # Default to Hold
            triggered_sl = False

            # --- 1. Stop Loss Logic (Highest Priority) ---
            if position > 0:
                # A. Fixed Percentage Stop Loss (Emergency parachute)
                if stop_loss_pct > 0 and exec_price < entry_price * (1 - stop_loss_pct):
                    triggered_sl = True

                # B. Darvas Box Stop (Trailing Stop)
                # We sell if price drops below the LAST confirmed box bottom
                elif trailing_stop_price > 0 and price < trailing_stop_price:
                    triggered_sl = True

                if triggered_sl:
                    capital += position * exec_price
                    position = 0
                    signal = -1

                    # Full Reset of State
                    state = "NO_BOX"
                    box_top = -1.0
                    box_bottom = -1.0
                    trailing_stop_price = -1.0
                    current_period_high = -1.0
                    current_period_low = 1e9
                    days_since_high = 0
                    days_since_low = 0

                    portfolio_value.append(capital)
                    signals.append(signal)
                    continue

            # --- 2. Box Formation Logic ---

            # State 1: Searching for a Ceiling (Top)
            if state == "NO_BOX":
                if price > current_period_high:
                    current_period_high = price
                    days_since_high = 0
                else:
                    days_since_high += 1

                if days_since_high == 3:  # Strictly 3 days of non-violation
                    box_top = current_period_high
                    state = "TOP_SET"
                    current_period_low = price  # Reset low search starting today
                    days_since_low = 0

            # State 2: Searching for a Floor (Bottom)
            elif state == "TOP_SET":
                # If price breaks the potential top, the top was invalid. Reset.
                if price > box_top:
                    current_period_high = price
                    days_since_high = 0
                    state = "NO_BOX"
                else:
                    if price < current_period_low:
                        current_period_low = price
                        days_since_low = 0
                    else:
                        days_since_low += 1

                    if days_since_low == 3:  # Strictly 3 days of non-violation
                        box_bottom = current_period_low
                        state = "BOX_ESTABLISHED"

                        # If we are ALREADY holding, we raise the stop loss (Pyramiding safety)
                        if position > 0:
                            trailing_stop_price = box_bottom

            # State 3: Box Complete - Wait for Breakout or Breakdown
            elif state == "BOX_ESTABLISHED":

                # Upside Breakout -> BUY
                if price > box_top:
                    if position == 0:
                        position = capital / exec_price
                        entry_price = exec_price
                        capital = 0
                        signal = 1
                        # Set initial stop loss at the bottom of the box we just broke out of
                        trailing_stop_price = box_bottom

                    # Whether we bought or held, the box is "broken" upwards.
                    # We start looking for a NEW box on top of this one.
                    state = "NO_BOX"
                    current_period_high = price
                    days_since_high = 0
                    current_period_low = 1e9

                # Downside Breakdown -> Reset logic (Stop Loss handled at top of loop)
                elif price < box_bottom:
                    # Even if we aren't holding, the box pattern failed. Reset.
                    state = "NO_BOX"
                    current_period_high = price
                    days_since_high = 0

            signals.append(signal)
            current_val = capital + (position * exec_price)
            portfolio_value.append(current_val)

        return signals, portfolio_value

    def calculate_atr(self, df, period=14):
        high = df['High']
        low = df['Low']
        close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - close).abs()
        tr3 = (low - close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        return atr

    def robust_strategy(self, data, test_indices, predicted_prices, risk_per_trade=0.02):
        """
        Robust ML-Enhanced Strategy with Risk Management.
        
        Args:
            data (pd.DataFrame): Full OHLC dataframe.
            test_indices (iterable): Indices corresponding to the test period in 'data'.
            predicted_prices (np.array): ML predictions for the test period.
            risk_per_trade (float): Fraction of capital to risk per trade (e.g., 0.02 for 2%).
            
        Returns:
            signals, portfolio_value
        """
        capital = self.initial_capital
        position = 0  # Shares
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        portfolio_value = [capital]
        signals = []

        # 1. Pre-calculate Indicators on full data
        # Trend Filter: 50-day SMA
        # Create a copy to avoid SettingWithCopyWarning on the original df
        data = data.copy()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        
        # Volatility: ATR
        data['ATR'] = self.calculate_atr(data)
        
        # We perform the simulation over the test indices
        # predicted_prices[i] corresponds to data.iloc[test_indices[i]]
        
        # Alignment check
        if len(test_indices) != len(predicted_prices):
            # Truncate to shorter
            min_len = min(len(test_indices), len(predicted_prices))
            test_indices = list(test_indices)[:min_len] # Ensure sliceable
            predicted_prices = predicted_prices[:min_len]

        for i, idx in enumerate(test_indices):
            # Current Market State (at close of day 'idx')
            
            # NOTE: We can't use future data.
            # unique 'idx' is the row in 'data'. 
            # predicted_prices[i] is the price we expect at `test_indices[i]`.
            
            if i == 0:
                # Can't trade at very first step if we need diff
                signals.append(0)
                portfolio_value.append(capital)
                continue
                
            current_idx = test_indices[i-1] # Today
            current_row = data.iloc[current_idx]
            current_price = current_row['Close']
            
            # Prediction for Tomorrow (which is index i)
            pred_next_price = predicted_prices[i]
            
            # Indicators (available Today)
            atr = current_row['ATR']
            sma_50 = current_row['SMA_50']
            
            if pd.isna(atr) or pd.isna(sma_50):
                # Not enough data for indicators
                signals.append(0)
                portfolio_value.append(capital + position * current_price)
                continue

            signal = 0
            
            # --- RISK MANAGEMENT CHECKS (Stop Loss / Take Profit) ---
            if position != 0:
                # Check High/Low of NEXT day (which is index i - the 'actual' price movement)
                next_day_row = data.iloc[test_indices[i]]
                next_close = next_day_row['Close']
                next_low = next_day_row['Low']
                next_high = next_day_row['High']
                
                # Check SL/TP hits intraday
                triggered_exit = False
                
                if position > 0: # Long
                    if next_low < stop_loss_price:
                        # Stopped out
                        exit_price = stop_loss_price 
                        # Gap handling
                        if next_day_row['Open'] < stop_loss_price:
                             exit_price = next_day_row['Open'] 
                        
                        capital += position * exit_price
                        position = 0
                        triggered_exit = True
                        signal = -1
                    elif next_high > take_profit_price:
                        # Take Profit
                        exit_price = take_profit_price
                        if next_day_row['Open'] > take_profit_price:
                            exit_price = next_day_row['Open']
                            
                        capital += position * exit_price
                        position = 0
                        triggered_exit = True
                        signal = -1
                
                # We update value *after* potential exit
                if triggered_exit:
                    portfolio_value.append(capital)
                    signals.append(signal)
                    continue
            
            # --- ENTRY LOGIC (If Flat) ---
            if position == 0:
                # 1. Regime Filter: Trend must be up
                trend_is_up = current_price > sma_50
                
                # 2. Prediction Confirmation
                # Predicted gain must be > 0.1 * ATR (Lowered from 0.5 to catch more moves)
                expected_move = pred_next_price - current_price
                is_meaningful_move = expected_move > (0.1 * atr)
                
                if trend_is_up and is_meaningful_move:
                    # BUY SIGNAL
                    risk_amount = capital * risk_per_trade
                    dist_sl = 2 * atr
                    
                    if dist_sl > 0:
                        shares_to_buy = risk_amount / dist_sl
                        
                        # Cap shares by available capital
                        max_shares = capital / current_price
                        shares = min(shares_to_buy, max_shares)
                        
                        if shares > 0:
                            position = shares
                            entry_price = current_price
                            capital -= shares * current_price
                            
                            # Set Exits
                            stop_loss_price = entry_price - (2 * atr)
                            take_profit_price = entry_price + (3 * atr) # 1.5R ratio
                            
                            signal = 1
            
            # Update Portfolio Value
            price_t = data.iloc[test_indices[i]]['Close'] # The close of the day we stepped into
            val = capital + (position * price_t)
            portfolio_value.append(val)
            signals.append(signal)

        return signals, portfolio_value
