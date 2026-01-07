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
        for i in range(len(predicted_prices)):
            price = actual_prices[i] if i < len(actual_prices) else predicted_prices[i]

            if i < long_window:
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

            # Crossover Check
            # Bullish Cross (Short > Long)
            if short_ma[i] > long_ma[i] and short_ma[i - 1] <= long_ma[i - 1]:
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
