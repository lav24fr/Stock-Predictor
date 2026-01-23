import pandas as pd
import numpy as np


class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def simple_strategy(self, actual_prices, predicted_prices, stop_loss_pct=0.0):
        actual_prices = np.array(actual_prices).flatten()
        predicted_prices = np.array(predicted_prices).flatten()
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        for i in range(len(actual_prices) - 1):
            curr_price = actual_prices[i]
            pred_next_price = predicted_prices[i+1]

            triggered_sl = False
            if position != 0 and stop_loss_pct > 0:
                if position > 0 and curr_price < entry_price * (1 - stop_loss_pct):
                    capital += position * curr_price
                    position = 0
                    triggered_sl = True
                    signals.append(-1)
                elif position < 0 and curr_price > entry_price * (1 + stop_loss_pct):
                    capital += position * curr_price
                    position = 0
                    triggered_sl = True
                    signals.append(1)

            if triggered_sl:
                portfolio_value.append(capital)
                continue

            diff = (pred_next_price - curr_price) / curr_price

            if diff > 0.005:
                if position < 0:
                    capital += position * curr_price
                    position = 0

                if position == 0:
                    position = capital / curr_price
                    entry_price = curr_price
                    capital = 0
                    signals.append(1)
                else:
                    signals.append(0)

            elif diff < -0.005:
                if position > 0:
                    capital += position * curr_price
                    position = 0

                if position == 0:
                    max_shares = capital / curr_price
                    position = -max_shares
                    entry_price = curr_price
                    capital += abs(position) * curr_price
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)

            current_val = capital + (position * curr_price)
            portfolio_value.append(current_val)

        return signals, portfolio_value

    def ma_crossover_strategy(self, actual_prices, predicted_prices, short_window=5, long_window=20, stop_loss_pct=0.0):
        actual_prices = np.array(actual_prices).flatten()
        predicted_prices = np.array(predicted_prices).flatten()
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        pred_series = pd.Series(predicted_prices)
        short_ma = pred_series.rolling(window=short_window).mean()
        long_ma = pred_series.rolling(window=long_window).mean()

        for i in range(len(predicted_prices) - 1):
            price = actual_prices[i] if i < len(actual_prices) else predicted_prices[i]

            if i + 1 < long_window:
                signals.append(0)
                portfolio_value.append(capital + (position * price))
                continue

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

            # Use current and previous values (no look-ahead)
            curr_short = short_ma[i]
            curr_long = long_ma[i]
            prev_short = short_ma[i-1] if i > 0 else short_ma[i]
            prev_long = long_ma[i-1] if i > 0 else long_ma[i]

            if curr_short > curr_long and prev_short <= prev_long:
                if position < 0:
                    capital += position * price
                    position = 0
                if position == 0:
                    position = capital / price
                    entry_price = price
                    capital = 0
                    signals.append(1)
                else:
                    signals.append(0)
            elif curr_short < curr_long and prev_short >= prev_long:
                if position > 0:
                    capital += position * price
                    position = 0
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
        actual_prices = np.array(actual_prices).flatten()
        predicted_prices = np.array(predicted_prices).flatten()
        
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        box_top = -1.0
        box_bottom = -1.0
        state = "NO_BOX"

        current_period_high = -1.0
        current_period_low = 1e9
        days_since_high = 0
        days_since_low = 0

        trailing_stop_price = -1.0

        prices = predicted_prices
        actuals = actual_prices

        for i in range(len(prices)):
            price = prices[i]
            exec_price = actuals[i] if i < len(actuals) else price

            signal = 0
            triggered_sl = False

            if position > 0:
                if stop_loss_pct > 0 and exec_price < entry_price * (1 - stop_loss_pct):
                    triggered_sl = True

                elif trailing_stop_price > 0 and price < trailing_stop_price:
                    triggered_sl = True

                if triggered_sl:
                    capital += position * exec_price
                    position = 0
                    signal = -1

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

            if state == "NO_BOX":
                if price > current_period_high:
                    current_period_high = price
                    days_since_high = 0
                else:
                    days_since_high += 1

                if days_since_high == 3:
                    box_top = current_period_high
                    state = "TOP_SET"
                    current_period_low = price
                    days_since_low = 0

            elif state == "TOP_SET":
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

                    if days_since_low == 3:
                        box_bottom = current_period_low
                        state = "BOX_ESTABLISHED"

                        if position > 0:
                            trailing_stop_price = box_bottom

            elif state == "BOX_ESTABLISHED":

                if price > box_top:
                    if position == 0:
                        position = capital / exec_price
                        entry_price = exec_price
                        capital = 0
                        signal = 1
                        trailing_stop_price = box_bottom

                    state = "NO_BOX"
                    current_period_high = price
                    days_since_high = 0
                    current_period_low = 1e9

                elif price < box_bottom:
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
        Robust strategy with ATR-based position sizing and dynamic stops.
        
        Fixed to avoid look-ahead bias:
        - Signals are generated at end of day t based on prediction for t+1
        - Entry execution happens at open of day t+1
        - Stop-loss/take-profit are checked using current bar's High/Low
          (simulating intraday limit/stop orders placed the previous day)
        """
        capital = self.initial_capital
        position = 0
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        portfolio_value = [capital]
        signals = []
        
        pending_entry = False
        pending_entry_atr = 0

        data = data.copy()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['ATR'] = self.calculate_atr(data)

        if len(test_indices) != len(predicted_prices):
            min_len = min(len(test_indices), len(predicted_prices))
            test_indices = list(test_indices)[:min_len]
            predicted_prices = predicted_prices[:min_len]

        for i, idx in enumerate(test_indices):
            current_row = data.iloc[idx]
            
            def get_scalar(val):
                """Convert potential Series to scalar."""
                if hasattr(val, 'item'):
                    return val.item()
                elif hasattr(val, 'iloc'):
                    return float(val.iloc[0]) if len(val) > 0 else float('nan')
                return float(val)
            
            current_open = get_scalar(current_row['Open'])
            current_high = get_scalar(current_row['High'])
            current_low = get_scalar(current_row['Low'])
            current_close = get_scalar(current_row['Close'])
            
            atr = get_scalar(current_row['ATR'])
            sma_50 = get_scalar(current_row['SMA_50'])
            
            signal = 0
            
            if pending_entry and position == 0:
                entry_price = current_open
                risk_amount = capital * risk_per_trade
                dist_sl = 2 * pending_entry_atr
                
                if dist_sl > 0:
                    shares_to_buy = risk_amount / dist_sl
                    max_shares = capital / entry_price
                    shares = min(shares_to_buy, max_shares)
                    
                    if shares > 0:
                        position = shares
                        capital -= shares * entry_price
                        
                        stop_loss_price = entry_price - (2 * pending_entry_atr)
                        take_profit_price = entry_price + (3 * pending_entry_atr)
                        
                        signal = 1
                
                pending_entry = False
                pending_entry_atr = 0
            
            if position > 0:
                triggered_exit = False
                exit_price = 0
                
                if current_low <= stop_loss_price:
                    if current_open <= stop_loss_price:
                        exit_price = current_open
                    else:
                        exit_price = stop_loss_price
                    triggered_exit = True
                    
                elif current_high >= take_profit_price:
                    if current_open >= take_profit_price:
                        exit_price = current_open
                    else:
                        exit_price = take_profit_price
                    triggered_exit = True
                
                if triggered_exit:
                    capital += position * exit_price
                    position = 0
                    signal = -1
                    stop_loss_price = 0
                    take_profit_price = 0
            
            if position == 0 and not pending_entry:
                if not (np.isnan(atr) or np.isnan(sma_50)):
                    trend_is_up = bool(current_close > sma_50)
                    
                    if i + 1 < len(predicted_prices):
                        pred_val = np.array(predicted_prices[i + 1]).flatten()
                        pred_next_price = float(pred_val[0]) if len(pred_val) > 0 else current_close
                        expected_move = pred_next_price - current_close
                        expected_pct_move = expected_move / current_close if current_close > 0 else 0
                        
                        # Relaxed conditions: either ATR-based OR percentage-based threshold
                        # ATR threshold lowered from 0.1 to 0.02 for more trades
                        is_meaningful_atr_move = atr > 0 and expected_move > (0.02 * atr)
                        is_meaningful_pct_move = expected_pct_move > 0.002  # 0.2% predicted up move
                        
                        if trend_is_up and (is_meaningful_atr_move or is_meaningful_pct_move):
                            pending_entry = True
                            pending_entry_atr = atr if atr > 0 else current_close * 0.02  # Fallback ATR
            
            val = capital + (position * current_close)
            portfolio_value.append(val)
            signals.append(signal)

        return signals, portfolio_value

