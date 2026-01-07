import pandas as pd


class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def simple_strategy(self, actual_prices, predicted_prices, stop_loss_pct=0.0):
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
        capital = self.initial_capital
        position = 0
        entry_price = 0
        portfolio_value = [capital]
        signals = []

        pred_series = pd.Series(predicted_prices.flatten())
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

            curr_short = short_ma[i+1]
            curr_long = long_ma[i+1]
            prev_short = short_ma[i]
            prev_long = long_ma[i]

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
            elif short_ma[i] < long_ma[i] and short_ma[i - 1] >= long_ma[i - 1]:
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

        prices = predicted_prices.flatten()
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
        capital = self.initial_capital
        position = 0
        entry_price = 0
        stop_loss_price = 0
        take_profit_price = 0
        portfolio_value = [capital]
        signals = []

        data = data.copy()
        data['SMA_50'] = data['Close'].rolling(window=50).mean()
        data['ATR'] = self.calculate_atr(data)

        if len(test_indices) != len(predicted_prices):
            min_len = min(len(test_indices), len(predicted_prices))
            test_indices = list(test_indices)[:min_len]
            predicted_prices = predicted_prices[:min_len]

        for i, idx in enumerate(test_indices):
            if i == 0:
                signals.append(0)
                portfolio_value.append(capital)
                continue
                
            current_idx = test_indices[i-1]
            current_row = data.iloc[current_idx]
            current_price = current_row['Close']
            
            pred_next_price = predicted_prices[i]
            
            atr = current_row['ATR']
            sma_50 = current_row['SMA_50']
            
            if pd.isna(atr) or pd.isna(sma_50):
                signals.append(0)
                portfolio_value.append(capital + position * current_price)
                continue

            signal = 0
            
            if position != 0:
                next_day_row = data.iloc[test_indices[i]]
                next_close = next_day_row['Close']
                next_low = next_day_row['Low']
                next_high = next_day_row['High']
                
                triggered_exit = False
                
                if position > 0:
                    if next_low < stop_loss_price:
                        exit_price = stop_loss_price
                        if next_day_row['Open'] < stop_loss_price:
                             exit_price = next_day_row['Open']
                        
                        capital += position * exit_price
                        position = 0
                        triggered_exit = True
                        signal = -1
                    elif next_high > take_profit_price:
                        exit_price = take_profit_price
                        if next_day_row['Open'] > take_profit_price:
                            exit_price = next_day_row['Open']
                            
                        capital += position * exit_price
                        position = 0
                        triggered_exit = True
                        signal = -1
                
                if triggered_exit:
                    portfolio_value.append(capital)
                    signals.append(signal)
                    continue
            
            if position == 0:
                trend_is_up = current_price > sma_50
                expected_move = pred_next_price - current_price
                is_meaningful_move = expected_move > (0.1 * atr)
                
                if trend_is_up and is_meaningful_move:
                    risk_amount = capital * risk_per_trade
                    dist_sl = 2 * atr
                    
                    if dist_sl > 0:
                        shares_to_buy = risk_amount / dist_sl
                        max_shares = capital / current_price
                        shares = min(shares_to_buy, max_shares)
                        
                        if shares > 0:
                            position = shares
                            entry_price = current_price
                            capital -= shares * current_price
                            
                            stop_loss_price = entry_price - (2 * atr)
                            take_profit_price = entry_price + (3 * atr)
                            
                            signal = 1
            
            price_t = data.iloc[test_indices[i]]['Close']
            val = capital + (position * price_t)
            portfolio_value.append(val)
            signals.append(signal)

        return signals, portfolio_value
