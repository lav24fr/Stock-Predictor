import pandas as pd
import numpy as np

class TradingStrategy:
    def __init__(self, initial_capital=10000):
        self.initial_capital = initial_capital

    def simple_strategy(self, actual_prices, predicted_prices):
        """
        Simple strategy: If predicted price > current price (with threshold), Buy.
        Otherwise Sell/Hold.
        """
        capital = self.initial_capital
        position = 0 # Number of shares
        portfolio_value = [capital]
        
        signals = [] # 1: Buy, -1: Sell, 0: Hold
        
        # Determine signals
        for i in range(len(actual_prices) - 1):
            curr_price = actual_prices[i]
            pred_next_price = predicted_prices[i] # Corresponds to prediction for i+1 (depending on alignment)
            
            # Simple logic: If we predict price goes up, buy
            if pred_next_price > curr_price * 1.005: # Threshold 0.5% gain
                if position == 0:
                    position = capital / curr_price
                    capital = 0
                    signals.append(1)
                else:
                    signals.append(0)
            elif pred_next_price < curr_price * 0.995: # Threshold 0.5% loss
                if position > 0:
                    capital = position * curr_price
                    position = 0
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
            
            # Update portfolio value
            current_val = capital + (position * curr_price)
            portfolio_value.append(current_val)
            
        return signals, portfolio_value

    def ma_crossover_strategy(self, actual_prices, predicted_prices, short_window=5, long_window=20):
        """
        Moving Average Crossover Strategy using Predicted Prices.
        Buy when Short MA crosses above Long MA.
        Sell when Short MA crosses below Long MA.
        """
        capital = self.initial_capital
        position = 0
        portfolio_value = [capital]
        signals = [] # 1: Buy, -1: Sell, 0: Hold
        
        # We need a history for MA, so we combine actual history (if available) or just run on the predicted window
        # For simplicity in this prototype, we'll calculate MAs on the 'predicted_prices' series directly
        # effectively assuming the prediction curve is the reality we are trading on.
        
        # Convert to Series for easy MA calculation
        pred_series = pd.Series(predicted_prices.flatten())
        short_ma = pred_series.rolling(window=short_window).mean()
        long_ma = pred_series.rolling(window=long_window).mean()
        
        # We can only trade after long_window available
        for i in range(len(predicted_prices)):
            price = actual_prices[i] if i < len(actual_prices) else predicted_prices[i]
            
            # Skip until we have enough data for MA
            if i < long_window:
                signals.append(0)
                portfolio_value.append(capital + (position * price))
                continue
                
            # Check Crossover
            # If Short > Long AND Short[prev] <= Long[prev] -> BUY
            if short_ma[i] > long_ma[i] and short_ma[i-1] <= long_ma[i-1]:
                if position == 0:
                    position = capital / price
                    capital = 0
                    signals.append(1)
                else:
                    signals.append(0)
            # If Short < Long AND Short[prev] >= Long[prev] -> SELL
            elif short_ma[i] < long_ma[i] and short_ma[i-1] >= long_ma[i-1]:
                if position > 0:
                    capital = position * price
                    position = 0
                    signals.append(-1)
                else:
                    signals.append(0)
            else:
                signals.append(0)
                
            current_val = capital + (position * price)
            portfolio_value.append(current_val)
            
        return signals, portfolio_value
