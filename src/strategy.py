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
